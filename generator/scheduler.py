from __future__ import annotations
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Callable, Iterable, Union, Set
from itertools import cycle, islice
import inspect
from generator.generator import (
    GpuContext,
    Gpr,
    GprRange,
    Vgpr,
    VgprRange,
    Sgpr,
    SgprRange,
    AccVgpr,
    AccVgprRange,
)
from generator.reg_allocator import VirtualGpr


class InstType(Enum):
    VMEM_LOAD = auto()      # Global memory buffer_load
    VMEM_STORE = auto()     # Global memory buffer_store
    LDS_READ = auto()       # Local Data Share ds_read
    LDS_WRITE = auto()      # Local Data Share ds_write
    MFMA_COMPUTE = auto()   # Matrix core compute (v_mfma_*)
    VALU = auto()           # General vector ALU instruction
    SALU = auto()           # General scalar ALU instruction
    WAITCNT = auto()        # Synchronization s_waitcnt
    BARRIER = auto()        # Workgroup barrier s_barrier
    OTHER = auto()


class BufferTokenType(Enum):
    VMEM_VGPR = auto()       # Global load destination VGPR buffer (g_buf[stage])
    LDS_PARTITION = auto()   # LDS memory partition (ping / pong)
    VALU_VGPR = auto()       # Registers feeding MFMA compute (valu_a/valu_b)


class BufferToken:
    """
    Represents a versioned memory or register buffer slot in multi-buffering.
    Used by the DAG scheduler to track cross-iteration RAW and WAR dependencies.
    """
    def __init__(self, token_type: BufferTokenType, slot_id: int, version: int = 0):
        self.token_type = token_type
        self.slot_id = slot_id
        self.version = version

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BufferToken):
            return False
        return (
            self.token_type == other.token_type
            and self.slot_id == other.slot_id
            and self.version == other.version
        )

    def __hash__(self) -> int:
        return hash((self.token_type, self.slot_id, self.version))

    def __repr__(self) -> str:
        return f"BufferToken({self.token_type.name}, slot={self.slot_id}, v={self.version})"


class InstructionNode:
    """
    Represents an instruction in the scheduling graph.
    Tracks def/use registers, instruction category, hardware latency, and buffer tokens.
    """
    def __init__(
        self,
        inst_type: InstType,
        emit_fn: Callable[[GpuContext], None] | Callable[[], None],
        def_regs: Optional[List[Union[Gpr, GprRange, VirtualGpr]]] = None,
        use_regs: Optional[List[Union[Gpr, GprRange, VirtualGpr]]] = None,
        produced_tokens: Optional[List[BufferToken]] = None,
        consumed_tokens: Optional[List[BufferToken]] = None,
        latency: int = 4,
        desc: str = "",
    ):
        self.inst_type = inst_type
        self.emit_fn = emit_fn
        self.def_regs = def_regs or []
        self.use_regs = use_regs or []
        self.produced_tokens = produced_tokens or []
        self.consumed_tokens = consumed_tokens or []
        self.latency = latency
        self.desc = desc

        # Graph connectivity for DAG scheduling
        self.predecessors: Set[InstructionNode] = set()
        self.successors: Set[InstructionNode] = set()
        self.priority: float = 0.0

    def emit(self, ctx: GpuContext):
        sig = inspect.signature(self.emit_fn)
        if len(sig.parameters) == 0:
            self.emit_fn()
        else:
            self.emit_fn(ctx)

    def __repr__(self) -> str:
        return f"InstructionNode({self.inst_type.name}, desc='{self.desc}')"


class SchedulingPolicy(Enum):
    ROUNDROBIN = auto()      # Original legacy round-robin interleaving (100% backward compatible)
    DAG_PIPELINE = auto()    # Dependency DAG + critical path list modulo scheduling
    INTERLEAVED = auto()     # Interleave memory reads, compute, and global loads (latency hiding)
    EARLY_ISSUE = auto()     # Issue all loads up front, wait, then execute compute
    SEQUENTIAL = auto()      # Issue in strictly sequential order


class WaitcntTracker:
    """
    Tracks in-flight VMEM and LDS instructions to optimize and minimize s_waitcnt stalls.
    Instead of inserting s_waitcnt immediately, waits are deferred until the cycle
    right before the destination register is consumed.
    """
    def __init__(self):
        self.active_vmem_loads: int = 0
        self.active_lgkm_ops: int = 0
        # Maps physical/virtual register index to the instruction type producing it
        self.pending_defs: Dict[int, InstType] = {}

    def _get_reg_keys(self, reg: Union[Gpr, GprRange, VirtualGpr]) -> List[int]:
        if isinstance(reg, VirtualGpr):
            if reg.physical_index is not None:
                return list(range(reg.physical_index, reg.physical_index + reg.size))
            return [hash(reg.name)]
        elif isinstance(reg, GprRange):
            return list(range(reg.index, reg.index + reg.size))
        elif isinstance(reg, Gpr):
            return [reg.index]
        return []

    def record_issue(self, node: InstructionNode):
        """Records an issued instruction and tracks its in-flight status."""
        if node.inst_type == InstType.VMEM_LOAD:
            self.active_vmem_loads += 1
            for reg in node.def_regs:
                for k in self._get_reg_keys(reg):
                    self.pending_defs[k] = InstType.VMEM_LOAD
        elif node.inst_type in (InstType.LDS_READ, InstType.LDS_WRITE):
            self.active_lgkm_ops += 1
            if node.inst_type == InstType.LDS_READ:
                for reg in node.def_regs:
                    for k in self._get_reg_keys(reg):
                        self.pending_defs[k] = InstType.LDS_READ
        elif node.inst_type == InstType.WAITCNT:
            pass

    def check_and_emit_wait_for_uses(self, ctx: GpuContext, node: InstructionNode):
        """
        Inspects node.use_regs. If any register is still pending from an in-flight
        load, emits an optimal s_waitcnt before this instruction executes.
        """
        need_vmem_wait = False
        need_lgkm_wait = False

        for reg in node.use_regs:
            for k in self._get_reg_keys(reg):
                producer = self.pending_defs.get(k)
                if producer == InstType.VMEM_LOAD:
                    need_vmem_wait = True
                elif producer == InstType.LDS_READ:
                    need_lgkm_wait = True

        if need_vmem_wait or need_lgkm_wait:
            vmcnt_arg = 0 if need_vmem_wait else None
            lgkmcnt_arg = 0 if need_lgkm_wait else None
            ctx.s_waitcnt(vmcnt=vmcnt_arg, lgkmcnt=lgkmcnt_arg)

            if need_vmem_wait:
                self.active_vmem_loads = 0
                self.pending_defs = {
                    k: v for k, v in self.pending_defs.items() if v != InstType.VMEM_LOAD
                }
            if need_lgkm_wait:
                self.active_lgkm_ops = 0
                self.pending_defs = {
                    k: v for k, v in self.pending_defs.items() if v != InstType.LDS_READ
                }

    def sync_all(self, ctx: GpuContext, vmcnt: bool = True, lgkmcnt: bool = True):
        """Forces synchronization of all outstanding memory operations."""
        v_arg = 0 if (vmcnt and self.active_vmem_loads > 0) else None
        l_arg = 0 if (lgkmcnt and self.active_lgkm_ops > 0) else None
        if v_arg is not None or l_arg is not None:
            ctx.s_waitcnt(vmcnt=v_arg, lgkmcnt=l_arg)
            if v_arg is not None:
                self.active_vmem_loads = 0
                self.pending_defs = {
                    k: v for k, v in self.pending_defs.items() if v != InstType.VMEM_LOAD
                }
            if l_arg is not None:
                self.active_lgkm_ops = 0
                self.pending_defs = {
                    k: v for k, v in self.pending_defs.items() if v != InstType.LDS_READ
                }


class ModuloPipelineScheduler:
    """
    Modulo software pipelining scheduler for GEMM mainloop.
    Supports both the original roundrobin interleaving (for full backward compatibility)
    and dependency-graph (DAG) priority list scheduling.
    """
    def __init__(
        self,
        policy: SchedulingPolicy = SchedulingPolicy.ROUNDROBIN,
        wave_tiling: Tuple[int, int] = (2, 2),
    ):
        self.policy = policy
        self.wave_tiling = wave_tiling

    @staticmethod
    def roundrobin(*iterables: Iterable[Optional[InstructionNode]]) -> Iterable[InstructionNode]:
        """Round-robin interleaves non-empty instructions across multiple streams."""
        iterators = [iter(it) for it in iterables]
        for num_active in range(len(iterators), 0, -1):
            cycle_iter = cycle(islice(iterators, num_active))
            while iterators:
                try:
                    it = next(cycle_iter)
                    item = next(it)
                    if item is not None:
                        yield item
                except StopIteration:
                    iterators = [i for i in iterators if i != it]
                    break

    def build_dag(self, all_nodes: List[InstructionNode]):
        """
        Builds def-use data dependency edges (RAW, WAR, WAW) and BufferToken edges
        between instruction nodes.
        """
        for node in all_nodes:
            node.predecessors.clear()
            node.successors.clear()

        tracker = WaitcntTracker()
        last_def_node: Dict[int, InstructionNode] = {}
        last_use_nodes: Dict[int, List[InstructionNode]] = {}
        token_producer: Dict[BufferToken, InstructionNode] = {}
        token_consumers: Dict[BufferToken, List[InstructionNode]] = {}

        for node in all_nodes:
            # 1. Register RAW dependencies (use after def)
            for reg in node.use_regs:
                for k in tracker._get_reg_keys(reg):
                    if k in last_def_node:
                        pred = last_def_node[k]
                        if pred is not node:
                            node.predecessors.add(pred)
                            pred.successors.add(node)

            # 2. Register WAR dependencies (def after use)
            for reg in node.def_regs:
                for k in tracker._get_reg_keys(reg):
                    if k in last_use_nodes:
                        for pred in last_use_nodes[k]:
                            if pred is not node:
                                node.predecessors.add(pred)
                                pred.successors.add(node)

            # 3. Register WAW dependencies (def after def)
            for reg in node.def_regs:
                for k in tracker._get_reg_keys(reg):
                    if k in last_def_node:
                        pred = last_def_node[k]
                        if pred is not node:
                            node.predecessors.add(pred)
                            pred.successors.add(node)

            # 4. BufferToken RAW (consumed token produced by earlier node)
            for tok in node.consumed_tokens:
                if tok in token_producer:
                    pred = token_producer[tok]
                    if pred is not node:
                        node.predecessors.add(pred)
                        pred.successors.add(node)

            # 5. BufferToken WAR (produced token must not overwrite before consumed)
            for tok in node.produced_tokens:
                if tok in token_consumers:
                    for pred in token_consumers[tok]:
                        if pred is not node:
                            node.predecessors.add(pred)
                            pred.successors.add(node)

            # Update book-keeping
            for reg in node.def_regs:
                for k in tracker._get_reg_keys(reg):
                    last_def_node[k] = node
                    last_use_nodes[k] = []
            for reg in node.use_regs:
                for k in tracker._get_reg_keys(reg):
                    last_use_nodes.setdefault(k, []).append(node)

            for tok in node.produced_tokens:
                token_producer[tok] = node
                token_consumers[tok] = []
            for tok in node.consumed_tokens:
                token_consumers.setdefault(tok, []).append(node)

        # Compute priority scores via reverse topological critical path height
        for node in reversed(all_nodes):
            max_succ_prio = max((s.priority for s in node.successors), default=0.0)
            type_weight = 1.0
            if node.inst_type == InstType.VMEM_LOAD:
                type_weight = 10.0
            elif node.inst_type == InstType.LDS_READ:
                type_weight = 4.0
            node.priority = node.latency * type_weight + max_succ_prio

    def schedule_loop_step(
        self,
        ctx: GpuContext,
        tracker: WaitcntTracker,
        lr_nodes_a: List[InstructionNode],
        lr_nodes_b: List[InstructionNode],
        mfma_nodes: List[InstructionNode],
        gl_nodes: Optional[List[InstructionNode]] = None,
        lw_nodes: Optional[List[InstructionNode]] = None,
    ):
        """
        Schedules a single unrolled K-step using the configured SchedulingPolicy.
        Supports ROUNDROBIN (legacy 100% compatibility), DAG_PIPELINE (priority list DAG),
        and EARLY_ISSUE.
        """
        gl_nodes = gl_nodes or []
        lw_nodes = lw_nodes or []

        if self.policy == SchedulingPolicy.ROUNDROBIN:
            # Original legacy round-robin interleaving
            mfma_iter = iter(mfma_nodes)
            lr_a_iter = iter(lr_nodes_a)
            lr_b_iter = iter(lr_nodes_b)
            gl_iter = iter(gl_nodes)
            lw_iter = iter(lw_nodes)

            for node in self.roundrobin(lr_a_iter, mfma_iter, lr_b_iter, mfma_iter, gl_iter, lw_iter):
                node.emit(ctx)

        elif self.policy in (SchedulingPolicy.DAG_PIPELINE, SchedulingPolicy.INTERLEAVED):
            # Dependency DAG + critical path priority list scheduling
            all_nodes: List[InstructionNode] = lr_nodes_a + lr_nodes_b + mfma_nodes + gl_nodes + lw_nodes
            if not all_nodes:
                return

            self.build_dag(all_nodes)

            in_degrees = {node: len(node.predecessors) for node in all_nodes}
            ready_nodes = [node for node in all_nodes if in_degrees[node] == 0]

            mfma_waited = False
            while ready_nodes:
                # Priority: higher critical-path priority first
                # Tie-breaking heuristic: favor LDS_READ over MFMA to issue reads early
                ready_nodes.sort(
                    key=lambda n: (
                        n.priority,
                        2 if n.inst_type == InstType.LDS_READ else (1 if n.inst_type == InstType.VMEM_LOAD else 0),
                    ),
                    reverse=True,
                )
                node = ready_nodes.pop(0)

                # Prior LDS reads must complete before MFMA compute can safely read operand registers
                if node.inst_type == InstType.MFMA_COMPUTE and not mfma_waited:
                    if lr_nodes_a or lr_nodes_b:
                        expected_lgkmcnt = len(lr_nodes_a) + len(lr_nodes_b)
                        ctx.s_waitcnt(lgkmcnt=expected_lgkmcnt)
                    mfma_waited = True

                # Emit with JIT waitcnt checking
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)

                for succ in node.successors:
                    in_degrees[succ] -= 1
                    if in_degrees[succ] == 0:
                        ready_nodes.append(succ)

        elif self.policy == SchedulingPolicy.EARLY_ISSUE:
            # Issue all memory loads early to maximize flight cycles
            for node in lr_nodes_a + lr_nodes_b + gl_nodes:
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)

            for node in mfma_nodes:
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)

            for node in lw_nodes:
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)

        else:  # SEQUENTIAL
            for node in lr_nodes_a + lr_nodes_b + mfma_nodes + gl_nodes + lw_nodes:
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)
