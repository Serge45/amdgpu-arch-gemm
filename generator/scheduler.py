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
        issue_latency: int = 4,
        desc: str = "",
    ):
        self.inst_type = inst_type
        self.emit_fn = emit_fn
        self.def_regs = def_regs or []
        self.use_regs = use_regs or []
        self.produced_tokens = produced_tokens or []
        self.consumed_tokens = consumed_tokens or []
        self.latency = latency
        self.issue_latency = issue_latency
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
    COLUMN_PIPELINE = auto() # Fine-grained per-column scoreboard pipelining (hides LDS latency)
    EARLY_ISSUE = auto()     # Issue all loads up front, wait, then execute compute
    SEQUENTIAL = auto()      # Issue in strictly sequential order


class InFlightOp:
    def __init__(self, node: InstructionNode, issue_cycle: int, finish_cycle: int, reg_keys: List[int]):
        self.node = node
        self.issue_cycle = issue_cycle
        self.finish_cycle = finish_cycle
        self.reg_keys = reg_keys


class WaitcntTracker:
    """
    Tracks in-flight VMEM and LDS instructions to optimize and minimize s_waitcnt stalls.
    Maintains a virtual cycle scoreboard to determine whether data has already arrived
    in hardware registers, emitting s_waitcnt with exact remaining counters only when
    strictly necessary.
    """
    def __init__(self):
        self.current_cycle: int = 0
        self.in_flight_lgkm: List[InFlightOp] = []
        self.in_flight_vmem: List[InFlightOp] = []
        self.reg_ready_cycle: Dict[int, int] = {}
        self.reg_producer: Dict[int, InFlightOp] = {}
        # Legacy tracking for compatibility
        self.active_vmem_loads: int = 0
        self.active_lgkm_ops: int = 0
        self.pending_defs: Dict[int, InstType] = {}

    def _get_reg_keys(self, reg: Union[Gpr, GprRange, VirtualGpr, str, int]) -> List[Any]:
        if isinstance(reg, str):
            return [reg]
        elif isinstance(reg, int):
            return [f"v{reg}"]
        elif isinstance(reg, VirtualGpr):
            if reg.physical_index is not None:
                return [f"v{reg.physical_index + i}" for i in range(reg.size)]
            return [reg.name]
        elif isinstance(reg, GprRange):
            prefix = "a" if "Acc" in type(reg).__name__ else ("s" if "Sgpr" in type(reg).__name__ else "v")
            return [f"{prefix}{reg.index + i}" for i in range(reg.size)]
        elif isinstance(reg, Gpr):
            prefix = "a" if "Acc" in type(reg).__name__ else ("s" if "Sgpr" in type(reg).__name__ else "v")
            return [f"{prefix}{reg.index}"]
        return []

    def record_issue(self, node: InstructionNode):
        """Records an issued instruction and tracks its in-flight status and latency."""
        self.current_cycle += getattr(node, "issue_latency", 2)
        finish_cycle = self.current_cycle + node.latency
        reg_keys: List[Any] = []
        for reg in node.def_regs:
            reg_keys.extend(self._get_reg_keys(reg))

        if node.inst_type == InstType.VMEM_LOAD:
            op = InFlightOp(node, self.current_cycle, finish_cycle, reg_keys)
            self.in_flight_vmem.append(op)
            for k in reg_keys:
                self.reg_ready_cycle[k] = finish_cycle
                self.reg_producer[k] = op
                self.pending_defs[k] = InstType.VMEM_LOAD
            self.active_vmem_loads = len(self.in_flight_vmem)

        elif node.inst_type in (InstType.LDS_READ, InstType.LDS_WRITE):
            op = InFlightOp(node, self.current_cycle, finish_cycle, reg_keys)
            self.in_flight_lgkm.append(op)
            for k in reg_keys:
                self.reg_ready_cycle[k] = finish_cycle
                self.reg_producer[k] = op
                self.pending_defs[k] = node.inst_type
            self.active_lgkm_ops = len(self.in_flight_lgkm)

        elif node.inst_type == InstType.MFMA_COMPUTE:
            for k in reg_keys:
                self.reg_ready_cycle[k] = finish_cycle

    def check_and_emit_wait_for_uses(self, ctx: GpuContext, node: InstructionNode):
        """
        Inspects node.use_regs. If any register is still pending from an in-flight
        load and has not completed at the current virtual cycle, emits an optimal
        s_waitcnt with the exact remaining counter threshold.
        """
        needed_lgkm_wait: Optional[int] = None
        needed_vmem_wait: Optional[int] = None
        target_cycle = self.current_cycle

        for reg in node.use_regs:
            for k in self._get_reg_keys(reg):
                op = self.reg_producer.get(k)
                if op is not None:
                    target_cycle = max(target_cycle, op.finish_cycle)
                    if op in self.in_flight_lgkm:
                        idx = self.in_flight_lgkm.index(op)
                        remaining = len(self.in_flight_lgkm) - 1 - idx
                        if needed_lgkm_wait is None or remaining < needed_lgkm_wait:
                            needed_lgkm_wait = remaining
                    elif op in self.in_flight_vmem:
                        idx = self.in_flight_vmem.index(op)
                        remaining = len(self.in_flight_vmem) - 1 - idx
                        if needed_vmem_wait is None or remaining < needed_vmem_wait:
                            needed_vmem_wait = remaining
                else:
                    # Fallback if pending_defs was set without InFlightOp
                    p_type = self.pending_defs.get(k)
                    if p_type == InstType.VMEM_LOAD:
                        needed_vmem_wait = 0
                    elif p_type == InstType.LDS_READ:
                        needed_lgkm_wait = 0

        if needed_vmem_wait is not None or needed_lgkm_wait is not None:
            ctx.s_waitcnt(vmcnt=needed_vmem_wait, lgkmcnt=needed_lgkm_wait)
            self.current_cycle = max(self.current_cycle, target_cycle)

            if needed_vmem_wait is not None:
                num_to_retire = len(self.in_flight_vmem) - needed_vmem_wait
                retired = self.in_flight_vmem[:num_to_retire]
                self.in_flight_vmem = self.in_flight_vmem[num_to_retire:]
                for op in retired:
                    for k in op.reg_keys:
                        if self.reg_producer.get(k) is op:
                            del self.reg_producer[k]
                            self.pending_defs.pop(k, None)
                self.active_vmem_loads = len(self.in_flight_vmem)

            if needed_lgkm_wait is not None:
                num_to_retire = len(self.in_flight_lgkm) - needed_lgkm_wait
                retired = self.in_flight_lgkm[:num_to_retire]
                self.in_flight_lgkm = self.in_flight_lgkm[num_to_retire:]
                for op in retired:
                    for k in op.reg_keys:
                        if self.reg_producer.get(k) is op:
                            del self.reg_producer[k]
                            self.pending_defs.pop(k, None)
                self.active_lgkm_ops = len(self.in_flight_lgkm)

    def wait_all_reads_for_nodes(self, ctx: GpuContext, nodes: List[InstructionNode]):
        """
        Batches waitcnt for all in-flight LDS reads consumed by a set of nodes (e.g. all MFMAs in a step).
        Emits AT MOST ONE s_waitcnt lgkmcnt(remaining) instead of separate waitcnts per instruction.
        """
        needed_lgkm_wait: Optional[int] = None
        target_cycle = self.current_cycle
        for node in nodes:
            for reg in node.use_regs:
                for k in self._get_reg_keys(reg):
                    op = self.reg_producer.get(k)
                    if op is not None and op in self.in_flight_lgkm:
                        target_cycle = max(target_cycle, op.finish_cycle)
                        idx = self.in_flight_lgkm.index(op)
                        remaining = len(self.in_flight_lgkm) - 1 - idx
                        if needed_lgkm_wait is None or remaining < needed_lgkm_wait:
                            needed_lgkm_wait = remaining
                    elif self.pending_defs.get(k) == InstType.LDS_READ:
                        needed_lgkm_wait = 0

        if needed_lgkm_wait is not None:
            ctx.s_waitcnt(lgkmcnt=needed_lgkm_wait)
            self.current_cycle = max(self.current_cycle, target_cycle)
            num_to_retire = len(self.in_flight_lgkm) - needed_lgkm_wait
            retired = self.in_flight_lgkm[:num_to_retire]
            self.in_flight_lgkm = self.in_flight_lgkm[num_to_retire:]
            for op in retired:
                for k in op.reg_keys:
                    if self.reg_producer.get(k) is op:
                        del self.reg_producer[k]
                        self.pending_defs.pop(k, None)
            self.active_lgkm_ops = len(self.in_flight_lgkm)

    def sync_all(self, ctx: GpuContext, vmcnt: bool = True, lgkmcnt: bool = True):
        """Forces synchronization of all outstanding memory operations."""
        v_arg = 0 if (vmcnt and len(self.in_flight_vmem) > 0) else None
        l_arg = 0 if (lgkmcnt and len(self.in_flight_lgkm) > 0) else None
        if v_arg is not None or l_arg is not None:
            ctx.s_waitcnt(vmcnt=v_arg, lgkmcnt=l_arg)
            if v_arg is not None:
                for op in self.in_flight_vmem:
                    self.current_cycle = max(self.current_cycle, op.finish_cycle)
                    for k in op.reg_keys:
                        if self.reg_producer.get(k) is op:
                            del self.reg_producer[k]
                            self.pending_defs.pop(k, None)
                self.in_flight_vmem.clear()
                self.active_vmem_loads = 0
            if l_arg is not None:
                for op in self.in_flight_lgkm:
                    self.current_cycle = max(self.current_cycle, op.finish_cycle)
                    for k in op.reg_keys:
                        if self.reg_producer.get(k) is op:
                            del self.reg_producer[k]
                            self.pending_defs.pop(k, None)
                self.in_flight_lgkm.clear()
                self.active_lgkm_ops = 0



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

        elif self.policy == SchedulingPolicy.INTERLEAVED:
            # Batched latency-hiding issue order:
            # 1. Issue next step LDS reads up front to start 40-cycle clock
            for node in (lr_nodes_a + lr_nodes_b):
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)

            # 2. Issue global memory loads if scheduled in this step
            for node in gl_nodes:
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)

            # 3. Issue MFMA compute for current step (hides LDS read latency)
            mfma_waited = False
            for node in mfma_nodes:
                if not mfma_waited:
                    tracker.wait_all_reads_for_nodes(ctx, mfma_nodes)
                    mfma_waited = True
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)

            # 4. Issue LDS writes if scheduled in this step
            for node in lw_nodes:
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)

        elif self.policy == SchedulingPolicy.DAG_PIPELINE:
            # Dependency DAG + critical path priority list scheduling
            all_nodes: List[InstructionNode] = lr_nodes_a + lr_nodes_b + gl_nodes + mfma_nodes + lw_nodes
            if not all_nodes:
                return

            self.build_dag(all_nodes)

            in_degrees = {node: len(node.predecessors) for node in all_nodes}
            ready_nodes = [node for node in all_nodes if in_degrees[node] == 0]

            mfma_waited = False
            while ready_nodes:
                # Priority: higher critical-path priority first
                # Tie-breaking heuristic: favor LDS_READ (3), VMEM_LOAD (2), MFMA (1) to maximize in-flight compute
                ready_nodes.sort(
                    key=lambda n: (
                        n.priority,
                        3 if n.inst_type == InstType.LDS_READ else (2 if n.inst_type == InstType.VMEM_LOAD else (1 if n.inst_type == InstType.MFMA_COMPUTE else 0)),
                    ),
                    reverse=True,
                )
                node = ready_nodes.pop(0)

                if node.inst_type == InstType.MFMA_COMPUTE and not mfma_waited:
                    tracker.wait_all_reads_for_nodes(ctx, mfma_nodes)
                    mfma_waited = True

                # Emit with latency-aware scoreboard waitcnt checking
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)

                for succ in node.successors:
                    in_degrees[succ] -= 1
                    if in_degrees[succ] == 0:
                        ready_nodes.append(succ)

        elif self.policy == SchedulingPolicy.COLUMN_PIPELINE:
            # Fine-grained per-column pipelining:
            # Break mfma_nodes into columns based on wave_tiling (wt0 x wt1).
            # In GEMM loop, MFMAs are generated column-major (j outer 0..wt1-1, i inner 0..wt0-1).
            wt0 = self.wave_tiling[0] if self.wave_tiling and len(self.wave_tiling) >= 1 else 1
            wt1 = self.wave_tiling[1] if self.wave_tiling and len(self.wave_tiling) >= 2 else 1
            if len(mfma_nodes) > 0 and len(mfma_nodes) % wt0 == 0:
                num_cols = len(mfma_nodes) // wt0
            else:
                num_cols = max(1, wt1)
                wt0 = max(1, len(mfma_nodes) // num_cols)

            mfma_cols = [
                mfma_nodes[c * wt0 : (c + 1) * wt0]
                for c in range(num_cols)
            ]

            # Evenly distribute LDS reads across columns to prevent LDS arbiter contention
            all_reads = lr_nodes_a + lr_nodes_b
            col_reads: List[List[InstructionNode]] = [[] for _ in range(num_cols)]
            for idx, r in enumerate(all_reads):
                col_reads[idx % num_cols].append(r)

            # Distribute global memory loads across columns
            col_gl: List[List[InstructionNode]] = [[] for _ in range(num_cols)]
            for idx, gl in enumerate(gl_nodes):
                col_gl[idx % num_cols].append(gl)

            # Distribute LDS writes across columns
            col_lw: List[List[InstructionNode]] = [[] for _ in range(num_cols)]
            for idx, lw in enumerate(lw_nodes):
                col_lw[idx % num_cols].append(lw)

            # Issue column by column:
            for c in range(num_cols):
                # 1. Issue memory operations assigned to this column
                for node in col_reads[c] + col_gl[c] + col_lw[c]:
                    tracker.check_and_emit_wait_for_uses(ctx, node)
                    node.emit(ctx)
                    tracker.record_issue(node)

                # 2. Wait ONLY for reads consumed by this column's MFMAs
                if mfma_cols[c]:
                    tracker.wait_all_reads_for_nodes(ctx, mfma_cols[c])

                # 3. Issue column c MFMAs
                for node in mfma_cols[c]:
                    tracker.check_and_emit_wait_for_uses(ctx, node)
                    node.emit(ctx)
                    tracker.record_issue(node)

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
