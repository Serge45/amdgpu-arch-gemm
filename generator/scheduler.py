from __future__ import annotations
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Callable, Iterable, Union, Set
from itertools import cycle, islice
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


class InstructionNode:
    """
    Represents an instruction in the scheduling graph.
    Tracks def/use registers, instruction category, and hardware latency.
    """
    def __init__(
        self,
        inst_type: InstType,
        emit_fn: Callable[[GpuContext], None],
        def_regs: Optional[List[Union[Gpr, GprRange, VirtualGpr]]] = None,
        use_regs: Optional[List[Union[Gpr, GprRange, VirtualGpr]]] = None,
        latency: int = 4,
        desc: str = "",
    ):
        self.inst_type = inst_type
        self.emit_fn = emit_fn
        self.def_regs = def_regs or []
        self.use_regs = use_regs or []
        self.latency = latency
        self.desc = desc

    def emit(self, ctx: GpuContext):
        self.emit_fn(ctx)

    def __repr__(self) -> str:
        return f"InstructionNode({self.inst_type.name}, desc='{self.desc}')"


class SchedulingPolicy(Enum):
    INTERLEAVED = auto()    # Interleave memory reads, compute, and global loads (latency hiding)
    EARLY_ISSUE = auto()    # Issue all loads up front, wait, then execute compute
    SEQUENTIAL = auto()     # Issue in strictly sequential order


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
            pass  # waitcnt adjustments handled separately

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
                # Clear pending vmem defs
                self.pending_defs = {
                    k: v for k, v in self.pending_defs.items() if v != InstType.VMEM_LOAD
                }
            if need_lgkm_wait:
                self.active_lgkm_ops = 0
                # Clear pending lds defs
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
    Interleaves memory transfers (Global Load, LDS Read, LDS Write) with
    matrix arithmetic (MFMA) to maximize memory latency hiding.
    """
    def __init__(
        self,
        policy: SchedulingPolicy = SchedulingPolicy.INTERLEAVED,
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
        Schedules a single unrolled K-step using the chosen latency hiding strategy.
        In CDNA architectures, issuing LDS reads BEFORE MFMAs allows the multi-cycle
        MFMA execution to hide LDS memory latency.
        """
        gl_nodes = gl_nodes or []
        lw_nodes = lw_nodes or []

        if self.policy == SchedulingPolicy.INTERLEAVED:
            # Optimal order: interleave LDS reads, MFMA compute, and global loads
            scheduled_nodes: List[InstructionNode] = []
            
            # 1. Round-robin interleave LDS read, MFMA, and Global loads
            mfma_iter = iter(mfma_nodes)
            lr_a_iter = iter(lr_nodes_a)
            lr_b_iter = iter(lr_nodes_b)
            gl_iter = iter(gl_nodes)
            lw_iter = iter(lw_nodes)

            for node in self.roundrobin(lr_a_iter, mfma_iter, lr_b_iter, mfma_iter, gl_iter, lw_iter):
                scheduled_nodes.append(node)

            # Emit instructions with deferred waitcnt checking
            for node in scheduled_nodes:
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)

        elif self.policy == SchedulingPolicy.EARLY_ISSUE:
            # Issue all memory loads early to maximize flight cycles
            for node in lr_nodes_a + lr_nodes_b + gl_nodes:
                tracker.check_and_emit_wait_for_uses(ctx, node)
                node.emit(ctx)
                tracker.record_issue(node)

            # Wait for LDS read before MFMA compute consumes operands
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
