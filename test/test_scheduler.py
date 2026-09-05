from __future__ import annotations
import pytest
from generator.scheduler import (
    InstructionNode,
    InstType,
    SchedulingPolicy,
    WaitcntTracker,
    ModuloPipelineScheduler,
)
from generator.generator import (
    GpuContext,
    Vgpr,
    VgprRange,
    Sgpr,
    SgprRange,
    AccVgpr,
    AccVgprRange,
)
from generator.atoms import MFMA_F32_32x32x2_F32
from generator.target_spec import GFX90A
from vm.gcn_virtual_machine import GcnVirtualMachine


def test_waitcnt_tracker_deferred_insertion():
    """
    Verify that WaitcntTracker tracks in-flight defs and inserts s_waitcnt
    deferred right before the register is consumed.
    """
    ctx = GpuContext()
    tracker = WaitcntTracker()

    dst_vmem = Vgpr(0)
    dst_lds = Vgpr(1)
    acc = AccVgprRange(0, 16)

    # 1. Global VMEM load into dst_vmem
    load_node = InstructionNode(
        inst_type=InstType.VMEM_LOAD,
        emit_fn=lambda c: c.v_mov_b32(dst_vmem, 1.0),
        def_regs=[dst_vmem],
        desc="vmem_load",
    )
    tracker.check_and_emit_wait_for_uses(ctx, load_node)
    load_node.emit(ctx)
    tracker.record_issue(load_node)

    assert tracker.active_vmem_loads == 1

    # 2. Independent instruction (no waitcnt should be emitted yet!)
    unrelated_node = InstructionNode(
        inst_type=InstType.VALU,
        emit_fn=lambda c: c.v_mov_b32(Vgpr(2), 2.0),
        def_regs=[Vgpr(2)],
        use_regs=[],
        desc="unrelated",
    )
    tracker.check_and_emit_wait_for_uses(ctx, unrelated_node)
    unrelated_node.emit(ctx)
    tracker.record_issue(unrelated_node)

    # Check instruction count: only v_mov_b32 instructions, NO s_waitcnt yet
    assert not any("s_waitcnt" in inst[0]() for inst in ctx.instructions)

    # 3. Instruction that consumes dst_vmem -> s_waitcnt vmcnt=0 MUST be emitted!
    consumer_node = InstructionNode(
        inst_type=InstType.LDS_WRITE,
        emit_fn=lambda c: c.ds_write_b32(Vgpr(2), dst_vmem, 0),
        def_regs=[],
        use_regs=[dst_vmem],
        desc="ds_write using vmem data",
    )
    tracker.check_and_emit_wait_for_uses(ctx, consumer_node)
    consumer_node.emit(ctx)
    tracker.record_issue(consumer_node)

    # Verify that s_waitcnt was inserted immediately before ds_write
    waitcnt_insts = [inst for inst in ctx.instructions if "s_waitcnt" in inst[0]()]
    assert len(waitcnt_insts) == 1
    assert "vmcnt(0)" in waitcnt_insts[0][0]()
    assert tracker.active_vmem_loads == 0


def test_interleaved_scheduling_order():
    """
    Verify that Interleaved scheduling interleaves LDS reads before/with MFMAs.
    """
    ctx = GpuContext()
    tracker = WaitcntTracker()
    scheduler = ModuloPipelineScheduler(policy=SchedulingPolicy.INTERLEAVED)

    lr_nodes_a = [
        InstructionNode(
            inst_type=InstType.LDS_READ,
            emit_fn=lambda c: c.comment("lr_a_0"),
            def_regs=[Vgpr(0)],
            desc="lr_a_0",
        )
    ]
    lr_nodes_b = [
        InstructionNode(
            inst_type=InstType.LDS_READ,
            emit_fn=lambda c: c.comment("lr_b_0"),
            def_regs=[Vgpr(1)],
            desc="lr_b_0",
        )
    ]
    mfma_nodes = [
        InstructionNode(
            inst_type=InstType.MFMA_COMPUTE,
            emit_fn=lambda c: c.comment("mfma_0"),
            def_regs=[AccVgprRange(0, 16)],
            use_regs=[Vgpr(0), Vgpr(1)],
            desc="mfma_0",
        )
    ]

    scheduler.schedule_loop_step(
        ctx=ctx,
        tracker=tracker,
        lr_nodes_a=lr_nodes_a,
        lr_nodes_b=lr_nodes_b,
        mfma_nodes=mfma_nodes,
    )

    inst_texts = [inst[0]() for inst in ctx.instructions]
    # Check that lr_a is issued before mfma
    lr_a_idx = next(i for i, t in enumerate(inst_texts) if "lr_a_0" in t)
    mfma_idx = next(i for i, t in enumerate(inst_texts) if "mfma_0" in t)
    assert lr_a_idx < mfma_idx


def test_scheduler_end_to_end_vm():
    """
    Verify full scheduled sequence running on GcnVirtualMachine.
    """
    ctx = GpuContext()
    tracker = WaitcntTracker()
    scheduler = ModuloPipelineScheduler(policy=SchedulingPolicy.INTERLEAVED)
    atom = MFMA_F32_32x32x2_F32()

    # Setup initial registers
    ctx.v_mov_b32(Vgpr(0), 1.5)
    ctx.v_mov_b32(Vgpr(1), 2.0)
    for i in range(16):
        ctx.v_accvgpr_write_b32(AccVgpr(i), 0.0)

    # Build scheduled nodes
    mfma_nodes = [
        InstructionNode(
            inst_type=InstType.MFMA_COMPUTE,
            emit_fn=lambda c: atom.emit(
                c,
                dst=AccVgprRange(0, 16),
                src_a=Vgpr(0),
                src_b=Vgpr(1),
                src_c=AccVgprRange(0, 16),
            ),
            def_regs=[AccVgprRange(0, 16)],
            use_regs=[Vgpr(0), Vgpr(1)],
            desc="mfma",
        )
    ]

    scheduler.schedule_loop_step(
        ctx=ctx,
        tracker=tracker,
        lr_nodes_a=[],
        lr_nodes_b=[],
        mfma_nodes=mfma_nodes,
    )

    # Run on GcnVirtualMachine
    vm = GcnVirtualMachine(
        num_total_sgpr=GFX90A.max_sgpr,
        num_total_vgpr=GFX90A.max_vgpr,
        wavefront_size=GFX90A.wavefront_size,
    )
    vm.run(ctx)

    # Verify computation result
    for tid in range(GFX90A.wavefront_size):
        for j in range(16):
            assert vm.a[j][tid] != 0


def test_buffer_token_and_dag_dependencies():
    """
    Verify BufferToken tracking and DAG edge generation for multi-buffer hazards.
    """
    from generator.scheduler import BufferToken, BufferTokenType

    tok_pong = BufferToken(BufferTokenType.LDS_PARTITION, slot_id=1, version=0)
    tok_ping = BufferToken(BufferTokenType.LDS_PARTITION, slot_id=0, version=0)

    # 1. Producer writes to LDS pong
    node_lw = InstructionNode(
        inst_type=InstType.LDS_WRITE,
        emit_fn=lambda c: c.comment("lw_pong"),
        produced_tokens=[tok_pong],
        desc="lw_pong",
    )
    # 2. Consumer reads from LDS pong
    node_lr = InstructionNode(
        inst_type=InstType.LDS_READ,
        emit_fn=lambda c: c.comment("lr_pong"),
        consumed_tokens=[tok_pong],
        desc="lr_pong",
    )
    # 3. Next iteration writes to LDS ping after reading ping
    node_lr_ping = InstructionNode(
        inst_type=InstType.LDS_READ,
        emit_fn=lambda c: c.comment("lr_ping"),
        consumed_tokens=[tok_ping],
        desc="lr_ping",
    )
    node_lw_ping = InstructionNode(
        inst_type=InstType.LDS_WRITE,
        emit_fn=lambda c: c.comment("lw_ping"),
        produced_tokens=[tok_ping],
        desc="lw_ping",
    )

    scheduler = ModuloPipelineScheduler(policy=SchedulingPolicy.DAG_PIPELINE)
    all_nodes = [node_lw, node_lr, node_lr_ping, node_lw_ping]
    scheduler.build_dag(all_nodes)

    # Verify RAW dependency: node_lw -> node_lr
    assert node_lw in node_lr.predecessors
    assert node_lr in node_lw.successors

    # Verify WAR dependency: node_lr_ping -> node_lw_ping
    assert node_lr_ping in node_lw_ping.predecessors
    assert node_lw_ping in node_lr_ping.successors


def test_roundrobin_policy_mode():
    """
    Verify that SchedulingPolicy.ROUNDROBIN correctly preserves the legacy round-robin interleaving.
    """
    ctx = GpuContext()
    tracker = WaitcntTracker()
    scheduler = ModuloPipelineScheduler(policy=SchedulingPolicy.ROUNDROBIN)

    lr_nodes = [
        InstructionNode(InstType.LDS_READ, emit_fn=lambda c: c.comment("lr_0")),
        InstructionNode(InstType.LDS_READ, emit_fn=lambda c: c.comment("lr_1")),
    ]
    mfma_nodes = [
        InstructionNode(InstType.MFMA_COMPUTE, emit_fn=lambda c: c.comment("mfma_0")),
        InstructionNode(InstType.MFMA_COMPUTE, emit_fn=lambda c: c.comment("mfma_1")),
    ]

    scheduler.schedule_loop_step(
        ctx=ctx,
        tracker=tracker,
        lr_nodes_a=lr_nodes,
        lr_nodes_b=[],
        mfma_nodes=mfma_nodes,
    )

    comments = [inst[0]() for inst in ctx.instructions if "//" in inst[0]()]
    assert comments == ["//lr_0", "//mfma_0", "//lr_1", "//mfma_1"]

