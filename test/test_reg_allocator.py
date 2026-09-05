from __future__ import annotations
import pytest
from generator.target_spec import GFX90A, GFX942, GFX1100
from generator.reg_allocator import (
    RegisterAllocator,
    VirtualVgpr,
    VirtualSgpr,
    VirtualAccVgpr,
)
from generator.generator import GpuContext, Vgpr, Sgpr, AccVgpr, VgprRange, SgprRange
from vm.gcn_virtual_machine import GcnVirtualMachine


def test_vgpr_2dword_even_alignment():
    """
    Verify that multi-dword VGPRs (size >= 2, including dwordx4) are strictly
    2-dword (even) aligned, and do not over-align to 4-dword.
    Also verify that alignment holes can be backfilled.
    """
    alloc = RegisterAllocator(target=GFX90A)

    # 1. Allocate a single dword VGPR -> should be index 0
    v0 = alloc.alloc_vgpr("v0", size=1)
    assert v0.physical_index == 0

    # 2. Allocate a 4-dword VGPR (dwordx4) -> skips odd index 1 and aligns to index 2 (even alignment),
    # NOT 4! This verifies no wasteful 4-dword over-alignment.
    v_x4 = alloc.alloc_vgpr("v_x4", size=4)
    assert v_x4.physical_index == 2
    assert v_x4.physical_index % 2 == 0
    assert v_x4.size == 4

    # 3. Allocate a 2-dword VGPR -> occupies index 6 (even: [6, 7])
    v_x2 = alloc.alloc_vgpr("v_x2", size=2)
    assert v_x2.physical_index == 6
    assert v_x2.physical_index % 2 == 0

    # 4. Allocate a 1-dword VGPR -> should backfill the alignment hole at index 1!
    v_hole = alloc.alloc_vgpr("v_hole", size=1)
    assert v_hole.physical_index == 1

    # 5. Allocate an 8-dword VGPR -> next available contiguous even chunk starts at index 8
    v_x8 = alloc.alloc_vgpr("v_x8", size=8)
    assert v_x8.physical_index == 8
    assert v_x8.physical_index % 2 == 0


def test_sgpr_srd_4dword_alignment():
    """
    Verify SGPR alignment:
    - size=1 -> align 1
    - size=2 -> align 2
    - size=4 (SRD) -> align 4 (s[4i:4i+3])
    - Holes at [2, 3] are successfully reused by 2-dword allocations
    """
    alloc = RegisterAllocator(target=GFX90A)

    s0 = alloc.alloc_sgpr("s0", size=1)
    assert s0.physical_index == 0

    # Next SRD (size=4) requires 4-dword alignment -> index 4 (skips 1, 2, 3)
    srd0 = alloc.alloc_sgpr("srd0", size=4)
    assert srd0.physical_index == 4
    assert srd0.physical_index % 4 == 0

    # size=2 requires 2-dword alignment -> successfully backfills free slot [2, 3]!
    s_ptr = alloc.alloc_sgpr("s_ptr", size=2)
    assert s_ptr.physical_index == 2
    assert s_ptr.physical_index % 2 == 0

    # size=1 backfills free slot 1!
    s1 = alloc.alloc_sgpr("s1", size=1)
    assert s1.physical_index == 1

    # Next 4-dword SRD goes to index 8
    srd1 = alloc.alloc_sgpr("srd1", size=4)
    assert srd1.physical_index == 8
    assert srd1.physical_index % 4 == 0


def test_scope_recycling_reuse():
    """
    Verify that temporary registers allocated within a scope are reclaimed upon exit,
    and reused by subsequent allocations.
    """
    alloc = RegisterAllocator(target=GFX90A)

    # Base allocation
    base_v = alloc.alloc_vgpr("base", size=4)
    assert base_v.physical_index == 0

    # Prologue scope
    with alloc.scope("prologue"):
        tmp0 = alloc.alloc_vgpr("tmp0", size=4)
        tmp1 = alloc.alloc_vgpr("tmp1", size=4)
        assert tmp0.physical_index == 4
        assert tmp1.physical_index == 8

    # Exiting scope: tmp0 and tmp1 should be freed.
    # New allocation in mainloop should reuse index 4!
    gl_data = alloc.alloc_vgpr("gl_data", size=4)
    assert gl_data.physical_index == 4


def test_explicit_aliasing():
    """
    Verify explicit aliasing (e.g., Epilogue reusing Global Load registers).
    """
    alloc = RegisterAllocator(target=GFX90A)

    gl_buf = alloc.alloc_vgpr("gl_buf", size=4)
    assert gl_buf.physical_index == 0

    # Epilogue aliases gl_buf
    epilogue_c = alloc.alias_vgpr("epilogue_c", gl_buf)
    assert epilogue_c.physical_index == gl_buf.physical_index
    assert epilogue_c.size == gl_buf.size


def test_diagnostics_and_occupancy():
    """
    Verify occupancy calculations on CDNA architecture.
    """
    alloc = RegisterAllocator(target=GFX90A, max_vgpr_budget=128)

    # Allocate 64 VGPRs
    alloc.alloc_vgpr("data", size=64)
    diag = alloc.get_diagnostics()
    assert diag["vgpr_used"] == 64
    assert diag["estimated_waves_per_simd"] == 4
    assert not diag["vgpr_budget_exceeded"]

    # Allocate another 64 VGPRs (total 128)
    alloc.alloc_vgpr("data2", size=64)
    diag = alloc.get_diagnostics()
    assert diag["vgpr_used"] == 128
    assert diag["estimated_waves_per_simd"] == 4
    assert not diag["vgpr_budget_exceeded"]

    # Allocate 1 more VGPR (total 130 because size=2 align=2)
    alloc.alloc_vgpr("data3", size=2)
    diag = alloc.get_diagnostics()
    assert diag["vgpr_used"] == 130
    assert diag["estimated_waves_per_simd"] == 2
    assert diag["vgpr_budget_exceeded"]


def test_lowering_and_vm_integration():
    """
    Verify lowering virtual registers to concrete Vgpr/Sgpr types and executing
    on the GcnVirtualMachine.
    """
    alloc = RegisterAllocator(target=GFX90A)

    v_dst = alloc.alloc_vgpr("v_dst", size=1)
    v_src = alloc.alloc_vgpr("v_src", size=1)
    s_val = alloc.alloc_sgpr("s_val", size=1)

    # Lower to concrete types
    real_v_dst = alloc.lower(v_dst)
    real_v_src = alloc.lower(v_src)
    real_s_val = alloc.lower(s_val)

    assert isinstance(real_v_dst, Vgpr)
    assert isinstance(real_v_src, Vgpr)
    assert isinstance(real_s_val, Sgpr)

    # Emit instructions using GpuContext
    context = GpuContext()
    context.s_mov_b32(real_s_val, 10)
    context.v_mov_b32(real_v_src, 5)
    context.v_add_u32(real_v_dst, real_v_src, real_s_val)

    # Run on GcnVirtualMachine
    vm = GcnVirtualMachine(
        num_total_sgpr=GFX90A.max_sgpr,
        num_total_vgpr=GFX90A.max_vgpr,
        wavefront_size=GFX90A.wavefront_size,
    )
    vm.run(context)

    # Verify simulated results across all 64 threads
    assert vm.s[real_s_val.index] == 10
    for tid in range(GFX90A.wavefront_size):
        assert vm.v[real_v_src.index][tid] == 5
        assert vm.v[real_dst_idx := real_v_dst.index][tid] == 15
