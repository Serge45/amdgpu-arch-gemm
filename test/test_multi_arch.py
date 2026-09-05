from __future__ import annotations
import pytest
import struct
import numpy as np
from generator.target_spec import GFX90A, GFX942, GFX1100, RegFileModel
from generator.atoms import MFMA_F32_32x32x2_F32, WMMA_F32_16x16x16_F16
from generator.reg_allocator import RegisterAllocator
from generator.layout import TiledMMA
from generator.generator import GpuContext, VgprRange, DataType
from generator.dsl import GemmKernel, Tensor, LayoutType
from vm.gcn_virtual_machine import GcnVirtualMachine


def test_cdna3_unified_register_spec():
    """
    Verify CDNA3 (GFX942) unified register file model and allocations.
    """
    assert GFX942.wavefront_size == 64
    assert GFX942.reg_file_model == RegFileModel.UNIFIED
    assert GFX942.max_vgpr == 512

    alloc = RegisterAllocator(target=GFX942, max_vgpr_budget=256)
    v_data = alloc.alloc_vgpr("v_big", size=256)
    assert v_data.physical_index == 0

    diag = alloc.get_diagnostics()
    assert diag["target"] == "gfx942"
    assert diag["vgpr_used"] == 256
    assert not diag["vgpr_budget_exceeded"]


def test_rdna3_wmma_wave32_atom():
    """
    Verify RDNA3 (GFX1100) target spec and WMMA atom characteristics.
    """
    assert GFX1100.wavefront_size == 32
    assert GFX1100.reg_file_model == RegFileModel.VGPR_ONLY
    assert GFX1100.max_agpr == 0

    atom = WMMA_F32_16x16x16_F16()
    assert atom.shape == (16, 16, 1, 16)
    assert atom.dest_reg_type.__name__ == "Vgpr"  # Pure VGPR, no AGPR

    # Thread coordinates for 32 threads
    # Thread 0: row = 0, col = 0
    assert atom.get_thread_coords_a(0) == (0, 0)
    # Thread 15: row = 15, col = 0
    assert atom.get_thread_coords_a(15) == (15, 0)
    # Thread 16: row = 0, col = 8
    assert atom.get_thread_coords_a(16) == (0, 8)
    # Thread 31: row = 15, col = 8
    assert atom.get_thread_coords_a(31) == (15, 8)


def test_rdna3_wmma_emulation_on_wave32_vm():
    """
    Verify RDNA3 Wave32 WMMA instruction emulation in GcnVirtualMachine.
    Computes D = A @ B + C for 16x16 matrices on 32 threads.
    """
    # Create random 16x16 FP16 matrices A and B, and FP32 matrix C
    np.random.seed(42)
    mat_a = np.random.randn(16, 16).astype(np.float16).astype(np.float32)
    mat_b = np.random.randn(16, 16).astype(np.float16).astype(np.float32)
    mat_c = np.random.randn(16, 16).astype(np.float32)

    vm = GcnVirtualMachine(
        num_total_sgpr=GFX1100.max_sgpr,
        num_total_vgpr=GFX1100.max_vgpr,
        wavefront_size=32,  # Wave32 mode
    )

    # Pack Matrix A into v[0:3] (4 VGPRs per thread)
    for lane in range(32):
        row = lane % 16
        k_base = (lane // 16) * 8
        for i in range(4):
            val0 = struct.pack("e", float(mat_a[row, k_base + i * 2]))
            val1 = struct.pack("e", float(mat_a[row, k_base + i * 2 + 1]))
            u32 = int.from_bytes(val0 + val1, "little")
            vm.v[i][lane] = u32

    # Pack Matrix B into v[4:7] (4 VGPRs per thread)
    for lane in range(32):
        col = lane % 16
        k_base = (lane // 16) * 8
        for i in range(4):
            val0 = struct.pack("e", float(mat_b[k_base + i * 2, col]))
            val1 = struct.pack("e", float(mat_b[k_base + i * 2 + 1, col]))
            u32 = int.from_bytes(val0 + val1, "little")
            vm.v[4 + i][lane] = u32

    # Pack Matrix C into v[8:15] (8 VGPRs per thread)
    for lane in range(32):
        row = lane % 16
        col_base = (lane // 16) * 8
        for i in range(8):
            val_f = mat_c[row, col_base + i]
            u32 = int.from_bytes(struct.pack("f", float(val_f)), "little")
            vm.v[8 + i][lane] = u32

    # Emit WMMA instruction
    ctx = GpuContext()
    ctx.v_wmma_f32_16x16x16_f16(
        dst=VgprRange(16, 8),
        src0=VgprRange(0, 4),
        src1=VgprRange(4, 4),
        src2=VgprRange(8, 8),
    )

    # Run on Wave32 virtual machine
    vm.run(ctx)

    # Read back matrix D from v[16:23]
    mat_d = np.zeros((16, 16), dtype=np.float32)
    for lane in range(32):
        row = lane % 16
        col_base = (lane // 16) * 8
        for i in range(8):
            val_u32 = vm.v[16 + i][lane]
            mat_d[row, col_base + i] = struct.unpack("f", int.to_bytes(val_u32, 4, "little"))[0]

    # Verify against reference calculation
    ref = mat_a @ mat_b + mat_c
    assert np.allclose(mat_d, ref, atol=1e-2)


def test_multi_arch_dsl_tiling():
    """
    Verify TiledMMA adaptation when targeting RDNA3 (Wave32) vs CDNA (Wave64).
    """
    atom_rdna = WMMA_F32_16x16x16_F16()
    tiled_rdna = TiledMMA(
        atom=atom_rdna,
        wave_group=(2, 2),
        wave_tiling=(2, 2),
        wavefront_size=32,  # RDNA Wave32
    )

    # 16 * 2 * 2 = 64
    assert tiled_rdna.block_tile == (64, 64)
    assert tiled_rdna.num_waves == 4
    assert tiled_rdna.num_threads == 128  # 4 * 32 = 128 threads!
    assert tiled_rdna.atom.dest_reg_type.__name__ == "Vgpr"
