from __future__ import annotations
import pytest
from generator.atoms import (
    MFMA_F32_32x32x2_F32,
    MFMA_F32_16x16x4_F32,
    BufferLoadAtom,
    DsWriteAtom,
    DsReadAtom,
)
from generator.layout import LdsLayout, LdsPaddingSolver
from generator.generator import (
    GpuContext,
    Vgpr,
    Sgpr,
    AccVgpr,
    VgprRange,
    SgprRange,
    AccVgprRange,
    DataType,
    GemmSolutionConfig,
)
from generator.target_spec import GFX90A
from vm.gcn_virtual_machine import GcnVirtualMachine


def test_mfma_32x32x2_thread_coordinates():
    atom = MFMA_F32_32x32x2_F32()
    assert atom.shape == (32, 32, 1, 2)
    assert atom.exec_cycles == 16
    assert atom.issue_cycles == 2

    # Matrix A coordinates: row = tid % 32, col = tid // 32
    assert atom.get_thread_coords_a(0) == (0, 0)
    assert atom.get_thread_coords_a(31) == (31, 0)
    assert atom.get_thread_coords_a(32) == (0, 1)
    assert atom.get_thread_coords_a(63) == (31, 1)

    # Matrix B coordinates: row = tid // 32, col = tid % 32
    assert atom.get_thread_coords_b(0) == (0, 0)
    assert atom.get_thread_coords_b(31) == (0, 31)
    assert atom.get_thread_coords_b(32) == (1, 0)
    assert atom.get_thread_coords_b(63) == (1, 31)


def test_mfma_16x16x4_thread_coordinates():
    atom = MFMA_F32_16x16x4_F32()
    assert atom.shape == (16, 16, 1, 4)

    # Matrix A coordinates: row = tid % 16, col = tid // 16
    assert atom.get_thread_coords_a(0) == (0, 0)
    assert atom.get_thread_coords_a(15) == (15, 0)
    assert atom.get_thread_coords_a(16) == (0, 1)
    assert atom.get_thread_coords_a(63) == (15, 3)

    # Matrix B coordinates: row = tid // 16, col = tid % 16
    assert atom.get_thread_coords_b(0) == (0, 0)
    assert atom.get_thread_coords_b(15) == (0, 15)
    assert atom.get_thread_coords_b(16) == (1, 0)
    assert atom.get_thread_coords_b(63) == (3, 15)


def test_lds_layout_addressing():
    # Leading dim 128, padding 4, element_bytes 4
    layout = LdsLayout(leading_dim=128, padding=4, element_bytes=4)
    assert layout.stride == 132

    # Coordinate (0, 0) -> offset 0
    assert layout.get_byte_offset(0, 0) == 0
    # Coordinate (row=4, col=1) -> (1 * 132 + 4) * 4 = 136 * 4 = 544
    assert layout.get_byte_offset(4, 1) == 544


def test_lds_padding_solver_equivalence():
    """
    Verify that LdsPaddingSolver produces the identical padding and 16-byte alignment
    as GemmSolutionConfig._auto_lds_pad_a and _auto_lds_pad_b in generator.py.
    """
    atom = MFMA_F32_32x32x2_F32()
    config = GemmSolutionConfig(
        a_type=DataType.FP32,
        b_type=DataType.FP32,
        cd_type=DataType.FP32,
        scalar_type=DataType.FP32,
        mfma=(32, 32, 2, 2),
        wave_group=(2, 2),
        wave_tiling=(2, 2),
        depth_k=16,
        trans_a=False,
        trans_b=False,
    )

    pad_a, conf_a = LdsPaddingSolver.solve_pad_a(atom, tile_m=config.tile_size[0])
    pad_b, conf_b = LdsPaddingSolver.solve_pad_b(atom, depth_k=config.depth_k)

    expected_pad_a = config._auto_lds_pad_a()
    expected_pad_b = config._auto_lds_pad_b()

    assert pad_a == expected_pad_a
    assert pad_b == expected_pad_b

    # Verify that strides are multiples of 4 elements (16 bytes)
    assert (config.tile_size[0] + pad_a) % 4 == 0
    assert (config.depth_k + pad_b) % 4 == 0


def test_atoms_with_gpu_context_and_vm():
    """
    Verify emission of MMAAtom and CopyAtom instructions into GpuContext,
    and simulate on GcnVirtualMachine.
    """
    ctx = GpuContext()
    atom = MFMA_F32_32x32x2_F32()
    load_atom = BufferLoadAtom(vector_dwords=4)
    write_atom = DsWriteAtom(vector_dwords=4)
    read_atom = DsReadAtom(vector_dwords=4)

    # Initialize registers
    ctx.v_mov_b32(Vgpr(0), 1.0)
    ctx.v_mov_b32(Vgpr(1), 2.0)
    for i in range(16):
        ctx.v_accvgpr_write_b32(AccVgpr(i), 0.0)

    # Emit MFMA with 16-register accumulator range
    atom.emit(
        ctx,
        dst=AccVgprRange(0, 16),
        src_a=Vgpr(0),
        src_b=Vgpr(1),
        src_c=AccVgprRange(0, 16),
    )

    vm = GcnVirtualMachine(
        num_total_sgpr=GFX90A.max_sgpr,
        num_total_vgpr=GFX90A.max_vgpr,
        wavefront_size=GFX90A.wavefront_size,
    )
    vm.run(ctx)

    # Verify that AccVgprRange was computed by the VM
    for tid in range(GFX90A.wavefront_size):
        for j in range(16):
            assert vm.a[j][tid] != 0


def test_tiled_mma_and_tiled_copy():
    """
    Verify hierarchical tiling math for TiledMMA and load partitioning for TiledCopy.
    """
    from generator.layout import TiledMMA, TiledCopy

    atom = MFMA_F32_32x32x2_F32()
    tiled_mma = TiledMMA(
        atom=atom,
        wave_group=(2, 2),
        wave_tiling=(2, 2),
        wavefront_size=64,
    )

    assert tiled_mma.atom_tile == (32, 32)
    assert tiled_mma.wave_tile == (64, 64)
    assert tiled_mma.block_tile == (128, 128)
    assert tiled_mma.num_waves == 4
    assert tiled_mma.num_threads == 256
    assert tiled_mma.get_wave_coords(0) == (0, 0)
    assert tiled_mma.get_wave_coords(1) == (1, 0)
    assert tiled_mma.get_wave_coords(2) == (0, 1)
    assert tiled_mma.get_wave_coords(3) == (1, 1)

    # Test TiledCopy with 128x16 tile, vector_bytes=16 (dwordx4), 256 workitems
    tiled_copy_a = TiledCopy(
        vector_bytes=16,
        tile_dim=128,
        depth_k=16,
        num_workitems=256,
        element_bytes=4,
    )
    # 128 * 4 = 512 bytes. 16 * 256 = 4096 bytes per workgroup wave load.
    # num_loads_0 = max(512 // 4096, 1) = 1
    # loads_per_row = 256 // (512 // 16) = 256 // 32 = 8
    # num_loads_1 = 16 // 8 = 2
    assert tiled_copy_a.num_loads_0 == 1
    assert tiled_copy_a.num_loads_1 == 2


def test_tiled_mma_agpr_validation():
    """
    Verify that TiledMMA computes accurate accumulator register usage and
    strictly validates against target AGPR limits.
    """
    from generator.layout import TiledMMA

    atom_32 = MFMA_F32_32x32x2_F32()
    assert atom_32.num_acc_regs == 16

    # 1. Valid wave_tiling (2, 2) on GFX90A: 2 * 2 * 16 = 64 AGPRs <= 256
    tiled_valid = TiledMMA(
        atom=atom_32,
        wave_group=(2, 2),
        wave_tiling=(2, 2),
        target=GFX90A,
    )
    assert tiled_valid.num_acc_regs_per_thread == 64

    # 2. Maximum safe rectangular wave_tiling (16, 1): 16 * 1 * 16 = 256 AGPRs <= 256
    tiled_max = TiledMMA(
        atom=atom_32,
        wave_group=(1, 1),
        wave_tiling=(16, 1),
        target=GFX90A,
    )
    assert tiled_max.num_acc_regs_per_thread == 256

    # 3. Invalid wave_tiling (8, 4) exceeding AGPR file: 32 * 16 = 512 AGPRs > 256
    with pytest.raises(ValueError, match="exceeds target 'gfx90a' limit of 256"):
        TiledMMA(
            atom=atom_32,
            wave_group=(1, 1),
            wave_tiling=(8, 4),
            target=GFX90A,
        )

    # 4. Valid wave_tiling (16, 4) with 16x16 atom: 16 * 4 * 4 = 256 AGPRs <= 256
    atom_16 = MFMA_F32_16x16x4_F32()
    assert atom_16.num_acc_regs == 4
    tiled_16_valid = TiledMMA(
        atom=atom_16,
        wave_group=(1, 1),
        wave_tiling=(16, 4),
        target=GFX90A,
    )
    assert tiled_16_valid.num_acc_regs_per_thread == 256

    # 5. Invalid wave_tiling (16, 8) with 16x16 atom: 128 * 4 = 512 AGPRs > 256
    with pytest.raises(ValueError, match="exceeds target 'gfx90a' limit of 256"):
        TiledMMA(
            atom=atom_16,
            wave_group=(1, 1),
            wave_tiling=(16, 8),
            target=GFX90A,
        )


def test_vectorized_ds_read_atoms_and_kernel():
    from generator.atoms import DsRead2Atom, DsReadAtom, MFMA_F32_16x16x16_F16
    from generator.dsl import GemmKernel, Tensor, LayoutType
    from generator.target_spec import GFX942

    # 1. Test DsRead2Atom emit
    ctx = GpuContext()
    ds2 = DsRead2Atom()
    ds2.emit(ctx, VgprRange(0, 4), Vgpr(4), offset0=0, offset1=4)
    asm_ds2 = ctx.materialize()
    assert "ds_read2_b64 v[0:3], v[4] offset0:0 offset1:4" in asm_ds2

    # 2. Test DsReadAtom(vector_dwords=4) emit with offset1
    ctx2 = GpuContext()
    ds4 = DsReadAtom(vector_dwords=4)
    ds4.emit(ctx2, VgprRange(0, 4), Vgpr(4), offset=0, offset1=4)
    asm_ds4 = ctx2.materialize()
    assert "ds_read2_b64 v[0:3], v[4] offset0:0 offset1:4" in asm_ds4

    # 3. Test GemmKernel auto-selection: GFX942 + FP16 + 16x16x16 MFMA + depth_k 32 -> vector_ds_read True
    kernel_auto = GemmKernel(name="test_auto_ds_read2", target=GFX942)
    kernel_auto.set_inputs(
        A=Tensor(shape=(128, 128), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(128, 128), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(128, 128), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_transposes(
        trans_a=True, trans_b=False
    ).set_tiling(
        block_tile=(128, 128, 32),
        wave_group=(2, 2),
        wave_tiling=(4, 4),
    ).bind_atoms(
        mma=MFMA_F32_16x16x16_F16()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=True,
    )
    asm_auto = kernel_auto.generate_assembly()
    assert "ds_read2_b64" in asm_auto

    # 4. Test GemmKernel explicit override: forcing 64-bit lds_read disables ds_read2_b64
    kernel_override = GemmKernel(name="test_override_ds_read", target=GFX942)
    kernel_override.set_inputs(
        A=Tensor(shape=(128, 128), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(128, 128), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(128, 128), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_transposes(
        trans_a=True, trans_b=False
    ).set_tiling(
        block_tile=(128, 128, 32),
        wave_group=(2, 2),
        wave_tiling=(4, 4),
    ).bind_atoms(
        mma=MFMA_F32_16x16x16_F16(),
        lds_read=DsReadAtom(vector_dwords=2),
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=True,
    )
    asm_override = kernel_override.generate_assembly()
    assert "ds_read_b64" in asm_override
    assert "ds_read2_b64" not in asm_override


def test_dual_port_lds_padding_solver():
    """
    Verify LdsPaddingSolver dual-port conflict modeling for vector_ds_read=True.
    """
    from generator.atoms import MFMA_F32_16x16x16_F16
    from generator.layout import LdsPaddingSolver

    atom = MFMA_F32_16x16x16_F16()
    # For HGEMM 256x128x64 with trans_a=True, trans_b=False
    pad_a, conf_a = LdsPaddingSolver.solve_pad_a(
        atom, tile_m=256, depth_k=64, trans_a=True, vector_ds_read=True
    )
    pad_b, conf_b = LdsPaddingSolver.solve_pad_b(
        atom, depth_k=64, tile_n=128, trans_b=False, vector_ds_read=True
    )

    # Padding must be a multiple of 8 elements (16 bytes for FP16)
    assert pad_a % 8 == 0
    assert pad_b % 8 == 0
    # Minimum conflict cycles for dual-port 64 dwords on 32 banks is 2
    assert conf_a == 2
    assert conf_b == 2
