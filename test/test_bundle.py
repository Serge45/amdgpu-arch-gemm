from __future__ import annotations
import pytest
from generator.dsl import GemmKernel, Tensor, LayoutType, GemmKernelBundle
from generator.atoms import MFMA_F32_32x32x2_F32, MFMA_F32_16x16x4_F32
from generator.target_spec import GFX90A
from generator.generator import DataType


def test_canonical_kernel_naming():
    # 1. Standard FP32 non-transpose, 256x128x16 single buffer
    k1 = GemmKernel(target=GFX90A)
    k1.set_inputs(
        A=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
        B=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
        C=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(256, 128, 16),
        wave_group=(2, 2),
        wave_tiling=(4, 2),
    ).bind_atoms(
        mma=MFMA_F32_32x32x2_F32()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=True,
    )
    assert k1.canonical_name == "sgemm_nn_b256x128x16_wg2x2_wt4x2_mfma32x32x2_sgl_vs2"
    assert k1.kernel_name == "sgemm_nn_b256x128x16_wg2x2_wt4x2_mfma32x32x2_sgl_vs2"

    # 2. Double buffer with depth_k=16
    k2 = GemmKernel(target=GFX90A)
    k2.set_inputs(
        A=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
        B=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
        C=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(128, 128, 16),
        wave_group=(2, 2),
        wave_tiling=(2, 2),
    ).bind_atoms(
        mma=MFMA_F32_32x32x2_F32()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=False,
    )
    assert k2.canonical_name == "sgemm_nn_b128x128x16_wg2x2_wt2x2_mfma32x32x2_dbl_vs2"

    # 3. Explicit name preservation
    k3 = GemmKernel(name="custom_gemm", target=GFX90A)
    assert k3.name == "custom_gemm"
    assert k3.kernel_name == "custom_gemm"


def test_gemm_kernel_bundle_assembly_generation():
    bundle = GemmKernelBundle("test_bundle", target=GFX90A)

    k1 = GemmKernel(target=GFX90A)
    k1.set_inputs(
        A=Tensor(shape=(1024, 1024), dtype=DataType.FP32),
        B=Tensor(shape=(1024, 1024), dtype=DataType.FP32),
        C=Tensor(shape=(1024, 1024), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(128, 128, 16),
        wave_group=(2, 2),
        wave_tiling=(2, 2),
    ).bind_atoms(
        mma=MFMA_F32_32x32x2_F32()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=True,
    )

    k2 = GemmKernel(target=GFX90A)
    k2.set_inputs(
        A=Tensor(shape=(1024, 1024), dtype=DataType.FP32),
        B=Tensor(shape=(1024, 1024), dtype=DataType.FP32),
        C=Tensor(shape=(1024, 1024), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(256, 128, 16),
        wave_group=(2, 2),
        wave_tiling=(4, 2),
    ).bind_atoms(
        mma=MFMA_F32_32x32x2_F32()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=False,
    )

    bundle.add(k1).add(k2)
    assert len(bundle) == 2

    # Verify query
    found = bundle.find_kernel(block_tile=(256, 128, 16), single_buffer_lds=False)
    assert found is not None
    assert found.canonical_name == k2.canonical_name

    # Generate bundle assembly
    asm = bundle.generate_assembly()
    assert f".globl {k1.canonical_name}" in asm
    assert f".globl {k2.canonical_name}" in asm
    assert f".type {k1.canonical_name},@function" in asm
    assert f".type {k2.canonical_name},@function" in asm
    assert f".amdhsa_kernel {k1.canonical_name}" in asm
    assert f".amdhsa_kernel {k2.canonical_name}" in asm
    assert f".name: {k1.canonical_name}" in asm
    assert f".name: {k2.canonical_name}" in asm
