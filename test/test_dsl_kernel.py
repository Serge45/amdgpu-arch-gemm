from __future__ import annotations
import pytest
import numpy as np
from generator.dsl import GemmKernel, Tensor, LayoutType
from generator.atoms import MFMA_F32_32x32x2_F32, MFMA_F32_16x16x4_F32
from generator.target_spec import GFX90A
from generator.generator import DataType


def test_gemm_kernel_declaration_and_diagnostics():
    """
    Verify fluent DSL API for declaring a GEMM kernel and inspecting compile-time diagnostics.
    """
    kernel = GemmKernel(name="my_sgemm", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=("M", "K"), dtype=DataType.FP32, layout=LayoutType.ROW_MAJOR),
        B=Tensor(shape=("K", "N"), dtype=DataType.FP32, layout=LayoutType.ROW_MAJOR),
        C=Tensor(shape=("M", "N"), dtype=DataType.FP32, layout=LayoutType.ROW_MAJOR),
    ).set_tiling(
        block_tile=(128, 128, 16),
        wave_group=(2, 2),
        wave_tiling=(2, 2),
    ).bind_atoms(
        mma=MFMA_F32_32x32x2_F32()
    ).set_schedule(
        vmem_stages=2,
        max_vgpr_budget=128,
    )

    diag = kernel.get_diagnostics()
    assert diag["kernel_name"] == "my_sgemm"
    assert diag["target"] == "gfx90a"
    assert diag["block_tile"] == (128, 128)
    assert diag["num_waves"] == 4
    assert diag["num_threads"] == 256
    assert diag["max_vgpr_budget"] == 128
    assert diag["lds_conflicts_a"] <= 2
    assert diag["lds_conflicts_b"] <= 4

    # Verify assembly generation
    asm = kernel.generate_assembly()
    assert len(asm) > 0
    assert "v_mfma_f32_32x32x2f32" in asm
    assert ".amdgcn_target \"amdgcn-amd-amdhsa--gfx90a:xnack-\"" in asm


def test_gemm_kernel_emulation():
    """
    Verify CPU emulation of GemmKernel against NumPy reference.
    """
    m, n, k = 16, 16, 64
    a_np = np.arange(0, m * k, 1, dtype=np.float32).reshape(m, k)
    b_np = np.arange(0, k * n, 1, dtype=np.float32).reshape(k, n)
    c_np = np.ones((m, n), dtype=np.float32)

    kernel = GemmKernel(name="sgemm_16x16x4_test", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(m, k), dtype=DataType.FP32),
        B=Tensor(shape=(k, n), dtype=DataType.FP32),
        C=Tensor(shape=(m, n), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(16, 16, 16),
        wave_group=(1, 1),
        wave_tiling=(1, 1),
    ).bind_atoms(
        mma=MFMA_F32_16x16x4_F32()
    ).set_schedule(
        vmem_stages=1,
    )

    # Emulate kernel execution on CPU virtual machine
    d_out = kernel.emulate(a_np, b_np, c_np, alpha=1.0, beta=1.0)
    ref = a_np @ b_np + c_np

    assert np.allclose(d_out, ref, atol=1e-4)


def test_gemm_kernel_epilogue_fusion():
    """
    Verify custom epilogue fusion (Bias + ReLU).
    """
    m, n, k = 16, 16, 64
    a_np = np.ones((m, k), dtype=np.float32) * 0.1
    b_np = np.ones((k, n), dtype=np.float32) * 0.2
    c_np = np.zeros((m, n), dtype=np.float32)

    kernel = GemmKernel(name="fused_sgemm", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(m, k), dtype=DataType.FP32),
        B=Tensor(shape=(k, n), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(16, 16, 16),
        wave_group=(1, 1),
        wave_tiling=(1, 1),
    ).bind_atoms(
        mma=MFMA_F32_16x16x4_F32()
    )

    # Register custom epilogue: Bias addition + ReLU
    bias = 0.5
    @kernel.epilogue
    def fused_bias_relu(d_matrix):
        return np.maximum(0, d_matrix + bias)

    d_out = kernel.emulate(a_np, b_np, c_np, alpha=1.0, beta=0.0)
    expected_ref = np.maximum(0, (a_np @ b_np) + bias)

    assert np.allclose(d_out, expected_ref, atol=1e-4)


def test_gemm_kernel_clang_compilation(tmp_path):
    """
    Verify that generated assembly compiles successfully into .o and .co
    via ROCm clang++ toolchain if available.
    """
    import os
    clang_path = "/opt/rocm/llvm/bin/clang++"
    if not os.path.exists(clang_path):
        pytest.skip(f"ROCm clang++ not found at {clang_path}")

    kernel = GemmKernel(name="compiled_sgemm_test", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(16, 64), dtype=DataType.FP32),
        B=Tensor(shape=(64, 16), dtype=DataType.FP32),
        C=Tensor(shape=(16, 16), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(16, 16, 16),
        wave_group=(1, 1),
        wave_tiling=(1, 1),
    ).bind_atoms(
        mma=MFMA_F32_16x16x4_F32()
    )

    out_dir = str(tmp_path)
    ret = kernel.compile(output_folder=out_dir)
    assert ret == 0

    co_file = os.path.join(out_dir, f"{kernel.name}.co")
    o_file = os.path.join(out_dir, f"{kernel.name}.o")
    toml_file = os.path.join(out_dir, f"{kernel.name}.toml")

    assert os.path.isfile(co_file)
    assert os.path.getsize(co_file) > 0
    assert os.path.isfile(o_file)
    assert os.path.isfile(toml_file)

