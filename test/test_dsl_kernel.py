from __future__ import annotations
import pytest
import numpy as np
from generator.dsl import GemmKernel, Tensor, LayoutType
from generator.atoms import MFMA_F32_32x32x2_F32, MFMA_F32_16x16x4_F32
from generator.target_spec import GFX90A
from generator.generator import DataType
from generator.scheduler import SchedulingPolicy


def test_gemm_kernel_declaration_and_diagnostics():
    """
    Verify fluent DSL API for declaring a GEMM kernel and inspecting compile-time diagnostics.
    """
    kernel = GemmKernel(name="my_sgemm", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=("M", "K"), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=("K", "N"), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=("M", "N"), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
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


def test_gemm_kernel_layout_configuration():
    """
    Verify that column-major (native AMDGPU BLAS) and row-major layouts
    correctly configure trans_a and trans_b flags.
    """
    # Test default column-major layout
    tensor_default = Tensor(shape=(16, 16), dtype=DataType.FP32)
    assert tensor_default.layout == LayoutType.COL_MAJOR

    kernel_col = GemmKernel(name="col_gemm", target=GFX90A)
    kernel_col.set_inputs(
        A=Tensor(shape=(16, 64), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(64, 16), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(16, 16), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    )
    config_col, _ = kernel_col.to_gemm_solution_config()
    assert config_col.trans_a is False
    assert config_col.trans_b is False

    # Test row-major layout
    kernel_row = GemmKernel(name="row_gemm", target=GFX90A)
    kernel_row.set_inputs(
        A=Tensor(shape=(16, 64), dtype=DataType.FP32, layout=LayoutType.ROW_MAJOR),
        B=Tensor(shape=(64, 16), dtype=DataType.FP32, layout=LayoutType.ROW_MAJOR),
        C=Tensor(shape=(16, 16), dtype=DataType.FP32, layout=LayoutType.ROW_MAJOR),
    )
    config_row, _ = kernel_row.to_gemm_solution_config()
    assert config_row.trans_a is True
    assert config_row.trans_b is True


def test_gemm_kernel_emulation():
    """
    Verify CPU emulation of GemmKernel against NumPy reference in column-major order.
    """
    m, n, k = 16, 16, 64
    a_np = np.arange(0, m * k, 1, dtype=np.float32).reshape(m, k, order="F")
    b_np = np.arange(0, k * n, 1, dtype=np.float32).reshape(k, n, order="F")
    c_np = np.ones((m, n), dtype=np.float32, order="F")

    kernel = GemmKernel(name="sgemm_16x16x4_test", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(m, k), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(k, n), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(m, n), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
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


def test_gemm_kernel_dag_pipeline_emulation():
    """
    Verify DAG + Modulo Pipeline scheduling on double-buffered (vmem_stages=2) GEMM.
    Ensures CPU emulation matches NumPy reference in column-major layout.
    """
    m, n, k = 16, 16, 64
    a_np = np.arange(0, m * k, 1, dtype=np.float32).reshape(m, k, order="F")
    b_np = np.arange(0, k * n, 1, dtype=np.float32).reshape(k, n, order="F")
    c_np = np.ones((m, n), dtype=np.float32, order="F")

    kernel = GemmKernel(name="sgemm_dag_pipeline_test", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(m, k), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(k, n), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(m, n), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_tiling(
        block_tile=(16, 16, 16),
        wave_group=(1, 1),
        wave_tiling=(1, 1),
    ).bind_atoms(
        mma=MFMA_F32_16x16x4_F32()
    ).set_schedule(
        vmem_stages=2,
        scheduling_policy=SchedulingPolicy.DAG_PIPELINE,
    )

    d_out = kernel.emulate(a_np, b_np, c_np, alpha=1.0, beta=1.0)
    ref = a_np @ b_np + c_np

    assert np.allclose(d_out, ref, atol=1e-3)


def test_gemm_kernel_epilogue_fusion():
    """
    Verify custom epilogue fusion (Bias + ReLU) with column-major matrices.
    """
    m, n, k = 16, 16, 64
    a_np = np.ones((m, k), dtype=np.float32, order="F") * 0.1
    b_np = np.ones((k, n), dtype=np.float32, order="F") * 0.2
    c_np = np.zeros((m, n), dtype=np.float32, order="F")

    kernel = GemmKernel(name="fused_sgemm", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(m, k), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(k, n), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
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
        A=Tensor(shape=(16, 64), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(64, 16), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(16, 16), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
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


def test_gemm_kernel_rectangular_macrotile():
    """
    Verify rectangular MacroTile (256, 32) configuration, LDS budgeting,
    and assembly generation.
    """
    kernel = GemmKernel(name="rect_256x32_test", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(256, 64), dtype=DataType.FP32),
        B=Tensor(shape=(64, 32), dtype=DataType.FP32),
        C=Tensor(shape=(256, 32), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(256, 32, 16),
        wave_group=(4, 1),
        wave_tiling=(2, 1),
    ).bind_atoms(
        mma=MFMA_F32_32x32x2_F32()
    ).set_schedule(
        vmem_stages=2,
    )

    diag = kernel.get_diagnostics()
    assert diag["block_tile"] == (256, 32)
    assert diag["wave_tile"] == (64, 32)
    assert diag["num_waves"] == 4
    assert diag["agpr_per_thread"] == 32
    assert diag["lds_usage_bytes"] <= 65536

    asm = kernel.generate_assembly()
    assert len(asm) > 0
    assert "v_mfma_f32_32x32x2f32" in asm


def test_gemm_kernel_rectangular_emulation():
    """
    Verify CPU virtual machine emulation of a rectangular tile against NumPy reference.
    """
    m, n, k = 32, 16, 64
    a_np = (np.arange(0, m * k, 1, dtype=np.float32).reshape(m, k, order="F")) * 0.01
    b_np = (np.arange(0, k * n, 1, dtype=np.float32).reshape(k, n, order="F")) * 0.01
    c_np = np.ones((m, n), dtype=np.float32, order="F")

    kernel = GemmKernel(name="rect_emul_test", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(m, k), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(k, n), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(m, n), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_tiling(
        block_tile=(32, 16, 16),
        wave_group=(1, 1),
        wave_tiling=(2, 1),
    ).bind_atoms(
        mma=MFMA_F32_16x16x4_F32()
    ).set_schedule(
        vmem_stages=2,
    )

    d_out = kernel.emulate(a_np, b_np, c_np, alpha=1.0, beta=1.0)
    ref = a_np @ b_np + c_np

    assert np.allclose(d_out, ref, atol=1e-2)


def test_gemm_kernel_agpr_overflow_rejection():
    """
    Verify that specifying wave_tiling that exceeds physical AGPR limit (256) is rejected.
    """
    kernel = GemmKernel(name="overflow_test", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(256, 64), dtype=DataType.FP32),
        B=Tensor(shape=(64, 256), dtype=DataType.FP32),
        C=Tensor(shape=(256, 256), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(256, 256, 16),
        wave_group=(1, 1),
        wave_tiling=(8, 4),  # 32 atoms * 16 regs = 512 AGPRs > 256
    ).bind_atoms(
        mma=MFMA_F32_32x32x2_F32()
    )

    with pytest.raises((ValueError, RuntimeError), match="exceeds"):
        kernel.to_gemm_solution_config()


def test_gemm_kernel_depth_k_32():
    """
    Verify double-buffered K=32 configuration fits in LDS budget and emits valid code.
    """
    kernel = GemmKernel(name="k32_128x64_test", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(128, 64), dtype=DataType.FP32),
        B=Tensor(shape=(64, 64), dtype=DataType.FP32),
        C=Tensor(shape=(128, 64), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(128, 64, 32),
        wave_group=(2, 2),
        wave_tiling=(2, 1),
    ).bind_atoms(
        mma=MFMA_F32_32x32x2_F32()
    ).set_schedule(
        vmem_stages=2,
    )
    kernel.depth_k = 32

    diag = kernel.get_diagnostics()
    assert diag["lds_usage_bytes"] <= 65536
    assert diag["agpr_per_thread"] == 32

    asm = kernel.generate_assembly()
    assert len(asm) > 0
    assert "v_mfma_f32_32x32x2f32" in asm


def test_gemm_kernel_single_buffer_lds_diagnostics_and_assembly():
    """
    Verify MacroTile (256, 256, 16) with single_buffer_lds fits within 64 KB LDS
    and emits valid AMDGPU assembly.
    """
    kernel = GemmKernel(name="single_lds_256x256_test", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(256, 64), dtype=DataType.FP32),
        B=Tensor(shape=(64, 256), dtype=DataType.FP32),
        C=Tensor(shape=(256, 256), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(256, 256, 16),
        wave_group=(2, 2),
        wave_tiling=(4, 4),
    ).bind_atoms(
        mma=MFMA_F32_32x32x2_F32()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=True,
    )

    diag = kernel.get_diagnostics()
    assert diag["block_tile"] == (256, 256)
    assert diag["agpr_per_thread"] == 256
    assert diag["single_buffer_lds"] is True
    # Verify LDS usage is strictly within 64 KB (36,864 Bytes)
    assert diag["lds_usage_bytes"] == 36864
    assert diag["lds_usage_bytes"] <= 65536

    asm = kernel.generate_assembly()
    assert len(asm) > 0
    assert "v_mfma_f32_32x32x2f32" in asm
    assert "s_barrier" in asm


def test_gemm_kernel_single_buffer_lds_emulation():
    """
    Verify CPU virtual machine emulation of a single-buffer LDS kernel against NumPy reference.
    """
    m, n, k = 16, 16, 64
    a_np = (np.arange(0, m * k, 1, dtype=np.float32).reshape(m, k, order="F")) * 0.01
    b_np = (np.arange(0, k * n, 1, dtype=np.float32).reshape(k, n, order="F")) * 0.01
    c_np = np.ones((m, n), dtype=np.float32, order="F")

    kernel = GemmKernel(name="single_lds_emul_test", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(m, k), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(k, n), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(m, n), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_tiling(
        block_tile=(16, 16, 16),
        wave_group=(1, 1),
        wave_tiling=(1, 1),
    ).bind_atoms(
        mma=MFMA_F32_16x16x4_F32()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=True,
    )

    d_out = kernel.emulate(a_np, b_np, c_np, alpha=1.0, beta=1.0)
    ref = a_np @ b_np + c_np

    assert np.allclose(d_out, ref, atol=1e-3)


