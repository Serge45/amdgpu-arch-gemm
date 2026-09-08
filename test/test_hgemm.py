import struct
import numpy as np
import pytest
from generator.generator import (
    GpuContext,
    GemmSolutionConfig,
    GemmOptimizations,
    DataType,
    FunctionArgument,
    gemm,
    AccVgprRange,
)
from generator.dsl import GemmKernel, Tensor, LayoutType
from generator.atoms import MFMA_F32_32x32x8_F16
from generator.target_spec import GFX90A
from vm.gcn_virtual_machine import GcnVirtualMachine


def test_hgemm_tn_assembly_and_compile():
    """
    Verify HGEMM TN kernel generation and local ROCm clang++ compilation.
    """
    kernel = GemmKernel(name="test_hgemm_tn_local", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(128, 64), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(64, 64), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(128, 64), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_transposes(
        trans_a=True, trans_b=False
    ).set_tiling(
        block_tile=(64, 64, 16),
        wave_group=(2, 1),
        wave_tiling=(1, 2),
    ).bind_atoms(
        mma=MFMA_F32_32x32x8_F16()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=True,
    )

    asm = kernel.generate_assembly()
    assert "v_mfma_f32_32x32x8f16" in asm
    assert "ds_read_b64" in asm
    assert "hgemm_tn" in kernel.canonical_name


def test_vm_hgemm_tn():
    """
    Test HGEMM TN execution in software GCN VM simulator against NumPy reference.
    Matrix A is FP16 (trans_a=True: shape (k, m) in memory).
    Matrix B is FP16 (trans_b=False: shape (k, n) in memory).
    Matrix C/D is FP32 (shape (m, n) in memory).
    """
    vm = GcnVirtualMachine(104, 256, 64)
    context = GpuContext()

    m, n, k = 32, 32, 32
    gemm_config = GemmSolutionConfig(
        a_type=DataType.FP16,
        b_type=DataType.FP16,
        cd_type=DataType.FP32,
        scalar_type=DataType.FP32,
        mfma=(32, 32, 1, 8),
        wave_group=(1, 1),
        wave_tiling=(1, 1),
        depth_k=16,
        trans_a=True,
        trans_b=False,
        vmem_stage=1,
        single_buffer_lds=True,
    )
    opt = GemmOptimizations(1)
    opt.plr = 1

    kern_args = [
        FunctionArgument("global_buffer", "a", None, 8),
        FunctionArgument("global_buffer", "b", None, 8),
        FunctionArgument("global_buffer", "c", None, 8),
        FunctionArgument("global_buffer", "d", None, 8),
        FunctionArgument("by_value", "m", None, 4),
        FunctionArgument("by_value", "n", None, 4),
        FunctionArgument("by_value", "k", None, 4),
        FunctionArgument("by_value", "lda", None, 4),
        FunctionArgument("by_value", "ldb", None, 4),
        FunctionArgument("by_value", "ldc", None, 4),
        FunctionArgument("by_value", "ldd", None, 4),
        FunctionArgument("by_value", "alpha", None, 4),
        FunctionArgument("by_value", "beta", None, 4),
        FunctionArgument("by_value", "numWorkgroupX", None, 4),
        FunctionArgument("by_value", "numWorkgroupY", None, 4),
    ]

    for i in range(vm.wavefront_size):
        vm.v[0][i] = i

    vm.s[0] = 0
    vm.s[1] = 0
    vm.s[2] = 0
    vm.s[3] = 0

    a_size = m * k * 2
    b_size = k * n * 2
    c_size = m * n * 4
    d_size = m * n * 4

    a_offset = 0
    b_offset = a_size
    c_offset = a_size + b_size
    d_offset = a_size + b_size + c_size

    lda = k  # trans_a
    ldb = k  # not trans_b
    ldc = m
    ldd = m

    vm.smem.mem[:8] = int.to_bytes(a_offset, 8, "little")
    vm.smem.mem[8:16] = int.to_bytes(b_offset, 8, "little")
    vm.smem.mem[16:24] = int.to_bytes(c_offset, 8, "little")
    vm.smem.mem[24:32] = int.to_bytes(d_offset, 8, "little")
    vm.smem.mem[32:36] = int.to_bytes(m, 4, "little")
    vm.smem.mem[36:40] = int.to_bytes(n, 4, "little")
    vm.smem.mem[40:44] = int.to_bytes(k, 4, "little")
    vm.smem.mem[44:48] = int.to_bytes(lda, 4, "little")
    vm.smem.mem[48:52] = int.to_bytes(ldb, 4, "little")
    vm.smem.mem[52:56] = int.to_bytes(ldc, 4, "little")
    vm.smem.mem[56:60] = int.to_bytes(ldd, 4, "little")
    vm.smem.mem[60:64] = struct.pack("f", 1.0)
    vm.smem.mem[64:68] = struct.pack("f", 0.0)
    vm.smem.mem[68:72] = int.to_bytes(1, 4, "little")
    vm.smem.mem[72:76] = int.to_bytes(1, 4, "little")

    np.random.seed(42)
    shape_a = (k, m)  # trans_a
    shape_b = (k, n)  # not trans_b

    a_mat = np.random.uniform(-1.0, 1.0, size=shape_a).astype(np.float16)
    b_mat = np.random.uniform(-1.0, 1.0, size=shape_b).astype(np.float16)
    c_mat = np.zeros((m, n), dtype=np.float32)

    vm.vmem.mem[a_offset:a_offset + a_size] = bytearray(a_mat.tobytes("F"))
    vm.vmem.mem[b_offset:b_offset + b_size] = bytearray(b_mat.tobytes("F"))
    vm.vmem.mem[c_offset:c_offset + c_size] = bytearray(c_mat.tobytes("F"))

    for i in range(16):
        for j in range(vm.wavefront_size):
            vm.a[i][j] = 0

    gemm(
        context,
        "gemm_hgemm_tn",
        "gfx90a:xnack-",
        gemm_config,
        opt,
        kern_args,
    )

    vm.run(context)

    # Reference calculation:
    op_a = a_mat.T.astype(np.float32)
    op_b = b_mat.astype(np.float32)
    ref_d = op_a @ op_b + c_mat

    raw_d = vm.vmem.mem[d_offset:d_offset + d_size]
    d_out = np.frombuffer(raw_d, dtype=np.float32).reshape((n, m)).T

    print("Ref D max:", np.max(np.abs(ref_d)), "Diff max:", np.max(np.abs(d_out - ref_d)))
    assert np.allclose(d_out, ref_d, atol=1e-2, rtol=1e-2)


def test_hgemm_bundle():
    """
    Test GemmKernelBundle with multiple HGEMM TN configurations,
    verifying canonical names, TOML serialization, and local clang++ compilation.
    """
    from generator.dsl import GemmKernelBundle
    import subprocess
    import tempfile
    import os

    bundle = GemmKernelBundle("test_hgemm_bundle", target=GFX90A)

    configs = [
        ((64, 64, 16), (2, 1), (1, 2)),
        ((128, 64, 16), (2, 2), (2, 1)),
        ((64, 128, 16), (2, 2), (1, 2)),
    ]

    for block_tile, wave_group, wave_tiling in configs:
        k = GemmKernel(target=GFX90A)
        k.set_inputs(
            A=Tensor(shape=(128, 64), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
            B=Tensor(shape=(64, 64), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
            C=Tensor(shape=(128, 64), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        ).set_transposes(
            trans_a=True, trans_b=False
        ).set_tiling(
            block_tile=block_tile,
            wave_group=wave_group,
            wave_tiling=wave_tiling,
        ).bind_atoms(
            mma=MFMA_F32_32x32x8_F16()
        ).set_schedule(
            vmem_stages=2,
            single_buffer_lds=True,
        )
        bundle.add(k)

    assert len(bundle) == 3
    for k in bundle.kernels.values():
        assert "hgemm_tn" in k.canonical_name

    asm = bundle.generate_assembly()
    assert "v_mfma_f32_32x32x8f16" in asm
    assert "ds_read_b64" in asm

    # Verify bundle compilation to .co and .toml using bundle.compile
    with tempfile.TemporaryDirectory() as tmpdir:
        ret = bundle.compile(output_folder=tmpdir)
        assert ret == 0

        co_path = os.path.join(tmpdir, f"{bundle.name}.co")
        toml_path = os.path.join(tmpdir, f"{bundle.name}.toml")

        assert os.path.exists(co_path)
        assert os.path.getsize(co_path) > 0

        assert os.path.exists(toml_path)
        with open(toml_path, "r") as f:
            content = f.read()
            assert "[kernels." in content
            assert "hgemm_tn" in content


def test_hgemm_16x16x16_assembly_and_compile():
    from generator.atoms import MFMA_F32_16x16x16_F16
    kernel = GemmKernel(name="test_hgemm_16x16x16_local", target=GFX90A)
    kernel.set_inputs(
        A=Tensor(shape=(128, 64), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(64, 64), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(128, 64), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_transposes(
        trans_a=True, trans_b=False
    ).set_tiling(
        block_tile=(64, 64, 16),
        wave_group=(2, 2),
        wave_tiling=(2, 2),
    ).bind_atoms(
        mma=MFMA_F32_16x16x16_F16()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=True,
    )

    asm = kernel.generate_assembly()
    assert "v_mfma_f32_16x16x16f16" in asm
    assert "ds_read_b64" in asm
    assert "mfma16x16x16" in kernel.canonical_name


def test_hgemm_160x128x64_assembly():
    from generator.atoms import MFMA_F32_16x16x16_F16, MFMA_F32_32x32x8_F16
    # Test with 16x16x16 MFMA (wg 2x2, wt 5x4)
    kernel_16x16 = GemmKernel(target=GFX90A)
    kernel_16x16.set_inputs(
        A=Tensor(shape=(2048, 2048), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(2048, 2048), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(2048, 2048), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_transposes(
        trans_a=True, trans_b=False
    ).set_workgroup_mapping(4).set_tiling(
        block_tile=(160, 128, 64),
        wave_group=(2, 2),
        wave_tiling=(5, 4),
    ).bind_atoms(
        mma=MFMA_F32_16x16x16_F16()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=True,
    )
    asm_16 = kernel_16x16.generate_assembly()
    assert "v_mfma_f32_16x16x16f16" in asm_16
    assert "s_cmp_ge_u32" in asm_16  # Epilogue store boundary guard
    assert "hgemm_tn_b160x128x64_wg2x2_wt5x4_mfma16x16x16_sgl_vs2_wgm4" == kernel_16x16.canonical_name


def test_vm_hgemm_tn_double_buffer():
    """
    Test HGEMM TN execution with double-buffered LDS in software GCN VM simulator.
    """
    vm = GcnVirtualMachine(104, 256, 64)
    context = GpuContext()

    m, n, k = 32, 32, 32
    gemm_config = GemmSolutionConfig(
        a_type=DataType.FP16,
        b_type=DataType.FP16,
        cd_type=DataType.FP32,
        scalar_type=DataType.FP32,
        mfma=(32, 32, 1, 8),
        wave_group=(1, 1),
        wave_tiling=(1, 1),
        depth_k=16,
        trans_a=True,
        trans_b=False,
        vmem_stage=2,
        single_buffer_lds=False,
    )
    opt = GemmOptimizations(2)
    opt.plr = 1

    kern_args = [
        FunctionArgument("global_buffer", "a", None, 8),
        FunctionArgument("global_buffer", "b", None, 8),
        FunctionArgument("global_buffer", "c", None, 8),
        FunctionArgument("global_buffer", "d", None, 8),
        FunctionArgument("by_value", "m", None, 4),
        FunctionArgument("by_value", "n", None, 4),
        FunctionArgument("by_value", "k", None, 4),
        FunctionArgument("by_value", "lda", None, 4),
        FunctionArgument("by_value", "ldb", None, 4),
        FunctionArgument("by_value", "ldc", None, 4),
        FunctionArgument("by_value", "ldd", None, 4),
        FunctionArgument("by_value", "alpha", None, 4),
        FunctionArgument("by_value", "beta", None, 4),
        FunctionArgument("by_value", "numWorkgroupX", None, 4),
        FunctionArgument("by_value", "numWorkgroupY", None, 4),
    ]

    for i in range(vm.wavefront_size):
        vm.v[0][i] = i

    vm.s[0] = vm.s[1] = vm.s[2] = vm.s[3] = 0

    a_size = m * k * 2
    b_size = k * n * 2
    c_size = m * n * 4
    d_size = m * n * 4

    a_offset = 0
    b_offset = a_size
    c_offset = a_size + b_size
    d_offset = a_size + b_size + c_size

    lda = k
    ldb = k
    ldc = m
    ldd = m

    vm.smem.mem[:8] = int.to_bytes(a_offset, 8, "little")
    vm.smem.mem[8:16] = int.to_bytes(b_offset, 8, "little")
    vm.smem.mem[16:24] = int.to_bytes(c_offset, 8, "little")
    vm.smem.mem[24:32] = int.to_bytes(d_offset, 8, "little")
    vm.smem.mem[32:36] = int.to_bytes(m, 4, "little")
    vm.smem.mem[36:40] = int.to_bytes(n, 4, "little")
    vm.smem.mem[40:44] = int.to_bytes(k, 4, "little")
    vm.smem.mem[44:48] = int.to_bytes(lda, 4, "little")
    vm.smem.mem[48:52] = int.to_bytes(ldb, 4, "little")
    vm.smem.mem[52:56] = int.to_bytes(ldc, 4, "little")
    vm.smem.mem[56:60] = int.to_bytes(ldd, 4, "little")
    vm.smem.mem[60:64] = struct.pack("f", 1.0)
    vm.smem.mem[64:68] = struct.pack("f", 0.0)
    vm.smem.mem[68:72] = int.to_bytes(1, 4, "little")
    vm.smem.mem[72:76] = int.to_bytes(1, 4, "little")

    np.random.seed(42)
    shape_a = (k, m)
    shape_b = (k, n)

    a_mat = np.random.uniform(-1.0, 1.0, size=shape_a).astype(np.float16)
    b_mat = np.random.uniform(-1.0, 1.0, size=shape_b).astype(np.float16)
    c_mat = np.zeros((m, n), dtype=np.float32)

    vm.vmem.mem[a_offset:a_offset + a_size] = bytearray(a_mat.tobytes("F"))
    vm.vmem.mem[b_offset:b_offset + b_size] = bytearray(b_mat.tobytes("F"))
    vm.vmem.mem[c_offset:c_offset + c_size] = bytearray(c_mat.tobytes("F"))

    for i in range(16):
        for j in range(vm.wavefront_size):
            vm.a[i][j] = 0

    gemm(context, "test_dbl_vm", "gfx90a:xnack-", gemm_config, opt, kern_args)
    vm.run(context)

    ref_d = a_mat.T.astype(np.float32) @ b_mat.astype(np.float32)
    d_bytes = vm.vmem.mem[d_offset:d_offset + d_size]
    gpu_d = np.frombuffer(d_bytes, dtype=np.float32).reshape((n, m)).T

    assert np.allclose(gpu_d, ref_d, atol=1e-2, rtol=1e-2)


def test_vm_hgemm_16x16x16_tn():
    """
    Test v_mfma_f32_16x16x16f16 atom execution in software GCN VM simulator.
    """
    vm = GcnVirtualMachine(104, 256, 64)
    context = GpuContext()

    m, n, k = 16, 16, 32
    gemm_config = GemmSolutionConfig(
        a_type=DataType.FP16,
        b_type=DataType.FP16,
        cd_type=DataType.FP32,
        scalar_type=DataType.FP32,
        mfma=(16, 16, 1, 16),
        wave_group=(1, 1),
        wave_tiling=(1, 1),
        depth_k=16,
        trans_a=True,
        trans_b=False,
        vmem_stage=1,
        single_buffer_lds=True,
    )
    opt = GemmOptimizations(1)
    opt.plr = 1

    kern_args = [
        FunctionArgument("global_buffer", "a", None, 8),
        FunctionArgument("global_buffer", "b", None, 8),
        FunctionArgument("global_buffer", "c", None, 8),
        FunctionArgument("global_buffer", "d", None, 8),
        FunctionArgument("by_value", "m", None, 4),
        FunctionArgument("by_value", "n", None, 4),
        FunctionArgument("by_value", "k", None, 4),
        FunctionArgument("by_value", "lda", None, 4),
        FunctionArgument("by_value", "ldb", None, 4),
        FunctionArgument("by_value", "ldc", None, 4),
        FunctionArgument("by_value", "ldd", None, 4),
        FunctionArgument("by_value", "alpha", None, 4),
        FunctionArgument("by_value", "beta", None, 4),
        FunctionArgument("by_value", "numWorkgroupX", None, 4),
        FunctionArgument("by_value", "numWorkgroupY", None, 4),
    ]

    for i in range(vm.wavefront_size):
        vm.v[0][i] = i

    vm.s[0] = vm.s[1] = vm.s[2] = vm.s[3] = 0

    a_size = m * k * 2
    b_size = k * n * 2
    c_size = m * n * 4
    d_size = m * n * 4

    a_offset = 0
    b_offset = a_size
    c_offset = a_size + b_size
    d_offset = a_size + b_size + c_size

    lda = k
    ldb = k
    ldc = m
    ldd = m

    vm.smem.mem[:8] = int.to_bytes(a_offset, 8, "little")
    vm.smem.mem[8:16] = int.to_bytes(b_offset, 8, "little")
    vm.smem.mem[16:24] = int.to_bytes(c_offset, 8, "little")
    vm.smem.mem[24:32] = int.to_bytes(d_offset, 8, "little")
    vm.smem.mem[32:36] = int.to_bytes(m, 4, "little")
    vm.smem.mem[36:40] = int.to_bytes(n, 4, "little")
    vm.smem.mem[40:44] = int.to_bytes(k, 4, "little")
    vm.smem.mem[44:48] = int.to_bytes(lda, 4, "little")
    vm.smem.mem[48:52] = int.to_bytes(ldb, 4, "little")
    vm.smem.mem[52:56] = int.to_bytes(ldc, 4, "little")
    vm.smem.mem[56:60] = int.to_bytes(ldd, 4, "little")
    vm.smem.mem[60:64] = struct.pack("f", 1.0)
    vm.smem.mem[64:68] = struct.pack("f", 0.0)
    vm.smem.mem[68:72] = int.to_bytes(1, 4, "little")
    vm.smem.mem[72:76] = int.to_bytes(1, 4, "little")

    np.random.seed(42)
    shape_a = (k, m)
    shape_b = (k, n)

    a_mat = np.random.uniform(-1.0, 1.0, size=shape_a).astype(np.float16)
    b_mat = np.random.uniform(-1.0, 1.0, size=shape_b).astype(np.float16)
    c_mat = np.zeros((m, n), dtype=np.float32)

    vm.vmem.mem[a_offset:a_offset + a_size] = bytearray(a_mat.tobytes("F"))
    vm.vmem.mem[b_offset:b_offset + b_size] = bytearray(b_mat.tobytes("F"))
    vm.vmem.mem[c_offset:c_offset + c_size] = bytearray(c_mat.tobytes("F"))

    for i in range(4):
        for j in range(vm.wavefront_size):
            vm.a[i][j] = 0

    gemm(context, "test_16x16_vm", "gfx90a:xnack-", gemm_config, opt, kern_args)
    vm.run(context)

    ref_d = a_mat.T.astype(np.float32) @ b_mat.astype(np.float32)
    d_bytes = vm.vmem.mem[d_offset:d_offset + d_size]
    gpu_d = np.frombuffer(d_bytes, dtype=np.float32).reshape((n, m)).T

    assert np.allclose(gpu_d, ref_d, atol=1e-2, rtol=1e-2)


