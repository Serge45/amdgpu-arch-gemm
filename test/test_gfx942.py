import os
import tempfile
import struct
import numpy as np
import pytest

from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import (
    MFMA_F32_32x32x2_F32,
    MFMA_F32_16x16x4_F32,
    MFMA_F32_32x32x8_F16,
    MFMA_F32_16x16x16_F16,
)
from generator.target_spec import GFX942
from generator.generator import (
    DataType,
    GpuContext,
    GemmSolutionConfig,
    GemmOptimizations,
    FunctionArgument,
    gemm,
    AccVgprRange,
)
from vm.gcn_virtual_machine import GcnVirtualMachine


def test_gfx942_sgemm_assembly_and_compile():
    """
    Verify FP32 (SGEMM) assembly generation and local clang++ compilation targeting gfx942.
    """
    kernel = GemmKernel(name="test_gfx942_sgemm", target=GFX942)
    kernel.set_inputs(
        A=Tensor(shape=(128, 128), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(128, 128), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(128, 128), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_transposes(
        trans_a=False, trans_b=False
    ).set_workgroup_mapping(
        8
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

    asm = kernel.generate_assembly()
    assert 'amdgcn-amd-amdhsa--gfx942' in asm
    assert 'v_mfma_f32_32x32x2f32' in asm
    assert '.amdhsa_accum_offset' in asm
    assert '.amdhsa_next_free_vgpr' in asm

    with tempfile.TemporaryDirectory() as tmpdir:
        ret = kernel.compile(tmpdir)
        assert ret == 0, f"Compilation failed with code {ret}"
        co_path = os.path.join(tmpdir, f"{kernel.name}.co")
        assert os.path.exists(co_path)
        assert os.path.getsize(co_path) > 0


def test_gfx942_hgemm_assembly_and_compile():
    """
    Verify FP16 (HGEMM) assembly generation and local clang++ compilation targeting gfx942.
    """
    kernel = GemmKernel(name="test_gfx942_hgemm", target=GFX942)
    kernel.set_inputs(
        A=Tensor(shape=(128, 128), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(128, 128), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(128, 128), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_transposes(
        trans_a=True, trans_b=False
    ).set_workgroup_mapping(
        8
    ).set_tiling(
        block_tile=(128, 128, 32),
        wave_group=(2, 2),
        wave_tiling=(4, 4),
    ).bind_atoms(
        mma=MFMA_F32_16x16x16_F16()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=False,
    )

    asm = kernel.generate_assembly()
    assert 'amdgcn-amd-amdhsa--gfx942' in asm
    assert 'v_mfma_f32_16x16x16f16' in asm
    assert ('ds_read_b64' in asm or 'ds_read2_b64' in asm)
    assert '.amdhsa_accum_offset' in asm

    with tempfile.TemporaryDirectory() as tmpdir:
        ret = kernel.compile(tmpdir)
        assert ret == 0, f"Compilation failed with code {ret}"
        co_path = os.path.join(tmpdir, f"{kernel.name}.co")
        assert os.path.exists(co_path)
        assert os.path.getsize(co_path) > 0


def test_gfx942_multi_kernel_bundle():
    """
    Verify GemmKernelBundle bundling multiple gfx942 kernels into a unified .co and .toml.
    """
    bundle = GemmKernelBundle("test_bundle_gfx942", target=GFX942)

    # 1. SGEMM Candidate
    k1 = GemmKernel(target=GFX942)
    k1.set_inputs(
        A=Tensor(shape=(2048, 2048), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(2048, 2048), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(2048, 2048), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_tiling(
        block_tile=(128, 64, 16),
        wave_group=(2, 2),
        wave_tiling=(2, 1),
    ).bind_atoms(
        mma=MFMA_F32_32x32x2_F32()
    ).set_schedule(vmem_stages=2, single_buffer_lds=False)
    bundle.add(k1)

    # 2. HGEMM Candidate
    k2 = GemmKernel(target=GFX942)
    k2.set_inputs(
        A=Tensor(shape=(2048, 2048), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        B=Tensor(shape=(2048, 2048), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
        C=Tensor(shape=(2048, 2048), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
    ).set_transposes(trans_a=True, trans_b=False).set_tiling(
        block_tile=(128, 64, 64),
        wave_group=(2, 2),
        wave_tiling=(4, 2),
    ).bind_atoms(
        mma=MFMA_F32_16x16x16_F16()
    ).set_schedule(vmem_stages=2, single_buffer_lds=False)
    bundle.add(k2)

    assert len(bundle) == 2

    with tempfile.TemporaryDirectory() as tmpdir:
        ret = bundle.compile(tmpdir)
        assert ret == 0

        co_path = os.path.join(tmpdir, f"{bundle.name}.co")
        toml_path = os.path.join(tmpdir, f"{bundle.name}.toml")

        assert os.path.exists(co_path)
        assert os.path.exists(toml_path)

        with open(toml_path, "r") as f:
            content = f.read()
            assert "[kernels." in content
            assert k1.canonical_name in content
            assert k2.canonical_name in content


def test_gfx942_vm_emulation_hgemm():
    """
    Verify GcnVirtualMachine correctly emulates a GFX942-compiled HGEMM instruction sequence.
    """
    vm = GcnVirtualMachine(104, 512, 64)  # 512 VGPR limit for GFX942
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
    a_mat = np.random.uniform(-1.0, 1.0, size=(k, m)).astype(np.float16)
    b_mat = np.random.uniform(-1.0, 1.0, size=(k, n)).astype(np.float16)
    c_mat = np.zeros((m, n), dtype=np.float32)

    vm.vmem.mem[a_offset:a_offset + a_size] = bytearray(a_mat.tobytes("F"))
    vm.vmem.mem[b_offset:b_offset + b_size] = bytearray(b_mat.tobytes("F"))
    vm.vmem.mem[c_offset:c_offset + c_size] = bytearray(c_mat.tobytes("F"))

    for i in range(4):
        for j in range(vm.wavefront_size):
            vm.a[i][j] = 0

    gemm(context, "test_gfx942_vm", "gfx942:xnack-", gemm_config, opt, kern_args)
    vm.run(context)

    ref_d = a_mat.T.astype(np.float32) @ b_mat.astype(np.float32)
    d_bytes = vm.vmem.mem[d_offset:d_offset + d_size]
    gpu_d = np.frombuffer(d_bytes, dtype=np.float32).reshape((n, m)).T

    assert np.allclose(gpu_d, ref_d, atol=1e-2, rtol=1e-2)


def test_gfx942_vm_emulation_sgemm():
    """
    Verify GcnVirtualMachine correctly emulates a GFX942-compiled SGEMM instruction sequence.
    """
    vm = GcnVirtualMachine(104, 512, 64)
    context = GpuContext()
    gemm_config = GemmSolutionConfig(
        DataType.FP32, DataType.FP32, DataType.FP32, DataType.FP32,
        (32, 32, 1, 2), (1, 1), (1, 1), 16, False, False
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
    m, n, k = 32, 32, 64
    for i in range(vm.wavefront_size): vm.v[0][i] = i
    vm.s[0] = vm.s[1] = vm.s[2] = vm.s[3] = 0
    a_offset, a_size = 0, m * k * 4
    b_offset, b_size = a_size, n * k * 4
    c_offset, c_size = a_size + b_size, n * m * 4
    d_offset, d_size = a_size + b_size + c_size, n * m * 4

    vm.smem.mem[:8] = int.to_bytes(a_offset, 8, "little")
    vm.smem.mem[8:16] = int.to_bytes(b_offset, 8, "little")
    vm.smem.mem[16:24] = int.to_bytes(c_offset, 8, "little")
    vm.smem.mem[24:32] = int.to_bytes(d_offset, 8, "little")
    vm.smem.mem[32:36] = int.to_bytes(m, 4, "little")
    vm.smem.mem[36:40] = int.to_bytes(n, 4, "little")
    vm.smem.mem[40:44] = int.to_bytes(k, 4, "little")
    vm.smem.mem[44:48] = int.to_bytes(m, 4, "little")
    vm.smem.mem[48:52] = int.to_bytes(k, 4, "little")
    vm.smem.mem[52:56] = int.to_bytes(m, 4, "little")
    vm.smem.mem[56:60] = int.to_bytes(m, 4, "little")
    vm.smem.mem[60:64] = struct.pack("f", 1.0)
    vm.smem.mem[64:68] = struct.pack("f", 0.0)
    vm.smem.mem[68:72] = int.to_bytes(1, 4, "little")
    vm.smem.mem[72:76] = int.to_bytes(1, 4, "little")

    a, b, c = np.arange(0, m*k, 1, dtype=np.float32), np.arange(0, n*k, 1, dtype=np.float32), np.zeros(m*n, dtype=np.float32)
    vm.vmem.mem[a_offset:a_offset+a_size] = bytearray(a)
    vm.vmem.mem[b_offset:b_offset+b_size] = bytearray(b)
    vm.vmem.mem[c_offset:c_offset+c_size] = bytearray(c)
    for i in range(16):
        for j in range(vm.wavefront_size): vm.a[i][j] = 0

    gemm(context, "test_sgemm_vm", "gfx942:xnack-", gemm_config, opt, kern_args)
    vm.run(context)
    raw_d = vm.vmem.mem[d_offset:d_offset+d_size]
    d_from_accvgpr = vm.accvgpr_to_ndarray(AccVgprRange(0, 16), 32, 32, 2)
    d = np.frombuffer(raw_d, dtype=np.float32).reshape(n, m)
    a_mat = a.reshape(k, m)
    b_mat = b.reshape(n, k)
    ref_d = (b_mat @ a_mat)
    assert np.allclose(d_from_accvgpr, d, 1e-5)
    assert np.allclose(d, ref_d, 1e-5)
