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
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle
from generator.atoms import MFMA_F32_32x32x2_F32, MFMA_F32_16x16x4_F32
from generator.target_spec import GFX90A
from vm.gcn_virtual_machine import GcnVirtualMachine


@pytest.mark.parametrize("trans_a, trans_b", [
    (False, False),  # NN
    (False, True),   # NT
    (True, False),   # TN
    (True, True),    # TT
])
@pytest.mark.parametrize("single_buffer_lds", [False, True])
def test_vm_gemm_transposes(trans_a: bool, trans_b: bool, single_buffer_lds: bool):
    """
    Test GEMM execution in software VM for all 4 transpose combinations:
    NN, NT, TN, TT against NumPy reference, testing both single and double buffered LDS.
    """
    vm = GcnVirtualMachine(104, 256, 64)
    context = GpuContext()

    m, n, k = 32, 32, 32
    gemm_config = GemmSolutionConfig(
        DataType.FP32,
        DataType.FP32,
        DataType.FP32,
        DataType.FP32,
        (32, 32, 1, 2),
        (1, 1),
        (1, 1),
        16,
        trans_a,
        trans_b,
        vmem_stage=1,
        single_buffer_lds=single_buffer_lds,
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

    # Setup thread IDs for wavefront
    for i in range(vm.wavefront_size):
        vm.v[0][i] = i

    # Base pointers in VM SMEM
    vm.s[0] = 0
    vm.s[1] = 0
    vm.s[2] = 0
    vm.s[3] = 0

    a_size = m * k * 4
    b_size = k * n * 4
    c_size = m * n * 4
    d_size = m * n * 4

    a_offset = 0
    b_offset = a_size
    c_offset = a_size + b_size
    d_offset = a_size + b_size + c_size

    lda = k if trans_a else m
    ldb = n if trans_b else k
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
    vm.smem.mem[64:68] = struct.pack("f", 1.0)
    vm.smem.mem[68:72] = int.to_bytes(1, 4, "little")
    vm.smem.mem[72:76] = int.to_bytes(1, 4, "little")

    # Generate matrix test data
    np.random.seed(42)
    # Underlying column-major memory layout:
    # If trans_a: A_mem has shape (k, m), element is A_mem[k_idx, m_idx]
    # If not trans_a: A_mem has shape (m, k), element is A_mem[m_idx, k_idx]
    shape_a = (k, m) if trans_a else (m, k)
    shape_b = (n, k) if trans_b else (k, n)

    a_mat = np.random.uniform(-1.0, 1.0, size=shape_a).astype(np.float32)
    b_mat = np.random.uniform(-1.0, 1.0, size=shape_b).astype(np.float32)
    c_mat = np.random.uniform(-1.0, 1.0, size=(m, n)).astype(np.float32)

    # Flatten in Fortran (column-major) order to match GPU memory layout
    a_bytes = a_mat.tobytes("F")
    b_bytes = b_mat.tobytes("F")
    c_bytes = c_mat.tobytes("F")

    vm.vmem.mem[a_offset:a_offset + a_size] = bytearray(a_bytes)
    vm.vmem.mem[b_offset:b_offset + b_size] = bytearray(b_bytes)
    vm.vmem.mem[c_offset:c_offset + c_size] = bytearray(c_bytes)

    gemm(
        context,
        f"gemm_{'t' if trans_a else 'n'}{'t' if trans_b else 'n'}",
        "gfx90a:xnack-",
        gemm_config,
        opt,
        kern_args,
    )

    vm.run(context)

    # Read output matrix D from VM memory
    raw_d = vm.vmem.mem[d_offset:d_offset + d_size]
    d_gpu = np.frombuffer(raw_d, dtype=np.float32).reshape((n, m)).T

    # Compute CPU reference according to transpose mode
    op_a = a_mat.T if trans_a else a_mat
    op_b = b_mat.T if trans_b else b_mat
    ref_d = op_a @ op_b + c_mat

    assert np.allclose(d_gpu, ref_d, atol=1e-4), (
        f"Mismatch in {'t' if trans_a else 'n'}{'t' if trans_b else 'n'} GEMM! "
        f"Max diff: {np.max(np.abs(d_gpu - ref_d))}"
    )


def test_dsl_transposes():
    """Verify that GemmKernel properly supports set_transposes and canonical naming."""
    k_nn = GemmKernel("test_nn", target=GFX90A).set_transposes(False, False)
    cfg_nn, _ = k_nn.to_gemm_solution_config()
    assert "sgemm_nn" in cfg_nn.canonical_name

    k_nt = GemmKernel("test_nt", target=GFX90A).set_transposes(False, True)
    cfg_nt, _ = k_nt.to_gemm_solution_config()
    assert "sgemm_nt" in cfg_nt.canonical_name

    k_tn = GemmKernel("test_tn", target=GFX90A).set_transposes(True, False)
    cfg_tn, _ = k_tn.to_gemm_solution_config()
    assert "sgemm_tn" in cfg_tn.canonical_name

    k_tt = GemmKernel("test_tt", target=GFX90A).set_transposes(True, True)
    cfg_tt, _ = k_tt.to_gemm_solution_config()
    assert "sgemm_tt" in cfg_tt.canonical_name


def test_bundle_with_all_transposes():
    """Verify bundling kernels of different transpose types in a GemmKernelBundle."""
    bundle = GemmKernelBundle("trans_bundle", target=GFX90A)
    for ta in [False, True]:
        for tb in [False, True]:
            k = GemmKernel(target=GFX90A).set_transposes(ta, tb).set_tiling(
                block_tile=(128, 128, 16),
                wave_group=(2, 2),
                wave_tiling=(2, 2),
            )
            bundle.add(k)

    assert len(bundle) == 4
    assert bundle.find_kernel(trans_a=False, trans_b=False) is not None
    assert bundle.find_kernel(trans_a=False, trans_b=True) is not None
    assert bundle.find_kernel(trans_a=True, trans_b=False) is not None
    assert bundle.find_kernel(trans_a=True, trans_b=True) is not None

    asm = bundle.generate_assembly()
    assert "sgemm_nn" in asm
    assert "sgemm_nt" in asm
    assert "sgemm_tn" in asm
    assert "sgemm_tt" in asm
