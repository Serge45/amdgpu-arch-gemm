import struct
import numpy as np
from vm.gcn_virtual_machine import GcnVirtualMachine
from generator.generator import AccVgpr, AccVgprRange, Vgpr, VgprRange, Sgpr, SgprRange, GpuContext

vm = GcnVirtualMachine(104, 256, 64)


def test_s_mov_b32():
    vm.s_mov_b32(Sgpr(3), 5)
    assert vm.s[3] == 5
    vm.s_mov_b32(Sgpr(2), Sgpr(3))
    assert vm.s[2] == 5


def test_s_mov_b64():
    vm.s_mov_b32(Sgpr(2), 5)
    vm.s_mov_b32(Sgpr(3), 6)
    vm.s_mov_b64(SgprRange(0, 2), SgprRange(2, 2))
    assert vm.s[0] == 5
    assert vm.s[1] == 6


def test_s_lshl_b32():
    vm.s_mov_b32(Sgpr(1), 1)
    vm.s_lshl_b32(Sgpr(0), Sgpr(1), 3)
    assert vm.s[0] == (1 << 3)


def test_s_lshr_b32():
    vm.s_mov_b32(Sgpr(1), 16)
    vm.s_lshr_b32(Sgpr(0), Sgpr(1), 3)
    assert vm.s[0] == (16 >> 3)


def test_s_mul_i32():
    vm.s_mov_b32(Sgpr(1), 16)
    vm.s_mul_i32(Sgpr(0), Sgpr(1), 3)
    assert vm.s[0] == 16 * 3


def test_s_add_i32():
    vm.s_mov_b32(Sgpr(1), 16)
    vm.s_add_i32(Sgpr(0), Sgpr(1), 3)
    assert vm.s[0] == 16 + 3


def test_s_sub_i32():
    vm.s_mov_b32(Sgpr(1), 16)
    vm.s_sub_i32(Sgpr(0), Sgpr(1), 3)
    assert vm.s[0] == 16 - 3


def test_s_and_b32():
    vm.s_mov_b32(Sgpr(1), 16)
    vm.s_and_b32(Sgpr(0), Sgpr(1), 3)
    assert vm.s[0] == 16 & 3


def test_simple_program():
    context = GpuContext()
    context.s_mov_b32(Sgpr(0), 9)
    context.s_mul_i32(Sgpr(0), Sgpr(0), 4)
    context.s_mov_b32(Sgpr(1), 3)
    context.s_add_i32(Sgpr(0), Sgpr(0), Sgpr(1))
    vm.run(context)
    assert vm.s[0] == 39

def test_s_div_u32():
    context = GpuContext()
    context.s_mov_b32(Sgpr(0), 9)
    context.s_mov_b32(Sgpr(1), 5)
    context.s_div_u32(Sgpr(2), Sgpr(3), Sgpr(0), Sgpr(1))
    vm.run(context)
    assert vm.s[2] == 1
    assert vm.s[3] == 4

def test_s_div_u32_equal():
    context = GpuContext()
    context.s_mov_b32(Sgpr(0), 5)
    context.s_mov_b32(Sgpr(1), 5)
    context.s_div_u32(Sgpr(2), Sgpr(3), Sgpr(0), Sgpr(1))
    vm.run(context)
    assert vm.s[2] == 1
    assert vm.s[3] == 0

def test_s_div_u32_zero_quotient():
    context = GpuContext()
    context.s_mov_b32(Sgpr(0), 4)
    context.s_mov_b32(Sgpr(1), 5)
    context.s_div_u32(Sgpr(2), Sgpr(3), Sgpr(0), Sgpr(1))
    vm.run(context)
    assert vm.s[2] == 0
    assert vm.s[3] == 4

def test_v_mov_b32_const():
    context = GpuContext()
    context.v_mov_b32(Vgpr(2), 4)
    vm.run(context)
    assert all(i == 4 for i in vm.v[2])

def test_v_mov_b32_float():
    context = GpuContext()
    context.v_mov_b32(Vgpr(2), 4.)
    vm.run(context)
    assert all(struct.unpack("f", int.to_bytes(i, 4, "little"))[0] == 4. for i in vm.v[2])

def test_v_mov_b32_sgpr():
    context = GpuContext()
    context.s_mov_b32(Sgpr(0), 9)
    context.v_mov_b32(Vgpr(2), Sgpr(0))
    vm.run(context)
    assert all(i == 9 for i in vm.v[2])

def test_v_mov_b32_vgpr():
    context = GpuContext()
    context.v_mov_b32(Vgpr(0), 5)
    context.v_mov_b32(Vgpr(2), Vgpr(0))
    vm.run(context)
    assert all(i == 5 for i in vm.v[2])

def test_v_add_u32():
    context = GpuContext()
    context.v_mov_b32(Vgpr(1), 2)
    context.v_add_u32(Vgpr(2), Vgpr(1), 4)
    vm.run(context)
    assert all(i == 6 for i in vm.v[2])

def test_v_lshl_b32():
    context = GpuContext()
    context.v_mov_b32(Vgpr(1), 2)
    context.v_lshlrev_b32(Vgpr(2), 3, Vgpr(1))
    vm.run(context)
    assert all(i == (2 << 3) for i in vm.v[2])

def test_v_lshr_b32():
    context = GpuContext()
    context.v_mov_b32(Vgpr(1), 16)
    context.v_lshrrev_b32(Vgpr(2), 3, Vgpr(1))
    vm.run(context)
    assert all(i == (16 >> 3) for i in vm.v[2])


def test_v_mul_lo_u32():
    context = GpuContext()
    context.v_mov_b32(Vgpr(1), 3)
    context.v_mul_lo_u32(Vgpr(0), Vgpr(1), 2)
    vm.run(context)
    assert all(i == 6 for i in vm.v[0])


def test_v_mul_f32():
    context = GpuContext()
    a = int.from_bytes(struct.pack("f", 3.14), "little")
    b = int.from_bytes(struct.pack("f", 55.66), "little")
    context.v_mov_b32(Vgpr(1), a)
    context.v_mul_f32(Vgpr(0), Vgpr(1), b)
    vm.run(context)
    assert all(abs(struct.unpack("f", int.to_bytes(i, 4, "little"))[0] - (3.14 * 55.66)) < 1e-5 for i in vm.v[0])

def test_v_fma_f32():
    context = GpuContext()
    a = int.from_bytes(struct.pack("f", 1.0), "little")
    b = int.from_bytes(struct.pack("f", 2.0), "little")
    c = int.from_bytes(struct.pack("f", 3.0), "little")
    context.v_mov_b32(Vgpr(1), a)
    context.v_mov_b32(Vgpr(2), b)
    context.v_mov_b32(Vgpr(0), c)
    context.v_fma_f32(Vgpr(0), Vgpr(1), Vgpr(2), Vgpr(0))
    vm.run(context)
    assert all(abs(struct.unpack("f", int.to_bytes(i, 4, "little"))[0]-(1.0*2.0+3.0)) < 1e-5 for i in vm.v[0])

def test_v_accvgpr_write_b32():
    context = GpuContext()
    context.v_mov_b32(Vgpr(1), 9)
    context.v_accvgpr_write_b32(AccVgpr(0), Vgpr(1))
    vm.run(context)
    assert all(i == 9 for i in vm.a[0])

def test_v_accvgpr_read_b32():
    context = GpuContext()
    context.v_accvgpr_write_b32(AccVgpr(1), 9)
    context.v_accvgpr_read_b32(Vgpr(0), AccVgpr(1))
    vm.run(context)
    assert all(i == 9 for i in vm.v[0])

def test_buffer_load_dword():
    for i in range(0, len(vm.vmem), 4):
        vm.vmem.mem[i:i+4] = int.to_bytes(i//4, 4, "little")

    context = GpuContext()
    context.s_mov_b32(Sgpr(0), 0)
    context.s_mov_b32(Sgpr(1), 0)
    context.s_mov_b32(Sgpr(2), len(vm.vmem))
    context.s_mov_b32(Sgpr(3), 0)
    context.s_mov_b32(Sgpr(4), 0)

    for i in range(len(vm.v[0])):
        vm.v[0][i] = i * 4

    context.buffer_load_dword(Vgpr(1), Vgpr(0), SgprRange(0, 4), Sgpr(4), 0)
    vm.run(context)

    for idx, val in enumerate(vm.v[1]):
        assert val == idx


def test_buffer_load_dwordx2():
    for i in range(0, len(vm.vmem), 8):
        vm.vmem.mem[i:i+8] = int.to_bytes(i//8, 8, "little")

    context = GpuContext()
    context.s_mov_b32(Sgpr(0), 0)
    context.s_mov_b32(Sgpr(1), 0)
    context.s_mov_b32(Sgpr(2), len(vm.vmem))
    context.s_mov_b32(Sgpr(3), 0)
    context.s_mov_b32(Sgpr(4), 0)

    for i in range(len(vm.v[0])):
        vm.v[0][i] = i * 8

    context.buffer_load_dwordx2(VgprRange(2, 2), Vgpr(0), SgprRange(0, 4), Sgpr(4), 0)
    vm.run(context)

    for idx, vals in enumerate(zip(vm.v[2], vm.v[3])):
        assert (vals[0] | (vals[1] << 32)) == idx

def test_buffer_load_dwordx4():
    for i in range(0, len(vm.vmem), 16):
        vm.vmem.mem[i:i+16] = int.to_bytes(i//16, 16, "little")

    context = GpuContext()
    context.s_mov_b32(Sgpr(0), 0)
    context.s_mov_b32(Sgpr(1), 0)
    context.s_mov_b32(Sgpr(2), len(vm.vmem))
    context.s_mov_b32(Sgpr(3), 0)
    context.s_mov_b32(Sgpr(4), 0)

    for i in range(len(vm.v[0])):
        vm.v[0][i] = i * 16

    context.buffer_load_dwordx4(VgprRange(2, 4), Vgpr(0), SgprRange(0, 4), Sgpr(4), 0)
    vm.run(context)

    for idx, vals in enumerate(zip(vm.v[2], vm.v[3], vm.v[4], vm.v[5])):
        assert (vals[0] | (vals[1] << 32) | (vals[2] << 64) | (vals[3] << 96)) == idx

def test_buffer_store_dword():
    context = GpuContext()
    context.s_mov_b32(Sgpr(0), 0)
    context.s_mov_b32(Sgpr(1), 0)
    context.s_mov_b32(Sgpr(2), len(vm.vmem))
    context.s_mov_b32(Sgpr(3), 0)
    context.s_mov_b32(Sgpr(4), 0)

    for i in range(len(vm.v[0])):
        vm.v[0][i] = i * 4
        vm.v[1][i] = i

    context.buffer_store_dword(Vgpr(1), Vgpr(0), SgprRange(0, 4), Sgpr(4), 0)
    vm.run(context)

    for i in range(0, vm.wavefront_size*4, 4):
        assert int.from_bytes(vm.vmem.mem[i:i+4], "little") == i // 4

def test_buffer_store_dwordx2():
    context = GpuContext()
    context.s_mov_b32(Sgpr(0), 0)
    context.s_mov_b32(Sgpr(1), 0)
    context.s_mov_b32(Sgpr(2), len(vm.vmem))
    context.s_mov_b32(Sgpr(3), 0)
    context.s_mov_b32(Sgpr(4), 0)

    for i in range(len(vm.v[0])):
        vm.v[0][i] = i * 8
        vm.v[2][i] = i
        vm.v[3][i] = i >> 32

    context.buffer_store_dwordx2(VgprRange(2, 2), Vgpr(0), SgprRange(0, 4), Sgpr(4), 0)
    vm.run(context)

    for i in range(0, vm.wavefront_size*8, 8):
        assert int.from_bytes(vm.vmem.mem[i:i+8], "little") == i // 8

def test_buffer_store_dwordx4():
    context = GpuContext()
    context.s_mov_b32(Sgpr(0), 0)
    context.s_mov_b32(Sgpr(1), 0)
    context.s_mov_b32(Sgpr(2), len(vm.vmem))
    context.s_mov_b32(Sgpr(3), 0)
    context.s_mov_b32(Sgpr(4), 0)

    for i in range(len(vm.v[0])):
        vm.v[0][i] = i * 16
        vm.v[2][i] = i
        vm.v[3][i] = i >> 32
        vm.v[4][i] = i >> 64
        vm.v[5][i] = i >> 96

    context.buffer_store_dwordx4(VgprRange(2, 4), Vgpr(0), SgprRange(0, 4), Sgpr(4), 0)
    vm.run(context)

    for i in range(0, vm.wavefront_size*16, 16):
        assert int.from_bytes(vm.vmem.mem[i:i+16], "little") == i // 16

def _test_ds_write_template(num_bytes_per_load: int):
    context = GpuContext()

    for i in range(len(vm.v[0])):
        vm.v[0][i] = i * num_bytes_per_load

    for i in range(len(vm.v[1])):
        vm.v[1][i] = i

        if num_bytes_per_load >= 8:
            vm.v[2][i] = (i >> 32)

        if num_bytes_per_load >= 16:
            vm.v[3][i] = (i >> 64)
            vm.v[4][i] = (i >> 96)

    if num_bytes_per_load == 4:
        context.ds_write_b32(Vgpr(0), Vgpr(1), 0)
    elif num_bytes_per_load == 8:
        context.ds_write_b64(Vgpr(0), VgprRange(1, 2), 0)
    else:
        context.ds_write_b128(Vgpr(0), VgprRange(1, 4), 0)
    context.s_waitcnt(vmcnt=None, lgkmcnt=0)
    vm.run(context)

    for i in range(vm.wavefront_size):
        assert int.from_bytes(vm.lds.mem[num_bytes_per_load*i:num_bytes_per_load*i+num_bytes_per_load], "little") == i

def test_ds_write_b32():
    _test_ds_write_template(4)

def test_ds_write_b64():
    _test_ds_write_template(8)

def test_ds_write_b128():
    _test_ds_write_template(16)

def _test_ds_read_template(num_bytes_per_load: int):
    for i in range(vm.wavefront_size):
        vm.lds.mem[i*num_bytes_per_load:(i+1)*num_bytes_per_load] = int.to_bytes(i, num_bytes_per_load, "little")
        vm.v[0][i] = i*num_bytes_per_load

    context = GpuContext()
    if num_bytes_per_load == 4:
        context.ds_read_b32(Vgpr(1), Vgpr(0), 0)
    elif num_bytes_per_load == 8:
        context.ds_read_b64(VgprRange(1, 2), Vgpr(0), 0)
    else:
        context.ds_read_b128(VgprRange(1, 4), Vgpr(0), 0)
    vm.run(context)

    for i in range(vm.wavefront_size):
        if num_bytes_per_load == 4:
            assert vm.v[1][i] == i
        elif num_bytes_per_load == 8:
            assert vm.v[1][i] == i
            assert vm.v[2][i] == (i >> 32)
        else:
            assert vm.v[1][i] == i
            assert vm.v[2][i] == (i >> 32)
            assert vm.v[3][i] == (i >> 64)
            assert vm.v[4][i] == (i >> 96)

def test_ds_read_b32():
    _test_ds_read_template(4)

def test_ds_read_b64():
    _test_ds_read_template(8)

def test_ds_read_b128():
    _test_ds_read_template(16)

def _test_s_load_template(num_bytes_per_load: int):
    context = GpuContext()
    vm.smem.mem[:num_bytes_per_load] = int.to_bytes(9, num_bytes_per_load, "little")
    context.s_mov_b32(Sgpr(0), 0)
    context.s_mov_b32(Sgpr(1), 0)

    if num_bytes_per_load == 4:
        context.s_load_dword(Sgpr(2), SgprRange(0, 2), 0)
        context.s_waitcnt(lgkmcnt=0)
        vm.run(context)
        assert vm.s[2] == 9
    elif num_bytes_per_load == 8:
        context.s_load_dwordx2(SgprRange(2, 2), SgprRange(0, 2), 0)
        context.s_waitcnt(lgkmcnt=0)
        vm.run(context)
        assert (vm.s[2] | (vm.s[3] << 32)) == 9
    elif num_bytes_per_load == 16:
        context.s_load_dwordx4(SgprRange(2, 4), SgprRange(0, 2), 0)
        context.s_waitcnt(lgkmcnt=0)
        vm.run(context)
        assert (vm.s[2] | (vm.s[3] << 32) | (vm.s[4] << 64) | (vm.s[5] << 96)) == 9

def test_s_load_dword():
    _test_s_load_template(4)

def test_s_load_dwordx2():
    _test_s_load_template(8)

def test_s_load_dwordx4():
    _test_s_load_template(16)

def test_v_mfma_f32_16x16x4f32():
    for i in range(vm.wavefront_size):
        v = int.from_bytes(struct.pack("f", float(i)), "little")
        vm.v[0][i] = v
        vm.v[1][i] = v
        vm.a[0][i] = v
        vm.a[1][i] = v
        vm.a[2][i] = v
        vm.a[3][i] = v

    c = vm.accvgpr_to_ndarray(AccVgprRange(0, 4), 16, 16, 4)
    context = GpuContext()
    context.v_mfma_f32_16x16x4f32(AccVgprRange(0, 4), Vgpr(0), Vgpr(1), AccVgprRange(0, 4))
    vm.run(context)
    d = vm.accvgpr_to_ndarray(AccVgprRange(0, 4), 16, 16, 4)
    a = vm.vgpr_to_ndarray(Vgpr(0), 16, 4)
    b = vm.vgpr_to_ndarray(Vgpr(1), 16, 4)
    ref = b.T @ a + c
    assert np.allclose(d, ref)

def test_v_mfma_f32_32x32x2f32():
    for i in range(vm.wavefront_size):
        v = int.from_bytes(struct.pack("f", float(i%7)), "little")
        vm.v[0][i] = v
        vm.v[1][i] = v
        for j in range(16):
            vm.a[j][i] = 0

    c = vm.accvgpr_to_ndarray(AccVgprRange(0, 16), 32, 32, 2)
    context = GpuContext()
    context.v_mfma_f32_32x32x2f32(AccVgprRange(0, 16), Vgpr(0), Vgpr(1), AccVgprRange(0, 16))
    vm.run(context)
    d = vm.accvgpr_to_ndarray(AccVgprRange(0, 16), 32, 32, 2)
    a = vm.vgpr_to_ndarray(Vgpr(0), 32, 2)
    b = vm.vgpr_to_ndarray(Vgpr(1), 32, 2)
    ref = b.T @ a + c
    assert np.allclose(d, ref)

def test_run_sgemm_16x16x4():
    from generator.generator import gemm, GemmOptimizations, GemmSolutionConfig, DataType, FunctionArgument
    gemm_config = GemmSolutionConfig(
        DataType.FP32,
        DataType.FP32,
        DataType.FP32,
        DataType.FP32,
        (16, 16, 1, 4),
        (1, 1),
        (1, 1),
        16,
        False,
        False,
    )
    opt = GemmOptimizations(1)
    opt.plr = 1
    context = GpuContext()
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
    m, n, k = 16, 16, 64

    #setup thread ID
    for i in range(vm.wavefront_size):
        vm.v[0][i] = i

    #setup kern arg addr
    vm.s[0] = 0
    vm.s[1] = 0
    vm.s[2] = 0
    vm.s[3] = 0
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
    vm.smem.mem[64:68] = struct.pack("f", 1.0)
    vm.smem.mem[68:72] = int.to_bytes(1, 4, "little")
    vm.smem.mem[72:76] = int.to_bytes(1, 4, "little")
    a, b, c = np.arange(0, m*k, 1, dtype=np.float32), np.arange(0, n*k, 1, dtype=np.float32), np.ones(m*n, dtype=np.float32)
    vm.vmem.mem[a_offset:a_offset+a_size] = bytearray(a)
    vm.vmem.mem[b_offset:b_offset+b_size] = bytearray(b)
    vm.vmem.mem[c_offset:c_offset+c_size] = bytearray(c)

    kern_src = gemm(
        context,
        "gemm",
        "gfx90a:xnack-",
        gemm_config,
        opt,
        kern_args
    )

    print(kern_src)

    vm.run(context)
    raw_d = vm.vmem.mem[d_offset:d_offset+d_size]
    d_from_accvgpr = vm.accvgpr_to_ndarray(AccVgprRange(0, 4), 16, 16, 4)
    d = np.frombuffer(raw_d, dtype=np.float32).reshape(n, m)
    a_mat = a.reshape(64, 16)
    b_mat = b.reshape(16, 64)
    c_mat = c.reshape(16, 16)
    ref_d = (b_mat @ a_mat) + c_mat
    assert np.allclose(d_from_accvgpr, d, 1e-5)
    assert np.allclose(d, ref_d, 1e-5)

def test_run_sgemm_32x32x2():
    from generator.generator import gemm, GemmOptimizations, GemmSolutionConfig, DataType, FunctionArgument
    gemm_config = GemmSolutionConfig(
        DataType.FP32,
        DataType.FP32,
        DataType.FP32,
        DataType.FP32,
        (32, 32, 1, 2),
        (1, 1),
        (1, 1),
        16,
        False,
        False,
    )
    opt = GemmOptimizations(1)
    opt.plr = 1
    context = GpuContext()
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

    #setup thread ID
    for i in range(vm.wavefront_size):
        vm.v[0][i] = i

    #setup kern arg addr
    vm.s[0] = 0
    vm.s[1] = 0
    vm.s[2] = 0
    vm.s[3] = 0
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
    vm.smem.mem[64:68] = struct.pack("f", 1.0)
    vm.smem.mem[68:72] = int.to_bytes(1, 4, "little")
    vm.smem.mem[72:76] = int.to_bytes(1, 4, "little")
    a, b, c = np.arange(0, m*k, 1, dtype=np.float32), np.arange(0, n*k, 1, dtype=np.float32), np.ones(m*n, dtype=np.float32)
    vm.vmem.mem[a_offset:a_offset+a_size] = bytearray(a)
    vm.vmem.mem[b_offset:b_offset+b_size] = bytearray(b)
    vm.vmem.mem[c_offset:c_offset+c_size] = bytearray(c)

    gemm(
        context,
        "gemm",
        "gfx90a:xnack-",
        gemm_config,
        opt,
        kern_args
    )

    vm.run(context)
    raw_d = vm.vmem.mem[d_offset:d_offset+d_size]
    d_from_accvgpr = vm.accvgpr_to_ndarray(AccVgprRange(0, 16), 32, 32, 2)
    d = np.frombuffer(raw_d, dtype=np.float32).reshape(n, m)
    a_mat = a.reshape(k, m)
    b_mat = b.reshape(n, k)
    c_mat = c.reshape(m, n)
    ref_d = (b_mat @ a_mat) + c_mat
    assert np.allclose(d_from_accvgpr, d, 1e-5)
    assert np.allclose(d, ref_d, 1e-5)
