from vm.gcn_virtual_machine import GcnVirtualMachine
from generator.generator import AccVgpr, Vgpr, Sgpr, SgprRange, GpuContext

vm = GcnVirtualMachine(104, 256, 256)


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
    context.v_mov_b32(Vgpr(1), 3.14)
    context.v_mul_f32(Vgpr(0), Vgpr(1), 55.66)
    vm.run(context)
    assert all(i == (3.14 * 55.66) for i in vm.v[0])

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