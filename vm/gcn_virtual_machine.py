import struct
import numpy as np
from generator.generator import (
    Vgpr,
    VgprRange,
    Sgpr,
    SgprRange,
    AccVgpr,
    AccVgprRange,
    GpuContext,
)


class GcnVirtualMachine:
    def __init__(self, num_total_sgpr: int, num_total_vgpr: int, wavefront_size: int):
        self.wavefront_size = wavefront_size
        self.s = [0] * num_total_sgpr
        self.v = [[0] * wavefront_size for _ in range(num_total_vgpr)]
        self.a = [[0] * wavefront_size for _ in range(num_total_vgpr)]
        self.scc = 0
        self.vcc = [0] * num_total_vgpr
        self.pc = 0
        self.pc_end = 0
        self.labels = {}
        self.smem = bytearray(1024)
        self.vmem = bytearray(8192)
        self.lds = bytearray(65536)

    def _accumulate_labels(self, context: GpuContext):
        self.label = {}

        for line_no, inst in enumerate(context.instructions):
            inst_str: str = inst[0]().split(" ")[0]
            if inst_str.endswith(":"):
                self.labels[inst_str[:-1]] = line_no

    def run(self, context: GpuContext):
        self._accumulate_labels(context)
        self.pc = 0
        self.pc_end = len(context.instructions)

        while self.pc < self.pc_end:
            inst = context.instructions[self.pc]
            inst_str: str = inst[0]().split(" ")[0]
            args = inst[1:]
            if hasattr(self, inst_str):
                f = getattr(self, inst_str)
                f(*args)
            elif inst_str.endswith(":"):
                print(f"Label found: {inst_str}")
            elif inst_str.startswith('//'):
                print(f"comment: {inst_str}")
            else:
                print(f"Unsupported instruction: {inst_str}")
            self.pc += 1

    def s_mov_b32(self, dst: Sgpr, src: Sgpr | int | float):
        val = src if isinstance(src, int) else self.s[src.index]
        self.s[dst.index] = val

    def s_mov_b64(self, dst: SgprRange, src: SgprRange):
        assert dst.size == src.size

        for i in range(dst.size):
            self.s[dst.index + i] = self.s[src.index + i]

    def s_lshl_b32(self, dst: Sgpr, src: Sgpr, shift: int | Sgpr):
        val = shift if isinstance(shift, int) else self.s[shift.index]
        self.s[dst.index] = self.s[src.index] << val

    def s_lshr_b32(self, dst: Sgpr, src: Sgpr, shift: int | Sgpr):
        val = shift if isinstance(shift, int) else self.s[shift.index]
        self.s[dst.index] = self.s[src.index] >> val

    def s_mul_i32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        val = src1 if isinstance(src1, int) else self.s[src1.index]
        self.s[dst.index] = self.s[src0.index] * val

    def s_add_i32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        val = src1 if isinstance(src1, int) else self.s[src1.index]
        self.s[dst.index] = self.s[src0.index] + val

    def s_sub_i32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        val = src1 if isinstance(src1, int) else self.s[src1.index]
        self.s[dst.index] = self.s[src0.index] - val

    def s_and_b32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        val = src1 if isinstance(src1, int) else self.s[src1.index]
        self.s[dst.index] = self.s[src0.index] & val

    def s_cmp_lt_u32(self, lhs: Sgpr | int, rhs: Sgpr | int):
        val_lhs = lhs if isinstance(lhs, int) else self.s[lhs.index]
        val_rhs = rhs if isinstance(rhs, int) else self.s[rhs.index]
        self.scc = int(val_lhs < val_rhs)

    def s_cmp_le_u32(self, lhs: Sgpr | int, rhs: Sgpr | int):
        val_lhs = lhs if isinstance(lhs, int) else self.s[lhs.index]
        val_rhs = rhs if isinstance(rhs, int) else self.s[rhs.index]
        self.scc = int(val_lhs <= val_rhs)

    def s_cmp_eq_u32(self, lhs: Sgpr | int, rhs: Sgpr | int):
        val_lhs = lhs if isinstance(lhs, int) else self.s[lhs.index]
        val_rhs = rhs if isinstance(rhs, int) else self.s[rhs.index]
        self.scc = int(val_lhs == val_rhs)

    def s_cselect_b32(self, dst: Sgpr, lhs: Sgpr | int, rhs: Sgpr | int):
        val_lhs = lhs if isinstance(lhs, int) else self.s[lhs.index]
        val_rhs = rhs if isinstance(rhs, int) else self.s[rhs.index]

        if self.scc:
            self.s[dst.index] = val_lhs
        else:
            self.s[dst.index] = val_rhs

    def s_cbranch_scc1(self, name: str):
        if self.scc:
            next_pc = self.labels[GpuContext.get_label_name(name)]
            self.pc = next_pc

    def s_branch(self, name: str):
        next_pc = self.labels[GpuContext.get_label_name(name)]
        self.pc = next_pc

    def s_barrier(self):
        pass

    def s_waitcnt(self, vmcnt: int = None, lgkmcnt: int = None):
        pass

    def s_endpgm(self):
        self.pc = self.pc_end

    def v_mov_b32(self, dst: Vgpr, src: Sgpr | Vgpr | int | float):
        if not isinstance(src, Vgpr):
            val = self.s[src.index] if isinstance(src, Sgpr) else src
            for i in range(self.wavefront_size):
                self.v[dst.index][i] = val
        else:
            self.v[dst.index] = self.v[src.index][:]

    def _get_v_inst_src_val(self, src: Vgpr | Sgpr | AccVgpr | int | float):
        if not isinstance(src, (Vgpr, AccVgpr)):
            val = self.s[src.index] if isinstance(src, Sgpr) else src
            return [val] * self.wavefront_size
        elif isinstance(src, AccVgpr):
            return self.a[src.index]
        else:
            return self.v[src.index]


    def v_and_b32(
        self, dst: Vgpr, src0: Vgpr | int | float, src1: Sgpr | Vgpr | int | float
    ):
        val0, val1 = self._get_v_inst_src_val(src0), self._get_v_inst_src_val(src1)
        for i in range(self.wavefront_size):
            self.v[dst.index][i] = val0[i] & val1[i]

    def v_add_u32(self, dst: Vgpr, src0: Vgpr | int, src1: Sgpr | Vgpr | int):
        val0, val1 = self._get_v_inst_src_val(src0), self._get_v_inst_src_val(src1)
        for i in range(self.wavefront_size):
            self.v[dst.index][i] = val0[i] + val1[i]

    def v_add_i32(self, dst: Vgpr, src0: Vgpr | int, src1: Sgpr | Vgpr | int):
        self.v_add_u32(dst, src0, src1)

    def v_lshlrev_b32(
        self, dst: Vgpr, shift: Vgpr | int | float, src: Sgpr | Vgpr | int | float
    ):
        shift_val, val = self._get_v_inst_src_val(shift), self._get_v_inst_src_val(src)
        for i in range(self.wavefront_size):
            self.v[dst.index][i] = val[i] << shift_val[i]

    def v_lshrrev_b32(
        self, dst: Vgpr, shift: Vgpr | int | float, src: Sgpr | Vgpr | int | float
    ):
        shift_val, val = self._get_v_inst_src_val(shift), self._get_v_inst_src_val(src)
        for i in range(self.wavefront_size):
            self.v[dst.index][i] = val[i] >> shift_val[i]

    def v_mul_lo_u32(self, dst: Vgpr, src0: Vgpr | int, src1: Sgpr | Vgpr | int):
        val0, val1 = self._get_v_inst_src_val(src0), self._get_v_inst_src_val(src1)
        for i in range(self.wavefront_size):
            self.v[dst.index][i] = val0[i] * val1[i]

    @staticmethod
    def float_to_gpr_val(f: float) -> int:
        return int.from_bytes(struct.pack("f", f), "little")

    @staticmethod
    def gpr_val_to_float(v: int) -> float:
        return struct.unpack("f", int.to_bytes(v, 4, "little"))[0]

    def v_mul_f32(self, dst: Vgpr, src0: Vgpr | float, src1: Sgpr | Vgpr | float):
        val0, val1 = self._get_v_inst_src_val(src0), self._get_v_inst_src_val(src1)
        val0, val1 = [GcnVirtualMachine.gpr_val_to_float(val0[i]) for i in range(self.wavefront_size)], [GcnVirtualMachine.gpr_val_to_float(val1[i]) for i in range(self.wavefront_size)]
        for i in range(self.wavefront_size):
            self.v[dst.index][i] = GcnVirtualMachine.float_to_gpr_val(val0[i] * val1[i])

    def v_fma_f32(
        self,
        dst: Vgpr,
        src0: Vgpr | float,
        src1: Sgpr | Vgpr | float,
        src2: Sgpr | Vgpr | float,
    ):
        a, b, c = self._get_v_inst_src_val(src0), self._get_v_inst_src_val(src1), self._get_v_inst_src_val(src2)
        a = [GcnVirtualMachine.gpr_val_to_float(i) for i in a]
        b = [GcnVirtualMachine.gpr_val_to_float(i) for i in b]
        c = [GcnVirtualMachine.gpr_val_to_float(i) for i in c]
        for i in range(self.wavefront_size):
            self.v[dst.index][i] = GcnVirtualMachine.float_to_gpr_val(a[i] * b[i] + c[i])

    def v_mov_b64(self, dst: VgprRange, src: VgprRange | int | float):
        pass
        

    def v_accvgpr_write_b32(self, dst: AccVgpr, src: int | Vgpr):
        val = self._get_v_inst_src_val(src)
        for i in range(self.wavefront_size):
            self.a[dst.index][i] = val[i]


    def v_accvgpr_read_b32(self, dst: Vgpr, src: AccVgpr):
        val = self._get_v_inst_src_val(src)
        for i in range(self.wavefront_size):
            self.v[dst.index][i] = val[i]

    def buffer_load_dword(
        self,
        dst: Vgpr,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        assert srd.size == 4
        srd0, srd1, srd2, _ = srd.split()
        base = self.s[srd0.index] | (self.s[srd1.index] << 32)
        voffset_val = self._get_v_inst_src_val(voffset)
        soffset_val = self._get_v_inst_src_val(soffset)
        num_bytes = self._get_v_inst_src_val(srd2)[0]
        for i in range(self.wavefront_size):
            addr = base + voffset_val[i] + soffset_val[i] + const_offset
            self.v[dst.index][i] = int.from_bytes(self.vmem[addr:addr+4], "little") if addr < num_bytes else 0

    def buffer_load_dwordx2(
        self,
        dst: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        dst0, dst1 = dst.split()
        self.buffer_load_dword(dst0, voffset, srd, soffset, const_offset)
        self.buffer_load_dword(dst1, voffset, srd, soffset, const_offset+4)

    def buffer_load_dwordx4(
        self,
        dst: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        dst0, dst1 = dst.split(num_comp=2)
        self.buffer_load_dwordx2(dst0, voffset, srd, soffset, const_offset)
        self.buffer_load_dwordx2(dst1, voffset, srd, soffset, const_offset+8)

    def buffer_store_dword(
        self,
        data: Vgpr,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        assert srd.size == 4
        srd0, srd1, srd2, _ = srd.split()
        base = self.s[srd0.index] | (self.s[srd1.index] << 32)
        val = self._get_v_inst_src_val(data)
        voffset_val = self._get_v_inst_src_val(voffset)
        soffset_val = self._get_v_inst_src_val(soffset)
        num_bytes = self._get_v_inst_src_val(srd2)[0]
        for i in range(self.wavefront_size):
            addr = base + voffset_val[i] + soffset_val[i] + const_offset
            if addr < num_bytes:
                self.vmem[addr:addr+4] = int.to_bytes(val[i], 4, "little")


    def buffer_store_dwordx2(
        self,
        data: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        dst0, dst1 = data.split()
        self.buffer_store_dword(dst0, voffset, srd, soffset, const_offset)
        self.buffer_store_dword(dst1, voffset, srd, soffset, const_offset+4)

    def buffer_store_dwordx4(
        self,
        data: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        dst0, dst1 = data.split(2)
        self.buffer_store_dwordx2(dst0, voffset, srd, soffset, const_offset)
        self.buffer_store_dwordx2(dst1, voffset, srd, soffset, const_offset+8)

    def ds_write_b32(self, dst: Vgpr, vdata: Vgpr, const_offset: int):
        val = self._get_v_inst_src_val(vdata)
        voffset_val = self._get_v_inst_src_val(dst)
        for i in range(self.wavefront_size):
            addr = voffset_val[i] + const_offset
            self.lds[addr:addr+4] = int.to_bytes(val[i], 4, "little")

    def ds_write_b64(self, dst: Vgpr, vdata: VgprRange, const_offset: int):
        d0, d1 = vdata.split()
        self.ds_write_b32(dst, d0, const_offset)
        self.ds_write_b32(dst, d1, const_offset+4)

    def ds_write_b128(self, dst: Vgpr, vdata: VgprRange, const_offset: int):
        d0, d1 = vdata.split(2)
        self.ds_write_b64(dst, d0, const_offset)
        self.ds_write_b64(dst, d1, const_offset+8)


    def ds_read_b32(self, dst: Vgpr, voffset: Vgpr, const_offset: int):
        voffset_val = self._get_v_inst_src_val(voffset)
        for i in range(self.wavefront_size):
            addr = voffset_val[i] + const_offset
            self.v[dst.index][i] = int.from_bytes(self.lds[addr:addr+4], "little")

    def ds_read_b64(self, dst: VgprRange, voffset: Vgpr, const_offset: int):
        d0, d1 = dst.split()
        self.ds_read_b32(d0, voffset, const_offset)
        self.ds_read_b32(d1, voffset, const_offset+4)

    def ds_read_b128(self, dst: VgprRange, voffset: Vgpr, const_offset: int):
        d0, d1 = dst.split(2)
        self.ds_read_b64(d0, voffset, const_offset)
        self.ds_read_b64(d1, voffset, const_offset+8)

    def s_load_dword(self, dst: Sgpr, src: SgprRange, offset: int):
        assert src.size == 2
        srd0, srd1 = src.split()
        addr0 = self._get_v_inst_src_val(srd0)[0]
        addr1 = self._get_v_inst_src_val(srd1)[0]
        addr = ((addr1 << 32) | addr0) + offset
        self.s[dst.index] = int.from_bytes(self.smem[addr:addr+4], "little")
    def s_load_dwordx2(self, dst: SgprRange, src: SgprRange, offset: int):
        d0, d1 = dst.split()
        self.s_load_dword(d0, src, offset)
        self.s_load_dword(d1, src, offset+4)

    def s_load_dwordx4(self, dst: SgprRange, src: SgprRange, offset: int):
        d0, d1 = dst.split(2)
        self.s_load_dwordx2(d0, src, offset)
        self.s_load_dwordx2(d1, src, offset+8)

    def accvgpr_to_ndarray(self, acc: AccVgprRange, mi_m: int, mi_n: int, mi_k: int, dtype_str="f"):
        num_vert_tiles = mi_m * mi_n // self.wavefront_size // 4
        result = []

        for tile in range(num_vert_tiles):
            c_val = [self._get_v_inst_src_val(i) for i in acc.split()[tile*4:tile*4+4]]
            c_val = [[struct.unpack(dtype_str, int.to_bytes(j, 4, "little"))[0] for j in v] for v in c_val]
            #[4, 64]
            c_val = np.array(c_val, dtype=np.float32)
            #[64, 4]
            c_val = c_val.transpose([1, 0])
            c_val = c_val.reshape([mi_n//num_vert_tiles//4, mi_m, 4])
            c_val = c_val.transpose([1, 0, 2])
            c_val = c_val.reshape([mi_m, mi_n//num_vert_tiles])#col-major
            result.append(c_val)
        return np.hstack(result)

    def vgpr_to_ndarray(self, v: Vgpr | VgprRange, m: int, k: int, dtype_str="f"):
        val = self._get_v_inst_src_val(v)
        val = [struct.unpack(dtype_str, int.to_bytes(i, 4, "little"))[0] for i in val]
        val = np.array(val, dtype=np.float32)
        return val.reshape([k, m])

    def v_mfma_f32_16x16x4f32(
        self, acc: AccVgprRange, a: Vgpr, b: Vgpr, c: AccVgprRange
    ):
        a_val = self.vgpr_to_ndarray(a, 16, 4)
        b_val = self.vgpr_to_ndarray(b, 16, 4)
        result = b_val.T @ a_val
        result = result
        c_val = self.accvgpr_to_ndarray(c, 16, 16, 4)
        result += c_val
        dst_acc_indices = [i.index for i in acc.split()]

        with np.nditer(result, flags=["multi_index"]) as it:
            for x in it:
                col, row = it.multi_index
                accvgpr_idx = row % 4
                lane_idx = (col % 16) + (row // 4) * 16
                val = int.from_bytes(struct.pack("f", float(x)), "little")
                self.a[dst_acc_indices[accvgpr_idx]][lane_idx] = val


    def v_mfma_f32_32x32x2f32(
        self, acc: AccVgprRange, a: Vgpr, b: Vgpr, c: AccVgprRange):
        a_val = self.vgpr_to_ndarray(a, 32, 2)
        b_val = self.vgpr_to_ndarray(b, 32, 2)
        result = b_val.T @ a_val
        result = result
        c_val = self.accvgpr_to_ndarray(c, 32, 32, 2)
        result += c_val
        dst_acc_indices = [i.index for i in acc.split()]

        with np.nditer(result, flags=["multi_index"]) as it:
            for x in it:
                col, row = it.multi_index
                accvgpr_idx = row % 4 + (row // 8) * 4
                lane_idx = (col % 32) + ((row % 8) // 4) * 32
                assert accvgpr_idx < 16
                assert lane_idx < self.wavefront_size
                val = int.from_bytes(struct.pack("f", float(x)), "little")
                self.a[dst_acc_indices[accvgpr_idx]][lane_idx] = val
