from __future__ import annotations
from typing import Optional, List, Tuple, Dict
from contextlib import contextmanager
from io import StringIO
from enum import IntEnum
import subprocess
from dataclasses import dataclass
import math
from itertools import cycle, islice
import argparse
import yaml
import tomli_w


def roundrobin(*iterables):
    iterators = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = cycle(islice(iterators, num_active))
        yield from map(next, iterators)


DEFAULT_CLANG_PATH = "/opt/rocm/llvm/bin/clang++"
MAX_LDS_NUM_BYTES = 65536


class Gpr:
    gpr_type: str = None

    def __init__(self, idx: Optional[int]):
        self.index = idx

    def __str__(self):
        return f"{self.gpr_type}[{self.index}]"


class GprRange:
    gpr_type: str = None
    underlying_gpr_type = Gpr

    def __init__(self, index: int, size: int):
        self.index = index
        self.size = size

    def __str__(self):
        return f"{self.gpr_type}[{self.index}:{self.index+self.size-1}]"

    def split(self, num_comp: int=1) -> List[Gpr]:
        if num_comp > 1:
            return [type(self)(self.index + i, num_comp) for i in range(0, self.size, num_comp)]
        else:
            return [type(self).underlying_gpr_type(self.index + i) for i in range(self.size)]


class Vgpr(Gpr):
    gpr_type: str = "v"


class Sgpr(Gpr):
    gpr_type: str = "s"


class AccVgpr(Gpr):
    gpr_type: str = "acc"

class VgprRange(GprRange):
    gpr_type: str = "v"
    underlying_gpr_type = Vgpr

class SgprRange(GprRange):
    gpr_type: str = "s"
    underlying_gpr_type = Sgpr


class AccVgprRange(GprRange):
    gpr_type: str = "acc"
    underlying_gpr_type = AccVgpr


class GprPool:
    gpr_type: str = None

    def __init__(self, size: int):
        self.size = size


class VgprPool(GprPool):
    gpr_type: str = "v"

    def __init__(self, size: int):
        super().__init__(size)
        self.pool = [i for i in range(size)]


class FunctionArgument:
    def __init__(self, typename: str, name: str, offset: Optional[int], num_bytes: int):
        self.typename = typename
        self.name = name
        self.offset = offset
        self.num_bytes = num_bytes
        self.address_space = "global" if typename == "global_buffer" else None


FunctionArgumentList = List[FunctionArgument]


def iter_kern_args(args: FunctionArgumentList):
    offset = 0
    for arg in args:
        yield (arg, arg.num_bytes, offset)
        offset += arg.offset


class FunctionMeta:
    def __init__(self, name: str, args: FunctionArgumentList):
        self.name = name
        self.kernarg_segment_size = 0
        self.group_segment_fixed_size = 0
        self.private_segment_fixed_size = 0
        self.kernarg_segment_align = 8
        self.wavefront_size = 64
        self.workgroup_size = 256
        self.sgpr_count = 0
        self.vgpr_count = 0
        self.agpr_count = 0
        self.args = args

    def normalized_args(self):
        offset = 0
        ret = []

        for arg in self.args:
            ret.append(
                {
                    ".size": arg.num_bytes,
                    ".offset": offset,
                    ".value_kind": arg.typename,
                    ".name": (
                        f"{arg.name}_val"
                        if arg.name.lower() in ("y", "n", "yes", "no", "true", "false", "on", "off")
                        else arg.name
                    ),
                }
            )

            if arg.address_space:
                ret[-1][".address_space"] = arg.address_space

            offset += arg.num_bytes

        self.kernarg_segment_size = offset
        return ret

    @property
    def argument_num_bytes(self):
        return sum(arg.num_bytes for arg in self.args)

    @property
    def argument_num_sgpr(self):
        return self.argument_num_bytes // 4

    def ro_data(self):
        return f"""
.rodata
.p2align 6
.amdhsa_kernel {self.name}
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_accum_offset {max((self.vgpr_count+3)//4*4, 4)}
  .amdhsa_group_segment_fixed_size {self.group_segment_fixed_size}
  .amdhsa_next_free_vgpr {self.vgpr_count+self.agpr_count}
  .amdhsa_next_free_sgpr {self.sgpr_count}
.end_amdhsa_kernel
"""

    def to_metadata_dict(self):
        args = self.normalized_args()
        return {
            ".name": self.name,
            ".symbol": f"{self.name}.kd",
            ".kernarg_segment_size": self.kernarg_segment_size,
            ".group_segment_fixed_size": self.group_segment_fixed_size,
            ".private_segment_fixed_size": self.private_segment_fixed_size,
            ".kernarg_segment_align": self.kernarg_segment_align,
            ".wavefront_size": self.wavefront_size,
            ".max_flat_workgroup_size": self.workgroup_size,
            ".sgpr_count": self.sgpr_count,
            ".vgpr_count": self.vgpr_count,
            ".agpr_count": self.agpr_count,
            ".args": args,
        }

    def __str__(self):
        ret = {
            "amdhsa.version": [1, 1],
            "amdhsa.kernels": [self.to_metadata_dict()],
        }

        return f".amdgpu_metadata\n---\n{yaml.dump(ret)}...\n.end_amdgpu_metadata"


class MultiKernelMeta:
    def __init__(self, metas: List[FunctionMeta]):
        self.metas = metas

    def ro_data(self) -> str:
        return "\n".join(meta.ro_data() for meta in self.metas)

    def __str__(self) -> str:
        ret = {
            "amdhsa.version": [1, 1],
            "amdhsa.kernels": [m.to_metadata_dict() for m in self.metas],
        }
        return f".amdgpu_metadata\n---\n{yaml.dump(ret)}...\n.end_amdgpu_metadata"


def count_calls(f):
    def wrapper(*args, **kwargs):
        wrapper._num_calls += 1
        return f(*args, **kwargs)

    wrapper._num_calls = 0
    return wrapper


def count_gprs(f):
    def wrapper(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, Vgpr):
                self.vgpr_counter = max(self.vgpr_counter, arg.index + 1)
                self.max_vgpr = max(self.max_vgpr, self.vgpr_counter)
            elif isinstance(arg, VgprRange):
                self.vgpr_counter = max(self.vgpr_counter, arg.index + arg.size)
                self.max_vgpr = max(self.max_vgpr, self.vgpr_counter)
            elif isinstance(arg, Sgpr):
                self.sgpr_counter = max(self.sgpr_counter, arg.index + 1)
                self.max_sgpr = max(self.max_sgpr, self.sgpr_counter)
            elif isinstance(arg, SgprRange):
                self.sgpr_counter = max(self.sgpr_counter, arg.index + arg.size)
                self.max_sgpr = max(self.max_sgpr, self.sgpr_counter)
            elif isinstance(arg, AccVgpr):
                self.agpr_counter = max(self.agpr_counter, arg.index + 1)
                self.max_agpr = max(self.max_agpr, self.agpr_counter)
            elif isinstance(arg, AccVgprRange):
                self.agpr_counter = max(self.agpr_counter, arg.index + arg.size)
                self.max_agpr = max(self.max_agpr, self.agpr_counter)

        return f(self, *args, **kwargs)

    return wrapper


class GpuContext:
    def __init__(self):
        self.content = StringIO()
        self.instructions = []
        self.sgpr_counter = 0
        self.vgpr_counter = 0
        self.agpr_counter = 0
        self.max_sgpr = 0
        self.max_vgpr = 0
        self.max_agpr = 0

    @staticmethod
    def get_label_name(name: str):
        return f"label_{name}"

    def label_name(self, name: str):
        return f"label_{name}"

    def label(self, name: str):
        self.instructions.append([lambda: f"{self.label_name(name)}:"])

    def comment(self, comment: str):
        self.instructions.append([lambda: f"//{comment}"])

    @count_gprs
    def buffer_load_dword(
        self,
        dst: Vgpr,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_load_dword {str(dst)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                dst,
                voffset,
                srd,
                soffset,
                const_offset,
            ]
        )

    @count_gprs
    def buffer_load_dwordx2(
        self,
        dst: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_load_dwordx2 {str(dst)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                dst,
                voffset,
                srd,
                soffset,
                const_offset,
            ]
        )

    @count_gprs
    def buffer_load_dwordx4(
        self,
        dst: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_load_dwordx4 {str(dst)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                dst,
                voffset,
                srd,
                soffset,
                const_offset,
            ]
        )

    def buffer_load_inst(self, num_dwords: int):
        if num_dwords == 1:
            return self.buffer_load_dword
        elif num_dwords == 2:
            return self.buffer_load_dwordx2
        elif num_dwords == 4:
            return self.buffer_load_dwordx4

    @count_gprs
    def buffer_store_dword(
        self,
        data: Vgpr,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_store_dword {str(data)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                data,
                voffset,
                srd,
                soffset,
                const_offset,
            ]
        )

    @count_gprs
    def buffer_store_dwordx2(
        self,
        data: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_store_dwordx2 {str(data)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                data,
                voffset,
                srd,
                soffset,
                const_offset,
            ]
        )

    @count_gprs
    def buffer_store_dwordx4(
        self,
        data: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_store_dwordx4 {str(data)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                data,
                voffset,
                srd,
                soffset,
                const_offset,
            ]
        )

    def buffer_store_inst(self, num_dwords: int):
        if num_dwords == 1:
            return self.buffer_store_dword
        elif num_dwords == 2:
            return self.buffer_store_dwordx2
        elif num_dwords == 4:
            return self.buffer_store_dwordx4

    @count_gprs
    def ds_write_b32(self, dst: Vgpr, vdata: Vgpr, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_write_b32 {str(dst)}, {str(vdata)}, offset:{const_offset}",
                dst,
                vdata,
                const_offset,
            ]
        )

    @count_gprs
    def ds_write_b64(self, dst: Vgpr, vdata: VgprRange, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_write_b64 {str(dst)}, {str(vdata)}, offset:{const_offset}",
                dst,
                vdata,
                const_offset,
            ]
        )

    @count_gprs
    def ds_write_b128(self, dst: Vgpr, vdata: VgprRange, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_write_b128 {str(dst)}, {str(vdata)}, offset:{const_offset}",
                dst,
                vdata,
                const_offset,
            ]
        )

    def ds_write_inst(self, num_bytes):
        if num_bytes == 4:
            return self.ds_write_b32
        elif num_bytes == 8:
            return self.ds_write_b64
        elif num_bytes == 16:
            return self.ds_write_b128

    @count_gprs
    def ds_read_b32(self, dst: Vgpr, voffset: Vgpr, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_read_b32 {str(dst)}, {str(voffset)}, offset:{const_offset}",
                dst,
                voffset,
                const_offset,
            ]
        )

    @count_gprs
    def ds_read_b64(self, dst: VgprRange, voffset: Vgpr, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_read_b64 {str(dst)}, {str(voffset)}, offset:{const_offset}",
                dst,
                voffset,
                const_offset,
            ]
        )

    @count_gprs
    def ds_read_b128(self, dst: VgprRange, voffset: Vgpr, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_read_b128 {str(dst)}, {str(voffset)}, offset:{const_offset}",
                dst,
                voffset,
                const_offset,
            ]
        )

    @count_gprs
    def ds_read2_b64(self, dst: VgprRange, voffset: Vgpr, offset0: int, offset1: int):
        self.instructions.append(
            [
                lambda: f"ds_read2_b64 {str(dst)}, {str(voffset)} offset0:{offset0} offset1:{offset1}",
                dst,
                voffset,
                offset0,
                offset1,
            ]
        )

    def ds_read_inst(self, num_bytes: int):
        if num_bytes == 4:
            return self.ds_read_b32
        elif num_bytes == 8:
            return self.ds_read_b64
        elif num_bytes == 16:
            return self.ds_read_b128

    @count_gprs
    def s_mov_b32(self, dst: Sgpr, src: Sgpr | int | float):
        self.instructions.append(
            [
                lambda: f"s_mov_b32 {str(dst)}, {str(src)}",
                dst,
                src,
            ]
        )

    @count_gprs
    def s_mov_b64(self, dst: SgprRange, src: SgprRange):
        self.instructions.append(
            [
                lambda: f"s_mov_b64 {str(dst)}, {str(src)}",
                dst,
                src,
            ]
        )

    @count_gprs
    def s_lshl_b32(self, dst: Sgpr, src: Sgpr, shift: int):
        self.instructions.append(
            [lambda: f"s_lshl_b32 {str(dst)}, {str(src)}, {shift}", dst, src, shift]
        )

    @count_gprs
    def s_lshr_b32(self, dst: Sgpr, src: Sgpr, shift: int):
        self.instructions.append(
            [lambda: f"s_lshr_b32 {str(dst)}, {str(src)}, {shift}", dst, src, shift]
        )

    @count_gprs
    def s_mul_i32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        self.instructions.append(
            [lambda: f"s_mul_i32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def s_add_i32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        self.instructions.append(
            [lambda: f"s_add_i32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def s_sub_i32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        self.instructions.append(
            [lambda: f"s_sub_i32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def s_and_b32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        self.instructions.append(
            [lambda: f"s_and_b32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def s_load_dword(self, dst: Sgpr, src: SgprRange, offset: int):
        self.instructions.append(
            [lambda: f"s_load_dword {str(dst)}, {str(src)} {offset}", dst, src, offset]
        )

    @count_gprs
    def s_load_dwordx2(self, dst: SgprRange, src: SgprRange, offset: int):
        self.instructions.append(
            [
                lambda: f"s_load_dwordx2 {str(dst)}, {str(src)} {offset}",
                dst,
                src,
                offset,
            ]
        )

    @count_gprs
    def s_load_dwordx4(self, dst: SgprRange, src: SgprRange, offset: int):
        self.instructions.append(
            [
                lambda: f"s_load_dwordx4 {str(dst)}, {str(src)} {offset}",
                dst,
                src,
                offset,
            ]
        )

    @count_calls
    @count_gprs
    def s_div_u32(
        self,
        dst: Sgpr,
        remainder: Sgpr,
        dividend: Sgpr,
        divisor: Sgpr,
        label_prefix: str = "",
    ):
        pfx = label_prefix
        end_label_name = f"{pfx}s_division_end_{self.s_div_u32._num_calls}"
        self.s_mov_b32(remainder, 0)
        self.s_cmp_eq_u32(dividend, divisor)
        self.s_cselect_b32(dst, 1, 0)
        self.s_cbranch_scc1(end_label_name)
        self.s_cmp_lt_u32(dividend, divisor)
        self.s_cselect_b32(dst, 0, 1)
        self.s_mov_b32(remainder, dividend)
        self.s_cbranch_scc1(end_label_name)
        div_beg_label_name = f"{pfx}s_division_shift_{self.s_div_u32._num_calls}"
        div_end_label_name = f"{pfx}s_division_shift_end_{self.s_div_u32._num_calls}"
        self.s_mov_b32(remainder, divisor)

        self.label(div_beg_label_name)
        self.s_cmp_lt_u32(dividend, remainder)
        self.s_cbranch_scc1(div_end_label_name)
        self.s_lshl_b32(dst, dst, 1)
        self.s_lshl_b32(remainder, remainder, 1)
        self.s_branch(div_beg_label_name)
        self.label(div_end_label_name)

        div_beg_sub_label_name = f"{pfx}s_division_sub_{self.s_div_u32._num_calls}"
        div_end_sub_label_name = f"{pfx}s_division_sub_end_{self.s_div_u32._num_calls}"
        self.label(div_beg_sub_label_name)
        self.s_cmp_lt_u32(remainder, dividend)
        self.s_cbranch_scc1(div_end_sub_label_name)
        self.s_sub_i32(remainder, remainder, divisor)
        self.s_sub_i32(dst, dst, 1)
        self.s_branch(div_beg_sub_label_name)
        self.label(div_end_sub_label_name)
        self.s_sub_i32(remainder, dividend, remainder)
        self.label(end_label_name)

    @count_gprs
    def s_cmp_lt_u32(self, lhs: Sgpr | int, rhs: Sgpr | int):
        self.instructions.append(
            [lambda: f"s_cmp_lt_u32 {str(lhs)}, {str(rhs)}", lhs, rhs]
        )

    @count_gprs
    def s_cmp_le_u32(self, lhs: Sgpr | int, rhs: Sgpr | int):
        self.instructions.append(
            [lambda: f"s_cmp_le_u32 {str(lhs)}, {str(rhs)}", lhs, rhs]
        )

    @count_gprs
    def s_cmp_eq_u32(self, lhs: Sgpr | int, rhs: Sgpr | int):
        self.instructions.append(
            [lambda: f"s_cmp_eq_u32 {str(lhs)}, {str(rhs)}", lhs, rhs]
        )

    @count_gprs
    def s_cmp_ge_u32(self, lhs: Sgpr | int, rhs: Sgpr | int):
        self.instructions.append(
            [lambda: f"s_cmp_ge_u32 {str(lhs)}, {str(rhs)}", lhs, rhs]
        )

    @count_gprs
    def s_cmp_gt_u32(self, lhs: Sgpr | int, rhs: Sgpr | int):
        self.instructions.append(
            [lambda: f"s_cmp_gt_u32 {str(lhs)}, {str(rhs)}", lhs, rhs]
        )

    @count_gprs
    def s_cselect_b32(self, dst: Sgpr, lhs: Sgpr | int, rhs: Sgpr | int):
        self.instructions.append(
            [lambda: f"s_cselect_b32 {str(dst)}, {str(lhs)}, {str(rhs)}", dst, lhs, rhs]
        )

    def s_waitcnt(self, vmcnt: int = None, lgkmcnt: int = None):
        assert (vmcnt, lgkmcnt) != (None, None)

        if lgkmcnt is not None:
            lgkmcnt = min(lgkmcnt, 15)

        def impl():
            args = []

            if vmcnt is not None:
                args.append(f"vmcnt({vmcnt})")

            if lgkmcnt is not None:
                args.append(f"lgkmcnt({lgkmcnt})")

            return " ".join(
                [
                    "s_waitcnt",
                ]
                + args
            )

        self.instructions.append([impl, vmcnt, lgkmcnt])

    def s_cbranch_scc1(self, name: str):
        self.instructions.append([lambda: f"s_cbranch_scc1 {self.label_name(name)}", name])

    def s_cbranch_scc0(self, name: str):
        self.instructions.append([lambda: f"s_cbranch_scc0 {self.label_name(name)}", name])

    def s_branch(self, name: str):
        self.instructions.append([lambda: f"s_branch {self.label_name(name)}", name])

    def s_barrier(self):
        self.instructions.append([lambda: "s_barrier"])

    def s_endpgm(self):
        self.instructions.append([lambda: "s_endpgm"])

    @count_gprs
    def v_mov_b32(self, dst: Vgpr, src: Sgpr | Vgpr | int | float):
        self.instructions.append(
            [lambda: f"v_mov_b32 {str(dst)}, {str(src)}", dst, src]
        )

    @count_gprs
    def v_and_b32(
        self, dst: Vgpr, src0: Vgpr | int | float, src1: Sgpr | Vgpr | int | float
    ):
        self.instructions.append(
            [lambda: f"v_and_b32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def v_add_u32(self, dst: Vgpr, src0: Vgpr | int, src1: Sgpr | Vgpr | int):
        self.instructions.append(
            [lambda: f"v_add_u32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def v_add_i32(self, dst: Vgpr, src0: Vgpr | int, src1: Sgpr | Vgpr | int):
        self.instructions.append(
            [lambda: f"v_add_i32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def v_lshlrev_b32(
        self, dst: Vgpr, shift: Vgpr | int | float, src: Sgpr | Vgpr | int | float
    ):
        self.instructions.append(
            [
                lambda: f"v_lshlrev_b32 {str(dst)}, {str(shift)}, {str(src)}",
                dst,
                shift,
                src,
            ]
        )

    @count_gprs
    def v_lshrrev_b32(
        self, dst: Vgpr, shift: Vgpr | int | float, src: Sgpr | Vgpr | int | float
    ):
        self.instructions.append(
            [
                lambda: f"v_lshrrev_b32 {str(dst)}, {str(shift)}, {str(src)}",
                dst,
                shift,
                src,
            ]
        )

    @count_gprs
    def v_mul_lo_u32(self, dst: Vgpr, src0: Vgpr | int, src1: Sgpr | Vgpr | int):
        self.instructions.append(
            [
                lambda: f"v_mul_lo_u32 {str(dst)}, {str(src0)}, {str(src1)}",
                dst,
                src0,
                src1,
            ]
        )

    @count_gprs
    def v_mul_f32(self, dst: Vgpr, src0: Vgpr | float, src1: Sgpr | Vgpr | float):
        self.instructions.append(
            [
                lambda: f"v_mul_f32 {str(dst)}, {str(src0)}, {str(src1)}",
                dst,
                src0,
                src1,
            ]
        )

    @count_gprs
    def v_fma_f32(
        self,
        dst: Vgpr,
        src0: Vgpr | float,
        src1: Sgpr | Vgpr | float,
        src2: Sgpr | Vgpr | float,
    ):
        self.instructions.append(
            [
                lambda: f"v_fma_f32 {str(dst)}, {str(src0)}, {str(src1)}, {str(src2)}",
                dst,
                src0,
                src1,
                src2,
            ]
        )

    @count_gprs
    def v_mov_b64(self, dst: VgprRange, src: VgprRange | int | float):
        self.instructions.append(
            [lambda: f"v_mov_b64 {str(dst)}, {str(src)}", dst, src]
        )

    @count_gprs
    def v_accvgpr_write_b32(self, dst: AccVgpr, src: int | Vgpr):
        self.instructions.append(
            [lambda: f"v_accvgpr_write_b32 {str(dst)}, {str(src)}", dst, src]
        )

    @count_gprs
    def v_accvgpr_read_b32(self, dst: Vgpr, src: AccVgpr):
        self.instructions.append(
            [lambda: f"v_accvgpr_read_b32 {str(dst)}, {str(src)}", dst, src]
        )

    @count_gprs
    def v_mfma_f32_16x16x4f32(
        self, acc: AccVgprRange, a: Vgpr, b: Vgpr, c: AccVgprRange
    ):
        self.instructions.append(
            [
                lambda: f"v_mfma_f32_16x16x4f32 {str(acc)}, {str(a)}, {str(b)}, {str(c)}",
                acc,
                a,
                b,
                c,
            ]
        )

    @count_gprs
    def v_mfma_f32_32x32x2f32(
        self, acc: AccVgprRange, a: Vgpr, b: Vgpr, c: AccVgprRange
    ):
        self.instructions.append(
            [
                lambda: f"v_mfma_f32_32x32x2f32 {str(acc)}, {str(a)}, {str(b)}, {str(c)}",
                acc,
                a,
                b,
                c,
            ]
        )

    @count_gprs
    def v_mfma_f32_32x32x8f16(
        self, acc: AccVgprRange, a: VgprRange, b: VgprRange, c: AccVgprRange
    ):
        self.instructions.append(
            [
                lambda: f"v_mfma_f32_32x32x8f16 {str(acc)}, {str(a)}, {str(b)}, {str(c)}",
                acc,
                a,
                b,
                c,
            ]
        )

    @count_gprs
    def v_mfma_f32_16x16x16f16(
        self, acc: AccVgprRange, a: VgprRange, b: VgprRange, c: AccVgprRange
    ):
        self.instructions.append(
            [
                lambda: f"v_mfma_f32_16x16x16f16 {str(acc)}, {str(a)}, {str(b)}, {str(c)}",
                acc,
                a,
                b,
                c,
            ]
        )

    @count_gprs
    def v_wmma_f32_16x16x16_f16(
        self, dst: VgprRange, src0: VgprRange, src1: VgprRange, src2: VgprRange
    ):
        self.instructions.append(
            [
                lambda: f"v_wmma_f32_16x16x16_f16 {str(dst)}, {str(src0)}, {str(src1)}, {str(src2)}",
                dst,
                src0,
                src1,
                src2,
            ]
        )

    def mfma_inst(self, mfma: Tuple[int, int, int, int]):
        if mfma == (16, 16, 1, 4):
            return self.v_mfma_f32_16x16x4f32
        elif mfma == (32, 32, 1, 2):
            return self.v_mfma_f32_32x32x2f32
        elif mfma == (32, 32, 1, 8):
            return self.v_mfma_f32_32x32x8f16
        elif mfma == (16, 16, 1, 16):
            return self.v_mfma_f32_16x16x16f16
        assert False, f"Unsupported mfma shape: {mfma}"

    def materialize(self):
        return "\n".join([inst[0]() for inst in self.instructions])


def gpu_function(func):
    def wrapper(context, *args, **kwargs):
        if context is None:
            context = GpuContext()
        return func(context, *args, **kwargs)

    return wrapper


class DataType(IntEnum):
    FP32 = 0
    FP16 = 1
    BF16 = 2
    INT8 = 3


def datatype_size(dtype: DataType):
    if dtype == DataType.FP32:
        return 4
    elif dtype in (DataType.FP16, DataType.BF16):
        return 2
    elif dtype == DataType.INT8:
        return 1

    assert False, "unrecognized type"


class GemmOptimizations:
    def __init__(
        self,
        level: int,
        wgm: int | None = None,
        plr: int | None = None,
        gw: int | None = None,
        map_k_idx: int | None = None,
        scheduling_policy: Optional[Any] = None,
    ):
        from generator.scheduler import SchedulingPolicy
        self.level = level
        self.wgm = 1
        self.plr = 0
        self.gw = 0
        self.map_k_idx = 0
        self.scheduling_policy = (
            scheduling_policy if scheduling_policy is not None else SchedulingPolicy.ROUNDROBIN
        )
        self._setup_optimizations()
        if wgm is not None:
            self.wgm = wgm
        if plr is not None:
            self.plr = plr
        if gw is not None:
            self.gw = gw
        if map_k_idx is not None:
            self.map_k_idx = map_k_idx

    def _setup_optimizations(self):
        if self.level != 0:
            self.plr = 1
            self.gw = 1


class GemmSolutionConfig:
    def __init__(
        self,
        a_type: DataType,
        b_type: DataType,
        cd_type: DataType,
        scalar_type: DataType,
        mfma: Tuple[int, int, int, int],
        wave_group: Tuple[int, int],
        wave_tiling: Tuple[int, int],
        depth_k: int,
        trans_a: bool,
        trans_b: bool,
        vmem_stage: int = 1,
        single_buffer_lds: bool = False,
        wgm: int = 1,
        barrier_reduction: bool = False,
        disperse_reads: bool = False,
        vector_ds_read: bool = False,
    ):
        self.a_type = a_type
        self.b_type = b_type
        self.cd_type = cd_type
        self.scalar_type = scalar_type
        self.wave_group = wave_group
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.mfma = mfma
        self.wave_tiling = wave_tiling
        self.depth_k = depth_k
        self.vmem_stage = vmem_stage
        self.single_buffer_lds = single_buffer_lds
        self.wgm = wgm
        self.barrier_reduction = barrier_reduction
        self.disperse_reads = disperse_reads
        self.vector_ds_read = vector_ds_read
        self.wavefront_size = 64
        self.name = None

        if self.lds_usage_bytes >= MAX_LDS_NUM_BYTES:
            raise RuntimeError(
                f"LDS usage exceeds {MAX_LDS_NUM_BYTES}: {self.lds_usage_bytes}"
            )

        if not all(
            (wave_group[i] & (wave_group[i] - 1)) == 0 for i in range(len(wave_group))
        ):
            raise RuntimeError(
                f"Invalid wave group: {wave_group}"
            )

        num_reg_per_thread = self.mfma[0] * self.mfma[1] // self.wavefront_size
        total_agpr = self.wave_tiling[0] * self.wave_tiling[1] * num_reg_per_thread
        if total_agpr > 256:
            raise RuntimeError(
                f"AGPR usage exceeds 256: {total_agpr} (wave_tiling={self.wave_tiling}, mfma={self.mfma})"
            )

    @property
    def tile_size(self) -> Tuple[int, int]:
        return (
            self.mfma[0] * self.wave_group[0] * self.wave_tiling[0],
            self.mfma[1] * self.wave_group[1] * self.wave_tiling[1],
        )

    @property
    def num_workitems(self):
        return self.wave_group[0] * self.wave_group[1] * self.wavefront_size

    @property
    def num_bytes_per_buffer_load(self) -> Tuple[int, int]:
        t0, t1 = self.tile_size

        def num_bytes_loads(t, k, dtype):
            num_bytes = t * k * datatype_size(dtype)
            assert num_bytes % self.num_workitems == 0
            return num_bytes // self.num_workitems

        num_bytes_load_a = num_bytes_loads(t0, self.depth_k, self.a_type)
        num_bytes_load_b = num_bytes_loads(t1, self.depth_k, self.b_type)
        def get_vec_width(num_bytes_load):
            for v in [16, 8, 4]:
                if num_bytes_load >= v and num_bytes_load % v == 0:
                    return v
            raise RuntimeError(
                f"Buffer load bytes {num_bytes_load} not divisible by 4, 8, or 16"
            )

        vec_a = get_vec_width(num_bytes_load_a)
        vec_b = get_vec_width(num_bytes_load_b)
        return vec_a, vec_b

    @property
    def num_dwords_per_buffer_load(self) -> Tuple[int, int]:
        NUM_BYTES_DWORD = 4
        b0, b1 = self.num_bytes_per_buffer_load
        b0 //= NUM_BYTES_DWORD
        b1 //= NUM_BYTES_DWORD
        if any((i & (i - 1)) for i in (b0, b1)):
            raise RuntimeError("Invalid buffer load")
        return b0, b1

    @property
    def num_elements_per_ds_read(self) -> Tuple[int, int]:
        if self.a_type == DataType.FP16 and self.mfma[3] >= 8:
            return 4, 4
        return 1, 1

    @property
    def num_bytes_per_ds_read(self) -> Tuple[int, int]:
        if self.vector_ds_read:
            return 16, 16
        return (
            self.num_elements_per_ds_read[0] * datatype_size(self.a_type),
            self.num_elements_per_ds_read[1] * datatype_size(self.b_type),
        )

    @property
    def lds_offset_bytes(self) -> Tuple[int, int]:
        stride_elem_a = (
            self.depth_k + self.lds_pad_bytes[0] // datatype_size(self.a_type)
            if self.trans_a
            else self.tile_size[0] + self.lds_pad_bytes[0] // datatype_size(self.a_type)
        )
        dim1_a = self.tile_size[0] if self.trans_a else self.depth_k
        return 0, stride_elem_a * dim1_a * datatype_size(self.a_type)

    @property
    def lds_swap_offset_bytes(self) -> int:
        stride_elem_a = (
            self.depth_k + self.lds_pad_bytes[0] // datatype_size(self.a_type)
            if self.trans_a
            else self.tile_size[0] + self.lds_pad_bytes[0] // datatype_size(self.a_type)
        )
        dim1_a = self.tile_size[0] if self.trans_a else self.depth_k
        size_a = stride_elem_a * dim1_a * datatype_size(self.a_type)

        stride_elem_b = (
            self.tile_size[1] + self.lds_pad_bytes[1] // datatype_size(self.b_type)
            if self.trans_b
            else self.depth_k + self.lds_pad_bytes[1] // datatype_size(self.b_type)
        )
        dim1_b = self.depth_k if self.trans_b else self.tile_size[1]
        size_b = stride_elem_b * dim1_b * datatype_size(self.b_type)

        return size_a + size_b

    @property
    def lds_partitions(self) -> int:
        return 1 if self.single_buffer_lds else (self.vmem_stage + 1)

    @property
    def lds_usage_bytes(self) -> int:
        return self.lds_partitions * self.lds_swap_offset_bytes

    @property
    def lds_pad_bytes(self) -> Tuple[int, int]:
        return (
            self._auto_lds_pad_a() * datatype_size(self.a_type),
            self._auto_lds_pad_b() * datatype_size(self.b_type),
        )

    def _auto_lds_pad_a(self) -> int:
        best_pad = 0
        min_conflict = 999
        elem_size = datatype_size(self.a_type)
        # Constrain to multiples of 4 elements (16-byte alignment for FP32, 8-byte for FP16)
        candidate_pads = [i * 4 for i in range(12)]
        num_elems_read = self.num_elements_per_ds_read[0]
        bytes_per_thread = self.num_bytes_per_ds_read[0]
        num_banks_per_thread = max(1, bytes_per_thread // 4)
        # MI300X has 32 banks, 4 bytes/bank -> 128 bytes/cycle DS port bandwidth.
        # Waves issue ds_read in beats of (128 // bytes_per_thread) threads.
        # Bank conflicts only occur within the same beat!
        threads_per_beat = max(1, 128 // bytes_per_thread)
        num_beats = max(1, self.wavefront_size // threads_per_beat)

        for pad in candidate_pads:
            if not self.trans_a:
                stride = self.tile_size[0] + pad
            else:
                stride = self.depth_k + pad
            max_conf_across_beats = 0
            for b in range(num_beats):
                bank_counts = {}
                for wt in range(b * threads_per_beat, (b + 1) * threads_per_beat):
                    t_row = wt & (self.mfma[0] - 1)
                    t_col = (wt // self.mfma[0]) * num_elems_read
                    if not self.trans_a:
                        base_addr = (t_col * stride + t_row) * elem_size
                    else:
                        base_addr = (t_row * stride + t_col) * elem_size
                    for b_off in range(num_banks_per_thread):
                        bank = ((base_addr + b_off * 4) // 4) % 32
                        bank_counts[bank] = bank_counts.get(bank, 0) + 1
                conf = max(bank_counts.values()) if bank_counts else 0
                if conf > max_conf_across_beats:
                    max_conf_across_beats = conf

            if max_conf_across_beats < min_conflict:
                min_conflict = max_conf_across_beats
                best_pad = pad
            elif max_conf_across_beats == min_conflict and pad < best_pad:
                best_pad = pad
        return best_pad

    @property
    def num_unrolled_iters(self) -> int:
        return self.depth_k // self.mfma[3]

    @property
    def canonical_name(self) -> str:
        type_prefixes = {
            DataType.FP32: "s",
            DataType.FP16: "h",
            DataType.BF16: "b",
            DataType.INT8: "i8",
        }
        p = type_prefixes.get(self.a_type, "custom")
        gemm_type = f"{p}gemm"

        ta = "t" if self.trans_a else "n"
        tb = "t" if self.trans_b else "n"
        layout_str = f"{ta}{tb}"

        mt0 = self.tile_size[0]
        mt1 = self.tile_size[1]
        k = self.depth_k
        tile_str = f"b{mt0}x{mt1}x{k}"

        wg_str = f"wg{self.wave_group[0]}x{self.wave_group[1]}"
        wt_str = f"wt{self.wave_tiling[0]}x{self.wave_tiling[1]}"
        mfma_str = f"mfma{self.mfma[0]}x{self.mfma[1]}x{self.mfma[3]}"
        buf_str = "sgl" if self.single_buffer_lds else "dbl"
        vs_str = f"vs{self.vmem_stage + 1}"
        wgm_str = f"_wgm{self.wgm}" if self.wgm > 1 else ""

        return f"{gemm_type}_{layout_str}_{tile_str}_{wg_str}_{wt_str}_{mfma_str}_{buf_str}_{vs_str}{wgm_str}"

    def to_dict(self) -> Dict:
        return {
            "a_type": int(self.a_type),
            "b_type": int(self.b_type),
            "cd_type": int(self.cd_type),
            "scalar_type": int(self.scalar_type),
            "wave_group": self.wave_group,
            "trans_a": self.trans_a,
            "trans_b": self.trans_b,
            "mfma": self.mfma,
            "wave_tiling": self.wave_tiling,
            "depth_k": self.depth_k,
            "vmem_stage": self.vmem_stage,
            "single_buffer_lds": self.single_buffer_lds,
            "wgm": int(self.wgm),
            "barrier_reduction": self.barrier_reduction,
            "disperse_reads": self.disperse_reads,
            "vector_ds_read": self.vector_ds_read,
            "wavefront_size": self.wavefront_size,
            "lds_usage_bytes": self.lds_usage_bytes,
            "name": self.name if self.name else "",
        }

    def from_dict(self, d):
        self.a_type = DataType(d["a_type"])
        self.b_type = DataType(d["b_type"])
        self.cd_type = DataType(d["cd_type"])
        self.scalar_type = DataType(d["scalar_type"])
        self.wave_group = d["wave_group"]
        self.trans_a = d["trans_a"]
        self.trans_b = d["trans_b"]
        self.mfma = d["mfma"]
        self.wave_tiling = d["wave_tiling"]
        self.depth_k = d["depth_k"]
        self.vmem_stage = d.get("vmem_stage", 1)
        self.single_buffer_lds = d.get("single_buffer_lds", False)
        self.wgm = d.get("wgm", 1)
        self.barrier_reduction = d.get("barrier_reduction", False)
        self.disperse_reads = d.get("disperse_reads", False)
        self.vector_ds_read = d.get("vector_ds_read", False)
        self.wavefront_size = d["wavefront_size"]
        self.lds_usage_bytes = d["lds_usage_bytes"]
        self.name = d["name"]

    def _auto_lds_pad_b(self) -> int:
        best_pad = 0
        min_conflict = 999
        elem_size = datatype_size(self.b_type)
        # Constrain to multiples of 4 elements (16-byte alignment for FP32, 8-byte for FP16)
        candidate_pads = [i * 4 for i in range(12)]
        num_elems_read = self.num_elements_per_ds_read[1]
        bytes_per_thread = self.num_bytes_per_ds_read[1]
        num_banks_per_thread = max(1, bytes_per_thread // 4)
        # MI300X has 32 banks, 4 bytes/bank -> 128 bytes/cycle DS port bandwidth.
        # Waves issue ds_read in beats of (128 // bytes_per_thread) threads.
        # Bank conflicts only occur within the same beat!
        threads_per_beat = max(1, 128 // bytes_per_thread)
        num_beats = max(1, self.wavefront_size // threads_per_beat)

        for pad in candidate_pads:
            if not self.trans_b:
                stride = self.depth_k + pad
            else:
                stride = self.tile_size[1] + pad
            max_conf_across_beats = 0
            for b in range(num_beats):
                bank_counts = {}
                for wt in range(b * threads_per_beat, (b + 1) * threads_per_beat):
                    t_col = wt & (self.mfma[1] - 1)
                    t_row = (wt // self.mfma[1]) * num_elems_read
                    if not self.trans_b:
                        base_addr = (t_col * stride + t_row) * elem_size
                    else:
                        base_addr = (t_row * stride + t_col) * elem_size
                    for b_off in range(num_banks_per_thread):
                        bank = ((base_addr + b_off * 4) // 4) % 32
                        bank_counts[bank] = bank_counts.get(bank, 0) + 1
                conf = max(bank_counts.values()) if bank_counts else 0
                if conf > max_conf_across_beats:
                    max_conf_across_beats = conf

            if max_conf_across_beats < min_conflict:
                min_conflict = max_conf_across_beats
                best_pad = pad
            elif max_conf_across_beats == min_conflict and pad < best_pad:
                best_pad = pad
        return best_pad
@gpu_function
def gemm(
    context: GpuContext,
    name: str,
    arch: str,
    config: GemmSolutionConfig,
    opt: GemmOptimizations,
    arguments: FunctionArgumentList,
    generate_parts: bool = False,
) -> str | Tuple[str, str, FunctionMeta]:
    meta = FunctionMeta(name, arguments)
    meta.group_segment_fixed_size = config.lds_usage_bytes
    meta.workgroup_size = (
        config.wave_group[0] * config.wave_group[1] * meta.wavefront_size
    )

    @dataclass
    class SgprAlloc:
        srd_a: int
        srd_b: int
        srd_c: int
        srd_d: int
        kern_args_addr: int
        wg_id_x: int
        wg_id_y: int
        m: int
        n: int
        k: int
        k_idx: int
        row_idx: int
        col_idx: int
        gl_offset_a: int
        gl_offset_b: int
        stride_a_0: int
        stride_a_1: int
        stride_b_0: int
        stride_b_1: int
        stride_c_0: int
        stride_c_1: int
        stride_d_0: int
        stride_d_1: int
        alpha: int
        beta: int
        lds_start_addr: int
        mapped_k_idx: int
        lds_read_ptr: int
        lds_write_ptr: int
        lds_read_diff: int
        lds_write_diff: int
        kern_args: int
        map_k_offset: int
        stride_a_1_bytes: int
        gl_offset_a_base: int
        gl_offset_b_base: int
        end: int

    @dataclass
    class VgprAlloc:
        t_id: int
        gl_offset_a: List[List[int]]
        gl_offset_b: List[List[int]]
        gl_offset_c: List[List[int]]
        gl_offset_d: List[List[int]]
        t_row: int
        t_col: int
        gl_data_a: List[List[List[int]]]
        gl_data_b: List[List[List[int]]]
        lw_addr_a: List[List[int]]
        lw_addr_b: List[List[int]]
        lr_addr_a: List[List[int]]
        lr_addr_b: List[List[int]]
        valu_a: List[List[List[int]]]
        valu_b: List[List[List[int]]]
        valu_c: List[List[int]]
        valu_d: List[List[int]]
        valu_acc: List[List[int]]
        w_id: int
        w_row: int
        w_col: int
        wt_id: int

    @dataclass
    class AgprAlloc:
        num_reg_per_thread: int
        num_reg_contiguous: int
        arpgs: List[List[List[int]]]

    def sgpr_alloc():
        # TODO: need clearer way to avoid overlapping
        end_arg = 36 + meta.argument_num_sgpr
        return SgprAlloc(
            srd_a=4,
            srd_b=8,
            srd_c=4,
            srd_d=8,
            kern_args_addr=0,
            wg_id_x=2,
            wg_id_y=3,
            m=12,
            n=13,
            k=14,
            k_idx=15,
            row_idx=16,
            col_idx=17,
            gl_offset_a=18,
            gl_offset_b=19,
            stride_a_0=20,
            stride_a_1=21,
            stride_b_0=22,
            stride_b_1=23,
            stride_c_0=24,
            stride_c_1=25,
            stride_d_0=26,
            stride_d_1=27,
            alpha=28,
            beta=29,
            lds_start_addr=30,
            mapped_k_idx = 32,
            lds_read_ptr=31,
            lds_write_ptr=33,
            lds_read_diff=34,
            lds_write_diff=35,
            kern_args=36,
            map_k_offset=end_arg,
            stride_a_1_bytes=end_arg + 1,
            gl_offset_a_base=end_arg + 2,
            gl_offset_b_base=end_arg + 3,
            end=end_arg + 4,
        )

    def vgpr_alloc(opt: GemmOptimizations):
        mt0, mt1 = config.tile_size
        depth_k = config.depth_k
        num_workitems = config.num_workitems

        vgpr_counter = 0
        t_id = 0
        vgpr_counter += 1
        w_id = vgpr_counter
        vgpr_counter += 1
        w_row = vgpr_counter
        vgpr_counter += 1
        w_col = vgpr_counter
        vgpr_counter += 1
        t_row = vgpr_counter
        vgpr_counter += 1
        t_col = vgpr_counter
        vgpr_counter += 1
        wt_id = vgpr_counter
        vgpr_counter += 1

        if vgpr_counter % 2:
            vgpr_counter += 1

        mac_vgpr_start = vgpr_counter

        glvw_bytes_a, glvw_bytes_b = config.num_bytes_per_buffer_load
        dim0_a = depth_k if config.trans_a else mt0
        dim1_a = mt0 if config.trans_a else depth_k
        dim0_b = mt1 if config.trans_b else depth_k
        dim1_b = depth_k if config.trans_b else mt1

        num_loads_a_0 = max(
            (dim0_a * datatype_size(config.a_type)) // (glvw_bytes_a * num_workitems), 1
        )
        num_loads_a_1 = dim1_a // (
            num_workitems // ((dim0_a * datatype_size(config.a_type)) // glvw_bytes_a)
        )
        num_loads_b_0 = max(
            (dim0_b * datatype_size(config.b_type)) // (glvw_bytes_b * num_workitems),
            1,
        )
        num_loads_b_1 = dim1_b // (
            num_workitems // ((dim0_b * datatype_size(config.b_type)) // glvw_bytes_b)
        )
        assert (num_loads_a_0, num_loads_a_1) != (0, 0)
        assert (num_loads_b_0, num_loads_b_1) != (0, 0)
        # print(f"num_lods_a: {(num_loads_a_0, num_loads_a_1)}")
        # print(f"num_lods_b: {(num_loads_b_0, num_loads_b_1)}")

        def gl_read_data(num_loads_0, num_loads_1, vw_num_vgpr, single_set=False):
            nonlocal vgpr_counter
            gl_datas = []

            for j in range(num_loads_1):
                indices = []
                for i in range(num_loads_0):
                    vgpr_base = vgpr_counter + vw_num_vgpr * (i + j * num_loads_0) if not single_set else vgpr_counter
                    indices.append(vgpr_base)
                gl_datas.append(indices)

            if single_set:
                vgpr_counter += vw_num_vgpr
            else:
                vgpr_counter += vw_num_vgpr * num_loads_0 * num_loads_1
            return gl_datas

        # Double-buffer gl_data_a and gl_data_b
        gl_data_a = []
        for _ in range(2):
            if (glvw_num_vgpr_a := (glvw_bytes_a // 4)) > 1:
                vgpr_counter = (vgpr_counter + 1) // 2 * 2
            gl_data_a.append(gl_read_data(num_loads_a_0, num_loads_a_1, glvw_num_vgpr_a))

        gl_data_b = []
        for _ in range(2):
            if (glvw_num_vgpr_b := (glvw_bytes_b // 4)) > 1:
                vgpr_counter = (vgpr_counter + 1) // 2 * 2
            gl_data_b.append(gl_read_data(num_loads_b_0, num_loads_b_1, glvw_num_vgpr_b))

        # print("gl data vgpr:")
        # print(gl_data_a)
        # print(gl_data_b)

        # vw == 1 since we only need 1 VGPR to store offset for each thread
        gl_voffset_a = gl_read_data(num_loads_a_0, num_loads_a_1, 1)
        gl_voffset_b = gl_read_data(num_loads_b_0, num_loads_b_1, 1)
        # print("gl offset vgpr:")
        # print(gl_voffset_a)
        # print(gl_voffset_b)

        lw_voffset_a = gl_read_data(num_loads_a_0, num_loads_a_1, 1)
        lw_voffset_b = gl_read_data(num_loads_b_0, num_loads_b_1, 1)
        # print("lw offset vgpr:")
        # print(lw_voffset_a)
        # print(lw_voffset_b)

        lr_addr_a = gl_read_data(config.wave_tiling[0], 1, 1)
        lr_addr_b = gl_read_data(1, config.wave_tiling[1], 1)

        # print("ds read addr")
        # print(lr_addr_a)
        # print(lr_addr_b)
        valu_num_vgpr_a = config.num_bytes_per_ds_read[0] // 4
        valu_num_vgpr_b = config.num_bytes_per_ds_read[1] // 4

        valu_a = []
        for _ in range(opt.plr + 1):
            if valu_num_vgpr_a >= 4:
                vgpr_counter = (vgpr_counter + 3) // 4 * 4
            elif valu_num_vgpr_a > 1:
                vgpr_counter = (vgpr_counter + 1) // 2 * 2
            valu_a.append(gl_read_data(config.wave_tiling[0], 1, valu_num_vgpr_a))

        valu_b = []
        for _ in range(opt.plr + 1):
            if valu_num_vgpr_b >= 4:
                vgpr_counter = (vgpr_counter + 3) // 4 * 4
            elif valu_num_vgpr_b > 1:
                vgpr_counter = (vgpr_counter + 1) // 2 * 2
            valu_b.append(gl_read_data(1, config.wave_tiling[1], valu_num_vgpr_b))

        # print("valu{a, b}")
        # print(valu_a)
        # print(valu_b)

        # release unused vgprs
        vgpr_counter = mac_vgpr_start

        #FIXME: not 4, should take MFMA instruction into consideration
        num_agpr_per_thread = config.mfma[0]*config.mfma[1]//config.wavefront_size
        single_set = (opt.gw > 0)
        valu_c = gl_read_data(config.wave_tiling[0], config.wave_tiling[1], num_agpr_per_thread, single_set)
        valu_d = gl_read_data(config.wave_tiling[0], config.wave_tiling[1], num_agpr_per_thread, single_set)

        # print("valu{c, d}")
        # print(valu_c)
        # print(valu_d)

        gl_voffset_c = gl_read_data(config.wave_tiling[0], config.wave_tiling[1], 1, single_set)
        gw_voffset_d = gl_read_data(config.wave_tiling[0], config.wave_tiling[1], 1, single_set)

        # print("voffset{c, d}")
        # print(gl_voffset_c)
        # print(gw_voffset_d)

        valu_acc = gl_read_data(config.wave_tiling[0], config.wave_tiling[1], num_agpr_per_thread, single_set)

        return VgprAlloc(
            t_id=t_id,
            gl_offset_a=gl_voffset_a,
            gl_offset_b=gl_voffset_b,
            gl_offset_c=gl_voffset_c,
            gl_offset_d=gw_voffset_d,
            t_row=t_row,
            t_col=t_col,
            gl_data_a=gl_data_a,
            gl_data_b=gl_data_b,
            lw_addr_a=lw_voffset_a,
            lw_addr_b=lw_voffset_b,
            lr_addr_a=lr_addr_a,
            lr_addr_b=lr_addr_b,
            valu_a=valu_a,
            valu_b=valu_b,
            valu_c=valu_c,
            valu_d=valu_d,
            valu_acc=valu_acc,
            w_id=w_id,
            w_row=w_row,
            w_col=w_col,
            wt_id=wt_id,
        )

    def agpr_alloc():
        num_reg_per_thread = config.mfma[0] * config.mfma[1] // config.wavefront_size
        num_reg_contiguous = min(4, num_reg_per_thread)
        agprs = AgprAlloc(num_reg_per_thread, num_reg_contiguous, [])

        for j in range(config.wave_tiling[1]):
            val = []
            for i in range(config.wave_tiling[0]):
                val.append(agprs.num_reg_per_thread * (i + j * config.wave_tiling[0]))
            agprs.arpgs.append(val)

        return agprs

    @contextmanager
    def alloc_tmp_sgpr(num_regs: int):
        sgpr = (
            SgprRange(context.sgpr_counter, num_regs)
            if num_regs > 1
            else Sgpr(context.sgpr_counter)
        )
        context.sgpr_counter += num_regs
        context.max_sgpr = max(context.max_sgpr, context.sgpr_counter)
        try:
            yield sgpr
        finally:
            context.sgpr_counter -= num_regs

    @contextmanager
    def alloc_tmp_vgpr(num_regs: int):
        vgpr = (
            VgprRange(context.sgpr_counter, num_regs)
            if num_regs > 1
            else Vgpr(context.sgpr_counter)
        )
        context.vgpr_counter += num_regs
        context.max_vgpr = max(context.max_vgpr, context.vgpr_counter)
        try:
            yield vgpr
        finally:
            context.vgpr_counter -= num_regs

    def header():
        return f"""
.amdgcn_target "amdgcn-amd-amdhsa--{arch}"
.text
"""

    def implementation(gemm_config: GemmSolutionConfig):
        lbl = lambda s: f".L_{name}_{s}"
        sgprs = sgpr_alloc()
        context.sgpr_counter = max(context.sgpr_counter, sgprs.end)
        num_sgpr_kernarg = meta.argument_num_sgpr
        kern_arg_sgpr_offset = 0
        context.label(lbl("load_args"))
        context.comment("Load all arguments")
        while num_sgpr_kernarg:
            if num_sgpr_kernarg >= 4:
                context.s_load_dwordx4(
                    SgprRange(sgprs.kern_args + kern_arg_sgpr_offset, 4),
                    SgprRange(sgprs.kern_args_addr, 2),
                    kern_arg_sgpr_offset * 4,
                )
                kern_arg_sgpr_offset += 4
                num_sgpr_kernarg -= 4
            elif num_sgpr_kernarg >= 2:
                context.s_load_dwordx2(
                    SgprRange(sgprs.kern_args + kern_arg_sgpr_offset, 2),
                    SgprRange(sgprs.kern_args_addr, 2),
                    kern_arg_sgpr_offset * 4,
                )
                kern_arg_sgpr_offset += 2
                num_sgpr_kernarg -= 2
            else:
                context.s_load_dword(
                    Sgpr(sgprs.kern_args + kern_arg_sgpr_offset),
                    SgprRange(sgprs.kern_args_addr, 2),
                    kern_arg_sgpr_offset * 4,
                )
                kern_arg_sgpr_offset += 1
                num_sgpr_kernarg -= 1
        context.s_waitcnt(lgkmcnt=0)

        # context.label("test_div")
        # with alloc_tmp_sgpr(4) as stmp:
        #     s0, s1, s2, s3 = stmp.split()
        #     context.s_mov_b32(s0, 5)
        #     context.s_mov_b32(s1, 1)
        #     context.s_div_u32(s2, s3, s0, s1)

        if opt.wgm > 1:
            context.label(lbl("wgm_beg"))
            assert (opt.wgm & opt.wgm - 1) == 0
            num_workgroups_x, num_workgroups_y = (
                sgprs.kern_args + 17,
                sgprs.kern_args + 18,
            )
            with alloc_tmp_sgpr(4) as stmps:
                stmp0, stmp1, stmp2, stmp3 = stmps.split()
                log_wgm = int(math.log2(opt.wgm))
                # z = x + y * nwg0
                context.s_mul_i32(stmp0, Sgpr(sgprs.wg_id_y), Sgpr(num_workgroups_x))
                context.s_add_i32(stmp0, stmp0, Sgpr(sgprs.wg_id_x))
                # x = (z % wgm) + z / wgm / nwg1 * wgm
                # y = (z / wgm) % n
                context.s_and_b32(Sgpr(sgprs.wg_id_x), stmp0, opt.wgm - 1)
                context.s_lshr_b32(stmp1, stmp0, log_wgm)
                context.s_div_u32(
                    stmp3, stmp2, stmp1, Sgpr(num_workgroups_y), label_prefix=lbl("")
                )
                context.s_mul_i32(stmp3, stmp3, opt.wgm)
                context.s_add_i32(Sgpr(sgprs.wg_id_x), Sgpr(sgprs.wg_id_x), stmp3)
                context.comment("wg_id_x")
                context.s_lshr_b32(Sgpr(sgprs.wg_id_y), stmp0, log_wgm)
                context.s_div_u32(
                    stmp2, stmp1, Sgpr(sgprs.wg_id_y), Sgpr(num_workgroups_y), label_prefix=lbl("")
                )
                context.comment("wg_id_y")
                context.s_mov_b32(Sgpr(sgprs.wg_id_y), stmp1)
            context.label(lbl("wgm_end"))

        context.comment("Setup Srd{A, B}")
        context.s_mov_b32(Sgpr(sgprs.srd_a + 3), 0x20000)
        context.s_mov_b32(Sgpr(sgprs.srd_b + 3), 0x20000)
        context.s_mov_b64(SgprRange(sgprs.srd_a, 2), SgprRange(sgprs.kern_args, 2))
        context.s_mov_b64(SgprRange(sgprs.srd_b, 2), SgprRange(sgprs.kern_args + 2, 2))
        context.comment("Setup sizes, m, n and k")
        context.s_mov_b32(Sgpr(sgprs.m), Sgpr(sgprs.kern_args + 8))
        context.s_mov_b32(Sgpr(sgprs.n), Sgpr(sgprs.kern_args + 9))
        context.s_mov_b32(Sgpr(sgprs.k), Sgpr(sgprs.kern_args + 10))
        context.s_mov_b32(Sgpr(sgprs.stride_a_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_a_1), Sgpr(sgprs.kern_args + 11))
        context.s_mov_b32(Sgpr(sgprs.stride_b_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_b_1), Sgpr(sgprs.kern_args + 12))
        context.s_mov_b32(Sgpr(sgprs.stride_c_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_c_1), Sgpr(sgprs.kern_args + 13))
        context.s_mov_b32(Sgpr(sgprs.stride_d_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_d_1), Sgpr(sgprs.kern_args + 14))

        context.label(lbl("setup_gl_offsets"))
        context.comment("Setup global read offsets")
        if (config.tile_size[0] & (config.tile_size[0] - 1)) == 0:
            context.s_lshl_b32(
                Sgpr(sgprs.row_idx),
                Sgpr(sgprs.wg_id_x),
                int(math.log2(config.tile_size[0])),
            )
        else:
            with alloc_tmp_sgpr(1) as stmp:
                context.s_mov_b32(stmp, config.tile_size[0])
                context.s_mul_i32(
                    Sgpr(sgprs.row_idx),
                    Sgpr(sgprs.wg_id_x),
                    stmp,
                )

        if (config.tile_size[1] & (config.tile_size[1] - 1)) == 0:
            context.s_lshl_b32(
                Sgpr(sgprs.col_idx),
                Sgpr(sgprs.wg_id_y),
                int(math.log2(config.tile_size[1])),
            )
        else:
            with alloc_tmp_sgpr(1) as stmp:
                context.s_mov_b32(stmp, config.tile_size[1])
                context.s_mul_i32(
                    Sgpr(sgprs.col_idx),
                    Sgpr(sgprs.wg_id_y),
                    stmp,
                )

        bpe_log_a = int(math.log2(datatype_size(gemm_config.a_type)))
        bpe_log_b = int(math.log2(datatype_size(gemm_config.b_type)))

        with alloc_tmp_sgpr(1) as tmp:
            if not config.trans_a:
                context.s_mul_i32(tmp, Sgpr(sgprs.stride_a_1), Sgpr(sgprs.k))
            else:
                context.s_mul_i32(tmp, Sgpr(sgprs.stride_a_1), Sgpr(sgprs.m))
            context.s_lshl_b32(Sgpr(sgprs.srd_a + 2), tmp, bpe_log_a)

            if not config.trans_b:
                context.s_mul_i32(tmp, Sgpr(sgprs.n), Sgpr(sgprs.stride_b_1))
            else:
                context.s_mul_i32(tmp, Sgpr(sgprs.k), Sgpr(sgprs.stride_b_1))
            context.s_lshl_b32(Sgpr(sgprs.srd_b + 2), tmp, bpe_log_b)

        if opt.map_k_idx:
            with alloc_tmp_sgpr(1) as tmp:
                context.s_and_b32(tmp, Sgpr(sgprs.wg_id_x), opt.map_k_idx - 1)
                context.s_mul_i32(Sgpr(sgprs.map_k_offset), tmp, config.depth_k)
            if not config.trans_a:
                context.s_lshl_b32(
                    Sgpr(sgprs.stride_a_1_bytes),
                    Sgpr(sgprs.stride_a_1),
                    bpe_log_a,
                )
            else:
                context.s_mov_b32(
                    Sgpr(sgprs.stride_a_1_bytes),
                    1 << bpe_log_a,
                )

        if not config.trans_a:
            context.s_mul_i32(
                Sgpr(sgprs.gl_offset_a), Sgpr(sgprs.row_idx), Sgpr(sgprs.stride_a_0)
            )
        else:
            context.s_mul_i32(
                Sgpr(sgprs.gl_offset_a), Sgpr(sgprs.row_idx), Sgpr(sgprs.stride_a_1)
            )
        context.s_lshl_b32(Sgpr(sgprs.gl_offset_a), Sgpr(sgprs.gl_offset_a), bpe_log_a)

        if not config.trans_b:
            context.s_mul_i32(
                Sgpr(sgprs.gl_offset_b), Sgpr(sgprs.col_idx), Sgpr(sgprs.stride_b_1)
            )
        else:
            context.s_mul_i32(
                Sgpr(sgprs.gl_offset_b), Sgpr(sgprs.col_idx), Sgpr(sgprs.stride_b_0)
            )
        context.s_lshl_b32(Sgpr(sgprs.gl_offset_b), Sgpr(sgprs.gl_offset_b), bpe_log_b)

        if opt.map_k_idx:
            context.s_mov_b32(Sgpr(sgprs.gl_offset_a_base), Sgpr(sgprs.gl_offset_a))
            context.s_mov_b32(Sgpr(sgprs.gl_offset_b_base), Sgpr(sgprs.gl_offset_b))
        context.s_mov_b32(Sgpr(sgprs.k_idx), 0)
        agprs = agpr_alloc()

        for col in agprs.arpgs:
            for row in col:
                for i in range(row, row + agprs.num_reg_per_thread):
                    context.v_accvgpr_write_b32(AccVgpr(i), 0)

        vgprs = vgpr_alloc(opt)

        def make_lw_a_write(row, vdata):
            return lambda: context.ds_write_inst(config.num_bytes_per_buffer_load[0])(
                Vgpr(row), vdata, config.lds_offset_bytes[0]
            )

        def make_lw_b_write(row, vdata):
            return lambda: context.ds_write_inst(config.num_bytes_per_buffer_load[1])(
                Vgpr(row), vdata, config.lds_offset_bytes[1]
            )

        def lw_a_gen(g_buf_idx: int = 0):
            for j, col in enumerate(vgprs.lw_addr_a):
                for i, row in enumerate(col):
                    vdata = (
                        VgprRange(
                            vgprs.gl_data_a[g_buf_idx][j][i],
                            config.num_bytes_per_buffer_load[0] // 4,
                        )
                        if config.num_bytes_per_buffer_load[0] > 4
                        else Vgpr(vgprs.gl_data_a[g_buf_idx][j][i])
                    )
                    yield make_lw_a_write(row, vdata)

        def lw_b_gen(g_buf_idx: int = 0):
            for j, col in enumerate(vgprs.lw_addr_b):
                for i, row in enumerate(col):
                    vdata = (
                        VgprRange(
                            vgprs.gl_data_b[g_buf_idx][j][i],
                            config.num_bytes_per_buffer_load[1] // 4,
                        )
                        if config.num_bytes_per_buffer_load[1] > 4
                        else Vgpr(vgprs.gl_data_b[g_buf_idx][j][i])
                    )
                    yield make_lw_b_write(row, vdata)

        def lw_a(g_buf_idx: int = 0):
            for inst in lw_a_gen(g_buf_idx):
                inst()

        def lw_b(g_buf_idx: int = 0):
            for inst in lw_b_gen(g_buf_idx):
                inst()
        context.label(lbl("addr_calculations"))
        gl_num_elements_a = config.num_bytes_per_buffer_load[0] // datatype_size(
            config.a_type
        )
        dim0_a = config.depth_k if config.trans_a else config.tile_size[0]
        dim1_a = config.tile_size[0] if config.trans_a else config.depth_k
        num_load_threads0_a = dim0_a // gl_num_elements_a
        num_load_threads1_a = config.num_workitems // num_load_threads0_a
        context.v_and_b32(
            Vgpr(vgprs.t_row),
            Vgpr(vgprs.t_id),
            dim0_a // gl_num_elements_a - 1,
        )
        context.v_mul_lo_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), gl_num_elements_a)
        context.v_lshrrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_a)), Vgpr(vgprs.t_id)
        )

        for j, col in enumerate(vgprs.gl_offset_a):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.comment(f"gl_addr_a_{i}_{j}")
                    context.s_mov_b32(stmp, j * num_load_threads1_a)
                    context.v_add_u32(
                        Vgpr(vgprs.gl_offset_a[j][i]), Vgpr(vgprs.t_col), stmp
                    )
                    context.v_mul_lo_u32(
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        Sgpr(sgprs.stride_a_1),
                    )
                    context.v_add_u32(
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        Vgpr(vgprs.t_row),
                    )
                    context.s_mov_b32(stmp, i * num_load_threads0_a * gl_num_elements_a)
                    context.v_add_u32(
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        stmp,
                    )
                    context.v_mul_lo_u32(
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        datatype_size(config.a_type),
                        Vgpr(vgprs.gl_offset_a[j][i]),
                    )

        gl_num_elements_b = config.num_bytes_per_buffer_load[1] // datatype_size(
            config.b_type
        )
        dim0_b = config.tile_size[1] if config.trans_b else config.depth_k
        dim1_b = config.depth_k if config.trans_b else config.tile_size[1]
        num_load_threads0_b = dim0_b // gl_num_elements_b
        num_load_threads1_b = config.num_workitems // num_load_threads0_b
        context.v_and_b32(
            Vgpr(vgprs.t_row), Vgpr(vgprs.t_id), dim0_b // gl_num_elements_b - 1
        )
        context.v_mul_lo_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), gl_num_elements_b)
        context.v_lshrrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_b)), Vgpr(vgprs.t_id)
        )

        for j, col in enumerate(vgprs.gl_offset_b):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.comment(f"gl_addr_b_{i}_{j}")
                    context.s_mov_b32(stmp, j * num_load_threads1_b)
                    context.v_add_u32(
                        Vgpr(vgprs.gl_offset_b[j][i]), Vgpr(vgprs.t_col), stmp
                    )
                    context.v_mul_lo_u32(
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        Sgpr(sgprs.stride_b_1),
                    )
                    context.v_add_u32(
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        Vgpr(vgprs.t_row),
                    )
                    context.s_mov_b32(stmp, i * num_load_threads0_b * gl_num_elements_b)
                    context.v_add_u32(
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        stmp,
                    )
                    context.v_mul_lo_u32(
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        datatype_size(config.b_type),
                        Vgpr(vgprs.gl_offset_b[j][i]),
                    )

        def make_gl_a_load(dst, j, i, num_dwords_per_load):
            return lambda: context.buffer_load_inst(num_dwords_per_load)(
                dst,
                Vgpr(vgprs.gl_offset_a[j][i]),
                SgprRange(sgprs.srd_a, 4),
                Sgpr(sgprs.gl_offset_a),
                0,
            )

        def make_gl_b_load(dst, j, i, num_dwords_per_load):
            return lambda: context.buffer_load_inst(num_dwords_per_load)(
                dst,
                Vgpr(vgprs.gl_offset_b[j][i]),
                SgprRange(sgprs.srd_b, 4),
                Sgpr(sgprs.gl_offset_b),
                0,
            )

        def gl_a_gen(g_buf_idx: int = 0):
            for j, col in enumerate(vgprs.gl_data_a[g_buf_idx]):
                for i, row in enumerate(col):
                    num_dwords_per_load = (
                        gl_num_elements_a * datatype_size(config.a_type) // 4
                    )
                    dst = (
                        VgprRange(row, num_dwords_per_load)
                        if num_dwords_per_load > 1
                        else Vgpr(row)
                    )
                    yield make_gl_a_load(dst, j, i, num_dwords_per_load)

        def gl_b_gen(g_buf_idx: int = 0):
            for j, col in enumerate(vgprs.gl_data_b[g_buf_idx]):
                for i, row in enumerate(col):
                    num_dwords_per_load = (
                        gl_num_elements_b * datatype_size(config.b_type) // 4
                    )
                    dst = (
                        VgprRange(row, num_dwords_per_load)
                        if num_dwords_per_load > 1
                        else Vgpr(row)
                    )
                    yield make_gl_b_load(dst, j, i, num_dwords_per_load)

        def gl_a(g_buf_idx: int = 0):
            context.comment(f"gl_a {g_buf_idx}")
            for inst in gl_a_gen(g_buf_idx):
                inst()

        def gl_b(g_buf_idx: int = 0):
            context.comment(f"gl_b {g_buf_idx}")
            for inst in gl_b_gen(g_buf_idx):
                inst()

        def set_next_gl_soffsets_from_k_idx(offset_idx=0):
            context.comment(f"map k index, offset_idx: {offset_idx}")
            with alloc_tmp_sgpr(1) as stmp:
                # mapped_k_idx = k_idx + map_k_offset + offset_idx
                context.s_add_i32(Sgpr(sgprs.mapped_k_idx), Sgpr(sgprs.k_idx), Sgpr(sgprs.map_k_offset))
                if offset_idx != 0:
                    context.s_add_i32(Sgpr(sgprs.mapped_k_idx), Sgpr(sgprs.mapped_k_idx), offset_idx)

                # wrap around if mapped_k_idx >= k
                context.comment(f"wrap around if mapped_k_idx >= k")
                context.s_sub_i32(stmp, Sgpr(sgprs.mapped_k_idx), Sgpr(sgprs.k))
                context.s_cmp_lt_u32(Sgpr(sgprs.mapped_k_idx), Sgpr(sgprs.k))
                context.s_cselect_b32(Sgpr(sgprs.mapped_k_idx), Sgpr(sgprs.mapped_k_idx), stmp)

                if not config.trans_a:
                    # gl_offset_a = gl_offset_a_base + mapped_k_idx * stride_a_1_bytes
                    context.s_mul_i32(stmp, Sgpr(sgprs.mapped_k_idx), Sgpr(sgprs.stride_a_1_bytes))
                else:
                    context.s_lshl_b32(stmp, Sgpr(sgprs.mapped_k_idx), bpe_log_a)
                context.s_add_i32(Sgpr(sgprs.gl_offset_a), Sgpr(sgprs.gl_offset_a_base), stmp)

                if not config.trans_b:
                    # gl_offset_b = gl_offset_b_base + (mapped_k_idx << bpe_log_b)
                    context.s_lshl_b32(stmp, Sgpr(sgprs.mapped_k_idx), bpe_log_b)
                else:
                    context.s_lshl_b32(stmp, Sgpr(sgprs.stride_b_1), bpe_log_b)
                    context.s_mul_i32(stmp, Sgpr(sgprs.mapped_k_idx), stmp)
                context.s_add_i32(Sgpr(sgprs.gl_offset_b), Sgpr(sgprs.gl_offset_b_base), stmp)

        def gl_increments():
            with alloc_tmp_sgpr(1) as stmp:
                context.comment("gl_offset increments for unrolled loop")
                if not config.trans_a:
                    context.s_mul_i32(
                        stmp,
                        Sgpr(sgprs.stride_a_1),
                        config.depth_k * datatype_size(config.a_type),
                    )
                    context.s_add_i32(
                        Sgpr(sgprs.gl_offset_a), Sgpr(sgprs.gl_offset_a), stmp
                    )
                else:
                    context.s_add_i32(
                        Sgpr(sgprs.gl_offset_a),
                        Sgpr(sgprs.gl_offset_a),
                        config.depth_k * datatype_size(config.a_type),
                    )

                if not config.trans_b:
                    context.s_add_i32(
                        Sgpr(sgprs.gl_offset_b),
                        Sgpr(sgprs.gl_offset_b),
                        config.depth_k * datatype_size(config.b_type),
                    )
                else:
                    context.s_mul_i32(
                        stmp,
                        Sgpr(sgprs.stride_b_1),
                        config.depth_k * datatype_size(config.b_type),
                    )
                    context.s_add_i32(
                        Sgpr(sgprs.gl_offset_b), Sgpr(sgprs.gl_offset_b), stmp
                    )

        def swap_lds_write_addr():
            if config.single_buffer_lds:
                return
            context.comment("swap ds write address")
            context.s_mov_b32(Sgpr(sgprs.lds_start_addr), config.lds_swap_offset_bytes)
            for j, col in enumerate(vgprs.lw_addr_a):
                for i, row in enumerate(col):
                    context.v_add_u32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_start_addr))

            for j, col in enumerate(vgprs.lw_addr_b):
                for i, row in enumerate(col):
                    context.v_add_u32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_start_addr))

        # First calculate the write offsets base (which starts at buffer index 0)
        context.comment("lw_a")
        context.v_and_b32(
            Vgpr(vgprs.t_row),
            Vgpr(vgprs.t_id),
            dim0_a // gl_num_elements_a - 1,
        )

        context.v_mul_lo_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), gl_num_elements_a)
        context.v_lshrrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_a)), Vgpr(vgprs.t_id)
        )

        stride_lds_elem_a = (
            config.depth_k + config.lds_pad_bytes[0] // datatype_size(config.a_type)
            if config.trans_a
            else config.tile_size[0] + config.lds_pad_bytes[0] // datatype_size(config.a_type)
        )
        for j, col in enumerate(vgprs.lw_addr_a):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.comment(f"lw_addr_a_{i}_{j}")
                    context.s_mov_b32(stmp, j * num_load_threads1_a)
                    context.v_add_u32(Vgpr(row), Vgpr(vgprs.t_col), stmp)
                    context.s_mov_b32(stmp, stride_lds_elem_a)
                    context.v_mul_lo_u32(Vgpr(row), Vgpr(row), stmp)
                    context.s_mov_b32(stmp, i * num_load_threads0_a * gl_num_elements_a)
                    context.v_add_u32(Vgpr(row), Vgpr(row), stmp)
                context.v_add_u32(Vgpr(row), Vgpr(row), Vgpr(vgprs.t_row))
                context.v_mul_lo_u32(Vgpr(row), Vgpr(row), datatype_size(config.a_type))

        context.comment("lw_b")
        context.v_and_b32(
            Vgpr(vgprs.t_row), Vgpr(vgprs.t_id), dim0_b // gl_num_elements_b - 1
        )
        context.v_mul_lo_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), gl_num_elements_b)
        context.v_lshrrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_b)), Vgpr(vgprs.t_id)
        )

        stride_lds_elem_b = (
            config.tile_size[1] + config.lds_pad_bytes[1] // datatype_size(config.b_type)
            if config.trans_b
            else config.depth_k + config.lds_pad_bytes[1] // datatype_size(config.b_type)
        )
        for j, col in enumerate(vgprs.lw_addr_b):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.comment(f"lw_addr_b_{i}_{j}")
                    context.s_mov_b32(stmp, j * num_load_threads1_b)
                    context.v_add_u32(Vgpr(row), Vgpr(vgprs.t_col), stmp)
                    context.s_mov_b32(stmp, stride_lds_elem_b)
                    context.v_mul_lo_u32(Vgpr(row), Vgpr(row), stmp)
                    context.s_mov_b32(stmp, i * num_load_threads0_b * gl_num_elements_b)
                    context.v_add_u32(Vgpr(row), Vgpr(row), stmp)
                context.v_add_u32(Vgpr(row), Vgpr(row), Vgpr(vgprs.t_row))
                context.v_mul_lo_u32(Vgpr(row), Vgpr(row), datatype_size(config.b_type))

        # Perform prefetch loop config.vmem_stage times
        if opt.map_k_idx:
            set_next_gl_soffsets_from_k_idx(0)

        for stage in range(config.vmem_stage):
            context.comment(f"prefetch stage {stage}")
            reg_buf_idx = stage % 2
            gl_a(reg_buf_idx)
            gl_b(reg_buf_idx)

            context.s_waitcnt(vmcnt=0)
            lw_a(reg_buf_idx)
            lw_b(reg_buf_idx)

            # Increment global offsets for the next stage tile
            if opt.map_k_idx:
                set_next_gl_soffsets_from_k_idx((stage + 1) * config.depth_k)
            else:
                gl_increments()

            # Swap write addresses to the next stage buffer
            swap_lds_write_addr()

        # Issue the final outstanding prefetch (stage = config.vmem_stage)
        # which will be in flight during the first outer loop iteration
        context.comment(f"outstanding prefetch stage {config.vmem_stage}")
        reg_buf_idx = config.vmem_stage % 2
        gl_a(reg_buf_idx)
        gl_b(reg_buf_idx)

        if opt.map_k_idx:
            set_next_gl_soffsets_from_k_idx((config.vmem_stage + 1) * config.depth_k)
        else:
            gl_increments()
        context.label(lbl("lds_wave_offsets"))
        context.comment("lds read addresses: wave offsets")
        context.v_lshrrev_b32(Vgpr(vgprs.w_id), 6, Vgpr(vgprs.t_id))
        context.v_and_b32(
            Vgpr(vgprs.wt_id), config.wavefront_size - 1, Vgpr(vgprs.t_id)
        )
        context.v_lshrrev_b32(
            Vgpr(vgprs.w_col), int(math.log2(config.wave_group[0])), Vgpr(vgprs.w_id)
        )
        context.v_mul_lo_u32(Vgpr(vgprs.w_col), Vgpr(vgprs.w_col), config.mfma[1])
        context.v_and_b32(Vgpr(vgprs.w_row), config.wave_group[0] - 1, Vgpr(vgprs.w_id))
        context.v_mul_lo_u32(Vgpr(vgprs.w_row), Vgpr(vgprs.w_row), config.mfma[0])

        context.comment("lds read addresses: thread offsets a")
        context.v_and_b32(Vgpr(vgprs.t_row), config.mfma[0] - 1, Vgpr(vgprs.wt_id))
        context.v_lshrrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(config.mfma[0])), Vgpr(vgprs.wt_id)
        )
        if config.num_elements_per_ds_read[0] > 1:
            context.v_lshlrev_b32(
                Vgpr(vgprs.t_col), int(math.log2(config.num_elements_per_ds_read[0])), Vgpr(vgprs.t_col)
            )
        context.v_add_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), Vgpr(vgprs.w_row))

        stride_lds_elem_a = (
            config.depth_k + config.lds_pad_bytes[0] // datatype_size(config.a_type)
            if config.trans_a
            else config.tile_size[0] + config.lds_pad_bytes[0] // datatype_size(config.a_type)
        )
        for j, col in enumerate(vgprs.lr_addr_a):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.s_mov_b32(stmp, stride_lds_elem_a)
                    if not config.trans_a:
                        context.v_mul_lo_u32(Vgpr(row), Vgpr(vgprs.t_col), stmp)
                        context.v_add_u32(Vgpr(row), Vgpr(row), Vgpr(vgprs.t_row))
                    else:
                        context.v_mul_lo_u32(Vgpr(row), Vgpr(vgprs.t_row), stmp)
                        context.v_add_u32(Vgpr(row), Vgpr(row), Vgpr(vgprs.t_col))
                    context.v_mul_lo_u32(
                        Vgpr(row), Vgpr(row), datatype_size(config.a_type)
                    )
                    context.s_mov_b32(stmp, config.wave_group[0] * config.mfma[0])
                    context.v_add_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), stmp)

        context.comment("lds read addresses: thread offsets b")
        context.v_and_b32(Vgpr(vgprs.t_col), config.mfma[1] - 1, Vgpr(vgprs.wt_id))
        context.v_lshrrev_b32(
            Vgpr(vgprs.t_row), int(math.log2(config.mfma[1])), Vgpr(vgprs.wt_id)
        )
        if config.num_elements_per_ds_read[1] > 1:
            context.v_lshlrev_b32(
                Vgpr(vgprs.t_row), int(math.log2(config.num_elements_per_ds_read[1])), Vgpr(vgprs.t_row)
            )
        context.v_add_u32(Vgpr(vgprs.t_col), Vgpr(vgprs.t_col), Vgpr(vgprs.w_col))

        stride_lds_elem_b = (
            config.tile_size[1] + config.lds_pad_bytes[1] // datatype_size(config.b_type)
            if config.trans_b
            else config.depth_k + config.lds_pad_bytes[1] // datatype_size(config.b_type)
        )
        for j, col in enumerate(vgprs.lr_addr_b):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.s_mov_b32(stmp, stride_lds_elem_b)
                    if not config.trans_b:
                        context.v_mul_lo_u32(Vgpr(row), Vgpr(vgprs.t_col), stmp)
                        context.v_add_u32(Vgpr(row), Vgpr(row), Vgpr(vgprs.t_row))
                    else:
                        context.v_mul_lo_u32(Vgpr(row), Vgpr(vgprs.t_row), stmp)
                        context.v_add_u32(Vgpr(row), Vgpr(row), Vgpr(vgprs.t_col))
                    context.v_mul_lo_u32(
                        Vgpr(row), Vgpr(row), datatype_size(config.b_type)
                    )
                    if config.vector_ds_read:
                        context.s_mov_b32(stmp, config.lds_offset_bytes[1])
                        context.v_add_u32(Vgpr(row), Vgpr(row), stmp)
                    context.s_mov_b32(stmp, config.wave_group[1] * config.mfma[1])
                    context.v_add_u32(Vgpr(vgprs.t_col), Vgpr(vgprs.t_col), stmp)

        context.comment("sync prefetch")
        context.s_waitcnt(lgkmcnt=0)
        context.s_barrier()

        unrolled_lr_offset_a, unrolled_lr_offset_b = (
            (0, 0) if config.vector_ds_read else config.lds_offset_bytes
        )

        step_k_bytes_a = (
            config.mfma[3]
            * (config.tile_size[0] + config.lds_pad_bytes[0] // datatype_size(config.a_type))
            * datatype_size(config.a_type)
            if not config.trans_a
            else config.mfma[3] * datatype_size(config.a_type)
        )
        step_k_bytes_b = (
            config.mfma[3] * datatype_size(config.b_type)
            if not config.trans_b
            else (
                config.mfma[3]
                * (config.tile_size[1] + config.lds_pad_bytes[1] // datatype_size(config.b_type))
                * datatype_size(config.b_type)
            )
        )

        def make_lr_a_read(row, j, i, offset):
            if config.vector_ds_read:
                dst = VgprRange(row, 4)
                offset0 = offset // 8
                offset1 = (offset + step_k_bytes_a) // 8
                return lambda: context.ds_read2_b64(
                    dst, Vgpr(vgprs.lr_addr_a[j][i]), offset0, offset1
                )
            dst = (
                Vgpr(row)
                if config.num_bytes_per_ds_read[0] == 4
                else VgprRange(row, config.num_bytes_per_ds_read[0] // 4)
            )
            return lambda: context.ds_read_inst(config.num_bytes_per_ds_read[0])(
                dst, Vgpr(vgprs.lr_addr_a[j][i]), offset
            )

        def make_lr_b_read(row, j, i, offset):
            if config.vector_ds_read:
                dst = VgprRange(row, 4)
                offset0 = offset // 8
                offset1 = (offset + step_k_bytes_b) // 8
                return lambda: context.ds_read2_b64(
                    dst, Vgpr(vgprs.lr_addr_b[j][i]), offset0, offset1
                )
            dst = (
                Vgpr(row)
                if config.num_bytes_per_ds_read[1] == 4
                else VgprRange(row, config.num_bytes_per_ds_read[1] // 4)
            )
            return lambda: context.ds_read_inst(config.num_bytes_per_ds_read[1])(
                dst, Vgpr(vgprs.lr_addr_b[j][i]), offset
            )

        def lr_a(k: int):
            nonlocal unrolled_lr_offset_a
            for j, col in enumerate(vgprs.valu_a[k]):
                for i, row in enumerate(col):
                    make_lr_a_read(row, j, i, unrolled_lr_offset_a)()
            if config.vector_ds_read:
                unrolled_lr_offset_a += 2 * step_k_bytes_a
            elif not config.trans_a:
                unrolled_lr_offset_a += (
                    config.mfma[3]
                    * (config.tile_size[0] + config.lds_pad_bytes[0] // datatype_size(config.a_type))
                    * datatype_size(config.a_type)
                )
            else:
                unrolled_lr_offset_a += config.mfma[3] * datatype_size(config.a_type)

        def lr_b(k: int):
            nonlocal unrolled_lr_offset_b
            for j, col in enumerate(vgprs.valu_b[k]):
                for i, row in enumerate(col):
                    make_lr_b_read(row, j, i, unrolled_lr_offset_b)()
            if config.vector_ds_read:
                unrolled_lr_offset_b += 2 * step_k_bytes_b
            elif not config.trans_b:
                unrolled_lr_offset_b += config.mfma[3] * datatype_size(config.b_type)
            else:
                unrolled_lr_offset_b += (
                    config.mfma[3]
                    * (config.tile_size[1] + config.lds_pad_bytes[1] // datatype_size(config.b_type))
                    * datatype_size(config.b_type)
                )

        def lr_b_col0(k: int):
            nonlocal unrolled_lr_offset_b
            if len(vgprs.valu_b[k]) > 0:
                col = vgprs.valu_b[k][0]
                j = 0
                for i, row in enumerate(col):
                    make_lr_b_read(row, j, i, unrolled_lr_offset_b)()
            if config.vector_ds_read:
                unrolled_lr_offset_b += 2 * step_k_bytes_b
            elif not config.trans_b:
                unrolled_lr_offset_b += config.mfma[3] * datatype_size(config.b_type)
            else:
                unrolled_lr_offset_b += (
                    config.mfma[3]
                    * (config.tile_size[1] + config.lds_pad_bytes[1] // datatype_size(config.b_type))
                    * datatype_size(config.b_type)
                )

        context.s_mov_b32(Sgpr(sgprs.lds_read_ptr), 0)
        context.s_mov_b32(Sgpr(sgprs.lds_write_ptr), 0 if config.single_buffer_lds else config.vmem_stage)
        if config.single_buffer_lds:
            context.s_mov_b32(Sgpr(sgprs.lds_read_diff), 0)
            context.s_mov_b32(Sgpr(sgprs.lds_write_diff), 0)
        elif config.vmem_stage == 1:
            context.s_mov_b32(Sgpr(sgprs.lds_read_diff), -config.lds_swap_offset_bytes)
            context.s_mov_b32(Sgpr(sgprs.lds_write_diff), config.lds_swap_offset_bytes)

        from generator.scheduler import SchedulingPolicy
        is_cross_plr = bool(
            opt.plr
            and getattr(opt, "scheduling_policy", None)
            in (SchedulingPolicy.ROUNDROBIN, SchedulingPolicy.INTERLEAVED, SchedulingPolicy.DAG_PIPELINE, SchedulingPolicy.COLUMN_PIPELINE)
        )

        plr_buf_idx = 0
        if is_cross_plr:
            if config.vector_ds_read:
                lr_a(0)
                lr_b(0)
            else:
                for u in range(opt.plr):
                    lr_a(plr_buf_idx)
                    if getattr(config, "disperse_reads", False) and opt.plr == 1:
                        lr_b_col0(plr_buf_idx)
                    else:
                        lr_b(plr_buf_idx)
                    plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)

        gl_insts_per_iter = {
            0: [[] for _ in range(config.num_unrolled_iters)],
            1: [[] for _ in range(config.num_unrolled_iters)],
        }
        num_gl_insts = 0
        if opt.level:
            for b in [0, 1]:
                gl_a_list = list(gl_a_gen(b))
                gl_b_list = list(gl_b_gen(b))
                all_gl = []
                from itertools import zip_longest
                for a_inst, b_inst in zip_longest(gl_a_list, gl_b_list):
                    if a_inst: all_gl.append(a_inst)
                    if b_inst: all_gl.append(b_inst)
                num_gl_insts = len(all_gl)
                max_issue_iters = config.num_unrolled_iters - opt.plr - 1
                if max_issue_iters <= 0:
                    max_issue_iters = 1
                num_target_iters = max(1, max_issue_iters // 2)
                for idx, inst in enumerate(all_gl):
                    iter_idx = idx % num_target_iters
                    gl_insts_per_iter[b][iter_idx].append(inst)

        def mfma(k: int, u: Optional[int] = None):
            sub_idx = 2 if (config.vector_ds_read and u is not None and u % 2 == 1) else 0
            for j, col in enumerate(agprs.arpgs):
                for i, row in enumerate(col):
                    if config.vector_ds_read:
                        src_a = VgprRange(vgprs.valu_a[k][0][i] + sub_idx, 2)
                        src_b = VgprRange(vgprs.valu_b[k][j][0] + sub_idx, 2)
                    else:
                        src_a = (
                            Vgpr(vgprs.valu_a[k][0][i])
                            if config.num_bytes_per_ds_read[0] == 4
                            else VgprRange(vgprs.valu_a[k][0][i], config.num_bytes_per_ds_read[0] // 4)
                        )
                        src_b = (
                            Vgpr(vgprs.valu_b[k][j][0])
                            if config.num_bytes_per_ds_read[1] == 4
                            else VgprRange(vgprs.valu_b[k][j][0], config.num_bytes_per_ds_read[1] // 4)
                        )
                    context.mfma_inst(config.mfma)(
                        AccVgprRange(row, agprs.num_reg_per_thread),
                        src_a,
                        src_b,
                        AccVgprRange(row, agprs.num_reg_per_thread),
                    )

        from generator.scheduler import InstructionNode, InstType

        def make_col_premise_reads():
            nodes = []
            if getattr(config, "disperse_reads", False) and opt.plr == 1 and not config.vector_ds_read:
                for j in range(1, len(vgprs.valu_b[0])):
                    for i, row in enumerate(vgprs.valu_b[0][j]):
                        fn = make_lr_b_read(row, j, i, config.lds_offset_bytes[1])
                        dst = (
                            Vgpr(row)
                            if config.num_bytes_per_ds_read[1] == 4
                            else VgprRange(row, config.num_bytes_per_ds_read[1] // 4)
                        )
                        nodes.append(
                            InstructionNode(
                                inst_type=InstType.LDS_READ,
                                emit_fn=fn,
                                latency=40,
                                issue_latency=4,
                                def_regs=[dst],
                                use_regs=[Vgpr(vgprs.lr_addr_b[j][i])],
                                desc=f"disperse_lr_b_step0_col{j}_{i}",
                            )
                        )
            return nodes

        def lr_a_gen(k: int):
            nonlocal unrolled_lr_offset_a
            for j, col in enumerate(vgprs.valu_a[k]):
                for i, row in enumerate(col):
                    yield make_lr_a_read(row, j, i, unrolled_lr_offset_a)
            if config.vector_ds_read:
                unrolled_lr_offset_a += 2 * step_k_bytes_a
            elif not config.trans_a:
                unrolled_lr_offset_a += (
                    config.mfma[3]
                    * (config.tile_size[0] + config.lds_pad_bytes[0] // datatype_size(config.a_type))
                    * datatype_size(config.a_type)
                )
            else:
                unrolled_lr_offset_a += config.mfma[3] * datatype_size(config.a_type)

        def lr_b_gen(k: int):
            nonlocal unrolled_lr_offset_b
            for j, col in enumerate(vgprs.valu_b[k]):
                for i, row in enumerate(col):
                    yield make_lr_b_read(row, j, i, unrolled_lr_offset_b)
            if config.vector_ds_read:
                unrolled_lr_offset_b += 2 * step_k_bytes_b
            elif not config.trans_b:
                unrolled_lr_offset_b += config.mfma[3] * datatype_size(config.b_type)
            else:
                unrolled_lr_offset_b += (
                    config.mfma[3]
                    * (config.tile_size[1] + config.lds_pad_bytes[1] // datatype_size(config.b_type))
                    * datatype_size(config.b_type)
                )

        def make_mfma_inst(row, k, j, i, u: Optional[int] = None):
            sub_idx = 2 if (config.vector_ds_read and u is not None and u % 2 == 1) else 0
            if config.vector_ds_read:
                src_a = VgprRange(vgprs.valu_a[k][0][i] + sub_idx, 2)
                src_b = VgprRange(vgprs.valu_b[k][j][0] + sub_idx, 2)
            else:
                src_a = (
                    Vgpr(vgprs.valu_a[k][0][i])
                    if config.num_bytes_per_ds_read[0] == 4
                    else VgprRange(vgprs.valu_a[k][0][i], config.num_bytes_per_ds_read[0] // 4)
                )
                src_b = (
                    Vgpr(vgprs.valu_b[k][j][0])
                    if config.num_bytes_per_ds_read[1] == 4
                    else VgprRange(vgprs.valu_b[k][j][0], config.num_bytes_per_ds_read[1] // 4)
                )
            return lambda: context.mfma_inst(config.mfma)(
                AccVgprRange(row, agprs.num_reg_per_thread),
                src_a,
                src_b,
                AccVgprRange(row, agprs.num_reg_per_thread),
            )


        def mfma_gen(k, u: Optional[int] = None):
            for j, col in enumerate(agprs.arpgs):
                for i, row in enumerate(col):
                    yield make_mfma_inst(row, k, j, i, u)

        def swap_lds_addr():
            if config.single_buffer_lds:
                return
            context.comment("dynamic swap lds read and write addresses")
            N = config.vmem_stage + 1
            offset = config.lds_swap_offset_bytes

            if N == 2:
                context.s_sub_i32(Sgpr(sgprs.lds_read_diff), 0, Sgpr(sgprs.lds_read_diff))
                context.s_sub_i32(Sgpr(sgprs.lds_write_diff), 0, Sgpr(sgprs.lds_write_diff))
            else:
                # 1. Update Read pointer and diff
                context.s_add_i32(Sgpr(sgprs.lds_read_ptr), Sgpr(sgprs.lds_read_ptr), 1)
                context.s_cmp_eq_u32(Sgpr(sgprs.lds_read_ptr), N)
                context.s_mov_b32(Sgpr(sgprs.lds_read_diff), offset)
                context.s_cselect_b32(Sgpr(sgprs.lds_read_diff), -(N - 1) * offset, Sgpr(sgprs.lds_read_diff))
                context.s_cselect_b32(Sgpr(sgprs.lds_read_ptr), 0, Sgpr(sgprs.lds_read_ptr))

                # 2. Update Write pointer and diff
                context.s_add_i32(Sgpr(sgprs.lds_write_ptr), Sgpr(sgprs.lds_write_ptr), 1)
                context.s_cmp_eq_u32(Sgpr(sgprs.lds_write_ptr), N)
                context.s_mov_b32(Sgpr(sgprs.lds_write_diff), offset)
                context.s_cselect_b32(Sgpr(sgprs.lds_write_diff), -(N - 1) * offset, Sgpr(sgprs.lds_write_diff))
                context.s_cselect_b32(Sgpr(sgprs.lds_write_ptr), 0, Sgpr(sgprs.lds_write_ptr))

            # 3. Apply Read difference to lr_addr_a/lr_addr_b
            for j, col in enumerate(vgprs.lr_addr_a):
                for i, row in enumerate(col):
                    context.v_add_i32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_read_diff))

            for j, col in enumerate(vgprs.lr_addr_b):
                for i, row in enumerate(col):
                    context.v_add_i32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_read_diff))

            # 4. Apply Write difference to lw_addr_a/lw_addr_b
            for j, col in enumerate(vgprs.lw_addr_a):
                for i, row in enumerate(col):
                    context.v_add_i32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_write_diff))

            for j, col in enumerate(vgprs.lw_addr_b):
                for i, row in enumerate(col):
                    context.v_add_i32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_write_diff))

        def generate_loop_iteration(g_buf_idx: int, jump_target: str = None):
            nonlocal plr_buf_idx, unrolled_lr_offset_a, unrolled_lr_offset_b
            context.comment(f"--- Loop Iteration (g_buf_idx = {g_buf_idx}) ---")
            if not is_cross_plr:
                unrolled_lr_offset_a, unrolled_lr_offset_b = (
                    (0, 0) if config.vector_ds_read else config.lds_offset_bytes
                )
                plr_buf_idx = 0
                for u in range(opt.plr):
                    lr_a(plr_buf_idx)
                    lr_b(plr_buf_idx)
                    plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)

            if opt.level:
                from generator.scheduler import SchedulingPolicy
                if opt.scheduling_policy == SchedulingPolicy.ROUNDROBIN:
                    for u in range(config.num_unrolled_iters):
                        next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                        mfma_iter = mfma_gen(u % (opt.plr + 1))
                        if u + opt.plr < config.num_unrolled_iters:
                            context.s_waitcnt(lgkmcnt=0)
                            gl_iter = iter(gl_insts_per_iter[1 - g_buf_idx][u])
                            for inst in roundrobin(
                                lr_a_gen(plr_buf_idx),
                                lr_b_gen(plr_buf_idx),
                                gl_iter,
                                mfma_iter,
                            ):
                                if inst:
                                    inst()
                        else:
                            if config.num_unrolled_iters - u == opt.plr:
                                context.s_waitcnt(lgkmcnt=0)
                                if config.single_buffer_lds:
                                    context.s_barrier()
                                context.s_waitcnt(vmcnt=num_gl_insts)
                                for inst in roundrobin(
                                    lw_a_gen(g_buf_idx),
                                    lw_b_gen(g_buf_idx),
                                    mfma_iter,
                                ):
                                    if inst:
                                        inst()
                            else:
                                context.s_waitcnt(lgkmcnt=0)
                                for inst in mfma_iter:
                                    if inst:
                                        inst()
                        plr_buf_idx = next_plr_buf_idx
                    swap_lds_addr()
                    context.s_waitcnt(lgkmcnt=0)
                    context.s_barrier()
                    if is_cross_plr:
                        unrolled_lr_offset_a, unrolled_lr_offset_b = (
                            (0, 0) if config.vector_ds_read else config.lds_offset_bytes
                        )
                        plr_buf_idx = 0
                        for u in range(opt.plr):
                            lr_a(plr_buf_idx)
                            lr_b(plr_buf_idx)
                            plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                else:
                    # DAG + Modulo Scheduling mode
                    from generator.scheduler import (
                        InstructionNode,
                        InstType,
                        BufferToken,
                        BufferTokenType,
                        WaitcntTracker,
                        ModuloPipelineScheduler,
                    )
                    dag_scheduler = ModuloPipelineScheduler(
                        policy=opt.scheduling_policy,
                        wave_tiling=config.wave_tiling,
                    )
                    loop_tracker = WaitcntTracker()

                    def make_mfma_node_list(k: int, u: int):
                        nodes = []
                        lat = 16 if (config.mfma[0] == 16 or config.a_type == DataType.FP16) else 32
                        sub_idx = 2 if (config.vector_ds_read and u % 2 == 1) else 0
                        for j, col in enumerate(agprs.arpgs):
                            for i, row in enumerate(col):
                                fn = make_mfma_inst(row, k, j, i, u)
                                if config.vector_ds_read:
                                    src_a = VgprRange(vgprs.valu_a[k][0][i] + sub_idx, 2)
                                    src_b = VgprRange(vgprs.valu_b[k][j][0] + sub_idx, 2)
                                else:
                                    src_a = (
                                        Vgpr(vgprs.valu_a[k][0][i])
                                        if config.num_bytes_per_ds_read[0] == 4
                                        else VgprRange(vgprs.valu_a[k][0][i], config.num_bytes_per_ds_read[0] // 4)
                                    )
                                    src_b = (
                                        Vgpr(vgprs.valu_b[k][j][0])
                                        if config.num_bytes_per_ds_read[1] == 4
                                        else VgprRange(vgprs.valu_b[k][j][0], config.num_bytes_per_ds_read[1] // 4)
                                    )
                                acc = AccVgprRange(row, agprs.num_reg_per_thread)
                                nodes.append(
                                    InstructionNode(
                                        inst_type=InstType.MFMA_COMPUTE,
                                        emit_fn=fn,
                                        latency=lat,
                                        issue_latency=4,
                                        def_regs=[acc],
                                        use_regs=[src_a, src_b, acc],
                                        desc=f"mfma_u{u}_{j}_{i}",
                                    )
                                )
                        return nodes

                    def make_lr_nodes_a(k: int, u: int, g_buf: int):
                        nodes = []
                        nonlocal unrolled_lr_offset_a
                        for j, col in enumerate(vgprs.valu_a[k]):
                            for i, row in enumerate(col):
                                fn = make_lr_a_read(row, j, i, unrolled_lr_offset_a)
                                dst = (
                                    VgprRange(row, 4)
                                    if config.vector_ds_read
                                    else (
                                        Vgpr(row)
                                        if config.num_bytes_per_ds_read[0] == 4
                                        else VgprRange(row, config.num_bytes_per_ds_read[0] // 4)
                                    )
                                )
                                nodes.append(
                                    InstructionNode(
                                        inst_type=InstType.LDS_READ,
                                        emit_fn=fn,
                                        latency=40,
                                        issue_latency=4,
                                        def_regs=[dst],
                                        use_regs=[Vgpr(vgprs.lr_addr_a[j][i])],
                                        consumed_tokens=[BufferToken(BufferTokenType.LDS_PARTITION, slot_id=g_buf, version=u)],
                                        desc=f"lr_a_u{u}_{j}_{i}",
                                    )
                                )
                        if config.vector_ds_read:
                            unrolled_lr_offset_a += 2 * step_k_bytes_a
                        elif not config.trans_a:
                            unrolled_lr_offset_a += (
                                config.mfma[3]
                                * (config.tile_size[0] + config.lds_pad_bytes[0] // datatype_size(config.a_type))
                                * datatype_size(config.a_type)
                            )
                        else:
                            unrolled_lr_offset_a += config.mfma[3] * datatype_size(config.a_type)
                        return nodes

                    def make_lr_nodes_b(k: int, u: int, g_buf: int):
                        nodes = []
                        nonlocal unrolled_lr_offset_b
                        for j, col in enumerate(vgprs.valu_b[k]):
                            for i, row in enumerate(col):
                                fn = make_lr_b_read(row, j, i, unrolled_lr_offset_b)
                                dst = (
                                    VgprRange(row, 4)
                                    if config.vector_ds_read
                                    else (
                                        Vgpr(row)
                                        if config.num_bytes_per_ds_read[1] == 4
                                        else VgprRange(row, config.num_bytes_per_ds_read[1] // 4)
                                    )
                                )
                                nodes.append(
                                    InstructionNode(
                                        inst_type=InstType.LDS_READ,
                                        emit_fn=fn,
                                        latency=40,
                                        issue_latency=4,
                                        def_regs=[dst],
                                        use_regs=[Vgpr(vgprs.lr_addr_b[j][i])],
                                        consumed_tokens=[BufferToken(BufferTokenType.LDS_PARTITION, slot_id=g_buf, version=u)],
                                        desc=f"lr_b_u{u}_{j}_{i}",
                                    )
                                )
                        if config.vector_ds_read:
                            unrolled_lr_offset_b += 2 * step_k_bytes_b
                        elif not config.trans_b:
                            unrolled_lr_offset_b += config.mfma[3] * datatype_size(config.b_type)
                        else:
                            unrolled_lr_offset_b += (
                                config.mfma[3]
                                * (config.tile_size[1] + config.lds_pad_bytes[1] // datatype_size(config.b_type))
                                * datatype_size(config.b_type)
                            )
                        return nodes

                    if is_cross_plr:
                        if config.vector_ds_read:
                            for j, col in enumerate(vgprs.valu_a[0]):
                                for i, row in enumerate(col):
                                    dst = VgprRange(row, 4)
                                    loop_tracker.record_issue(InstructionNode(InstType.LDS_READ, lambda: None, latency=40, def_regs=[dst], desc=f"prologue_lr_a_0"))
                            for j, col in enumerate(vgprs.valu_b[0]):
                                for i, row in enumerate(col):
                                    dst = VgprRange(row, 4)
                                    loop_tracker.record_issue(InstructionNode(InstType.LDS_READ, lambda: None, latency=40, def_regs=[dst], desc=f"prologue_lr_b_0"))
                        else:
                            for p in range(opt.plr):
                                for j, col in enumerate(vgprs.valu_a[p]):
                                    for i, row in enumerate(col):
                                        dst = Vgpr(row) if config.num_bytes_per_ds_read[0] == 4 else VgprRange(row, config.num_bytes_per_ds_read[0] // 4)
                                        loop_tracker.record_issue(InstructionNode(InstType.LDS_READ, lambda: None, latency=40, def_regs=[dst], desc=f"prologue_lr_a_{p}"))
                                for j, col in enumerate(vgprs.valu_b[p]):
                                    if getattr(config, "disperse_reads", False) and opt.plr == 1 and j > 0:
                                        continue
                                    for i, row in enumerate(col):
                                        dst = Vgpr(row) if config.num_bytes_per_ds_read[1] == 4 else VgprRange(row, config.num_bytes_per_ds_read[1] // 4)
                                        loop_tracker.record_issue(InstructionNode(InstType.LDS_READ, lambda: None, latency=40, def_regs=[dst], desc=f"prologue_lr_b_{p}"))

                    for u in range(config.num_unrolled_iters):
                        next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                        mfma_buf_idx = ((u // 2) % 2) if config.vector_ds_read else (u % (opt.plr + 1))
                        mfma_nodes = make_mfma_node_list(mfma_buf_idx, u)
                        col_premise_reads = make_col_premise_reads() if u == 0 else []

                        if u + opt.plr < config.num_unrolled_iters:
                            if config.vector_ds_read:
                                if u % 2 == 1:
                                    next_pair_buf = ((u + 1) // 2) % 2
                                    lr_nodes_a = make_lr_nodes_a(next_pair_buf, u, g_buf_idx)
                                    lr_nodes_b = make_lr_nodes_b(next_pair_buf, u, g_buf_idx)
                                else:
                                    lr_nodes_a = []
                                    lr_nodes_b = []
                            else:
                                lr_nodes_a = make_lr_nodes_a(plr_buf_idx, u, g_buf_idx)
                                lr_nodes_b = make_lr_nodes_b(plr_buf_idx, u, g_buf_idx)
                            gl_nodes = [
                                InstructionNode(
                                    inst_type=InstType.VMEM_LOAD,
                                    emit_fn=inst,
                                    latency=300,
                                    produced_tokens=[BufferToken(BufferTokenType.VMEM_VGPR, slot_id=1 - g_buf_idx, version=u)],
                                    desc=f"gl_u{u}",
                                )
                                for inst in gl_insts_per_iter[1 - g_buf_idx][u]
                            ]
                            dag_scheduler.schedule_loop_step(
                                ctx=context,
                                tracker=loop_tracker,
                                lr_nodes_a=lr_nodes_a,
                                lr_nodes_b=lr_nodes_b,
                                mfma_nodes=mfma_nodes,
                                gl_nodes=gl_nodes,
                                col_premise_reads=col_premise_reads,
                            )
                        else:
                            if config.num_unrolled_iters - u == opt.plr:
                                if config.single_buffer_lds and not getattr(config, "barrier_reduction", False):
                                    context.s_waitcnt(lgkmcnt=0)
                                    context.s_barrier()
                                context.s_waitcnt(vmcnt=num_gl_insts)
                                lw_nodes_a = [
                                    InstructionNode(
                                        inst_type=InstType.LDS_WRITE,
                                        emit_fn=inst,
                                        latency=20,
                                        consumed_tokens=[BufferToken(BufferTokenType.VMEM_VGPR, slot_id=g_buf_idx, version=u)],
                                        produced_tokens=[BufferToken(BufferTokenType.LDS_PARTITION, slot_id=1 - g_buf_idx, version=u)],
                                        desc=f"lw_a_u{u}",
                                    )
                                    for inst in lw_a_gen(g_buf_idx)
                                ]
                                lw_nodes_b = [
                                    InstructionNode(
                                        inst_type=InstType.LDS_WRITE,
                                        emit_fn=inst,
                                        latency=20,
                                        consumed_tokens=[BufferToken(BufferTokenType.VMEM_VGPR, slot_id=g_buf_idx, version=u)],
                                        produced_tokens=[BufferToken(BufferTokenType.LDS_PARTITION, slot_id=1 - g_buf_idx, version=u)],
                                        desc=f"lw_b_u{u}",
                                    )
                                    for inst in lw_b_gen(g_buf_idx)
                                ]
                                dag_scheduler.schedule_loop_step(
                                    ctx=context,
                                    tracker=loop_tracker,
                                    lr_nodes_a=[],
                                    lr_nodes_b=[],
                                    mfma_nodes=mfma_nodes,
                                    lw_nodes=lw_nodes_a + lw_nodes_b,
                                )
                            else:
                                for node in mfma_nodes:
                                    loop_tracker.check_and_emit_wait_for_uses(context, node)
                                    node.emit(context)
                                    loop_tracker.record_issue(node)
                        plr_buf_idx = next_plr_buf_idx
                    swap_lds_addr()
                    context.s_waitcnt(lgkmcnt=0)
                    context.s_barrier()
                    if is_cross_plr:
                        unrolled_lr_offset_a, unrolled_lr_offset_b = (
                            (0, 0) if config.vector_ds_read else config.lds_offset_bytes
                        )
                        if config.vector_ds_read:
                            lr_a(0)
                            lr_b(0)
                            for j, col in enumerate(vgprs.valu_a[0]):
                                for i, row in enumerate(col):
                                    dst = VgprRange(row, 4)
                                    loop_tracker.record_issue(InstructionNode(InstType.LDS_READ, lambda: None, latency=40, def_regs=[dst], desc="cross_lr_a_0"))
                            for j, col in enumerate(vgprs.valu_b[0]):
                                for i, row in enumerate(col):
                                    dst = VgprRange(row, 4)
                                    loop_tracker.record_issue(InstructionNode(InstType.LDS_READ, lambda: None, latency=40, def_regs=[dst], desc="cross_lr_b_0"))
                        else:
                            plr_buf_idx = 0
                            for u in range(opt.plr):
                                lr_a(plr_buf_idx)
                                if getattr(config, "disperse_reads", False) and opt.plr == 1:
                                    lr_b_col0(plr_buf_idx)
                                else:
                                    lr_b(plr_buf_idx)
                                for j, col in enumerate(vgprs.valu_a[plr_buf_idx]):
                                    for i, row in enumerate(col):
                                        dst = Vgpr(row) if config.num_bytes_per_ds_read[0] == 4 else VgprRange(row, config.num_bytes_per_ds_read[0] // 4)
                                        loop_tracker.record_issue(InstructionNode(InstType.LDS_READ, lambda: None, latency=40, def_regs=[dst], desc=f"cross_lr_a_{plr_buf_idx}"))
                                for j, col in enumerate(vgprs.valu_b[plr_buf_idx]):
                                    if getattr(config, "disperse_reads", False) and opt.plr == 1 and j > 0:
                                        continue
                                    for i, row in enumerate(col):
                                        dst = Vgpr(row) if config.num_bytes_per_ds_read[1] == 4 else VgprRange(row, config.num_bytes_per_ds_read[1] // 4)
                                        loop_tracker.record_issue(InstructionNode(InstType.LDS_READ, lambda: None, latency=40, def_regs=[dst], desc=f"cross_lr_b_{plr_buf_idx}"))
                                plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
            elif opt.plr:
                for u in range(config.num_unrolled_iters):
                    next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                    if u + opt.plr < config.num_unrolled_iters:
                        lr_a(plr_buf_idx)
                        lr_b(plr_buf_idx)
                        context.s_waitcnt(lgkmcnt=opt.plr * (config.wave_tiling[0] + config.wave_tiling[1]))
                    else:
                        context.s_waitcnt(lgkmcnt=0)
                    mfma_buf_idx = ((u // 2) % 2) if config.vector_ds_read else (u % (opt.plr + 1))
                    mfma(mfma_buf_idx, u)
                    plr_buf_idx = next_plr_buf_idx
            else:
                for u in range(config.num_unrolled_iters):
                    lr_a(plr_buf_idx)
                    lr_b(plr_buf_idx)
                    context.s_waitcnt(lgkmcnt=0)
                    mfma_buf_idx = ((u // 2) % 2) if config.vector_ds_read else (u % (opt.plr + 1))
                    mfma(mfma_buf_idx, u)
                    next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                    plr_buf_idx = next_plr_buf_idx

            if opt.level == 0:
                gl_a(1 - g_buf_idx)
                gl_b(1 - g_buf_idx)
                context.comment("ds write")
                context.s_waitcnt(vmcnt=0)
                lw_a(g_buf_idx)
                lw_b(g_buf_idx)
                context.comment("swap lds")
                swap_lds_addr()
                context.comment("wait for ds writes")
                context.s_waitcnt(lgkmcnt=0)
                context.s_barrier()

            with alloc_tmp_sgpr(1) as stmp:
                context.s_add_i32(Sgpr(sgprs.k_idx), Sgpr(sgprs.k_idx), config.depth_k)

                if opt.map_k_idx:
                    set_next_gl_soffsets_from_k_idx(2 * config.depth_k)
                else:
                    gl_increments()

                context.s_add_i32(stmp, Sgpr(sgprs.k_idx), config.depth_k)
                context.s_cmp_lt_u32(stmp, Sgpr(sgprs.k))
                if jump_target:
                    context.s_cbranch_scc1(jump_target)
                else:
                    context.s_cbranch_scc0(lbl("prefetch_last_loop"))

        start_buf_idx = config.vmem_stage % 2
        if start_buf_idx == 0:
            context.label(lbl("outer_loop"))
            generate_loop_iteration(0)
            generate_loop_iteration(1, jump_target=lbl("outer_loop"))
        else:
            context.label(lbl("outer_loop"))
            generate_loop_iteration(1)
            generate_loop_iteration(0, jump_target=lbl("outer_loop"))
        context.label(lbl("prefetch_last_loop"))
        context.comment("prefetch last loop")

        if not is_cross_plr:
            unrolled_lr_offset_a, unrolled_lr_offset_b = (
                (0, 0) if config.vector_ds_read else config.lds_offset_bytes
            )
            plr_buf_idx = 0
            if config.vector_ds_read:
                lr_a(0)
                lr_b(0)
            else:
                for u in range(opt.plr):
                    lr_a(plr_buf_idx)
                    lr_b(plr_buf_idx)
                    plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)

        if opt.level:
            from generator.scheduler import SchedulingPolicy
            if opt.scheduling_policy == SchedulingPolicy.ROUNDROBIN:
                for u in range(config.num_unrolled_iters):
                    context.s_waitcnt(lgkmcnt=0)
                    next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                    mfma_buf_idx = ((u // 2) % 2) if config.vector_ds_read else (u % (opt.plr + 1))
                    mfma_iter = mfma_gen(mfma_buf_idx, u)
                    if u + opt.plr < config.num_unrolled_iters:
                        if config.vector_ds_read:
                            if u % 2 == 1:
                                next_pair_buf = ((u + 1) // 2) % 2
                                for inst in roundrobin(
                                    lr_a_gen(next_pair_buf),
                                    lr_b_gen(next_pair_buf),
                                    mfma_iter,
                                ):
                                    if inst:
                                        inst()
                            else:
                                for inst in mfma_iter:
                                    if inst:
                                        inst()
                        else:
                            for inst in roundrobin(
                                lr_a_gen(plr_buf_idx),
                                lr_b_gen(plr_buf_idx),
                                mfma_iter,
                            ):
                                if inst:
                                    inst()
                    else:
                        for inst in mfma_iter:
                            if inst:
                                inst()
                    plr_buf_idx = next_plr_buf_idx
            else:
                from generator.scheduler import (
                    InstructionNode,
                    InstType,
                    BufferToken,
                    BufferTokenType,
                    WaitcntTracker,
                    ModuloPipelineScheduler,
                )
                dag_scheduler = ModuloPipelineScheduler(
                    policy=opt.scheduling_policy,
                    wave_tiling=config.wave_tiling,
                )
                loop_tracker = WaitcntTracker()
                for u in range(config.num_unrolled_iters):
                    context.s_waitcnt(lgkmcnt=0)
                    next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                    mfma_buf_idx = ((u // 2) % 2) if config.vector_ds_read else (u % (opt.plr + 1))
                    mfma_iter = list(mfma_gen(mfma_buf_idx, u))
                    mfma_nodes = [
                        InstructionNode(
                            inst_type=InstType.MFMA_COMPUTE,
                            emit_fn=inst,
                            latency=16 if config.mfma[0] == 16 else 32,
                            desc=f"mfma_u{u}",
                        )
                        for inst in mfma_iter
                    ]
                    col_premise_reads = make_col_premise_reads() if u == 0 else []
                    if u + opt.plr < config.num_unrolled_iters:
                        if config.vector_ds_read:
                            if u % 2 == 1:
                                next_pair_buf = ((u + 1) // 2) % 2
                                lr_nodes_a = [
                                    InstructionNode(
                                        inst_type=InstType.LDS_READ,
                                        emit_fn=inst,
                                        latency=40,
                                        consumed_tokens=[BufferToken(BufferTokenType.LDS_PARTITION, slot_id=0, version=u)],
                                        desc=f"lr_a_u{u}",
                                    )
                                    for inst in lr_a_gen(next_pair_buf)
                                ]
                                lr_nodes_b = [
                                    InstructionNode(
                                        inst_type=InstType.LDS_READ,
                                        emit_fn=inst,
                                        latency=40,
                                        consumed_tokens=[BufferToken(BufferTokenType.LDS_PARTITION, slot_id=0, version=u)],
                                        desc=f"lr_b_u{u}",
                                    )
                                    for inst in lr_b_gen(next_pair_buf)
                                ]
                            else:
                                lr_nodes_a = []
                                lr_nodes_b = []
                        else:
                            lr_nodes_a = [
                                InstructionNode(
                                    inst_type=InstType.LDS_READ,
                                    emit_fn=inst,
                                    latency=40,
                                    consumed_tokens=[BufferToken(BufferTokenType.LDS_PARTITION, slot_id=0, version=u)],
                                    desc=f"lr_a_u{u}",
                                )
                                for inst in lr_a_gen(plr_buf_idx)
                            ]
                            lr_nodes_b = [
                                InstructionNode(
                                    inst_type=InstType.LDS_READ,
                                    emit_fn=inst,
                                    latency=40,
                                    consumed_tokens=[BufferToken(BufferTokenType.LDS_PARTITION, slot_id=0, version=u)],
                                    desc=f"lr_b_u{u}",
                                )
                                for inst in lr_b_gen(plr_buf_idx)
                            ]
                        dag_scheduler.schedule_loop_step(
                            ctx=context,
                            tracker=loop_tracker,
                            lr_nodes_a=lr_nodes_a,
                            lr_nodes_b=lr_nodes_b,
                            mfma_nodes=mfma_nodes,
                            col_premise_reads=col_premise_reads,
                        )
                    else:
                        for node in mfma_nodes:
                            node.emit(context)
                    plr_buf_idx = next_plr_buf_idx
        elif opt.plr:
            for u in range(config.num_unrolled_iters):
                context.s_waitcnt(lgkmcnt=0)
                mfma_buf_idx = ((u // 2) % 2) if config.vector_ds_read else (u % (opt.plr + 1))
                mfma(mfma_buf_idx, u)
                next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                if u + opt.plr < config.num_unrolled_iters:
                    if config.vector_ds_read:
                        if u % 2 == 1:
                            next_pair_buf = ((u + 1) // 2) % 2
                            lr_a(next_pair_buf)
                            lr_b(next_pair_buf)
                    else:
                        lr_a(plr_buf_idx)
                        lr_b(plr_buf_idx)
                plr_buf_idx = next_plr_buf_idx
        else:
            for u in range(config.num_unrolled_iters):
                lr_a(plr_buf_idx)
                lr_b(plr_buf_idx)
                context.s_waitcnt(lgkmcnt=0)
                mfma_buf_idx = ((u // 2) % 2) if config.vector_ds_read else (u % (opt.plr + 1))
                mfma(mfma_buf_idx, u)
                next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                plr_buf_idx = next_plr_buf_idx

        context.s_mov_b32(Sgpr(sgprs.alpha), Sgpr(sgprs.kern_args + 15))
        context.s_mov_b32(Sgpr(sgprs.beta), Sgpr(sgprs.kern_args + 16))
        context.comment("setup srd{c, d}")
        context.s_mov_b64(SgprRange(sgprs.srd_c, 2), SgprRange(sgprs.kern_args + 4, 2))
        context.s_mov_b64(SgprRange(sgprs.srd_d, 2), SgprRange(sgprs.kern_args + 6, 2))
        context.s_mov_b32(Sgpr(sgprs.srd_c + 3), 0x20000)
        context.s_mov_b32(Sgpr(sgprs.srd_d + 3), 0x20000)
        context.s_mov_b32(Sgpr(sgprs.srd_c + 2), Sgpr(sgprs.m))
        context.s_mul_i32(
            Sgpr(sgprs.srd_c + 2), Sgpr(sgprs.srd_c + 2), Sgpr(sgprs.stride_c_1)
        )
        context.s_mul_i32(
            Sgpr(sgprs.srd_c + 2), Sgpr(sgprs.srd_c + 2), datatype_size(config.cd_type)
        )
        context.s_mov_b32(Sgpr(sgprs.srd_d + 2), Sgpr(sgprs.m))
        context.s_mul_i32(
            Sgpr(sgprs.srd_d + 2), Sgpr(sgprs.srd_d + 2), Sgpr(sgprs.stride_d_1)
        )
        context.s_mul_i32(
            Sgpr(sgprs.srd_d + 2), Sgpr(sgprs.srd_d + 2), datatype_size(config.cd_type)
        )
        # re-use
        gl_offset_c = sgprs.gl_offset_a
        gw_offset_d = sgprs.gl_offset_b
        context.s_mul_i32(
            Sgpr(gl_offset_c), Sgpr(sgprs.col_idx), Sgpr(sgprs.stride_c_1)
        )
        context.s_add_i32(Sgpr(gl_offset_c), Sgpr(gl_offset_c), Sgpr(sgprs.row_idx))
        context.s_mul_i32(
            Sgpr(gl_offset_c), Sgpr(gl_offset_c), datatype_size(config.cd_type)
        )
        context.s_mul_i32(
            Sgpr(gw_offset_d), Sgpr(sgprs.col_idx), Sgpr(sgprs.stride_d_1)
        )
        context.s_add_i32(Sgpr(gw_offset_d), Sgpr(gw_offset_d), Sgpr(sgprs.row_idx))
        context.s_mul_i32(
            Sgpr(gw_offset_d), Sgpr(gw_offset_d), datatype_size(config.cd_type)
        )

        def gw_naive():
            for j, col in enumerate(vgprs.gl_offset_d):
                for i, row in enumerate(col):
                    context.comment(f"gw_addr_{i}_{j}")
                    with alloc_tmp_sgpr(1) as stmp:
                        context.v_and_b32(
                            Vgpr(vgprs.t_col), config.mfma[1] - 1, Vgpr(vgprs.wt_id)
                        )
                        context.s_mov_b32(stmp, j * config.wave_group[1] * config.mfma[1])
                        context.v_add_u32(Vgpr(vgprs.t_col), Vgpr(vgprs.t_col), stmp)
                        context.v_lshrrev_b32(
                            Vgpr(vgprs.t_row),
                            int(math.log2(config.mfma[1])),
                            Vgpr(vgprs.wt_id),
                        )
                        context.v_mul_lo_u32(Vgpr(vgprs.t_row), 4, Vgpr(vgprs.t_row))
                        context.s_mov_b32(stmp, i * config.wave_group[0] * config.mfma[0])
                        context.v_add_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), stmp)
                        context.v_add_i32(
                            Vgpr(vgprs.t_col), Vgpr(vgprs.t_col), Vgpr(vgprs.w_col)
                        )
                        context.v_add_i32(
                            Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), Vgpr(vgprs.w_row)
                        )
                        context.comment(f"setup voffset_c_{i}_{j}")
                        context.v_mul_lo_u32(
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            Vgpr(vgprs.t_col),
                            Sgpr(sgprs.stride_c_1),
                        )
                        context.v_add_u32(
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            Vgpr(vgprs.t_row),
                        )
                        context.v_mul_lo_u32(
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            datatype_size(config.cd_type),
                        )
                        context.comment(f"setup voffset_d_{i}_{j}")
                        context.v_mul_lo_u32(
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            Vgpr(vgprs.t_col),
                            Sgpr(sgprs.stride_d_1),
                        )
                        context.v_add_u32(
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            Vgpr(vgprs.t_row),
                        )
                        context.v_mul_lo_u32(
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            datatype_size(config.cd_type),
                        )

            context.label(lbl("gw"))
            for j, col in enumerate(agprs.arpgs):
                for i, row in enumerate(col):
                    context.comment(f"gw_{i}_{j}")
                    skip_lbl = lbl(f"skip_gw_{i}_{j}")
                    needs_boundary_check = (config.tile_size[0] & (config.tile_size[0] - 1)) != 0
                    if needs_boundary_check:
                        with alloc_tmp_sgpr(1) as stmp:
                            context.s_add_i32(stmp, Sgpr(sgprs.row_idx), i * config.wave_group[0] * config.mfma[0])
                            context.s_cmp_ge_u32(stmp, Sgpr(sgprs.m))
                            context.s_cbranch_scc1(skip_lbl)

                    for r in range(agprs.num_reg_per_thread):
                        context.v_accvgpr_read_b32(
                            Vgpr(vgprs.valu_acc[j][i] + r), AccVgpr(row + r)
                        )
                    
                    for l in range(0, agprs.num_reg_per_thread, agprs.num_reg_contiguous):
                        context.buffer_load_inst(agprs.num_reg_contiguous)(
                            VgprRange(vgprs.valu_c[j][i]+l, agprs.num_reg_contiguous),
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            SgprRange(sgprs.srd_c, 4),
                            Sgpr(gl_offset_c),
                            0,
                        )

                        if agprs.num_reg_per_thread // agprs.num_reg_contiguous > 1:
                            with alloc_tmp_sgpr(1) as stmp:
                                increments = config.wavefront_size//config.mfma[1]*agprs.num_reg_contiguous
                                context.s_mul_i32(stmp, Sgpr(sgprs.stride_c_0), increments)
                                context.s_mul_i32(stmp, stmp, datatype_size(config.cd_type))
                                context.v_add_u32(Vgpr(vgprs.gl_offset_c[j][i]), Vgpr(vgprs.gl_offset_c[j][i]), stmp)
                    for r in range(agprs.num_reg_per_thread):
                        context.v_mul_f32(
                            Vgpr(vgprs.valu_acc[j][i] + r),
                            Vgpr(vgprs.valu_acc[j][i] + r),
                            Sgpr(sgprs.alpha),
                        )
                    context.s_waitcnt(vmcnt=0)
                    for r in range(agprs.num_reg_per_thread):
                        context.v_fma_f32(
                            Vgpr(vgprs.valu_acc[j][i] + r),
                            Sgpr(sgprs.beta),
                            Vgpr(vgprs.valu_c[j][i] + r),
                            Vgpr(vgprs.valu_acc[j][i] + r),
                        )

                    for l in range(0, agprs.num_reg_per_thread, agprs.num_reg_contiguous):
                        context.buffer_store_inst(agprs.num_reg_contiguous)(
                            VgprRange(vgprs.valu_acc[j][i]+l, agprs.num_reg_contiguous),
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            SgprRange(sgprs.srd_d, 4),
                            Sgpr(gw_offset_d),
                            0,
                        )

                        if agprs.num_reg_per_thread // agprs.num_reg_contiguous > 1:
                            with alloc_tmp_sgpr(1) as stmp:
                                increments = config.wavefront_size//config.mfma[1]*agprs.num_reg_contiguous
                                context.s_mul_i32(stmp, Sgpr(sgprs.stride_d_0), increments)
                                context.s_mul_i32(stmp, stmp, datatype_size(config.cd_type))
                                context.v_add_u32(Vgpr(vgprs.gl_offset_d[j][i]), Vgpr(vgprs.gl_offset_d[j][i]), stmp)

                    if needs_boundary_check:
                        context.label(skip_lbl)

        def gw_vgpr_minimized():
            for j, col in enumerate(vgprs.gl_offset_d):
                for i, row in enumerate(col):
                    context.comment(f"gw_addr_{i}_{j}")
                    with alloc_tmp_sgpr(1) as stmp:
                        context.v_and_b32(
                            Vgpr(vgprs.t_col), config.mfma[1] - 1, Vgpr(vgprs.wt_id)
                        )
                        context.s_mov_b32(stmp, j * config.wave_group[1] * config.mfma[1])
                        context.v_add_u32(Vgpr(vgprs.t_col), Vgpr(vgprs.t_col), stmp)
                        context.v_lshrrev_b32(
                            Vgpr(vgprs.t_row),
                            int(math.log2(config.mfma[1])),
                            Vgpr(vgprs.wt_id),
                        )
                        context.v_mul_lo_u32(Vgpr(vgprs.t_row), 4, Vgpr(vgprs.t_row))
                        context.s_mov_b32(stmp, i * config.wave_group[0] * config.mfma[0])
                        context.v_add_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), stmp)
                        context.v_add_i32(
                            Vgpr(vgprs.t_col), Vgpr(vgprs.t_col), Vgpr(vgprs.w_col)
                        )
                        context.v_add_i32(
                            Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), Vgpr(vgprs.w_row)
                        )
                        context.comment(f"setup voffset_c_{i}_{j}")
                        context.v_mul_lo_u32(
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            Vgpr(vgprs.t_col),
                            Sgpr(sgprs.stride_c_1),
                        )
                        context.v_add_u32(
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            Vgpr(vgprs.t_row),
                        )
                        context.v_mul_lo_u32(
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            datatype_size(config.cd_type),
                        )
                        context.comment(f"setup voffset_d_{i}_{j}")
                        context.v_mul_lo_u32(
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            Vgpr(vgprs.t_col),
                            Sgpr(sgprs.stride_d_1),
                        )
                        context.v_add_u32(
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            Vgpr(vgprs.t_row),
                        )
                        context.v_mul_lo_u32(
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            datatype_size(config.cd_type),
                        )
                    context.comment(f"gw_{i}_{j}")
                    skip_lbl = lbl(f"skip_gw_min_{i}_{j}")
                    needs_boundary_check = (config.tile_size[0] & (config.tile_size[0] - 1)) != 0
                    if needs_boundary_check:
                        with alloc_tmp_sgpr(1) as stmp:
                            context.s_add_i32(stmp, Sgpr(sgprs.row_idx), i * config.wave_group[0] * config.mfma[0])
                            context.s_cmp_ge_u32(stmp, Sgpr(sgprs.m))
                            context.s_cbranch_scc1(skip_lbl)

                    for r in range(agprs.num_reg_per_thread):
                        context.v_accvgpr_read_b32(
                            Vgpr(vgprs.valu_acc[j][i] + r), AccVgpr(agprs.arpgs[j][i] + r)
                        )
                    
                    for l in range(0, agprs.num_reg_per_thread, agprs.num_reg_contiguous):
                        context.buffer_load_inst(agprs.num_reg_contiguous)(
                            VgprRange(vgprs.valu_c[j][i]+l, agprs.num_reg_contiguous),
                            Vgpr(vgprs.gl_offset_c[j][i]),
                            SgprRange(sgprs.srd_c, 4),
                            Sgpr(gl_offset_c),
                            0,
                        )

                        if agprs.num_reg_per_thread // agprs.num_reg_contiguous > 1:
                            with alloc_tmp_sgpr(1) as stmp:
                                increments = config.wavefront_size//config.mfma[1]*agprs.num_reg_contiguous
                                context.s_mul_i32(stmp, Sgpr(sgprs.stride_c_0), increments)
                                context.s_mul_i32(stmp, stmp, datatype_size(config.cd_type))
                                context.v_add_u32(Vgpr(vgprs.gl_offset_c[j][i]), Vgpr(vgprs.gl_offset_c[j][i]), stmp)
                    for r in range(agprs.num_reg_per_thread):
                        context.v_mul_f32(
                            Vgpr(vgprs.valu_acc[j][i] + r),
                            Vgpr(vgprs.valu_acc[j][i] + r),
                            Sgpr(sgprs.alpha),
                        )
                    context.s_waitcnt(vmcnt=0)
                    for r in range(agprs.num_reg_per_thread):
                        context.v_fma_f32(
                            Vgpr(vgprs.valu_acc[j][i] + r),
                            Sgpr(sgprs.beta),
                            Vgpr(vgprs.valu_c[j][i] + r),
                            Vgpr(vgprs.valu_acc[j][i] + r),
                        )

                    for l in range(0, agprs.num_reg_per_thread, agprs.num_reg_contiguous):
                        context.buffer_store_inst(agprs.num_reg_contiguous)(
                            VgprRange(vgprs.valu_acc[j][i]+l, agprs.num_reg_contiguous),
                            Vgpr(vgprs.gl_offset_d[j][i]),
                            SgprRange(sgprs.srd_d, 4),
                            Sgpr(gw_offset_d),
                            0,
                        )

                        if agprs.num_reg_per_thread // agprs.num_reg_contiguous > 1:
                            with alloc_tmp_sgpr(1) as stmp:
                                increments = config.wavefront_size//config.mfma[1]*agprs.num_reg_contiguous
                                context.s_mul_i32(stmp, Sgpr(sgprs.stride_d_0), increments)
                                context.s_mul_i32(stmp, stmp, datatype_size(config.cd_type))
                                context.v_add_u32(Vgpr(vgprs.gl_offset_d[j][i]), Vgpr(vgprs.gl_offset_d[j][i]), stmp)

                    if needs_boundary_check:
                        context.label(skip_lbl)
        if opt.gw == 0:
            gw_naive()
        else:
            gw_vgpr_minimized()

        context.s_endpgm()
        return context.materialize()

    def body():
        return f"""
.globl {name}
.p2align 8
.type {name},@function
{name}:
{implementation(config)}
.L{name}_end:
    .size {name}, .L{name}_end - {name}
"""

    body_str = body()
    meta.sgpr_count = context.sgpr_counter
    meta.vgpr_count = context.vgpr_counter
    meta.agpr_count = context.agpr_counter

    if generate_parts:
        return body_str, meta.ro_data(), meta

    context.content.write(header())
    context.content.write(body_str)
    context.content.write(meta.ro_data())
    context.content.write(str(meta))
    return context.content.getvalue()


def generate_bundle_assembly(
    arch: str,
    kernel_parts: List[Tuple[str, str, FunctionMeta]],
) -> str:
    parts = [
        f'.amdgcn_target "amdgcn-amd-amdhsa--{arch}"\n.text\n'
    ]
    for body_str, _, _ in kernel_parts:
        parts.append(body_str)
    for _, rodata_str, _ in kernel_parts:
        parts.append(rodata_str)
    multi_meta = MultiKernelMeta([meta for _, _, meta in kernel_parts])
    parts.append(str(multi_meta))
    return "\n".join(parts)


def compile_bundle(
    bundle_name: str,
    bundle_asm: str,
    arch: str,
    output_folder: str,
    configs: Dict[str, GemmSolutionConfig],
) -> int:
    import os
    os.makedirs(output_folder, exist_ok=True)
    asm_file = f"{output_folder}/{bundle_name}.s"
    obj_file = f"{output_folder}/{bundle_name}.o"
    co_file = f"{output_folder}/{bundle_name}.co"
    toml_file = f"{output_folder}/{bundle_name}.toml"

    with open(asm_file, "w") as f:
        f.write(bundle_asm)
        f.flush()
        ret = subprocess.run(
            [
                DEFAULT_CLANG_PATH,
                "-x",
                "assembler",
                "-target",
                "amdgcn-amd-amdhsa",
                "-mcode-object-version=4",
                f"-mcpu={arch}",
                "-mwavefrontsize64",
                "-c",
                "-g",
                f.name,
                "-o",
                obj_file,
            ]
        )
        if ret.returncode != 0:
            return ret.returncode

        ret = subprocess.run(
            [
                DEFAULT_CLANG_PATH,
                "-target",
                "amdgcn-amd-amdhsa",
                obj_file,
                "-o",
                co_file,
            ]
        )
        if ret.returncode != 0:
            return ret.returncode

    toml_dict = {
        "kernels": {name: cfg.to_dict() for name, cfg in configs.items()}
    }
    with open(toml_file, "wb") as f:
        tomli_w.dump(toml_dict, f)

    return 0


def compile(kern_name: str, kern_str: str, arch: str, output_folder: str, gemm_config: GemmSolutionConfig):
    import os
    os.makedirs(output_folder, exist_ok=True)
    with open(f"{output_folder}/{kern_name}.s", "w") as f:
        f.write(kern_str)
        f.flush()
        ret = subprocess.run(
            [
                DEFAULT_CLANG_PATH,
                "-x",
                "assembler",
                "-target",
                "amdgcn-amd-amdhsa",
                "-mcode-object-version=4",
                f"-mcpu={arch}",
                "-mwavefrontsize64",
                "-c",
                "-g",
                f.name,
                "-o",
                f"{output_folder}/{kern_name}.o",
            ]
        )
        ret = subprocess.run(
            [
                DEFAULT_CLANG_PATH,
                "-target",
                "amdgcn-amd-amdhsa",
                f"{output_folder}/{kern_name}.o",
                "-o",
                f"{output_folder}/{kern_name}.co",
            ]
        )

    config_dict = gemm_config.to_dict()
    config_dict["name"] = kern_name
    config_dict["kernels"] = {kern_name: gemm_config.to_dict()}
    with open(f"{output_folder}/{kern_name}.toml", "wb") as f:
        tomli_w.dump(config_dict, f)

    return ret.returncode

if __name__ == "__main__":
    gemm_config = GemmSolutionConfig(
        DataType.FP32,
        DataType.FP32,
        DataType.FP32,
        DataType.FP32,
        # (16, 16, 1, 4),
        (32, 32, 1, 2),
        (2, 2),
        (4, 2),
        16,
        False,
        False,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument(dest="output_folder", action="store", type=str, help="Output folder")
    ap.add_argument(
        "--arch", dest="arch", action="store", choices=["gfx90a", "gfx90a:xnack-", "gfx942"]
    )
    args = ap.parse_args()

    arch = args.arch
    output_folder = args.output_folder

    opt = GemmOptimizations(1)
    opt.plr = 1
    asm_str = gemm(
        None,
        "generated_gemm",
        arch,
        gemm_config,
        opt,
        [
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
        ],
    )

    compile("generated_gemm", asm_str, arch, output_folder, gemm_config)
