from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Type, Union, Optional
from generator.generator import (
    GpuContext,
    Vgpr,
    Sgpr,
    AccVgpr,
    VgprRange,
    SgprRange,
    AccVgprRange,
    DataType,
)


class MMAAtom(ABC):
    """
    Abstract base class for matrix multiplication-accumulation hardware instructions.
    Encapsulates operand shapes, thread coordinate mapping, latencies, and instruction emission.
    """
    name: str
    shape: Tuple[int, int, int, int]  # (m, n, k, num_sub_k)
    dtype_a: DataType
    dtype_b: DataType
    dtype_c: DataType
    dest_reg_type: Type[Union[AccVgpr, Vgpr]]
    exec_cycles: int
    issue_cycles: int

    @abstractmethod
    def get_thread_coords_a(self, thread_id: int) -> Tuple[int, int]:
        """
        Returns (row, col) coordinates within the atom tile held by thread_id for Matrix A.
        """
        pass

    @abstractmethod
    def get_thread_coords_b(self, thread_id: int) -> Tuple[int, int]:
        """
        Returns (row, col) coordinates within the atom tile held by thread_id for Matrix B.
        """
        pass

    @abstractmethod
    def emit(
        self,
        ctx: GpuContext,
        dst: Union[AccVgpr, Vgpr, AccVgprRange, VgprRange],
        src_a: Union[Vgpr, VgprRange],
        src_b: Union[Vgpr, VgprRange],
        src_c: Union[AccVgpr, Vgpr, AccVgprRange, VgprRange],
    ):
        """
        Emits the assembly instruction into GpuContext.
        """
        pass


class MFMA_F32_32x32x2_F32(MMAAtom):
    """
    CDNA v_mfma_f32_32x32x2f32 instruction.
    Computes a 32x32x2 matrix tile for FP32.
    """
    name = "v_mfma_f32_32x32x2f32"
    shape = (32, 32, 1, 2)
    dtype_a = DataType.FP32
    dtype_b = DataType.FP32
    dtype_c = DataType.FP32
    dest_reg_type = AccVgpr
    exec_cycles = 16
    issue_cycles = 2

    def get_thread_coords_a(self, thread_id: int) -> Tuple[int, int]:
        # t_row = thread_id & (mfma_m - 1), t_col = thread_id // mfma_m
        return thread_id & 31, thread_id // 32

    def get_thread_coords_b(self, thread_id: int) -> Tuple[int, int]:
        # For Matrix B: t_row = thread_id // mfma_n, t_col = thread_id & (mfma_n - 1)
        return thread_id // 32, thread_id & 31

    def emit(
        self,
        ctx: GpuContext,
        dst: Union[AccVgpr, Vgpr, AccVgprRange, VgprRange],
        src_a: Union[Vgpr, VgprRange],
        src_b: Union[Vgpr, VgprRange],
        src_c: Union[AccVgpr, Vgpr, AccVgprRange, VgprRange],
    ):
        ctx.v_mfma_f32_32x32x2f32(dst, src_a, src_b, src_c)


class MFMA_F32_16x16x4_F32(MMAAtom):
    """
    CDNA v_mfma_f32_16x16x4f32 instruction.
    Computes a 16x16x4 matrix tile for FP32.
    """
    name = "v_mfma_f32_16x16x4f32"
    shape = (16, 16, 1, 4)
    dtype_a = DataType.FP32
    dtype_b = DataType.FP32
    dtype_c = DataType.FP32
    dest_reg_type = AccVgpr
    exec_cycles = 16
    issue_cycles = 2

    def get_thread_coords_a(self, thread_id: int) -> Tuple[int, int]:
        return thread_id & 15, thread_id // 16

    def get_thread_coords_b(self, thread_id: int) -> Tuple[int, int]:
        return thread_id // 16, thread_id & 15

    def emit(
        self,
        ctx: GpuContext,
        dst: Union[AccVgpr, Vgpr, AccVgprRange, VgprRange],
        src_a: Union[Vgpr, VgprRange],
        src_b: Union[Vgpr, VgprRange],
        src_c: Union[AccVgpr, Vgpr, AccVgprRange, VgprRange],
    ):
        ctx.v_mfma_f32_16x16x4f32(dst, src_a, src_b, src_c)


class CopyAtom(ABC):
    """
    Abstract base class for data movement instructions (Global -> VGPR, LDS -> VGPR, VGPR -> LDS).
    """
    vector_dwords: int
    vector_bytes: int

    def __init__(self, vector_dwords: int):
        self.vector_dwords = vector_dwords
        self.vector_bytes = vector_dwords * 4


class BufferLoadAtom(CopyAtom):
    """
    Encapsulates global memory buffer_load operations (1, 2, or 4 dwords).
    """
    def emit(
        self,
        ctx: GpuContext,
        dst: Union[Vgpr, VgprRange],
        vaddr: Vgpr,
        srsrc: SgprRange,
        soffset: Union[Sgpr, int] = 0,
        offset: int = 0,
    ):
        if self.vector_dwords == 1:
            ctx.buffer_load_dword(dst, vaddr, srsrc, soffset, offset)
        elif self.vector_dwords == 2:
            ctx.buffer_load_dwordx2(dst, vaddr, srsrc, soffset, offset)
        elif self.vector_dwords == 4:
            ctx.buffer_load_dwordx4(dst, vaddr, srsrc, soffset, offset)
        else:
            raise ValueError(f"Unsupported buffer_load vector width: {self.vector_dwords}")


class DsWriteAtom(CopyAtom):
    """
    Encapsulates Local Data Share (LDS) write operations (32, 64, or 128 bits).
    """
    def emit(
        self,
        ctx: GpuContext,
        vaddr: Vgpr,
        vdata: Union[Vgpr, VgprRange],
        offset: int = 0,
    ):
        if self.vector_dwords == 1:
            ctx.ds_write_b32(vaddr, vdata, offset)
        elif self.vector_dwords == 2:
            ctx.ds_write_b64(vaddr, vdata, offset)
        elif self.vector_dwords == 4:
            ctx.ds_write_b128(vaddr, vdata, offset)
        else:
            raise ValueError(f"Unsupported ds_write vector width: {self.vector_dwords}")


class DsReadAtom(CopyAtom):
    """
    Encapsulates Local Data Share (LDS) read operations (32, 64, or 128 bits).
    """
    def emit(
        self,
        ctx: GpuContext,
        vdst: Union[Vgpr, VgprRange],
        vaddr: Vgpr,
        offset: int = 0,
    ):
        if self.vector_dwords == 1:
            ctx.ds_read_b32(vdst, vaddr, offset)
        elif self.vector_dwords == 2:
            ctx.ds_read_b64(vdst, vaddr, offset)
        elif self.vector_dwords == 4:
            ctx.ds_read_b128(vdst, vaddr, offset)
        else:
            raise ValueError(f"Unsupported ds_read vector width: {self.vector_dwords}")
