from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager
from generator.target_spec import TargetSpec, GFX90A
from generator.generator import (
    Vgpr,
    VgprRange,
    Sgpr,
    SgprRange,
    AccVgpr,
    AccVgprRange,
    Gpr,
    GprRange,
)


class VirtualGpr:
    def __init__(
        self,
        name: str,
        size: int = 1,
        align: int = 1,
        reg_type: str = "v",
        physical_index: Optional[int] = None,
    ):
        self.name = name
        self.size = size
        self.align = align
        self.reg_type = reg_type
        self.physical_index = physical_index

    @property
    def is_allocated(self) -> bool:
        return self.physical_index is not None

    def lower(self) -> Union[Gpr, GprRange]:
        if self.physical_index is None:
            raise RuntimeError(f"Virtual register '{self.name}' has not been allocated!")
        
        if self.reg_type == "v":
            if self.size == 1:
                return Vgpr(self.physical_index)
            return VgprRange(self.physical_index, self.size)
        elif self.reg_type == "s":
            if self.size == 1:
                return Sgpr(self.physical_index)
            return SgprRange(self.physical_index, self.size)
        elif self.reg_type == "acc":
            if self.size == 1:
                return AccVgpr(self.physical_index)
            return AccVgprRange(self.physical_index, self.size)
        else:
            raise ValueError(f"Unknown register type: {self.reg_type}")

    def split(self, num_comp: int = 1) -> List[VirtualGpr]:
        if self.physical_index is None:
            raise RuntimeError(f"Cannot split unallocated virtual register '{self.name}'")
        if self.size % num_comp != 0:
            raise ValueError(f"Cannot evenly split size {self.size} by {num_comp}")
        
        sub_regs = []
        for i in range(0, self.size, num_comp):
            sub_regs.append(
                VirtualGpr(
                    name=f"{self.name}_{i // num_comp}",
                    size=num_comp,
                    align=num_comp if num_comp > 1 else 1,
                    reg_type=self.reg_type,
                    physical_index=self.physical_index + i,
                )
            )
        return sub_regs

    def __str__(self) -> str:
        if self.physical_index is not None:
            if self.size == 1:
                return f"{self.reg_type}[{self.physical_index}]/*{self.name}*/"
            return f"{self.reg_type}[{self.physical_index}:{self.physical_index + self.size - 1}]/*{self.name}*/"
        return f"${self.name}({self.reg_type}, size={self.size})"


class VirtualVgpr(VirtualGpr):
    def __init__(self, name: str, size: int = 1, align: int = 1, physical_index: Optional[int] = None):
        super().__init__(name=name, size=size, align=align, reg_type="v", physical_index=physical_index)


class VirtualSgpr(VirtualGpr):
    def __init__(self, name: str, size: int = 1, align: int = 1, physical_index: Optional[int] = None):
        super().__init__(name=name, size=size, align=align, reg_type="s", physical_index=physical_index)


class VirtualAccVgpr(VirtualGpr):
    def __init__(self, name: str, size: int = 1, align: int = 1, physical_index: Optional[int] = None):
        super().__init__(name=name, size=size, align=align, reg_type="acc", physical_index=physical_index)


class AlignedChunkPool:
    """
    Manages allocation of physical register slots respecting alignment constraints.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.occupied = [False] * capacity
        self.high_watermark = 0

    def allocate(self, size: int, align: int) -> int:
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        if align <= 0 or (align & (align - 1)) != 0:
            raise ValueError(f"Align must be a power of two, got {align}")

        # Scan for first contiguous chunk satisfying alignment
        start = 0
        while start + size <= self.capacity:
            if start % align != 0:
                start += align - (start % align)
                continue

            # Check if all slots in [start, start + size) are free
            free = True
            for i in range(size):
                if self.occupied[start + i]:
                    start += i + 1
                    free = False
                    break

            if free:
                for i in range(size):
                    self.occupied[start + i] = True
                self.high_watermark = max(self.high_watermark, start + size)
                return start

        raise RuntimeError(
            f"Failed to allocate {size} registers with align={align}. "
            f"Capacity={self.capacity}, High Watermark={self.high_watermark}"
        )

    def free(self, start: int, size: int):
        if start < 0 or start + size > self.capacity:
            raise ValueError(f"Invalid range to free: [{start}, {start + size})")
        for i in range(size):
            self.occupied[start + i] = False


class RegisterAllocator:
    """
    Structured register allocator for AMDGPU kernels.
    Supports:
    - Alignment (2-dword alignment for multi-dword VGPRs, 4-dword for SGPR SRDs)
    - Scoped lifetime recycling (freeing temporaries when exiting scopes)
    - Explicit aliasing (time-multiplexed reuse between disjoint phases)
    - Occupancy diagnostics
    """
    def __init__(
        self,
        target: TargetSpec = GFX90A,
        max_vgpr_budget: Optional[int] = None,
        max_sgpr_budget: Optional[int] = None,
        max_agpr_budget: Optional[int] = None,
    ):
        self.target = target
        self.max_vgpr_budget = max_vgpr_budget or target.max_vgpr
        self.max_sgpr_budget = max_sgpr_budget or target.max_sgpr
        self.max_agpr_budget = max_agpr_budget or target.max_agpr

        self.vgpr_pool = AlignedChunkPool(self.target.max_vgpr)
        self.sgpr_pool = AlignedChunkPool(self.target.max_sgpr)
        self.agpr_pool = AlignedChunkPool(self.target.max_agpr)

        self.current_scope: Optional[str] = None
        self.scope_regs: Dict[str, List[VirtualGpr]] = {}
        self.allocated_regs: List[VirtualGpr] = []

    def alloc_vgpr(
        self, name: str, size: int = 1, align: Optional[int] = None
    ) -> VirtualVgpr:
        # Default alignment for multi-dword VGPR (size >= 2) is 2 (even alignment)
        if align is None:
            align = 2 if size >= 2 else 1

        idx = self.vgpr_pool.allocate(size, align)
        reg = VirtualVgpr(name=name, size=size, align=align, physical_index=idx)
        self._record_allocation(reg)
        return reg

    def alloc_sgpr(
        self, name: str, size: int = 1, align: Optional[int] = None
    ) -> VirtualSgpr:
        # Default alignment: 4 for SRDs (>= 4 dwords), 2 for 64-bit (>= 2 dwords), 1 for single dwords
        if align is None:
            if size >= 4:
                align = 4
            elif size >= 2:
                align = 2
            else:
                align = 1

        idx = self.sgpr_pool.allocate(size, align)
        reg = VirtualSgpr(name=name, size=size, align=align, physical_index=idx)
        self._record_allocation(reg)
        return reg

    def alloc_agpr(
        self, name: str, size: int = 1, align: Optional[int] = None
    ) -> VirtualAccVgpr:
        if align is None:
            align = 2 if size >= 2 else 1

        idx = self.agpr_pool.allocate(size, align)
        reg = VirtualAccVgpr(name=name, size=size, align=align, physical_index=idx)
        self._record_allocation(reg)
        return reg

    def alias_vgpr(self, name: str, target_reg: VirtualVgpr) -> VirtualVgpr:
        """
        Creates an explicit alias that shares the physical registers of target_reg.
        Useful for Epilogue reusing Global Load registers.
        """
        if target_reg.physical_index is None:
            raise RuntimeError(f"Cannot alias unallocated register '{target_reg.name}'")
        alias_reg = VirtualVgpr(
            name=name,
            size=target_reg.size,
            align=target_reg.align,
            physical_index=target_reg.physical_index,
        )
        self.allocated_regs.append(alias_reg)
        return alias_reg

    def free(self, reg: VirtualGpr):
        if reg.physical_index is None:
            return

        if reg.reg_type == "v":
            self.vgpr_pool.free(reg.physical_index, reg.size)
        elif reg.reg_type == "s":
            self.sgpr_pool.free(reg.physical_index, reg.size)
        elif reg.reg_type == "acc":
            self.agpr_pool.free(reg.physical_index, reg.size)
        
        reg.physical_index = None

    @contextmanager
    def scope(self, name: str):
        """
        Context manager for scoped allocations. All registers allocated within
        the scope are automatically freed upon exiting the context.
        """
        prev_scope = self.current_scope
        self.current_scope = name
        if name not in self.scope_regs:
            self.scope_regs[name] = []

        try:
            yield
        finally:
            for reg in self.scope_regs[name]:
                self.free(reg)
            self.scope_regs[name].clear()
            self.current_scope = prev_scope

    def _record_allocation(self, reg: VirtualGpr):
        self.allocated_regs.append(reg)
        if self.current_scope is not None:
            self.scope_regs[self.current_scope].append(reg)

    def lower(self, reg: VirtualGpr) -> Union[Gpr, GprRange]:
        return reg.lower()

    def get_diagnostics(self) -> Dict[str, Any]:
        vgprs_used = self.vgpr_pool.high_watermark
        sgprs_used = self.sgpr_pool.high_watermark
        agprs_used = self.agpr_pool.high_watermark

        # Calculate CDNA occupancy
        # CDNA supports up to 4 waves/SIMD when VGPR <= 128, 2 waves when VGPR <= 256.
        if vgprs_used <= 128:
            waves_per_simd = 4
        elif vgprs_used <= 256:
            waves_per_simd = 2
        else:
            waves_per_simd = 0

        vgpr_budget_exceeded = vgprs_used > self.max_vgpr_budget

        return {
            "target": self.target.name,
            "vgpr_used": vgprs_used,
            "vgpr_capacity": self.target.max_vgpr,
            "vgpr_budget": self.max_vgpr_budget,
            "vgpr_budget_exceeded": vgpr_budget_exceeded,
            "sgpr_used": sgprs_used,
            "sgpr_capacity": self.target.max_sgpr,
            "agpr_used": agprs_used,
            "agpr_capacity": self.target.max_agpr,
            "estimated_waves_per_simd": waves_per_simd,
        }

    def summary(self) -> str:
        diag = self.get_diagnostics()
        return (
            f"[Register Allocator Summary - {diag['target'].upper()}]\n"
            f"  VGPRs: {diag['vgpr_used']} / {diag['vgpr_capacity']} "
            f"(Budget: {diag['vgpr_budget']}) "
            f"{'[EXCEEDED]' if diag['vgpr_budget_exceeded'] else '[OK]'}\n"
            f"  SGPRs: {diag['sgpr_used']} / {diag['sgpr_capacity']}\n"
            f"  AGPRs: {diag['agpr_used']} / {diag['agpr_capacity']}\n"
            f"  Estimated Occupancy: {diag['estimated_waves_per_simd']} Waves / SIMD"
        )
