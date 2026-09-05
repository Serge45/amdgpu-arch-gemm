from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class ArchitectureFamily(Enum):
    CDNA = auto()
    RDNA = auto()


class RegFileModel(Enum):
    DISJOINT_AGPR_VGPR = auto()  # CDNA1/2: Separate 256 VGPR + 256 AGPR
    UNIFIED = auto()             # CDNA3/4: Unified 512 physical vector register pool
    VGPR_ONLY = auto()           # RDNA3/4: No AGPR, VGPR only


@dataclass(frozen=True)
class TargetSpec:
    name: str
    family: ArchitectureFamily
    wavefront_size: int
    max_vgpr: int
    max_sgpr: int
    max_agpr: int
    reg_file_model: RegFileModel
    lds_num_banks: int
    lds_bank_bytes: int = 4
    max_lds_bytes: int = 65536


# CDNA 2 (MI200 series, e.g. MI210 / MI250X) - Current project default hardware
GFX90A = TargetSpec(
    name="gfx90a",
    family=ArchitectureFamily.CDNA,
    wavefront_size=64,
    max_vgpr=256,
    max_sgpr=104,
    max_agpr=256,
    reg_file_model=RegFileModel.DISJOINT_AGPR_VGPR,
    lds_num_banks=64,
    lds_bank_bytes=4,
    max_lds_bytes=65536,
)

# CDNA 3 (MI300 series, e.g. MI300A / MI300X)
GFX942 = TargetSpec(
    name="gfx942",
    family=ArchitectureFamily.CDNA,
    wavefront_size=64,
    max_vgpr=512,
    max_sgpr=104,
    max_agpr=256,
    reg_file_model=RegFileModel.UNIFIED,
    lds_num_banks=64,
    lds_bank_bytes=4,
    max_lds_bytes=65536,
)

# RDNA 3 (e.g. RX 7900 XTX)
GFX1100 = TargetSpec(
    name="gfx1100",
    family=ArchitectureFamily.RDNA,
    wavefront_size=32,
    max_vgpr=256,
    max_sgpr=104,
    max_agpr=0,
    reg_file_model=RegFileModel.VGPR_ONLY,
    lds_num_banks=32,
    lds_bank_bytes=4,
    max_lds_bytes=65536,
)
