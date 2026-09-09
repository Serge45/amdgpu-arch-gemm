from __future__ import annotations
from typing import Tuple, List, Dict, Optional
from generator.atoms import MMAAtom


class LdsLayout:
    """
    Manages LDS memory geometry, addressing, and coordinate mapping for matrix tiles.
    """
    def __init__(self, leading_dim: int, padding: int = 0, element_bytes: int = 4):
        self.leading_dim = leading_dim
        self.padding = padding
        self.stride = leading_dim + padding
        self.element_bytes = element_bytes

    def get_byte_offset(self, row: int, col: int) -> int:
        """
        Calculates byte offset in LDS for coordinate (row, col).
        """
        return (col * self.stride + row) * self.element_bytes


class LdsPaddingSolver:
    """
    Solves for optimal LDS padding to eliminate or minimize bank conflicts
    while strictly preserving 16-byte vector alignment.
    """
    @staticmethod
    def solve_pad_a(
        atom: MMAAtom,
        tile_m: int,
        depth_k: Optional[int] = None,
        wavefront_size: int = 64,
        num_banks: int = 32,
        element_bytes: Optional[int] = None,
        trans_a: bool = False,
    ) -> Tuple[int, int]:
        """
        Solves for optimal padding for Matrix A using beat-level conflict modeling.
        Returns: (best_pad, min_conflicts)
        """
        from generator.generator import datatype_size
        elem_bytes = element_bytes if element_bytes is not None else datatype_size(atom.dtype_a)
        best_pad = 0
        min_conflicts = 999

        # Constrain to multiples of 4 elements (16 bytes for FP32, 8 bytes for FP16)
        candidate_pads = [i * 4 for i in range(12)]
        num_bytes_read = 8 if (atom.shape[3] >= 8 and elem_bytes == 2) else 4
        num_banks_per_thread = max(1, num_bytes_read // 4)
        threads_per_beat = max(1, (num_banks * 4) // num_bytes_read)
        num_beats = max(1, wavefront_size // threads_per_beat)

        for pad in candidate_pads:
            if not trans_a:
                stride = tile_m + pad
            else:
                stride = (depth_k if depth_k is not None else atom.shape[3]) + pad
            max_conf_across_beats = 0
            for b in range(num_beats):
                bank_counts: Dict[int, int] = {}
                for wt in range(b * threads_per_beat, (b + 1) * threads_per_beat):
                    row, col = atom.get_thread_coords_a(wt)
                    if not trans_a:
                        addr = (col * stride + row) * elem_bytes
                    else:
                        addr = (row * stride + col) * elem_bytes
                    for b_off in range(num_banks_per_thread):
                        bank = ((addr + b_off * 4) // 4) % num_banks
                        bank_counts[bank] = bank_counts.get(bank, 0) + 1
                conf = max(bank_counts.values()) if bank_counts else 0
                if conf > max_conf_across_beats:
                    max_conf_across_beats = conf

            if max_conf_across_beats < min_conflicts:
                min_conflicts = max_conf_across_beats
                best_pad = pad
            elif max_conf_across_beats == min_conflicts and pad < best_pad:
                best_pad = pad

        return best_pad, min_conflicts

    @staticmethod
    def solve_pad_b(
        atom: MMAAtom,
        depth_k: int,
        tile_n: Optional[int] = None,
        wavefront_size: int = 64,
        num_banks: int = 32,
        element_bytes: Optional[int] = None,
        trans_b: bool = False,
    ) -> Tuple[int, int]:
        """
        Solves for optimal padding for Matrix B using beat-level conflict modeling.
        Returns: (best_pad, min_conflicts)
        """
        from generator.generator import datatype_size
        elem_bytes = element_bytes if element_bytes is not None else datatype_size(atom.dtype_b)
        best_pad = 0
        min_conflicts = 999

        # Constrain to multiples of 4 elements (16 bytes for FP32, 8 bytes for FP16)
        candidate_pads = [i * 4 for i in range(12)]
        num_bytes_read = 8 if (atom.shape[3] >= 8 and elem_bytes == 2) else 4
        num_banks_per_thread = max(1, num_bytes_read // 4)
        threads_per_beat = max(1, (num_banks * 4) // num_bytes_read)
        num_beats = max(1, wavefront_size // threads_per_beat)

        for pad in candidate_pads:
            if not trans_b:
                stride = depth_k + pad
            else:
                stride = (tile_n if tile_n is not None else 128) + pad
            max_conf_across_beats = 0
            for b in range(num_beats):
                bank_counts: Dict[int, int] = {}
                for wt in range(b * threads_per_beat, (b + 1) * threads_per_beat):
                    row, col = atom.get_thread_coords_b(wt)
                    if not trans_b:
                        addr = (col * stride + row) * elem_bytes
                    else:
                        addr = (row * stride + col) * elem_bytes
                    for b_off in range(num_banks_per_thread):
                        bank = ((addr + b_off * 4) // 4) % num_banks
                        bank_counts[bank] = bank_counts.get(bank, 0) + 1
                conf = max(bank_counts.values()) if bank_counts else 0
                if conf > max_conf_across_beats:
                    max_conf_across_beats = conf

            if max_conf_across_beats < min_conflicts:
                min_conflicts = max_conf_across_beats
                best_pad = pad
            elif max_conf_across_beats == min_conflicts and pad < best_pad:
                best_pad = pad

        return best_pad, min_conflicts


class TiledMMA:
    """
    Hierarchical matrix multiply-accumulate tile.
    Decomposes a Workgroup Tile (BlockTile) into Wave Tiles, which are further
    decomposed into Atom Tiles (MMAAtom).
    """
    def __init__(
        self,
        atom: MMAAtom,
        wave_group: Tuple[int, int],
        wave_tiling: Tuple[int, int],
        wavefront_size: int = 64,
        target: Optional[Any] = None,
    ):
        self.atom = atom
        self.wave_group = wave_group
        self.wave_tiling = wave_tiling
        self.wavefront_size = wavefront_size
        self.target = target
        if self.target is not None:
            self.validate(self.target)

    @property
    def num_acc_regs_per_thread(self) -> int:
        """Total accumulator registers per thread required for this tiled MMA."""
        return self.wave_tiling[0] * self.wave_tiling[1] * self.atom.num_acc_regs

    def validate(self, target: Optional[Any] = None) -> None:
        """Validates that AGPR/VGPR requirements do not exceed hardware limits."""
        tgt = target or self.target
        if tgt is not None:
            from generator.target_spec import RegFileModel
            max_limit = (
                tgt.max_vgpr
                if tgt.reg_file_model == RegFileModel.VGPR_ONLY
                else tgt.max_agpr
            )
            if self.num_acc_regs_per_thread > max_limit:
                raise ValueError(
                    f"Accumulator register requirement ({self.num_acc_regs_per_thread}) exceeds "
                    f"target '{tgt.name}' limit of {max_limit} for wave_tiling={self.wave_tiling}."
                )

    @property
    def atom_tile(self) -> Tuple[int, int]:
        """Shape of a single MMA instruction execution: (m, n)"""
        return self.atom.shape[0], self.atom.shape[1]

    @property
    def wave_tile(self) -> Tuple[int, int]:
        """Matrix tile size computed by a single wave across all its iterations"""
        return (
            self.atom.shape[0] * self.wave_tiling[0],
            self.atom.shape[1] * self.wave_tiling[1],
        )

    @property
    def block_tile(self) -> Tuple[int, int]:
        """Total matrix tile size computed by the entire workgroup"""
        return (
            self.atom.shape[0] * self.wave_group[0] * self.wave_tiling[0],
            self.atom.shape[1] * self.wave_group[1] * self.wave_tiling[1],
        )

    @property
    def num_waves(self) -> int:
        return self.wave_group[0] * self.wave_group[1]

    @property
    def num_threads(self) -> int:
        return self.num_waves * self.wavefront_size

    def get_wave_coords(self, wave_id: int) -> Tuple[int, int]:
        """Returns (wave_row, wave_col) for wave_id within wave_group."""
        w_row = wave_id % self.wave_group[0]
        w_col = wave_id // self.wave_group[0]
        return w_row, w_col


class TiledCopy:
    """
    Manages vector global load partitioning across threads in a workgroup.
    """
    def __init__(
        self,
        vector_bytes: int,
        tile_dim: int,
        depth_k: int,
        num_workitems: int,
        element_bytes: int = 4,
    ):
        self.vector_bytes = vector_bytes
        self.tile_dim = tile_dim
        self.depth_k = depth_k
        self.num_workitems = num_workitems
        self.element_bytes = element_bytes

    @property
    def num_loads_0(self) -> int:
        """Number of vector load iterations along dimension 0"""
        bytes_dim0 = self.tile_dim * self.element_bytes
        total_load_bytes = self.vector_bytes * self.num_workitems
        return max(bytes_dim0 // total_load_bytes, 1)

    @property
    def num_loads_1(self) -> int:
        """Number of vector load iterations along dimension 1"""
        bytes_dim0 = self.tile_dim * self.element_bytes
        loads_per_row = self.num_workitems // (bytes_dim0 // self.vector_bytes)
        return self.depth_k // loads_per_row
