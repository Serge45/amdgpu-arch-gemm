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
    while strictly preserving 16-byte (4-element) vector alignment.
    """
    @staticmethod
    def solve_pad_a(
        atom: MMAAtom,
        tile_m: int,
        wavefront_size: int = 64,
        num_banks: int = 32,
        element_bytes: int = 4,
    ) -> Tuple[int, int]:
        """
        Solves for optimal padding for Matrix A.
        Returns: (best_pad, min_conflicts)
        """
        best_pad = 0
        min_conflicts = 999

        # Stride must be a multiple of 4 elements to preserve 16-byte vector alignment
        candidate_pads = [0, 4, 8, 12, 16]
        for pad in candidate_pads:
            stride = tile_m + pad
            bank_counts: Dict[int, int] = {}

            for wt in range(wavefront_size):
                row, col = atom.get_thread_coords_a(wt)
                addr = (col * stride + row) * element_bytes
                bank = (addr // element_bytes) % num_banks
                bank_counts[bank] = bank_counts.get(bank, 0) + 1

            max_conf = max(bank_counts.values()) if bank_counts else 0
            if max_conf < min_conflicts:
                min_conflicts = max_conf
                best_pad = pad
            elif max_conf == min_conflicts and pad < best_pad:
                best_pad = pad

        return best_pad, min_conflicts

    @staticmethod
    def solve_pad_b(
        atom: MMAAtom,
        depth_k: int,
        wavefront_size: int = 64,
        num_banks: int = 32,
        element_bytes: int = 4,
    ) -> Tuple[int, int]:
        """
        Solves for optimal padding for Matrix B.
        Returns: (best_pad, min_conflicts)
        """
        best_pad = 0
        min_conflicts = 999

        candidate_pads = [0, 4, 8, 12, 16]
        for pad in candidate_pads:
            stride = depth_k + pad
            bank_counts: Dict[int, int] = {}

            for wt in range(wavefront_size):
                row, col = atom.get_thread_coords_b(wt)
                addr = (col * stride + row) * element_bytes
                bank = (addr // element_bytes) % num_banks
                bank_counts[bank] = bank_counts.get(bank, 0) + 1

            max_conf = max(bank_counts.values()) if bank_counts else 0
            if max_conf < min_conflicts:
                min_conflicts = max_conf
                best_pad = pad
            elif max_conf == min_conflicts and pad < best_pad:
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
    ):
        self.atom = atom
        self.wave_group = wave_group
        self.wave_tiling = wave_tiling
        self.wavefront_size = wavefront_size

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
