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
