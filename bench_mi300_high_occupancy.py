from __future__ import annotations
import os
import sys
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import (
    MFMA_F32_32x32x8_F16,
    MFMA_F32_16x16x16_F16,
)
from generator.target_spec import GFX942
from generator.generator import DataType


def build_occupancy_bundle(output_dir: str = "out_mi300_high_occ") -> GemmKernelBundle:
    print("=== Building MI300 High-Occupancy / Deep-K HGEMM Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_high_occ_gfx942", target=GFX942)

    # Format: (dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buffer_lds)
    candidates = [
        # Current Champion from Round 1
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 256, 16), (2, 2), (8, 8), 8, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 256, 16), (2, 2), (8, 8), 4, False),

        # Section 1: 8-Wave Workgroups (wg4x2 & wg2x4, 2 waves per SIMD unit!)
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (4, 2), (2, 2), 1, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (4, 2), (2, 2), 4, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (4, 2), (2, 2), 8, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (4, 2), (2, 2), 16, False),

        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 4), (2, 2), 1, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 4), (2, 2), 4, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 4), (2, 2), 8, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 4), (2, 2), 16, False),

        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 32), (4, 2), (4, 4), 4, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 32), (4, 2), (4, 4), 8, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 4), (4, 4), 4, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 4), (4, 4), 8, False),

        # Section 2: 16-Wave Workgroups (wg4x4, 4 waves per SIMD unit!)
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (4, 4), (2, 1), 4, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (4, 4), (2, 1), 8, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (4, 4), (1, 2), 4, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (4, 4), (1, 2), 8, False),

        # Section 3: Deep K=64 with 1024 WGs (128x128x64)
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 64), (2, 2), (4, 4), 4, True),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 64), (2, 2), (4, 4), 8, True),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 64), (2, 2), (4, 4), 16, True),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 128, 64), (2, 2), (2, 2), 4, True),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 128, 64), (2, 2), (2, 2), 8, True),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 128, 64), (2, 2), (2, 2), 16, True),

        # Section 4: Deep K=128 with 2048 WGs (128x64x128 & 64x128x128)
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 64, 128), (2, 2), (4, 2), 4, True),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 64, 128), (2, 2), (4, 2), 8, True),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 64, 128), (2, 2), (4, 2), 16, True),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (64, 128, 128), (2, 2), (2, 4), 8, True),
    ]

    for dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buffer_lds in candidates:
        kernel = GemmKernel(target=GFX942)
        kernel.set_inputs(
            A=Tensor(shape=(4096, 4096), dtype=dtype, layout=LayoutType.COL_MAJOR),
            B=Tensor(shape=(4096, 4096), dtype=dtype, layout=LayoutType.COL_MAJOR),
            C=Tensor(shape=(4096, 4096), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        ).set_transposes(
            trans_a=trans_a, trans_b=trans_b
        ).set_workgroup_mapping(
            wgm
        ).set_tiling(
            block_tile=block_tile,
            wave_group=wave_group,
            wave_tiling=wave_tiling,
        ).bind_atoms(
            mma=mma_atom
        ).set_schedule(
            vmem_stages=2,
            single_buffer_lds=single_buffer_lds,
        )
        bundle.add(kernel)

    print(f"\nBundle contains {len(bundle)} elite high-occupancy kernels.", flush=True)
    t0 = time.time()
    ret = bundle.compile(output_dir)
    t_compile = time.time() - t0
    print(f"Compilation finished in {t_compile:.2f}s with status code {ret}", flush=True)
    return bundle


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_mi300_high_occ"
    build_occupancy_bundle(out_dir)
