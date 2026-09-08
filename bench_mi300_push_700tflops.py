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


def build_push700_bundle(output_dir: str = "out_mi300_push700") -> GemmKernelBundle:
    print("=== Building MI300 4K HGEMM Push-700TFLOPS Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_push700_gfx942", target=GFX942)

    # Format: (dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buffer_lds)
    candidates = [
        # Baseline champion from initial sweep
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (2, 2), (4, 2), 8, True),

        # Pillar 1: 256x128x32 & 128x256x32 DOUBLE BUFFER LDS (60 KB LDS <= 64 KB, 1 barrier per iter)
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (2, 2), (4, 2), 1, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (2, 2), (4, 2), 4, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (2, 2), (4, 2), 8, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (2, 2), (4, 2), 16, False),

        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 2), (2, 4), 1, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 2), (2, 4), 4, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 2), (2, 4), 8, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 2), (2, 4), 16, False),

        # Pillar 1 (16x16 atom): 256x128x32 & 128x256x32 DOUBLE BUFFER LDS
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 32), (2, 2), (8, 4), 4, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 32), (2, 2), (8, 4), 8, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 32), (2, 2), (8, 4), 16, False),

        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 2), (4, 8), 4, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 2), (4, 8), 8, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 2), (4, 8), 16, False),

        # Pillar 2: 256x256 GIANT TILES (256 AGPRs / 512 total registers, 48 KB LDS dbl)
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 256, 16), (2, 2), (4, 4), 1, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 256, 16), (2, 2), (4, 4), 4, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 256, 16), (2, 2), (4, 4), 8, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 256, 16), (2, 2), (4, 4), 16, False),

        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 256, 16), (2, 2), (8, 8), 4, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 256, 16), (2, 2), (8, 8), 8, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 256, 16), (2, 2), (8, 8), 16, False),

        # Pillar 3: Deep K=64 Unroll (8 steps/iteration)
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (2, 2), (4, 2), 4, True),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (2, 2), (4, 2), 8, True),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (2, 2), (4, 2), 16, True),

        # Pillar 4: Symmetric 160x160x32 DOUBLE BUFFER LDS (50 KB LDS)
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (160, 160, 32), (2, 2), (5, 5), 4, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (160, 160, 32), (2, 2), (5, 5), 8, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (160, 160, 32), (2, 2), (5, 5), 16, False),
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

    print(f"\nBundle contains {len(bundle)} elite kernels.", flush=True)
    t0 = time.time()
    ret = bundle.compile(output_dir)
    t_compile = time.time() - t0
    print(f"Compilation finished in {t_compile:.2f}s with status code {ret}", flush=True)
    return bundle


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_mi300_push700"
    build_push700_bundle(out_dir)
