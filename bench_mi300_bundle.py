from __future__ import annotations
import os
import sys
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import (
    MFMA_F32_32x32x2_F32,
    MFMA_F32_16x16x4_F32,
    MFMA_F32_32x32x8_F16,
    MFMA_F32_16x16x16_F16,
)
from generator.target_spec import GFX942
from generator.generator import DataType


def build_mi300_bundle(output_dir: str = "out_mi300_bundle") -> GemmKernelBundle:
    print("=== Building MI300 (CDNA 3 / GFX942) Kernel Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_gemm_bundle_gfx942", target=GFX942)

    # 1. SGEMM Candidates (FP32 inputs, FP32 accumulate, FP32 output)
    # Target: 304 CUs (MI300X) / 228 CUs (MI300A)
    # Format: (dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buffer_lds)
    sgemm_candidates = [
        # --- 128x128 (1024 WGs at 4K = 3.37 waves/CU on MI300X, 95.2% efficiency) ---
        (DataType.FP32, False, False, MFMA_F32_32x32x2_F32(), (128, 128, 16), (2, 2), (2, 2), 1, False),
        (DataType.FP32, False, False, MFMA_F32_32x32x2_F32(), (128, 128, 16), (2, 2), (2, 2), 4, False),
        (DataType.FP32, False, False, MFMA_F32_32x32x2_F32(), (128, 128, 16), (2, 2), (2, 2), 8, False),
        (DataType.FP32, False, False, MFMA_F32_32x32x2_F32(), (128, 128, 16), (2, 2), (2, 2), 16, False),

        # --- 128x64 & 64x128 (2048 WGs at 4K = 6.74 waves/CU, 98.5% efficiency) ---
        (DataType.FP32, False, False, MFMA_F32_32x32x2_F32(), (128, 64, 16), (2, 2), (2, 1), 1, False),
        (DataType.FP32, False, False, MFMA_F32_32x32x2_F32(), (128, 64, 16), (2, 2), (2, 1), 8, False),
        (DataType.FP32, False, False, MFMA_F32_32x32x2_F32(), (128, 64, 16), (2, 2), (2, 1), 16, False),
        (DataType.FP32, False, False, MFMA_F32_32x32x2_F32(), (64, 128, 16), (2, 2), (1, 2), 8, False),

        # --- 256x128 & 128x256 (512 WGs at 4K = large tile data reuse) ---
        (DataType.FP32, False, False, MFMA_F32_32x32x2_F32(), (256, 128, 16), (2, 2), (4, 2), 1, True),
        (DataType.FP32, False, False, MFMA_F32_32x32x2_F32(), (256, 128, 16), (2, 2), (4, 2), 8, True),
        (DataType.FP32, False, False, MFMA_F32_32x32x2_F32(), (128, 256, 16), (2, 2), (2, 4), 8, True),
    ]

    # 2. HGEMM TN Candidates (FP16 inputs, FP32 accumulate, FP32 output)
    hgemm_candidates = [
        # --- 128x128 Deep K Double Buffer (1024 WGs, 45 KB LDS, 1 barrier) ---
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 32), (2, 2), (4, 4), 1, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 32), (2, 2), (4, 4), 4, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 32), (2, 2), (4, 4), 8, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 32), (2, 2), (4, 4), 16, False),

        # --- 128x64 & 64x128 K=64 Double Buffer (2048 WGs at 4K, 512 WGs at 2K) ---
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 64, 64), (2, 2), (4, 2), 1, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 64, 64), (2, 2), (4, 2), 8, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 64, 64), (2, 2), (4, 2), 16, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (64, 128, 64), (2, 2), (2, 4), 8, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (64, 128, 64), (2, 2), (2, 4), 16, False),

        # --- 160x128x32 & 128x160x32 Double Buffer (45 KB LDS, 1 barrier) ---
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (160, 128, 32), (1, 4), (5, 1), 4, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (160, 128, 32), (1, 4), (5, 1), 8, False),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 160, 32), (4, 1), (1, 5), 8, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (160, 128, 32), (2, 2), (5, 4), 8, False),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 160, 32), (2, 2), (4, 5), 8, False),

        # --- 128x256 & 256x128 Peak FLOPs Leaders ---
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 2), (2, 4), 1, True),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 2), (2, 4), 8, True),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 2), (2, 4), 16, True),
        (DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (2, 2), (4, 2), 8, True),
        (DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 2), (4, 8), 8, True),
    ]

    all_candidates = sgemm_candidates + hgemm_candidates

    for dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buffer_lds in all_candidates:
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
        print(f"Added Kernel: {kernel.canonical_name}", flush=True)
        bundle.add(kernel)

    print(f"\nMI300 Bundle contains {len(bundle)} kernels.", flush=True)
    t0 = time.time()
    ret = bundle.compile(output_dir)
    t_compile = time.time() - t0
    assert ret == 0, f"Compilation failed with code {ret}"
    print(f"Compilation finished in {t_compile:.2f}s! Artifacts in {output_dir}", flush=True)
    return bundle


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_mi300_bundle"
    build_mi300_bundle(out_dir)
