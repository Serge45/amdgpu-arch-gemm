from __future__ import annotations
import os
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import MFMA_F32_32x32x8_F16, MFMA_F32_16x16x16_F16
from generator.target_spec import GFX90A
from generator.generator import DataType


def build_hgemm_bundle(output_dir: str = "out_hgemm_tn_bundle") -> GemmKernelBundle:
    print("=== Building HGEMM TN Kernel Bundle ===", flush=True)

    bundle = GemmKernelBundle("hgemm_tn_bundle_gfx90a", target=GFX90A)

    # Format: (atom, block_tile, wave_group, wave_tiling, wgm, single_buffer_lds)
    candidates = [
        # === 160x128x64 & 128x160x64 Candidates (208 Workgroups = 2.0 WGs/CU on MI210) ===
        # --- 32x32x8 MFMA ---
        (MFMA_F32_32x32x8_F16(), (160, 128, 64), (1, 4), (5, 1), 1, True),
        (MFMA_F32_32x32x8_F16(), (160, 128, 64), (1, 4), (5, 1), 4, True),
        (MFMA_F32_32x32x8_F16(), (160, 128, 64), (1, 4), (5, 1), 8, True),
        (MFMA_F32_32x32x8_F16(), (128, 160, 64), (4, 1), (1, 5), 1, True),
        (MFMA_F32_32x32x8_F16(), (128, 160, 64), (4, 1), (1, 5), 4, True),
        (MFMA_F32_32x32x8_F16(), (128, 160, 64), (4, 1), (1, 5), 8, True),

        # --- 16x16x16 MFMA (Balanced wave_group=(2, 2)) ---
        (MFMA_F32_16x16x16_F16(), (160, 128, 64), (2, 2), (5, 4), 1, True),
        (MFMA_F32_16x16x16_F16(), (160, 128, 64), (2, 2), (5, 4), 4, True),
        (MFMA_F32_16x16x16_F16(), (160, 128, 64), (2, 2), (5, 4), 8, True),
        (MFMA_F32_16x16x16_F16(), (128, 160, 64), (2, 2), (4, 5), 1, True),
        (MFMA_F32_16x16x16_F16(), (128, 160, 64), (2, 2), (4, 5), 4, True),
        (MFMA_F32_16x16x16_F16(), (128, 160, 64), (2, 2), (4, 5), 8, True),

        # === 160x128x32 & 128x160x32 Candidates (Double Buffer LDS & Single Buffer) ===
        # --- 32x32x8 MFMA ---
        (MFMA_F32_32x32x8_F16(), (160, 128, 32), (1, 4), (5, 1), 1, False),
        (MFMA_F32_32x32x8_F16(), (160, 128, 32), (1, 4), (5, 1), 4, False),
        (MFMA_F32_32x32x8_F16(), (160, 128, 32), (1, 4), (5, 1), 8, False),
        (MFMA_F32_32x32x8_F16(), (128, 160, 32), (4, 1), (1, 5), 1, False),
        (MFMA_F32_32x32x8_F16(), (128, 160, 32), (4, 1), (1, 5), 4, False),
        (MFMA_F32_32x32x8_F16(), (128, 160, 32), (4, 1), (1, 5), 8, False),
        (MFMA_F32_32x32x8_F16(), (160, 128, 32), (1, 4), (5, 1), 4, True),
        (MFMA_F32_32x32x8_F16(), (128, 160, 32), (4, 1), (1, 5), 4, True),

        # --- 16x16x16 MFMA ---
        (MFMA_F32_16x16x16_F16(), (160, 128, 32), (2, 2), (5, 4), 1, False),
        (MFMA_F32_16x16x16_F16(), (160, 128, 32), (2, 2), (5, 4), 4, False),
        (MFMA_F32_16x16x16_F16(), (160, 128, 32), (2, 2), (5, 4), 8, False),
        (MFMA_F32_16x16x16_F16(), (128, 160, 32), (2, 2), (4, 5), 1, False),
        (MFMA_F32_16x16x16_F16(), (128, 160, 32), (2, 2), (4, 5), 4, False),
        (MFMA_F32_16x16x16_F16(), (128, 160, 32), (2, 2), (4, 5), 8, False),

        # === 32x32x8 Baseline Leaders ===
        (MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 2), (2, 4), 1, True),
        (MFMA_F32_32x32x8_F16(), (256, 128, 32), (2, 2), (4, 2), 1, True),
        (MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 2), (2, 4), 4, True),
        (MFMA_F32_32x32x8_F16(), (128, 64, 64), (2, 2), (2, 1), 1, True),
        (MFMA_F32_32x32x8_F16(), (128, 64, 64), (2, 2), (2, 1), 8, True),
        (MFMA_F32_32x32x8_F16(), (128, 64, 64), (2, 2), (2, 1), 1, False),
        (MFMA_F32_32x32x8_F16(), (256, 256, 32), (4, 2), (2, 4), 1, True),
        (MFMA_F32_32x32x8_F16(), (256, 256, 32), (4, 2), (2, 4), 4, True),

        # === New 16x16x16 Candidates (K=16 per MFMA) ===
        # --- K=64 Deep Unroll (num_unrolled_iters = 4, 768 cycles MFMA hides VMEM) ---
        (MFMA_F32_16x16x16_F16(), (128, 128, 64), (2, 2), (4, 4), 1, True),
        (MFMA_F32_16x16x16_F16(), (128, 128, 64), (2, 2), (4, 4), 4, True),
        (MFMA_F32_16x16x16_F16(), (128, 128, 64), (2, 2), (4, 4), 8, True),
        (MFMA_F32_16x16x16_F16(), (128, 64, 64), (2, 2), (4, 2), 1, True),
        (MFMA_F32_16x16x16_F16(), (128, 64, 64), (2, 2), (4, 2), 4, True),
        (MFMA_F32_16x16x16_F16(), (128, 64, 64), (2, 2), (4, 2), 8, True),
        (MFMA_F32_16x16x16_F16(), (128, 64, 64), (2, 2), (4, 2), 1, False),
        (MFMA_F32_16x16x16_F16(), (64, 128, 64), (2, 2), (2, 4), 1, True),
        (MFMA_F32_16x16x16_F16(), (64, 128, 64), (2, 2), (2, 4), 4, True),
        (MFMA_F32_16x16x16_F16(), (64, 128, 64), (2, 2), (2, 4), 8, True),
        (MFMA_F32_16x16x16_F16(), (64, 128, 64), (2, 2), (2, 4), 1, False),

        # --- K=32 Double Buffer & Single Buffer (num_unrolled_iters = 2) ---
        (MFMA_F32_16x16x16_F16(), (128, 128, 32), (2, 2), (4, 4), 1, False),
        (MFMA_F32_16x16x16_F16(), (128, 128, 32), (2, 2), (4, 4), 4, False),
        (MFMA_F32_16x16x16_F16(), (128, 128, 32), (2, 2), (4, 4), 8, False),
        (MFMA_F32_16x16x16_F16(), (128, 128, 32), (2, 2), (4, 4), 1, True),
        (MFMA_F32_16x16x16_F16(), (128, 128, 32), (2, 2), (4, 4), 4, True),
        (MFMA_F32_16x16x16_F16(), (128, 64, 32), (2, 2), (4, 2), 1, False),
        (MFMA_F32_16x16x16_F16(), (128, 64, 32), (2, 2), (4, 2), 4, False),
        (MFMA_F32_16x16x16_F16(), (128, 64, 32), (2, 2), (4, 2), 8, False),
        (MFMA_F32_16x16x16_F16(), (64, 128, 32), (2, 2), (2, 4), 1, False),
        (MFMA_F32_16x16x16_F16(), (64, 128, 32), (2, 2), (2, 4), 4, False),
        (MFMA_F32_16x16x16_F16(), (64, 128, 32), (2, 2), (2, 4), 8, False),
        (MFMA_F32_16x16x16_F16(), (64, 64, 32), (2, 2), (2, 2), 1, False),

        # --- 128x256 & 256x128 with K=32 (num_unrolled_iters = 2, 512 cycles MFMA) ---
        (MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 2), (4, 8), 1, True),
        (MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 2), (4, 8), 4, True),
        (MFMA_F32_16x16x16_F16(), (256, 128, 32), (2, 2), (8, 4), 1, True),
        (MFMA_F32_16x16x16_F16(), (256, 128, 32), (2, 2), (8, 4), 4, True),
    ]

    for mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buffer_lds in candidates:
        kernel = GemmKernel(target=GFX90A)
        kernel.set_inputs(
            A=Tensor(shape=(4096, 4096), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
            B=Tensor(shape=(4096, 4096), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
            C=Tensor(shape=(4096, 4096), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        ).set_transposes(
            trans_a=True, trans_b=False
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

    print(f"\nBundle contains {len(bundle)} kernels.", flush=True)
    t0 = time.time()
    ret = bundle.compile(output_dir)
    t_compile = time.time() - t0
    assert ret == 0, f"Compilation failed with code {ret}"
    print(f"Compilation finished in {t_compile:.2f}s! Artifacts in {output_dir}", flush=True)
    return bundle


if __name__ == "__main__":
    build_hgemm_bundle()
