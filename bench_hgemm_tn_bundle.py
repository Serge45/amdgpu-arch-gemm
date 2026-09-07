from __future__ import annotations
import os
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import MFMA_F32_32x32x8_F16
from generator.target_spec import GFX90A
from generator.generator import DataType


def build_hgemm_bundle(output_dir: str = "out_hgemm_tn_bundle") -> GemmKernelBundle:
    print("=== Building HGEMM TN Kernel Bundle ===", flush=True)

    bundle = GemmKernelBundle("hgemm_tn_bundle_gfx90a", target=GFX90A)

    # Fine-tuned candidate configs for MI210 2K & 4K optimization
    candidates = [
        # Top 4K candidates (depth_k = 32)
        ((256, 128, 32), (2, 2), (4, 2), 1),
        ((256, 128, 32), (2, 2), (4, 2), 2),
        ((256, 128, 32), (2, 2), (4, 2), 4),
        ((256, 128, 32), (2, 2), (4, 2), 8),
        ((128, 256, 32), (2, 2), (2, 4), 1),
        ((128, 256, 32), (2, 2), (2, 4), 2),
        ((128, 256, 32), (2, 2), (2, 4), 4),
        ((128, 256, 32), (2, 2), (2, 4), 8),

        # Top 2K & high-throughput candidates (depth_k = 64)
        # b128x64x64 (512 workgroups for 2K, 4.92 WGs/CU)
        ((128, 64, 64), (2, 2), (2, 1), 1),
        ((128, 64, 64), (2, 2), (2, 1), 2),
        ((128, 64, 64), (2, 2), (2, 1), 4),
        ((128, 64, 64), (2, 2), (2, 1), 8),
        ((128, 64, 64), (2, 2), (2, 1), 16),

        # b64x128x64 (512 workgroups for 2K)
        ((64, 128, 64), (2, 2), (1, 2), 1),
        ((64, 128, 64), (2, 2), (1, 2), 2),
        ((64, 128, 64), (2, 2), (1, 2), 4),
        ((64, 128, 64), (2, 2), (1, 2), 8),
        ((64, 128, 64), (2, 2), (1, 2), 16),

        # b128x128x64 (256 workgroups for 2K)
        ((128, 128, 64), (2, 2), (2, 2), 1),
        ((128, 128, 64), (2, 2), (2, 2), 2),
        ((128, 128, 64), (2, 2), (2, 2), 4),
        ((128, 128, 64), (2, 2), (2, 2), 8),
        ((128, 128, 64), (2, 2), (2, 2), 16),

        # Alternative wave groupings
        ((128, 64, 64), (4, 1), (1, 2), 4),
        ((128, 64, 64), (4, 1), (1, 2), 8),
        ((64, 128, 64), (1, 4), (2, 1), 4),
        ((64, 128, 64), (1, 4), (2, 1), 8),

        # High occupancy b64x64x64
        ((64, 64, 64), (2, 2), (1, 1), 1),
        ((64, 64, 64), (2, 2), (1, 1), 4),
        ((64, 64, 64), (2, 2), (1, 1), 8),
    ]

    for block_tile, wave_group, wave_tiling, wgm in candidates:
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
            mma=MFMA_F32_32x32x8_F16()
        ).set_schedule(
            vmem_stages=2,
            single_buffer_lds=True,
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
