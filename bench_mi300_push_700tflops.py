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
from generator.scheduler import SchedulingPolicy


def build_push700_bundle(output_dir: str = "out_mi300_push700") -> GemmKernelBundle:
    print("=== Building MI300 4K HGEMM Push-700TFLOPS Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_push700_gfx942", target=GFX942)

    # Format: (name, dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buffer_lds, policy)
    candidates = [
        # 1. Previous Champion Baseline (Round-Robin, 356 TFLOPS)
        ("hgemm_b256x128x64_wt4x2_mfma32x32x8_sgl_wgm16_rr",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (2, 2), (4, 2), 16, True, SchedulingPolicy.ROUNDROBIN),

        # 2. Champion with INTERLEAVED Scheduler (Push towards 500-700 TFLOPS)
        ("hgemm_b256x128x64_wt4x2_mfma32x32x8_sgl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (2, 2), (4, 2), 16, True, SchedulingPolicy.INTERLEAVED),

        # 3. 8-Wave workgroup (512 threads, halved reg pressure: 122 regs) with INTERLEAVED
        ("hgemm_b256x128x64_wg4x2_wt2x2_mfma32x32x8_sgl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (4, 2), (2, 2), 16, True, SchedulingPolicy.INTERLEAVED),

        # 4. Double Buffer LDS (60 KB LDS <= 64 KB, 1 barrier/iter) with INTERLEAVED
        ("hgemm_b256x128x32_wt4x2_mfma32x32x8_dbl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (2, 2), (4, 2), 16, False, SchedulingPolicy.INTERLEAVED),

        # 5. Transposed Tile 128x256x32 Double Buffer with INTERLEAVED
        ("hgemm_b128x256x32_wt2x4_mfma32x32x8_dbl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 2), (2, 4), 16, False, SchedulingPolicy.INTERLEAVED),

        # 6. MFMA 16x16x16 Atom (2x FP16 Rate) Double Buffer with INTERLEAVED
        ("hgemm_b256x128x32_wt8x4_mfma16x16x16_dbl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 32), (2, 2), (8, 4), 16, False, SchedulingPolicy.INTERLEAVED),

        # 7. MFMA 16x16x16 Atom Transposed Tile Double Buffer with INTERLEAVED
        ("hgemm_b128x256x32_wt4x8_mfma16x16x16_dbl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 2), (4, 8), 16, False, SchedulingPolicy.INTERLEAVED),

        # 8. Symmetric 160x160x32 Double Buffer with INTERLEAVED
        ("hgemm_b160x160x32_wt5x5_mfma16x16x16_dbl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (160, 160, 32), (2, 2), (5, 5), 16, False, SchedulingPolicy.INTERLEAVED),
    ]

    for name, dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buffer_lds, policy in candidates:
        kernel = GemmKernel(name=name, target=GFX942)
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
            scheduling_policy=policy,
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
