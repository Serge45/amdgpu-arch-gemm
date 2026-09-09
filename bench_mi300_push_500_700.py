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


def build_push500_700_bundle(output_dir: str = "out_mi300_push500_700") -> GemmKernelBundle:
    print("=== Building MI300 4K HGEMM Push-500-700 TFLOPS Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_push500_700_gfx942", target=GFX942)

    # Format: (name, dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buffer_lds, policy)
    candidates = [
        # --- Group A: 8-Wave Workgroup Champions (32x32x8 MFMA, low register pressure) ---
        ("hgemm_b256x128x64_wg4x2_wt2x2_mfma32x32x8_sgl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (4, 2), (2, 2), 16, True, SchedulingPolicy.INTERLEAVED),
        ("hgemm_b256x128x64_wg4x2_wt2x2_mfma32x32x8_sgl_wgm8_ilv",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (4, 2), (2, 2), 8, True, SchedulingPolicy.INTERLEAVED),
        ("hgemm_b256x128x64_wg4x2_wt2x2_mfma32x32x8_sgl_wgm4_ilv",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (4, 2), (2, 2), 4, True, SchedulingPolicy.INTERLEAVED),

        # Transposed 128x256x64 (wg2x4)
        ("hgemm_b128x256x64_wg2x4_wt2x2_mfma32x32x8_sgl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 64), (2, 4), (2, 2), 16, True, SchedulingPolicy.INTERLEAVED),
        ("hgemm_b128x256x64_wg2x4_wt2x2_mfma32x32x8_sgl_wgm8_ilv",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 64), (2, 4), (2, 2), 8, True, SchedulingPolicy.INTERLEAVED),

        # --- Group B: 16x16x16 MFMA Atom (2x FP16 Issue Rate, 8-wave workgroup) ---
        ("hgemm_b256x128x64_wg4x2_wt4x4_mfma16x16x16_sgl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 16, True, SchedulingPolicy.INTERLEAVED),
        ("hgemm_b256x128x64_wg4x2_wt4x4_mfma16x16x16_sgl_wgm8_ilv",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 8, True, SchedulingPolicy.INTERLEAVED),
        ("hgemm_b128x256x64_wg2x4_wt4x4_mfma16x16x16_sgl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 64), (2, 4), (4, 4), 16, True, SchedulingPolicy.INTERLEAVED),
        ("hgemm_b128x256x64_wg2x4_wt4x4_mfma16x16x16_sgl_wgm8_ilv",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 64), (2, 4), (4, 4), 8, True, SchedulingPolicy.INTERLEAVED),

        # --- Group C: High-Occupancy 128x128 Tiles (1024 Workgroups = 3.37 WGs/CU) ---
        ("hgemm_b128x128x64_wg2x2_wt2x2_mfma32x32x8_sgl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 128, 64), (2, 2), (2, 2), 16, True, SchedulingPolicy.INTERLEAVED),
        ("hgemm_b128x128x64_wg2x2_wt2x2_mfma32x32x8_sgl_wgm8_ilv",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 128, 64), (2, 2), (2, 2), 8, True, SchedulingPolicy.INTERLEAVED),
        ("hgemm_b128x128x64_wg2x2_wt4x4_mfma16x16x16_sgl_wgm16_ilv",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 64), (2, 2), (4, 4), 16, True, SchedulingPolicy.INTERLEAVED),
        ("hgemm_b128x128x64_wg2x2_wt4x4_mfma16x16x16_sgl_wgm8_ilv",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 64), (2, 2), (4, 4), 8, True, SchedulingPolicy.INTERLEAVED),

        # --- Group D: DAG Pipeline Policy on Top Tiles ---
        ("hgemm_b256x128x64_wg4x2_wt2x2_mfma32x32x8_sgl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (4, 2), (2, 2), 16, True, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x256x64_wg2x4_wt2x2_mfma32x32x8_sgl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 64), (2, 4), (2, 2), 16, True, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b256x128x64_wg4x2_wt4x4_mfma16x16x16_sgl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 16, True, SchedulingPolicy.DAG_PIPELINE),
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

    print(f"\nBundle contains {len(bundle)} elite push kernels.", flush=True)
    t0 = time.time()
    ret = bundle.compile(output_dir)
    t_compile = time.time() - t0
    print(f"Compilation finished in {t_compile:.2f}s with status code {ret}", flush=True)
    return bundle


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_mi300_push500_700"
    build_push500_700_bundle(out_dir)
