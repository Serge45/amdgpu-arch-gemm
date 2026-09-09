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


def build_push600_bundle(output_dir: str = "out_mi300_push600") -> GemmKernelBundle:
    print("=== Building MI300 Push-600 TFLOPS Exploration Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_push600_gfx942", target=GFX942)

    candidates = [
        # --- Current Champion ---
        ("hgemm_b256x128x64_wg4x2_wt4x4_mfma16x16x16_sgl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 16, True, SchedulingPolicy.DAG_PIPELINE),

        # --- Double Buffer LDS on 256x128 and 128x256 (60 KB LDS) ---
        ("hgemm_b256x128x32_wg4x2_wt4x4_mfma16x16x16_dbl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 32), (4, 2), (4, 4), 16, False, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b256x128x32_wg4x2_wt2x2_mfma32x32x8_dbl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 32), (4, 2), (2, 2), 16, False, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x256x32_wg2x4_wt4x4_mfma16x16x16_dbl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 4), (4, 4), 16, False, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x256x32_wg2x4_wt2x2_mfma32x32x8_dbl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 32), (2, 4), (2, 2), 16, False, SchedulingPolicy.DAG_PIPELINE),

        # --- 128x128 High Occupancy (1024 WGs at 4K = 3.37 WGs/CU) ---
        ("hgemm_b128x128x64_wg4x2_wt2x4_mfma16x16x16_sgl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 64), (4, 2), (2, 4), 16, True, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x128x64_wg2x2_wt4x4_mfma16x16x16_sgl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 64), (2, 2), (4, 4), 16, True, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x128x32_wg2x2_wt4x4_mfma16x16x16_dbl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 32), (2, 2), (4, 4), 16, False, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x128x32_wg4x2_wt2x4_mfma16x16x16_dbl_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 32), (4, 2), (2, 4), 16, False, SchedulingPolicy.DAG_PIPELINE),
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
        diag = kernel.get_diagnostics()
        print(f"Adding {name}: waves={diag['num_waves']}, LDS={diag['lds_usage_bytes']}B, dbl={not single_buffer_lds}, AGPR={diag['agpr_per_thread']}")
        bundle.add(kernel)

    print(f"\nBundle contains {len(bundle)} kernels.", flush=True)
    t0 = time.time()
    ret = bundle.compile(output_dir)
    t_compile = time.time() - t0
    print(f"Compilation finished in {t_compile:.2f}s with status code {ret}", flush=True)
    return bundle


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_mi300_push600"
    build_push600_bundle(out_dir)
