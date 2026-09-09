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


def build_tune600_bundle(output_dir: str = "out_mi300_tune600") -> GemmKernelBundle:
    print("=== Building MI300 Tune-600 TFLOPS Candidate Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_tune600_gfx942", target=GFX942)

    # (name, dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buf, vmem_stages, policy)
    candidates = [
        # --- Group 1: Deep Pipelined VMEM on 256x128x64 (8 Waves, MFMA 16x16x16) ---
        ("hgemm_b256x128x64_wg4x2_wt4x4_mfma16x16x16_sgl_vs2_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 16, True, 2, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b256x128x64_wg4x2_wt4x4_mfma16x16x16_sgl_vs3_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 16, True, 3, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b256x128x64_wg4x2_wt4x4_mfma16x16x16_sgl_vs4_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 16, True, 4, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b256x128x64_wg4x2_wt4x4_mfma16x16x16_sgl_vs3_wgm8_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 8, True, 3, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b256x128x64_wg4x2_wt4x4_mfma16x16x16_sgl_vs3_wgm4_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 4, True, 3, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b256x128x64_wg4x2_wt4x4_mfma16x16x16_sgl_vs4_wgm8_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 8, True, 4, SchedulingPolicy.DAG_PIPELINE),

        # --- Group 2: Deep Pipelined VMEM on 128x256x64 (8 Waves, MFMA 16x16x16) ---
        ("hgemm_b128x256x64_wg2x4_wt4x4_mfma16x16x16_sgl_vs2_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 64), (2, 4), (4, 4), 16, True, 2, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x256x64_wg2x4_wt4x4_mfma16x16x16_sgl_vs3_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 64), (2, 4), (4, 4), 16, True, 3, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x256x64_wg2x4_wt4x4_mfma16x16x16_sgl_vs4_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 64), (2, 4), (4, 4), 16, True, 4, SchedulingPolicy.DAG_PIPELINE),

        # --- Group 3: Double Buffer LDS Candidates (Ping-Pong LDS, K=32, vs=2) ---
        ("hgemm_b256x128x32_wg4x2_wt4x4_mfma16x16x16_dbl_vs2_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 32), (4, 2), (4, 4), 16, False, 2, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b256x128x32_wg4x2_wt4x4_mfma16x16x16_dbl_vs2_wgm8_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 32), (4, 2), (4, 4), 8, False, 2, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x256x32_wg2x4_wt4x4_mfma16x16x16_dbl_vs2_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 4), (4, 4), 16, False, 2, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x256x32_wg2x4_wt4x4_mfma16x16x16_dbl_vs2_wgm8_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 256, 32), (2, 4), (4, 4), 8, False, 2, SchedulingPolicy.DAG_PIPELINE),

        # --- Group 4: MFMA 32x32x8 Atom with Deep Pipelining ---
        ("hgemm_b256x128x64_wg4x2_wt2x2_mfma32x32x8_sgl_vs3_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (4, 2), (2, 2), 16, True, 3, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b256x128x64_wg4x2_wt2x2_mfma32x32x8_sgl_vs4_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (256, 128, 64), (4, 2), (2, 2), 16, True, 4, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x256x64_wg2x4_wt2x2_mfma32x32x8_sgl_vs3_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_32x32x8_F16(), (128, 256, 64), (2, 4), (2, 2), 16, True, 3, SchedulingPolicy.DAG_PIPELINE),

        # --- Group 5: High-Occupancy 128x128 Candidates (1024 Workgroups) ---
        ("hgemm_b128x128x64_wg4x2_wt2x4_mfma16x16x16_sgl_vs3_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 64), (4, 2), (2, 4), 16, True, 3, SchedulingPolicy.DAG_PIPELINE),
        ("hgemm_b128x128x64_wg2x2_wt4x4_mfma16x16x16_sgl_vs3_wgm16_dag",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (128, 128, 64), (2, 2), (4, 4), 16, True, 3, SchedulingPolicy.DAG_PIPELINE),
    ]

    for name, dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buf, vs, policy in candidates:
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
            vmem_stages=vs,
            single_buffer_lds=single_buf,
            scheduling_policy=policy,
        )
        diag = kernel.get_diagnostics()
        print(f"Adding {name}: waves={diag['num_waves']}, LDS={diag['lds_usage_bytes']}B, vs={vs}, AGPR={diag['agpr_per_thread']}")
        bundle.add(kernel)

    print(f"\nBundle contains {len(bundle)} tuning kernels.", flush=True)
    t0 = time.time()
    ret = bundle.compile(output_dir)
    t_compile = time.time() - t0
    print(f"Compilation finished in {t_compile:.2f}s with status code {ret}", flush=True)
    return bundle


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_mi300_tune600"
    build_tune600_bundle(out_dir)
