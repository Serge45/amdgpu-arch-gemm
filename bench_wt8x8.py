from __future__ import annotations
import os
import sys
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import MFMA_F32_16x16x16_F16
from generator.target_spec import GFX942
from generator.generator import DataType
from generator.scheduler import SchedulingPolicy


def build_and_bench_wt8x8(output_dir: str = "out_wt8x8"):
    print("=== Building MI300X 4-Wave / WT [8, 8] Benchmark Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_wt8x8_gfx942", target=GFX942)

    candidates = [
        # 1. Baseline 8-wave (wt4x4, 256x128x64) - Current 476 TFLOPS Champion
        ("hgemm_256x128x64_8w_wt4x4_base",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 16, True, True, False, False, False, SchedulingPolicy.COLUMN_PIPELINE, 2),

        # 2. 8-wave with ds_read_b128 K-Interleaved + disperse_reads
        ("hgemm_256x128x64_8w_wt4x4_b128_disp",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 16, True, True, True, False, True, SchedulingPolicy.COLUMN_PIPELINE, 2),

        # 3. 4-wave wt8x8, Block Tile 256x256x32, Scalar ds_read_b64 + disperse_reads (vs2)
        ("hgemm_256x256x32_4w_wt8x8_sca_disp_vs2",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 256, 32), (2, 2), (8, 8), 4, True, True, True, False, False, SchedulingPolicy.COLUMN_PIPELINE, 2),

        # 4. 4-wave wt8x8, Block Tile 256x256x32, 128-bit ds_read_b128 + disperse_reads (vs2)
        ("hgemm_256x256x32_4w_wt8x8_b128_disp_vs2",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 256, 32), (2, 2), (8, 8), 4, True, True, True, False, True, SchedulingPolicy.COLUMN_PIPELINE, 2),

        # 5. 4-wave wt8x8, Block Tile 256x256x32, Scalar ds_read_b64 + disperse_reads (vs3 - Triple-Buffered VMEM!)
        ("hgemm_256x256x32_4w_wt8x8_sca_disp_vs3",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 256, 32), (2, 2), (8, 8), 4, True, True, True, False, False, SchedulingPolicy.COLUMN_PIPELINE, 3),

        # 6. 4-wave wt8x8, Block Tile 256x256x32, 128-bit ds_read_b128 + disperse_reads (vs3 - Triple-Buffered VMEM Champion!)
        ("hgemm_256x256x32_4w_wt8x8_b128_disp_vs3",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 256, 32), (2, 2), (8, 8), 4, True, True, True, False, True, SchedulingPolicy.COLUMN_PIPELINE, 3),

        # 7. 4-wave wt8x8, Block Tile 256x256x32, 128-bit ds_read_b128 (nodisp, vs3)
        ("hgemm_256x256x32_4w_wt8x8_b128_nodisp_vs3",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 256, 32), (2, 2), (8, 8), 4, True, True, False, False, True, SchedulingPolicy.COLUMN_PIPELINE, 3),

        # 8. 4-wave wt8x4, Block Tile 256x128x64, 128-bit ds_read_b128 + disperse_reads (vs2)
        ("hgemm_256x128x64_4w_wt8x4_b128_disp_vs2",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (2, 2), (8, 4), 8, True, True, True, False, True, SchedulingPolicy.COLUMN_PIPELINE, 2),
    ]

    for name, dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buf, bar_red, disperse, vec_ds, ds_b128, policy, vs in candidates:
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
            barrier_reduction=bar_red,
            disperse_reads=disperse,
            vector_ds_read=vec_ds,
            ds_read_b128=ds_b128,
            scheduling_policy=policy,
        )
        diag = kernel.get_diagnostics()
        print(f"Adding {name}: waves={diag['num_waves']}, agpr={diag['agpr_per_thread']}, padA={diag['lds_pad_a']} (conf={diag['lds_conflicts_a']}), padB={diag['lds_pad_b']} (conf={diag['lds_conflicts_b']}), LDS={diag['lds_usage_bytes']}B, vec_ds={vec_ds}, b128={ds_b128}, vs={vs}")
        bundle.add(kernel)

    print(f"\nBundle contains {len(bundle)} kernels.", flush=True)
    t0 = time.time()
    ret = bundle.compile(output_dir)
    t_compile = time.time() - t0
    print(f"Compilation finished in {t_compile:.2f}s with status code {ret}", flush=True)
    assert ret == 0, f"Compilation failed with code {ret}"

    # Fast Validation at 1024 x 1024 x 1024
    print("\n" + "="*80, flush=True)
    print("VALIDATING ACCURACY at 1024 x 1024 x 1024", flush=True)
    print("="*80, flush=True)
    results_val = bundle.benchmark(
        m=1024, n=1024, k=1024,
        warmup_runs=1, num_runs=1,
        runner_bin="./build/GeneratorRunner",
        output_folder=output_dir,
        validate=True,
    )
    print("\n--- Validation Results ---")
    print(results_val)

    # Benchmark 4K (4096 x 4096 x 4096)
    print("\n" + "="*80, flush=True)
    print("BENCHMARKING 4096 x 4096 x 4096 (FP16 HGEMM)", flush=True)
    print("="*80, flush=True)
    results_4k = bundle.benchmark(
        m=4096, n=4096, k=4096,
        warmup_runs=10, num_runs=50,
        runner_bin="./build/GeneratorRunner",
        output_folder=output_dir,
        validate=False,
    )
    print("\n--- 4K Results ---")
    print(results_4k)

    # Benchmark 8K (8192 x 8192 x 8192)
    print("\n" + "="*80, flush=True)
    print("BENCHMARKING 8192 x 8192 x 8192 (FP16 HGEMM)", flush=True)
    print("="*80, flush=True)
    results_8k = bundle.benchmark(
        m=8192, n=8192, k=8192,
        warmup_runs=10, num_runs=50,
        runner_bin="./build/GeneratorRunner",
        output_folder=output_dir,
        validate=False,
    )
    print("\n--- 8K Results ---")
    print(results_8k)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_wt8x8"
    build_and_bench_wt8x8(out_dir)
