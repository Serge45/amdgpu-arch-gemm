from __future__ import annotations
import os
import sys
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import MFMA_F32_16x16x16_F16
from generator.target_spec import GFX942
from generator.generator import DataType
from generator.scheduler import SchedulingPolicy


def build_and_bench_next_step_bundle(output_dir: str = "out_next_step"):
    print("=== Building MI300 Next-Step Optimization Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_next_step_gfx942", target=GFX942)

    candidates = [
        # 1. Barrier-Reduced COLUMN_PIPELINE: 256x128x64, 8 waves, wt4x4, mfma16x16x16, single buffer LDS (1 barrier per iter)
        ("hgemm_256x128x64_col_nobar1",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 16, True, True, False, SchedulingPolicy.COLUMN_PIPELINE),

        # 2. Barrier-Reduced + Dispersed Reads COLUMN_PIPELINE: cuts post-barrier LDS stampede!
        ("hgemm_256x128x64_col_disperse",
         DataType.FP16, True, False, MFMA_F32_16x16x16_F16(), (256, 128, 64), (4, 2), (4, 4), 16, True, True, True, SchedulingPolicy.COLUMN_PIPELINE),
    ]

    for name, dtype, trans_a, trans_b, mma_atom, block_tile, wave_group, wave_tiling, wgm, single_buf, bar_red, disperse, policy in candidates:
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
            single_buffer_lds=single_buf,
            barrier_reduction=bar_red,
            disperse_reads=disperse,
            scheduling_policy=policy,
        )
        diag = kernel.get_diagnostics()
        print(f"Adding {name}: waves={diag['num_waves']}, padA={diag['lds_pad_a']} (conf={diag['lds_conflicts_a']}), padB={diag['lds_pad_b']} (conf={diag['lds_conflicts_b']}), LDS={diag['lds_usage_bytes']}B, bar_red={bar_red}")
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
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_next_step"
    build_and_bench_next_step_bundle(out_dir)
