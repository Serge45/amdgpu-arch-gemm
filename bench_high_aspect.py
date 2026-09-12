import sys
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import MFMA_F32_16x16x16_F16
from generator.target_spec import GFX942
from generator.generator import DataType
from generator.scheduler import SchedulingPolicy


def build_and_bench_high_aspect(output_dir: str = "out_high_aspect"):
    print("=== Building MI300X High-Aspect-Ratio Benchmark Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_high_aspect_gfx942", target=GFX942)

    configs = [
        # 1. Previous 4-wave champion: wt8x8, 256x256x32 (wg2x2) -> 256 WGs at 4K (48 CUs idle)
        ("hgemm_256x256x32_4w_wt8x8_b128_ref",
         (256, 256, 32), (2, 2), (8, 8), False, True, True),

        # 2. High-aspect-ratio 4-wave: wt4x10, 256x160x64 (wg4x1) -> 416 WGs at 4K (100% CU saturation)
        ("hgemm_256x160x64_4w_wt4x10_single_lds",
         (256, 160, 64), (4, 1), (4, 10), True, False, False),

        # 3. High-aspect-ratio 4-wave: wt4x12, 256x192x64 (wg4x1) -> 352 WGs at 4K (100% CU saturation)
        ("hgemm_256x192x64_4w_wt4x12_single_lds",
         (256, 192, 64), (4, 1), (4, 12), True, False, False),

        # 4. High-aspect-ratio 4-wave: wt4x15, 256x240x32 (wg4x1) -> 288 WGs at 4K
        ("hgemm_256x240x32_4w_wt4x15_single_lds",
         (256, 240, 32), (4, 1), (4, 15), True, False, False),

        # 5. 4-wave wt4x16, 256x256x32 (wg4x1)
        ("hgemm_256x256x32_4w_wt4x16_single_lds",
         (256, 256, 32), (4, 1), (4, 16), True, False, False),
    ]

    for name, block_tile, wave_group, wave_tiling, single_lds, vec_ds, ds_b128 in configs:
        kernel = GemmKernel(name=name, target=GFX942)
        kernel.set_inputs(
            A=Tensor(shape=(4096, 4096), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
            B=Tensor(shape=(4096, 4096), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
            C=Tensor(shape=(4096, 4096), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        ).set_transposes(
            trans_a=True, trans_b=False
        ).set_workgroup_mapping(
            4
        ).set_tiling(
            block_tile=block_tile,
            wave_group=wave_group,
            wave_tiling=wave_tiling,
        ).bind_atoms(
            mma=MFMA_F32_16x16x16_F16()
        ).set_schedule(
            vmem_stages=2,
            single_buffer_lds=True,
            barrier_reduction=True,
            disperse_reads=True,
            vector_ds_read=vec_ds,
            ds_read_b128=ds_b128,
            single_lds_base=single_lds,
            scheduling_policy=SchedulingPolicy.COLUMN_PIPELINE,
        )
        diag = kernel.get_diagnostics()
        print(f"Adding {name}: waves={diag['num_waves']}, padA={diag['lds_pad_a']}, padB={diag['lds_pad_b']}, LDS={diag['lds_usage_bytes']}B")
        bundle.add(kernel)

    print(f"\nBundle contains {len(bundle)} kernels. Compiling...", flush=True)
    t0 = time.time()
    ret = bundle.compile(output_dir)
    t_compile = time.time() - t0
    print(f"Compilation finished in {t_compile:.2f}s with status code {ret}", flush=True)
    assert ret == 0, f"Compilation failed with code {ret}"

    # Validation at 1024 x 1024 x 1024
    print("\n" + "=" * 80, flush=True)
    print("BENCHMARKING 1024 x 1024 x 1024", flush=True)
    print("=" * 80, flush=True)
    results_val = bundle.benchmark(
        m=1024, n=1024, k=1024,
        warmup_runs=5, num_runs=20,
        runner_bin="./build/GeneratorRunner",
        output_folder=output_dir,
        validate=False,
    )
    print("\n--- 1K Results ---")
    print(results_val)

    # Benchmark 4K (4096 x 4096 x 4096)
    print("\n" + "=" * 80, flush=True)
    print("BENCHMARKING 4096 x 4096 x 4096 (FP16 HGEMM)", flush=True)
    print("=" * 80, flush=True)
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
    print("\n" + "=" * 80, flush=True)
    print("BENCHMARKING 8192 x 8192 x 8192 (FP16 HGEMM)", flush=True)
    print("=" * 80, flush=True)
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
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_high_aspect"
    build_and_bench_high_aspect(out_dir)
