import sys
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import MFMA_F32_16x16x16_F16
from generator.target_spec import GFX942
from generator.generator import DataType
from generator.scheduler import SchedulingPolicy


def run_b128_benchmark(output_dir: str = "out_b128"):
    print("=== Building MI300X ds_read_b128 Benchmark Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_b128_gfx942", target=GFX942)

    configs = [
        # 1. 256x192x64 with ds_read_b64 (Current Champion: 417 TFLOPS)
        ("hgemm_256x192x64_wt4x12_b64",
         (256, 192, 64), (4, 1), (4, 12), True, False, False),

        # 2. 256x160x64 with ds_read_b64 (410 TFLOPS)
        ("hgemm_256x160x64_wt4x10_b64",
         (256, 160, 64), (4, 1), (4, 10), True, False, False),

        # 3. 256x160x64 with ds_read_b128 (NEW: 252 VGPRs, 128-bit vector read!)
        ("hgemm_256x160x64_wt4x10_b128",
         (256, 160, 64), (4, 1), (4, 10), True, True, True),

        # 4. 256x128x64 with ds_read_b128 (NEW: 216 VGPRs, 128-bit vector read!)
        ("hgemm_256x128x64_wt4x8_b128",
         (256, 128, 64), (4, 1), (4, 8), True, True, True),

        # 5. 256x256x32 with ds_read_b128 (wg2x2, wt8x8, 398 TFLOPS)
        ("hgemm_256x256x32_wt8x8_b128",
         (256, 256, 32), (2, 2), (8, 8), True, True, True),
    ]

    for name, block_tile, wave_group, wave_tiling, disp, vec, b128 in configs:
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
            disperse_reads=disp,
            vector_ds_read=vec,
            ds_read_b128=b128,
            single_lds_base=True,
            scheduling_policy=SchedulingPolicy.COLUMN_PIPELINE,
        )
        diag = kernel.get_diagnostics()
        print(f"Adding {name} (b128={b128}): waves={diag['num_waves']}, LDS={diag['lds_usage_bytes']}B")
        bundle.add(kernel)

    print(f"\nBundle contains {len(bundle)} kernels. Compiling...", flush=True)
    t0 = time.time()
    ret = bundle.compile(output_dir)
    t_compile = time.time() - t0
    print(f"Compilation finished in {t_compile:.2f}s with status code {ret}", flush=True)
    assert ret == 0, f"Compilation failed with code {ret}"

    for size in [1024, 4096, 8192]:
        print("\n" + "=" * 80, flush=True)
        print(f"BENCHMARKING {size} x {size} x {size} (FP16 HGEMM)", flush=True)
        print("=" * 80, flush=True)
        res = bundle.benchmark(
            m=size, n=size, k=size,
            warmup_runs=10, num_runs=50,
            runner_bin="./build/GeneratorRunner",
            output_folder=output_dir,
            validate=(size == 1024),
        )
        print(f"\n--- {size} Results ---")
        print(res)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_b128"
    run_b128_benchmark(out_dir)
