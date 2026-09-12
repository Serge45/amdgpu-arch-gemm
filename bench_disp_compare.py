import sys
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import MFMA_F32_16x16x16_F16
from generator.target_spec import GFX942
from generator.generator import DataType
from generator.scheduler import SchedulingPolicy


def run_disp_benchmark(output_dir: str = "out_disp_compare"):
    print("=== Building MI300X Disperse Reads Comparison Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_disp_compare_gfx942", target=GFX942)

    configs = [
        # 1. 256x192x64 with disperse_reads=True (Current Champion: 417 TFLOPS)
        ("hgemm_256x192x64_wt4x12_disp_true",
         (256, 192, 64), (4, 1), (4, 12), True),

        # 2. 256x192x64 with disperse_reads=False (New: Zero-stall prefetch)
        ("hgemm_256x192x64_wt4x12_disp_false",
         (256, 192, 64), (4, 1), (4, 12), False),

        # 3. 256x160x64 with disperse_reads=True
        ("hgemm_256x160x64_wt4x10_disp_true",
         (256, 160, 64), (4, 1), (4, 10), True),

        # 4. 256x160x64 with disperse_reads=False
        ("hgemm_256x160x64_wt4x10_disp_false",
         (256, 160, 64), (4, 1), (4, 10), False),
    ]

    for name, block_tile, wave_group, wave_tiling, disp in configs:
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
            vector_ds_read=False,
            ds_read_b128=False,
            single_lds_base=True,
            scheduling_policy=SchedulingPolicy.COLUMN_PIPELINE,
        )
        diag = kernel.get_diagnostics()
        print(f"Adding {name} (disp={disp}): waves={diag['num_waves']}, LDS={diag['lds_usage_bytes']}B")
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
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_disp_compare"
    run_disp_benchmark(out_dir)
