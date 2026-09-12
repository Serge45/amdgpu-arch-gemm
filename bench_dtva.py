import sys
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import MFMA_F32_16x16x16_F16
from generator.target_spec import GFX942
from generator.generator import DataType
from generator.scheduler import SchedulingPolicy


def run_dtva_benchmark(output_dir: str = "out_dtva"):
    print("=== Building MI300X Direct-To-VGPR A (DTVA) Benchmark Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_dtva_gfx942", target=GFX942)

    configs = [
        # 1. Baseline champion: 256x192x64 with ds_read_b64 (417 TFLOPS)
        ("hgemm_256x192x64_wt4x12_b64_base",
         (256, 192, 64), (4, 1), (4, 12), False, False, False),

        # 2. DTVA: 256x160x64 wt4x10 (176 Arch VGPRs, 22.5 KB LDS)
        ("hgemm_256x160x64_wt4x10_dtva",
         (256, 160, 64), (4, 1), (4, 10), True, True, True),

        # 3. DTVA: 256x192x64 wt4x12 (204 Arch VGPRs, 27 KB LDS)
        ("hgemm_256x192x64_wt4x12_dtva",
         (256, 192, 64), (4, 1), (4, 12), True, True, True),

        # 4. DTVA: 256x224x64 wt4x14 (228 Arch VGPRs, 31.5 KB LDS)
        ("hgemm_256x224x64_wt4x14_dtva",
         (256, 224, 64), (4, 1), (4, 14), True, True, True),

        # 5. DTVA: 256x256x64 wt4x16 (256 Arch VGPRs, 36 KB LDS)
        ("hgemm_256x256x64_wt4x16_dtva",
         (256, 256, 64), (4, 1), (4, 16), True, True, True),
    ]

    for name, block_tile, wave_group, wave_tiling, dtva, vec, b128 in configs:
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
            vector_ds_read=vec,
            ds_read_b128=b128,
            single_lds_base=True,
            direct_to_vgpr_a=dtva,
            scheduling_policy=SchedulingPolicy.COLUMN_PIPELINE,
        )
        diag = kernel.get_diagnostics()
        print(f"Adding {name}: dtva={dtva}, b128={b128}, LDS={diag['lds_usage_bytes']}B")
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
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_dtva"
    run_dtva_benchmark(out_dir)
