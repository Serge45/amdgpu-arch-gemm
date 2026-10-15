import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generator.dsl import GemmKernel, Tensor, GemmKernelBundle, LayoutType
from generator.atoms import MFMA_F32_16x16x16_F16
from generator.target_spec import GFX942
from generator.generator import DataType
from generator.scheduler import SchedulingPolicy


def run_dtva_benchmark(output_dir: str = "out_dtva_v4"):
    print("=== Building MI300X DTVA v4 (Waitcnt Optimized) Benchmark Bundle ===", flush=True)
    bundle = GemmKernelBundle("mi300_dtva_v4_gfx942", target=GFX942)

    configs = [
        # 1. Baseline champion: 256x192x64 with ds_read_b64 (417 TFLOPS)
        ("hgemm_256x192x64_wt4x12_b64_base",
         (256, 192, 64), (4, 1), (4, 12), False, True, 2, False, False, None),

        # 2. wt4x14 Single-buffer (pad_b=None auto 8)
        ("hgemm_256x224x64_wt4x14_dtva_sgl_p8",
         (256, 224, 64), (4, 1), (4, 14), True, True, 2, True, True, None),

        # 3. wt4x14 Single-buffer (pad_b=16 stride 2560)
        ("hgemm_256x224x64_wt4x14_dtva_sgl_p16",
         (256, 224, 64), (4, 1), (4, 14), True, True, 2, True, True, 16),

        # 5. wt8x7 Single-buffer (pad_b=None auto 8)
        ("hgemm_512x112x64_wt8x7_dtva_sgl_p8",
         (512, 112, 64), (4, 1), (8, 7), True, True, 2, True, True, None),

        # 6. wt8x7 Single-buffer (pad_b=16 stride 2560 matches hipBLASLt)
        ("hgemm_512x112x64_wt8x7_dtva_sgl_p16",
         (512, 112, 64), (4, 1), (8, 7), True, True, 2, True, True, 16),

        # 7. wt8x7 Double-buffer (pad_b=16 stride 2560)
        ("hgemm_512x112x64_wt8x7_dtva_dbl_p16",
         (512, 112, 64), (4, 1), (8, 7), True, False, 1, True, True, 16),
    ]

    for name, block_tile, wave_group, wave_tiling, dtva, sgl_lds, stages, vec, b128, pad_b in configs:
        wgm_val = 8 if block_tile[0] == 512 else 16
        kernel = GemmKernel(name=name, target=GFX942)
        kernel.set_inputs(
            A=Tensor(shape=(4096, 4096), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
            B=Tensor(shape=(4096, 4096), dtype=DataType.FP16, layout=LayoutType.COL_MAJOR),
            C=Tensor(shape=(4096, 4096), dtype=DataType.FP32, layout=LayoutType.COL_MAJOR),
        ).set_transposes(
            trans_a=True, trans_b=False
        ).set_workgroup_mapping(
            wgm_val, wgm_xcc=8
        ).set_tiling(
            block_tile=block_tile,
            wave_group=wave_group,
            wave_tiling=wave_tiling,
        ).bind_atoms(
            mma=MFMA_F32_16x16x16_F16()
        ).set_schedule(
            vmem_stages=stages,
            single_buffer_lds=sgl_lds,
            barrier_reduction=True,
            disperse_reads=(not dtva),
            vector_ds_read=vec,
            ds_read_b128=b128,
            single_lds_base=True,
            direct_to_vgpr_a=dtva,
            lds_pad_b=pad_b,
            scheduling_policy=SchedulingPolicy.COLUMN_PIPELINE,
        )
        diag = kernel.get_diagnostics()
        print(f"Adding {name}: dtva={dtva}, sgl_lds={sgl_lds}, LDS={diag['lds_usage_bytes']}B, AGPR={diag['agpr_per_thread']}")
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
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out_dtva_v3"
    run_dtva_benchmark(out_dir)
