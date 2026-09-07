from __future__ import annotations
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle
from generator.atoms import MFMA_F32_32x32x2_F32
from generator.target_spec import GFX90A
from generator.generator import DataType


def main():
    print("=== Multi-Kernel Bundle with Workgroup Mapping (WGM) Hardware Benchmark ===", flush=True)

    bundle = GemmKernelBundle("gemm_wgm_bundle_gfx90a", target=GFX90A)

    m, n, k = 2048, 2048, 2048

    for wgm in [1, 2, 4, 8]:
        kernel = GemmKernel(target=GFX90A)
        kernel.set_inputs(
            A=Tensor(shape=(m, k), dtype=DataType.FP32),
            B=Tensor(shape=(k, n), dtype=DataType.FP32),
            C=Tensor(shape=(m, n), dtype=DataType.FP32),
        ).set_workgroup_mapping(
            wgm
        ).set_tiling(
            block_tile=(128, 64, 16),
            wave_group=(2, 2),
            wave_tiling=(2, 1),
        ).bind_atoms(
            mma=MFMA_F32_32x32x2_F32()
        ).set_schedule(
            vmem_stages=2,
            single_buffer_lds=True,
        )
        print(f"[WGM={wgm}] Added Kernel: {kernel.canonical_name}", flush=True)
        bundle.add(kernel)

    print(f"\nBundle contains {len(bundle)} kernels.", flush=True)

    print("\nCompiling bundle into single code object (gemm_wgm_bundle_gfx90a.co)...", flush=True)
    t0 = time.time()
    ret = bundle.compile("out_wgm_bundle")
    t_compile = time.time() - t0
    assert ret == 0, f"Compilation failed with code {ret}"
    print(f"Compilation finished in {t_compile:.2f} seconds!", flush=True)

    # Benchmark all kernels in-process with shared GPU buffers
    print(f"\nRunning in-process batch benchmark for all WGM kernels ({m} x {n} x {k})...", flush=True)
    t0 = time.time()
    results = bundle.benchmark(
        m=m, n=n, k=k,
        warmup_runs=5,
        num_runs=20,
        validate=True,
        output_folder="out_wgm_bundle",
    )
    t_bench = time.time() - t0

    print("\nBenchmark & Validation Results Summary:")
    print(results)
    print(f"\nTotal batch benchmark time: {t_bench:.2f} seconds!")


if __name__ == "__main__":
    main()
