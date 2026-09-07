from __future__ import annotations
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle
from generator.atoms import MFMA_F32_32x32x2_F32
from generator.target_spec import GFX90A
from generator.generator import DataType


def main():
    print("=== Multi-Kernel Bundle & In-Process Batch Benchmark Demo ===", flush=True)

    bundle = GemmKernelBundle("gemm_pack_gfx90a", target=GFX90A)

    candidate_specs = [
        ("rect_256x128_k16_sgl", (256, 128, 16), (2, 2), (4, 2), True),
        ("rect_256x128_k32_sgl", (256, 128, 32), (2, 2), (4, 2), True),
        ("rect_128x256_k32_sgl", (128, 256, 32), (2, 2), (2, 4), True),
        ("square_128x128_k16_sgl", (128, 128, 16), (2, 2), (2, 2), True),
        ("square_128x128_k32_sgl", (128, 128, 32), (2, 2), (2, 2), True),
    ]

    for label, bt, wg, wt, sgl in candidate_specs:
        kernel = GemmKernel(target=GFX90A)  # Name auto-derived as canonical signature
        kernel.set_inputs(
            A=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
            B=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
            C=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
        ).set_tiling(
            block_tile=bt,
            wave_group=wg,
            wave_tiling=wt,
        ).bind_atoms(
            mma=MFMA_F32_32x32x2_F32()
        ).set_schedule(
            vmem_stages=2,
            single_buffer_lds=sgl,
        )
        print(f"Added Kernel: {kernel.canonical_name}", flush=True)
        bundle.add(kernel)

    print(f"\nBundle contains {len(bundle)} kernels.", flush=True)

    # Compile all kernels in ONE clang invocation
    print("\nCompiling bundle into single code object (gemm_pack_gfx90a.co)...", flush=True)
    t0 = time.time()
    ret = bundle.compile("out_bundle_demo")
    t_compile = time.time() - t0
    assert ret == 0, f"Compilation failed with code {ret}"
    print(f"Compilation finished in {t_compile:.2f} seconds!", flush=True)

    # Benchmark all kernels in-process with shared GPU buffers
    print("\nRunning in-process batch benchmark for all kernels (4096 x 4096)...", flush=True)
    t0 = time.time()
    results = bundle.benchmark(4096, 4096, 4096, warmup_runs=5, num_runs=20, output_folder="out_bundle_demo")
    t_bench = time.time() - t0

    print("\nBenchmark Results Summary:")
    print(results)
    print(f"\nTotal batch benchmark time: {t_bench:.2f} seconds!")

    best = results.best()
    if best:
        print(f"\nFastest Kernel: {best['name']} -> {best['latency_ms']} ms ({best['tflops']} TFLOPS)")

    # Test runtime lookup by DSL criteria
    print("\nTesting Runtime Lookup by DSL parameters:")
    matched = bundle.find_kernel(block_tile=(256, 128, 16), depth_k=16, single_buffer_lds=True)
    assert matched is not None
    print(f"Found matching kernel: {matched.canonical_name}")


if __name__ == "__main__":
    main()
