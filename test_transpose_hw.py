from __future__ import annotations
import time
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle
from generator.atoms import MFMA_F32_32x32x2_F32
from generator.target_spec import GFX90A
from generator.generator import DataType


def main():
    print("=== Transpose Modes (NN, NT, TN, TT) Hardware Verification ===", flush=True)

    bundle = GemmKernelBundle("gemm_transposes_gfx90a", target=GFX90A)

    # Test all 4 combinations
    modes = [
        ("NN", False, False),
        ("NT", False, True),
        ("TN", True, False),
        ("TT", True, True),
    ]

    m, n, k = 2048, 2048, 2048

    for name, trans_a, trans_b in modes:
        kernel = GemmKernel(target=GFX90A)
        kernel.set_inputs(
            A=Tensor(shape=(k, m) if trans_a else (m, k), dtype=DataType.FP32),
            B=Tensor(shape=(n, k) if trans_b else (k, n), dtype=DataType.FP32),
            C=Tensor(shape=(m, n), dtype=DataType.FP32),
        ).set_transposes(
            trans_a, trans_b
        ).set_tiling(
            block_tile=(256, 128, 16),
            wave_group=(2, 2),
            wave_tiling=(4, 2),
        ).bind_atoms(
            mma=MFMA_F32_32x32x2_F32()
        ).set_schedule(
            vmem_stages=2,
            single_buffer_lds=True,
        )
        print(f"[{name}] Added Kernel: {kernel.canonical_name}", flush=True)
        bundle.add(kernel)

    print(f"\nBundle contains {len(bundle)} kernels.", flush=True)

    print("\nCompiling bundle into single code object (gemm_transposes_gfx90a.co)...", flush=True)
    t0 = time.time()
    ret = bundle.compile("out_transpose_test")
    t_compile = time.time() - t0
    assert ret == 0, f"Compilation failed with code {ret}"
    print(f"Compilation finished in {t_compile:.2f} seconds!", flush=True)

    # Benchmark and validate against CPU reference (validate=True)
    print(f"\nRunning in-process batch benchmark with accuracy validation ({m} x {n} x {k})...", flush=True)
    t0 = time.time()
    results = bundle.benchmark(
        m=m, n=n, k=k,
        warmup_runs=3,
        num_runs=10,
        validate=True,
        output_folder="out_transpose_test",
    )
    t_bench = time.time() - t0

    print("\nBenchmark & Validation Results Summary:")
    print(results)
    print(f"\nTotal execution time: {t_bench:.2f} seconds!")


if __name__ == "__main__":
    main()
