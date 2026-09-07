from __future__ import annotations
import os
import subprocess
from generator.dsl import GemmKernel, Tensor, LayoutType
from generator.atoms import MFMA_F32_32x32x2_F32
from generator.target_spec import GFX90A
from generator.generator import DataType


def run_benchmark():
    configs = [
        ("square_128x128_k16", (128, 128, 16), (2, 2), (2, 2), 16),
        ("rect_256x64_k16", (256, 64, 16), (2, 2), (4, 1), 16),
        ("rect_256x128_k16", (256, 128, 16), (2, 2), (4, 2), 16),
        ("rect_256x32_k16", (256, 32, 16), (4, 1), (2, 1), 16),
        ("k32_128x64_k32", (128, 64, 32), (2, 2), (2, 1), 32),
        ("k32_64x128_k32", (64, 128, 32), (2, 2), (1, 2), 32),
    ]

    out_folder = "out_bench"
    os.makedirs(out_folder, exist_ok=True)

    matrix_size = 4096

    header = f"{'Config Name':<22} | {'Block Tile':<14} | {'LDS (B)':<8} | {'AGPR':<5} | {'Latency (ms)':<12} | {'TFLOPS':<8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for name, bt, wg, wt, dk in configs:
        kernel = GemmKernel(name="generated_gemm", target=GFX90A)
        kernel.set_inputs(
            A=Tensor(shape=(matrix_size, matrix_size), dtype=DataType.FP32),
            B=Tensor(shape=(matrix_size, matrix_size), dtype=DataType.FP32),
            C=Tensor(shape=(matrix_size, matrix_size), dtype=DataType.FP32),
        ).set_tiling(
            block_tile=bt,
            wave_group=wg,
            wave_tiling=wt,
        ).bind_atoms(
            mma=MFMA_F32_32x32x2_F32()
        ).set_schedule(
            vmem_stages=2,
        )
        kernel.depth_k = dk

        diag = kernel.get_diagnostics()
        lds_b = diag["lds_usage_bytes"]
        agpr = diag["agpr_per_thread"]

        ret = kernel.compile(output_folder=out_folder)
        if ret != 0:
            print(f"{name:<22} | {str(bt):<14} | {lds_b:<8} | {agpr:<5} | {'FAILED (compile)':<12} | {'-':<8}", flush=True)
            continue

        co_path = os.path.join(out_folder, "generated_gemm.co")
        toml_path = os.path.join(out_folder, "generated_gemm.toml")

        # Direct GPU benchmark: 5 warmup runs, 20 benchmark runs
        bench_cmd = f"build/GeneratorRunner {co_path} {toml_path} {matrix_size} {matrix_size} {matrix_size} 5 20 0"
        bench_res = subprocess.run(bench_cmd, shell=True, capture_output=True, text=True)

        dur_ms = "N/A"
        tflops = "N/A"
        for line in bench_res.stdout.splitlines():
            if "ASM gemm:" in line:
                dur_ms = line.split(":")[-1].replace("ms", "").strip()
            elif "Gflops:" in line:
                gflops_val = float(line.split(":")[-1].strip())
                tflops = f"{gflops_val / 1000.0:.2f}"

        print(f"{name:<22} | {str(bt):<14} | {lds_b:<8} | {agpr:<5} | {dur_ms:<12} | {tflops:<8}", flush=True)


if __name__ == "__main__":
    run_benchmark()
