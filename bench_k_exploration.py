from __future__ import annotations
import os
import subprocess
from generator.dsl import GemmKernel, Tensor
from generator.atoms import MFMA_F32_32x32x2_F32
from generator.target_spec import GFX90A
from generator.generator import DataType


def run_benchmark():
    # List of configs to compare:
    # (name, block_tile, wave_group, wave_tiling, depth_k, single_buffer_lds)
    configs = [
        # Baselines (Double-buffer, K=16)
        ("rect_256x128_k16_dbl", (256, 128, 16), (2, 2), (4, 2), 16, False),
        ("rect_128x256_k16_dbl", (128, 256, 16), (2, 2), (2, 4), 16, False),
        ("square_128x128_k16_dbl", (128, 128, 16), (2, 2), (2, 2), 16, False),

        # Direct K=16 Single-buffer comparison
        ("rect_256x128_k16_sgl", (256, 128, 16), (2, 2), (4, 2), 16, True),
        ("square_128x128_k16_sgl", (128, 128, 16), (2, 2), (2, 2), 16, True),
        ("square_256x256_k16_sgl", (256, 256, 16), (2, 2), (4, 4), 16, True),

        # K=32 Single-buffer (Testing hypothesis: larger depth_k amortizing barriers & increasing reuse)
        ("rect_256x128_k32_sgl", (256, 128, 32), (2, 2), (4, 2), 32, True),
        ("rect_128x256_k32_sgl", (128, 256, 32), (2, 2), (2, 4), 32, True),
        ("square_128x128_k32_sgl", (128, 128, 32), (2, 2), (2, 2), 32, True),
    ]

    out_folder = "out_bench_k"
    os.makedirs(out_folder, exist_ok=True)

    header = f"{'Config Name':<24} | {'Block Tile':<14} | {'Buffering':<8} | {'LDS (B)':<8} | {'VGPR':<5} | {'AGPR':<5} | {'Latency':<10} | {'TFLOPS':<8} | {'Validation'}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for name, bt, wg, wt, dk, sgl in configs:
        # Note: Name MUST be 'generated_gemm' because GeneratorRunner hardcodes 'generated_gemm' as symbol name
        kernel = GemmKernel(name="generated_gemm", target=GFX90A)
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

        diag = kernel.get_diagnostics()
        lds_b = diag["lds_usage_bytes"]
        agpr = diag["agpr_per_thread"]

        asm = kernel.generate_assembly()
        vgpr_count = "?"
        for line in asm.splitlines():
            if ".vgpr_count" in line:
                vgpr_count = line.split(":")[-1].strip()
                break

        cfg_out_dir = os.path.join(out_folder, name)
        os.makedirs(cfg_out_dir, exist_ok=True)
        ret = kernel.compile(output_folder=cfg_out_dir)
        if ret != 0:
            print(f"{name:<24} | {str(bt):<14} | {'Single' if sgl else 'Double':<8} | {lds_b:<8} | {vgpr_count:<5} | {agpr:<5} | {'COMPILE ERR':<10} | {'-':<8} | {'N/A'}", flush=True)
            continue

        co_path = os.path.join(cfg_out_dir, "generated_gemm.co")
        toml_path = os.path.join(cfg_out_dir, "generated_gemm.toml")

        # 1. Fast functional correctness check on 1024x1024x1024 matrix (1,048,576 output elements, fast on CPU)
        val_cmd = f"build/GeneratorRunner {co_path} {toml_path} 1024 1024 1024 1 1 1"
        val_res = subprocess.run(val_cmd, shell=True, capture_output=True, text=True)
        val_status = "PASS"
        for line in val_res.stdout.splitlines():
            if "# of mismatches:" in line:
                val_status = line.split(":")[-1].strip()
            elif "failed" in line.lower() or "error" in line.lower():
                val_status = line.strip()

        # 2. Performance benchmark on 4096x4096x4096 (5 warmup, 20 runs)
        bench_cmd = f"build/GeneratorRunner {co_path} {toml_path} 4096 4096 4096 5 20 0"
        bench_res = subprocess.run(bench_cmd, shell=True, capture_output=True, text=True)

        dur_ms = "N/A"
        tflops = "N/A"
        for line in bench_res.stdout.splitlines():
            if "ASM gemm:" in line:
                dur_ms = line.split(":")[-1].strip()
            elif "Gflops:" in line:
                gflops_val = float(line.split(":")[-1].strip())
                tflops = f"{gflops_val / 1000.0:.2f}"

        buf_label = "Single" if sgl else "Double"
        print(f"{name:<24} | {str(bt):<14} | {buf_label:<8} | {lds_b:<8} | {vgpr_count:<5} | {agpr:<5} | {dur_ms:<10} | {tflops:<8} | {val_status}", flush=True)


if __name__ == "__main__":
    run_benchmark()
