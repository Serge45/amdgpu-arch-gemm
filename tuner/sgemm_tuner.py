from __future__ import annotations
import argparse
import os
import math
from itertools import product
from typing import List
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle
from generator.atoms import MFMA_F32_32x32x2_F32, MFMA_F32_16x16x4_F32
from generator.target_spec import GFX90A
from generator.generator import DataType, GemmSolutionConfig, MAX_LDS_NUM_BYTES


SUPPORTED_WAVE_GROUPS = [(1, 1), (1, 2), (2, 1), (2, 2), (4, 2), (2, 4)]
SUPPORTED_WAVE_TILINGS = [(i, j) for i in [1, 2, 4] for j in [1, 2, 4]]
SUPPORTED_DEPTH_K = [16, 32]
SUPPORTED_SINGLE_BUFFER_LDS = [True, False]
SUPPORTED_WGM = [1, 2, 4, 8]


def find_valid_candidates(
    m: int,
    n: int,
    k: int,
    trans_a: bool = False,
    trans_b: bool = False,
    wgm_options: List[int] | None = None,
) -> List[dict]:
    candidates = []
    wgm_list = wgm_options if wgm_options is not None else [1]
    for wg, wt, dk, sgl, wgm in product(
        SUPPORTED_WAVE_GROUPS,
        SUPPORTED_WAVE_TILINGS,
        SUPPORTED_DEPTH_K,
        SUPPORTED_SINGLE_BUFFER_LDS,
        wgm_list,
    ):
        bt = (32 * wg[0] * wt[0], 32 * wg[1] * wt[1], dk)
        # Check divisibility
        if (m % bt[0]) != 0 or (n % bt[1]) != 0 or (k % bt[2]) != 0:
            continue

        nwg0 = m // bt[0]
        if nwg0 % wgm != 0:
            continue

        try:
            cfg = GemmSolutionConfig(
                DataType.FP32,
                DataType.FP32,
                DataType.FP32,
                DataType.FP32,
                (32, 32, 1, 2),
                wg,
                wt,
                dk,
                trans_a,
                trans_b,
                vmem_stage=1,
                single_buffer_lds=sgl,
                wgm=wgm,
            )
            if cfg.num_bytes_per_buffer_load != (16, 16):
                continue
            if cfg.lds_usage_bytes >= MAX_LDS_NUM_BYTES:
                continue

            candidates.append({
                "block_tile": bt,
                "wave_group": wg,
                "wave_tiling": wt,
                "depth_k": dk,
                "single_buffer_lds": sgl,
                "wgm": wgm,
            })
        except Exception:
            continue

    return candidates


def main():
    ap = argparse.ArgumentParser(description="Fast AMDGPU GEMM Tuner with Multi-Kernel Bundling")
    ap.add_argument("--m", dest="m", type=int, default=4096, help="Matrix M dimension")
    ap.add_argument("--n", dest="n", type=int, default=4096, help="Matrix N dimension")
    ap.add_argument("--k", dest="k", type=int, default=4096, help="Matrix K dimension")
    ap.add_argument("--trans-a", dest="trans_a", action="store_true", help="Transpose matrix A")
    ap.add_argument("--trans-b", dest="trans_b", action="store_true", help="Transpose matrix B")
    ap.add_argument("--wgm", dest="wgm", type=int, default=1, help="Workgroup mapping factor (1, 2, 4, 8) or 0 to sweep all")
    ap.add_argument("--validate", dest="validate", action="store_true", help="Validate results against CPU GEMM")
    ap.add_argument("--bench", dest="bench", type=str, default="build/GeneratorRunner", help="Path to GeneratorRunner")
    ap.add_argument("--output-folder", dest="output_folder", type=str, default="out_tuner", help="Output directory")
    ap.add_argument("--bundle-size", dest="bundle_size", type=int, default=20, help="Kernels per compiled bundle")
    ap.add_argument("--warmup", dest="warmup", type=int, default=5, help="Warmup iterations")
    ap.add_argument("--runs", dest="runs", type=int, default=20, help="Benchmark iterations")
    args = ap.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    wgm_options = SUPPORTED_WGM if args.wgm == 0 else [args.wgm]
    mode = f"{'T' if args.trans_a else 'N'}{'T' if args.trans_b else 'N'}"
    print(f"=== Fast GEMM Auto-Tuner ({mode}, M={args.m}, N={args.n}, K={args.k}, WGM={wgm_options}) ===", flush=True)
    candidates = find_valid_candidates(args.m, args.n, args.k, args.trans_a, args.trans_b, wgm_options)
    print(f"Discovered {len(candidates)} valid candidate configurations.", flush=True)

    # Chunk into bundles
    bundle_size = args.bundle_size
    bundle_chunks = [candidates[i:i + bundle_size] for i in range(0, len(candidates), bundle_size)]
    print(f"Partitioned candidates into {len(bundle_chunks)} bundles of up to {bundle_size} kernels.\n", flush=True)

    all_results = []

    for b_idx, chunk in enumerate(bundle_chunks):
        bundle_name = f"tuner_bundle_{b_idx}"
        bundle = GemmKernelBundle(name=bundle_name, target=GFX90A)

        for spec in chunk:
            kernel = GemmKernel(target=GFX90A)
            kernel.set_inputs(
                A=Tensor(shape=(args.k, args.m) if args.trans_a else (args.m, args.k), dtype=DataType.FP32),
                B=Tensor(shape=(args.n, args.k) if args.trans_b else (args.k, args.n), dtype=DataType.FP32),
                C=Tensor(shape=(args.m, args.n), dtype=DataType.FP32),
            ).set_transposes(
                args.trans_a, args.trans_b
            ).set_workgroup_mapping(
                spec["wgm"]
            ).set_tiling(
                block_tile=spec["block_tile"],
                wave_group=spec["wave_group"],
                wave_tiling=spec["wave_tiling"],
            ).bind_atoms(
                mma=MFMA_F32_32x32x2_F32()
            ).set_schedule(
                vmem_stages=2,
                single_buffer_lds=spec["single_buffer_lds"],
            )
            bundle.add(kernel)

        print(f"--- Compiling Bundle {b_idx + 1}/{len(bundle_chunks)} ({len(bundle)} kernels) ---", flush=True)
        ret = bundle.compile(output_folder=args.output_folder)
        if ret != 0:
            print(f"Failed to compile bundle {bundle_name}, skipping...", flush=True)
            continue

        print(f"--- Benchmarking Bundle {b_idx + 1}/{len(bundle_chunks)} in-process ---", flush=True)
        results = bundle.benchmark(
            m=args.m,
            n=args.n,
            k=args.k,
            warmup_runs=args.warmup,
            num_runs=args.runs,
            runner_bin=args.bench,
            output_folder=args.output_folder,
            validate=args.validate,
        )

        for res in results.all():
            print(f"  {res['name']:<55} | {str(res['latency_ms']) + ' ms':<10} | {res['tflops']} TFLOPS", flush=True)
            all_results.append(res)

    print("\n" + "=" * 80, flush=True)
    print("=== Auto-Tuning Summary ===", flush=True)
    print("=" * 80, flush=True)
    valid_results = [
        r for r in all_results
        if isinstance(r.get("tflops"), (int, float))
        and math.isfinite(r["tflops"])
        and isinstance(r.get("latency_ms"), (int, float))
        and r.get("latency_ms", 0) > 0
    ]
    valid_results.sort(key=lambda x: x["tflops"], reverse=True)

    header = f"{'Rank':<5} | {'Kernel Name':<55} | {'Latency':<10} | {'TFLOPS':<8}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(valid_results[:10]):
        print(f"{i + 1:<5} | {r['name']:<55} | {str(r['latency_ms']) + ' ms':<10} | {r['tflops']:<8}")

    if valid_results:
        best = valid_results[0]
        print(f"\n[WINNER] {best['name']} -> {best['latency_ms']} ms ({best['tflops']} TFLOPS)")


if __name__ == "__main__":
    main()
