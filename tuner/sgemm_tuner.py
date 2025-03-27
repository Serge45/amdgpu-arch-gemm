import argparse
import subprocess
from itertools import product
from generator.generator import (
    compile,
    gemm,
    GemmSolutionConfig,
    GemmOptimizations,
    DataType,
    FunctionArgument,
)

SUPPORTED_WAVE_GROUPS = [(1, 1), (1, 2), (2, 1), (2, 2)]
SUPPORTED_WAVE_TILINGS = [(i, j) for i in range(1, 5) for j in range(1, 5)]
SUPPORTED_MFMAS = [(16, 16, 1, 4), (32, 32, 1, 2)]
SUPPORTED_DEPTH_K = [8, 16, 32]
VERBOSE = False

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", dest="m", type=int, required=True, action="store")
    ap.add_argument("--n", dest="n", type=int, required=True, action="store")
    ap.add_argument("--k", dest="k", type=int, required=True, action="store")
    ap.add_argument("--bench", dest="bench", type=str, required=True, action="store")
    ap.add_argument("--output-folder", dest="output_folder", type=str, required=True, action="store")
    args = ap.parse_args()
    m: int = args.m
    n: int = args.n
    k: int = args.k
    bench: str = args.bench
    output_folder: str = args.output_folder
    best_gflops = 0
    best_config = None

    for wg, wt, mfma, depth_k in product(
        SUPPORTED_WAVE_GROUPS,
        SUPPORTED_WAVE_TILINGS,
        SUPPORTED_MFMAS,
        SUPPORTED_DEPTH_K,
    ):
        try:
            config = GemmSolutionConfig(
                DataType.FP32,
                DataType.FP32,
                DataType.FP32,
                DataType.FP32,
                mfma,
                wg,
                wt,
                depth_k,
                False,
                False,
            )
            opt = GemmOptimizations(1)
            kern_name = "generated_gemm"
            asm = gemm(
                None,
                kern_name,
                "gfx90a:xnack-",
                config,
                opt,
                [
                    FunctionArgument("global_buffer", "a", None, 8),
                    FunctionArgument("global_buffer", "b", None, 8),
                    FunctionArgument("global_buffer", "c", None, 8),
                    FunctionArgument("global_buffer", "d", None, 8),
                    FunctionArgument("by_value", "m", None, 4),
                    FunctionArgument("by_value", "n", None, 4),
                    FunctionArgument("by_value", "k", None, 4),
                    FunctionArgument("by_value", "lda", None, 4),
                    FunctionArgument("by_value", "ldb", None, 4),
                    FunctionArgument("by_value", "ldc", None, 4),
                    FunctionArgument("by_value", "ldd", None, 4),
                    FunctionArgument("by_value", "alpha", None, 4),
                    FunctionArgument("by_value", "beta", None, 4),
                    FunctionArgument("by_value", "numWorkgroupX", None, 4),
                    FunctionArgument("by_value", "numWorkgroupY", None, 4),
                ],
            )
        except RuntimeError as e:
            if VERBOSE:
                print(e)
            continue

        if (m % config.tile_size[0]) or (n % config.tile_size[1]) or (k % config.depth_k):
            if VERBOSE:
                print(f"{m}, {n}, {k} were not divisible by {config.tile_size[0]}, {config.tile_size[1]}, {config.depth_k}")
            continue

        if compile(kern_name, asm, "gfx90a:xnack-", output_folder, config):
            if VERBOSE:
                print(f"compile error")
            continue
        else:
            ret = subprocess.run([bench, f"{output_folder}/{kern_name}.co", f"{output_folder}/{kern_name}.toml", str(m), str(n), str(k), "5", "10"], stdout=subprocess.PIPE)
            if not ret.returncode:
                out = ret.stdout.decode()
                gflops = float(out.split("\n")[-4].split(":")[-1])
                if gflops > best_gflops:
                    best_config = config
                best_gflops = max(gflops, best_gflops)
                print(f"Gflops: {gflops}")

    print(f"Best: {best_gflops} Gflops, MT: {best_config.tile_size}, DK: {best_config.depth_k}")
