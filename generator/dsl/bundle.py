from __future__ import annotations
import os
import subprocess
from typing import Dict, List, Optional, Any
from generator.dsl.kernel import GemmKernel
from generator.target_spec import TargetSpec, GFX90A
from generator.generator import (
    generate_bundle_assembly,
    compile_bundle,
    GemmSolutionConfig,
)


import math


class BenchmarkResults:
    def __init__(self, entries: List[Dict[str, Any]]):
        self.entries = entries

    def all(self) -> List[Dict[str, Any]]:
        return self.entries

    def best(self) -> Optional[Dict[str, Any]]:
        if not self.entries:
            return None
        valid = [
            e for e in self.entries
            if isinstance(e.get("tflops"), (int, float))
            and math.isfinite(e["tflops"])
            and isinstance(e.get("latency_ms"), (int, float))
            and e.get("latency_ms", 0) > 0
        ]
        if not valid:
            return None
        return max(valid, key=lambda x: x["tflops"])

    def __repr__(self) -> str:
        header = f"{'Kernel Name':<55} | {'Latency':<10} | {'TFLOPS':<8}"
        lines = [header, "-" * len(header)]
        for e in self.entries:
            lat = f"{e.get('latency_ms', 'N/A')} ms" if isinstance(e.get('latency_ms'), (int, float)) else str(e.get('latency_ms', 'N/A'))
            lines.append(f"{e.get('name', ''):<55} | {lat:<10} | {str(e.get('tflops', 'N/A')):<8}")
        return "\n".join(lines)


class GemmKernelBundle:
    """
    Manages a collection of GemmKernel instances, allowing them to be:
    - Generated as a single multi-kernel assembly file (.s).
    - Assembled and linked into a single AMDGPU Code Object (.co) in ONE clang++ invocation.
    - Serialized into a unified multi-kernel catalog (.toml).
    - Benchmarked in-process using GeneratorRunner with pre-allocated GPU VRAM buffers.
    - Queried at runtime by DSL parameters.
    """
    def __init__(self, name: str = "gemm_bundle", target: TargetSpec = GFX90A):
        self.name = name
        self.target = target
        self.kernels: Dict[str, GemmKernel] = {}

    def add(self, kernel: GemmKernel) -> GemmKernelBundle:
        """Adds a GemmKernel to the bundle. Uses kernel.kernel_name as key."""
        kernel_name = kernel.kernel_name
        self.kernels[kernel_name] = kernel
        return self

    def get(self, name: str) -> Optional[GemmKernel]:
        """Retrieves a kernel by its symbol name."""
        return self.kernels.get(name)

    def find_kernel(self, **criteria) -> Optional[GemmKernel]:
        """
        Query a kernel from the bundle matching DSL criteria:
        e.g. bundle.find_kernel(block_tile=(256, 128, 16), depth_k=16, single_buffer_lds=True)
        """
        for k in self.kernels.values():
            match = True
            for attr, expected_val in criteria.items():
                actual_val = getattr(k, attr, None)
                if actual_val != expected_val:
                    match = False
                    break
            if match:
                return k
        return None

    def generate_assembly(self) -> str:
        """Generates unified multi-kernel assembly containing all kernels in the bundle."""
        kernel_parts = []
        for k in self.kernels.values():
            body_str, rodata_str, meta = k.generate_assembly(generate_parts=True)
            kernel_parts.append((body_str, rodata_str, meta))
        arch_str = f"{self.target.name}:xnack-"
        return generate_bundle_assembly(arch_str, kernel_parts)

    def compile(self, output_folder: str = "out_bundle", arch: Optional[str] = None) -> int:
        """
        Compiles all bundled kernels into a single .co and writes a multi-kernel .toml catalog.
        Invokes clang++ only once.
        """
        os.makedirs(output_folder, exist_ok=True)
        arch_str = arch or f"{self.target.name}:xnack-"
        bundle_asm = self.generate_assembly()
        configs = {
            name: k.to_gemm_solution_config()[0]
            for name, k in self.kernels.items()
        }
        return compile_bundle(self.name, bundle_asm, arch_str, output_folder, configs)

    def benchmark(
        self,
        m: int = 4096,
        n: int = 4096,
        k: int = 4096,
        warmup_runs: int = 5,
        num_runs: int = 20,
        runner_bin: str = "build/GeneratorRunner",
        output_folder: str = "out_bundle",
        kernel_name: Optional[str] = None,
        validate: bool = False,
    ) -> BenchmarkResults:
        """
        Benchmarks bundled kernels in-process using GeneratorRunner.
        """
        co_path = os.path.join(output_folder, f"{self.name}.co")
        toml_path = os.path.join(output_folder, f"{self.name}.toml")

        target_arg = kernel_name if kernel_name else "--all"
        val_flag = "1" if validate else "0"

        cmd = f"{runner_bin} {co_path} {toml_path} {m} {n} {k} {warmup_runs} {num_runs} {val_flag} {target_arg}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        entries = []
        for line in res.stdout.splitlines():
            if "[BATCH_BENCH]" in line:
                parts = line.replace("[BATCH_BENCH]", "").strip().split("|")
                if len(parts) >= 3:
                    k_name = parts[0].strip()
                    lat_str = parts[1].replace("ms", "").strip()
                    gflops_str = parts[2].replace("Gflops", "").strip()
                    try:
                        lat_val = float(lat_str)
                        gflops_val = float(gflops_str)
                        tflops_val = round(gflops_val / 1000.0, 2)
                    except ValueError:
                        lat_val = lat_str
                        tflops_val = "N/A"
                    entries.append({
                        "name": k_name,
                        "latency_ms": lat_val,
                        "tflops": tflops_val,
                    })

        return BenchmarkResults(entries)

    def __len__(self) -> int:
        return len(self.kernels)

    def __iter__(self):
        return iter(self.kernels.values())
