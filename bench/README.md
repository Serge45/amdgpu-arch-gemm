# Benchmark Suite

This directory contains consolidated benchmark and profiling scripts for AMD Instinct (MI300X gfx942 / MI210 gfx90a) GEMM kernels.

## Scripts

- **`bench_dtva.py`**: Benchmarks high-performance Direct-to-VGPR Matrix A (DTVA) HGEMM kernels, comparing single-buffer and double-buffer LDS with WGM/WGMXCC on MI300X.
- **`bench_hipblaslt.cpp`**: Standalone C++ benchmark for hipBLASLt GEMM on MI300X.
- **`bench_hipblas.cpp`**: Standalone C++ benchmark for rocBLAS/hipBLAS on AMD GPUs.
- **`profile_suite.py`**: Profiling script using `rocprof` to collect hardware PMC counters (MFMA busy cycles, LDS bank conflicts, wave wait cycles).

## Usage

From the project root:

```bash
# Run DTVA HGEMM benchmark
python3 bench/bench_dtva.py [output_dir]

# Run PMC profiling
python3 bench/profile_suite.py
```
