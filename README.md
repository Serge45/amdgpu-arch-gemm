# AMD GPU Architecture GEMM (amdgpu-arch-gemm)

An AMD GCN/CDNA assembly generator, virtual machine simulator, and runner driver for General Matrix Multiply (GEMM) kernels on AMD GPUs (supporting `gfx90a` and `gfx942` architectures).

This repository contains:
1. **GEMM DSL & GCN Assembly Generator** (Python): Declarative kernel specification (`GemmKernel`) and multi-kernel bundling (`GemmKernelBundle`), generating AMD assembly (`.s`) and compiling to Code Objects (`.co`) and TOML configurations.
2. **GCN Virtual Machine Simulator** (Python): Emulates execution of GCN assembly instructions in software, allowing instruction-level SGEMM accuracy and register verification **without requiring a physical AMD GPU**.
3. **C++ Runner Driver** (HIP / C++): Loads compiled Code Objects (`.co`) and performs high-speed in-process batch benchmarking and validation on AMD GPU hardware.
4. **Auto-Tuner** (Python): High-throughput grid search auto-tuner leveraging multi-kernel bundles to search across tiling, wave groups, depthK, and LDS buffering strategies.

---

## Directory Structure

```
amdgpu-arch-gemm/
├── generator/
│   ├── dsl/
│   │   ├── kernel.py          # Declarative GemmKernel DSL interface
│   │   ├── bundle.py          # GemmKernelBundle for multi-kernel compilation & batch runs
│   │   ├── tensor.py          # Tensor descriptors & data types
│   │   └── operation.py       # Matmul operation definitions
│   ├── atoms.py               # Hardware MFMA compute atoms (e.g. 32x32x2, 16x16x4)
│   ├── target_spec.py         # Target GPU architectures (gfx90a, gfx942)
│   ├── scheduler.py           # Instruction schedulers (interleaved, round-robin, DAG modulo)
│   ├── reg_allocator.py       # Register allocation & liveness tracking
│   └── generator.py           # Core AST instruction generation & bundle assembly logic
├── vm/
│   └── gcn_virtual_machine.py # Python-based GCN instruction emulator & simulator
├── runner/
│   ├── Utils/                 # TOML and buffer helper headers for HIP
│   ├── generator_runner.cpp   # C++ driver supporting single-kernel and multi-kernel batch benchmarking
│   └── CMakeLists.txt         # Build definition for the runner
├── tuner/
│   └── sgemm_tuner.py         # High-throughput bundled auto-tuner over GEMM configurations
├── test/
│   ├── test_bundle.py         # Multi-kernel bundle and canonical name tests
│   ├── test_dsl_kernel.py     # GEMM DSL interface unit tests
│   ├── test_atoms_and_layout.py # MFMA atoms and layout verification
│   ├── test_multi_arch.py     # Architecture compatibility tests (gfx90a vs gfx942)
│   ├── test_reg_allocator.py  # Register allocator and liveness tests
│   ├── test_scheduler.py      # Instruction scheduler tests
│   ├── test_sgemm.py          # Assembly generation unit tests
│   └── test_vm.py             # Emulation correctness tests for individual/GEMM instructions
├── requirements.txt           # Production packages (tomli, tomli_w, pyyaml)
└── requirements-test.txt      # Testing packages (pytest)
```

---

## Getting Started

### 1. Requirements

* **For Assembly Generation and Software Simulation (No GPU required):**
  * Python 3.10+
  * Dependencies:
    ```bash
    pip install -r requirements.txt -r requirements-test.txt
    ```

* **For Hardware Compilation and Execution (AMD GPU with ROCm required):**
  * ROCm stack installed (expected compiler path: `/opt/rocm/llvm/bin/clang++`)
  * CMake 3.16+

### 2. Running the Tests & Software VM Simulator
To run the full test suite (96 unit tests verifying instruction-level emulation, register allocation, schedulers, DSL, transpose modes, workgroup mapping, and software SGEMM simulation):
```bash
PYTHONPATH=. pytest
```
*No GPU is needed for this step! This enables developing and testing new instructions or logic changes locally.*

### 3. Declarative GEMM DSL & Multi-Kernel Bundling

The declarative DSL allows concise specification of GEMM kernels and bundling multiple candidate configurations into a single Code Object (`.co`):

```python
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle
from generator.atoms import MFMA_F32_32x32x2_F32
from generator.target_spec import GFX90A
from generator.generator import DataType

# Create a multi-kernel bundle
bundle = GemmKernelBundle("my_gemm_bundle", target=GFX90A)

# Define candidates with different block tiles or buffering strategies
for sgl in [True, False]:
    kernel = GemmKernel(
        A=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
        B=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
        C=Tensor(shape=(4096, 4096), dtype=DataType.FP32),
    ).set_tiling(
        block_tile=(256, 128, 16),
        wave_group=(2, 2),
        wave_tiling=(4, 2),
    ).bind_atoms(
        mma=MFMA_F32_32x32x2_F32()
    ).set_schedule(
        vmem_stages=2,
        single_buffer_lds=sgl,
    )
    bundle.add(kernel)

# Compile all kernels into a single .co with 1 clang++ invocation!
bundle.compile(output_folder="out_bundle")
```

### 4. Canonical Kernel Naming Convention

Each kernel is assigned a canonical symbol name that uniquely identifies its configuration:
```
{precision}_{transA}{transB}_b{M}x{N}x{K}_wg{wg0}x{wg1}_wt{wt0}x{wt1}_{atom}_{lds_buf}_vs{stages}
```
* **Example**: `sgemm_nn_b256x128x16_wg2x2_wt4x2_mfma32x32x2_sgl_vs2`
* **Fields**:
  - `precision`: `sgemm` (FP32), `hgemm` (FP16), etc.
  - `transA/B`: `nn`, `nt`, `tn`, `tt` matrix transpose layouts.
  - `block_tile`: `b256x128x16` (M x N x K MacroTile).
  - `wave_group`: `wg2x2` (Wavefront grid per workgroup).
  - `wave_tiling`: `wt4x2` (MFMA iterations per wave).
  - `atom`: MFMA instruction type (e.g. `mfma32x32x2`, `mfma16x16x4`).
  - `lds_buf`: `sgl` (single buffer LDS) or `dbl` (double buffer LDS).
  - `vmem_stage`: `vs1`, `vs2` (global memory prefetch pipelining stages).

### 5. Building the C++ Runner Driver
To compile the high-performance benchmark runner:
```bash
mkdir -p build && cd build
cmake ../runner
make
```
This builds `GeneratorRunner`.

### 6. Running and Benchmarking on Real Hardware
Run the compiled benchmark executable using:
```bash
./GeneratorRunner <path_to_co> <path_to_toml> M N K <num_warmup_runs> <num_benchmark_runs> <validation_flag> [kernel_name|--all]
```
* **Single Kernel Execution**:
  ```bash
  ./GeneratorRunner out/generated_gemm.co out/generated_gemm.toml 4096 4096 4096 10 100 1
  ```
* **Multi-Kernel Batch Execution (All bundled kernels in-process)**:
  ```bash
  ./GeneratorRunner out_bundle/my_gemm_bundle.co out_bundle/my_gemm_bundle.toml 4096 4096 4096 5 20 0 --all
  ```
* **Run a Specific Bundled Kernel by Name**:
  ```bash
  ./GeneratorRunner out_bundle/my_gemm_bundle.co out_bundle/my_gemm_bundle.toml 4096 4096 4096 5 20 0 sgemm_nn_b256x128x16_wg2x2_wt4x2_mfma32x32x2_sgl_vs2
  ```

### 7. High-Throughput Auto-Tuning
Use `sgemm_tuner.py` to explore combinations of MacroTiles, wave groups, depthK, and single/double buffered LDS. The tuner groups candidates into multi-kernel bundles to compile and benchmark with minimal overhead:
```bash
PYTHONPATH=. python tuner/sgemm_tuner.py --m 4096 --n 4096 --k 4096 --bundle-size 6 --bench ./build/GeneratorRunner --output-folder ./out_tuner
```

---

## Supported Architectures and Configurations
* **Architectures**: `gfx90a` (CDNA 2, e.g. MI210 / MI250) and `gfx942` (CDNA 3, e.g. MI300 series).
* **Instruction Set**: Uses MFMA (Matrix Fused-Multiply Add) instructions like `v_mfma_f32_16x16x4f32` and `v_mfma_f32_32x32x2f32` for FP32 GEMM computation.
* **Layouts**: All 4 Column-Major matrix layout transpose modes (`NN`, `NT`, `TN`, `TT`) are fully supported with vectorized loads and stores.

---

## Performance Benchmarks on AMD Instinct MI210

Benchmarked on physical **AMD Instinct MI210** hardware (CDNA 2 `gfx90a`, 104 Compute Units, 64 GB HBM2e, FP32 MFMA Peak ~45.3 TFLOPS @ 1.7 GHz engine clock) against **AMD rocBLAS** (vendor-optimized BLAS library).

### 1. Throughput Comparison at a Glance

```
4K SGEMM (M=N=K=4096) — Throughput (TFLOPS)
  rocBLAS ref:  [███████████████████████████████     ] 33.60 TFLOPS
  Our NN:       [████████████████████████████████    ] 34.54 TFLOPS (+2.8% vs rocBLAS)
  Our NT:       [████████████████████████████████    ] 34.74 TFLOPS (+3.4% vs rocBLAS)
  Our TN:       [███████████████████████████████▌    ] 33.85 TFLOPS (+0.7% vs rocBLAS)
  Our TT:       [███████████████████████████████▋    ] 34.06 TFLOPS (+1.4% vs rocBLAS)

2K SGEMM (M=N=K=2048) — Throughput (TFLOPS)
  rocBLAS ref:  [████████████████████████████████    ] 35.00 TFLOPS
  Our b256x128: [█████████████████████               ] 22.68 TFLOPS (Quantization stall: 80 CUs idle)
  Our NN (wgm8):[█████████████████████████████▊      ] 32.52 TFLOPS (92.9% of rocBLAS)
  Our NT (wgm8):[████████████████████████████▉      ] 31.58 TFLOPS (90.2% of rocBLAS)
  Our TN (wgm8):[███████████████████████████        ] 29.69 TFLOPS (84.8% of rocBLAS)
  Our TT (wgm8):[████████████████████████████       ] 30.51 TFLOPS (87.2% of rocBLAS)
```

### 2. 4K Matrix Evaluation ($4096 \times 4096 \times 4096$)

* **Kernel Configuration**: `b256x128x16_wg2x2_wt4x2_mfma32x32x2_sgl_vs2`
* **Grid Occupancy**: $16 \times 32 = 512$ workgroups dispatched across 104 CUs ($4.92$ waves/CU $\implies$ **98.5% scheduling efficiency**).
* **Pipeline**: Single-buffer LDS (`sgl`) with 2-stage VMEM prefetching (`vs2`).
* **Memory Symmetry**: Preserves 128-bit vectorized global and LDS memory operations across all transpose modes without bank conflicts.

| Transpose Mode | Kernel Symbol | Time (ms) | Throughput (TFLOPS) | rocBLAS Baseline | Relative to rocBLAS | Numerical Verification |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **NN** | `sgemm_nn_b256x128x16_wg2x2_wt4x2_mfma32x32x2_sgl_vs2` | 3.979 ms | **34.54 TFLOPS** | ~33.60 TFLOPS | **102.8%** (+2.8%) | Exact Match (CPU / ROCm) |
| **NT** | `sgemm_nt_b256x128x16_wg2x2_wt4x2_mfma32x32x2_sgl_vs2` | 3.956 ms | **34.74 TFLOPS** | ~33.60 TFLOPS | **103.4%** (+3.4%) | Exact Match (CPU / ROCm) |
| **TN** | `sgemm_tn_b256x128x16_wg2x2_wt4x2_mfma32x32x2_sgl_vs2` | 4.060 ms | **33.85 TFLOPS** | ~33.60 TFLOPS | **100.7%** (+0.7%) | Exact Match (CPU / ROCm) |
| **TT** | `sgemm_tt_b256x128x16_wg2x2_wt4x2_mfma32x32x2_sgl_vs2` | 4.035 ms | **34.06 TFLOPS** | ~33.60 TFLOPS | **101.4%** (+1.4%) | Exact Match (CPU / ROCm) |

> **Highlight**: The custom generated assembly outperforms rocBLAS across all 4 matrix transpose orientations at 4K, achieving up to **76.7% of the theoretical physical FP32 MFMA roofline** of the MI210.

---

### 3. 2K Matrix Evaluation ($2048 \times 2048 \times 2048$)

#### Wave Quantization (Tail Wave Stall)
At $2048^3$, applying the large `b256x128` tile produces $(2048/256) \times (2048/128) = 8 \times 16 = \mathbf{128}$ workgroups:
1. **Wave 1**: 104 workgroups saturate all 104 CUs (100% CU utilization).
2. **Wave 2**: The remaining $128 - 104 = 24$ workgroups execute, leaving **80 CUs completely idle** (23% CU utilization).
3. Overall CU scheduling efficiency drops to $(104 + 24) / (2 \times 104) = \mathbf{61.5\%}$, reducing throughput to **22.68 TFLOPS**.

#### Optimization: MacroTile Scaling + Workgroup Mapping (WGM)
To eliminate tail wave stalls and optimize cache reuse:
1. **Block Tile Scaling**: `b128x64` increases the grid to $(2048/128) \times (2048/64) = 16 \times 32 = \mathbf{512}$ workgroups ($512 / 104 = 4.92$ waves/CU), immediately recovering scheduling efficiency to **98.5%** and throughput to >32 TFLOPS.
2. **Workgroup Mapping (`wgm=8`)**: Re-indexes the linear hardware dispatch order into 2D block columns ($8 \times 8$ WGs). Concurrently executing CUs share active lines in the 8 MB L2 cache for both matrices A and B, minimizing external HBM transactions.

```mermaid
graph LR
    subgraph Linear 1D Grid Launch
        A1["WG(0,0)"] --> A2["WG(1,0)"] --> A3["WG(2,0)"] --> A4["WG(3,0) ..."]
        style A1 fill:#ffebee,stroke:#c62828
        style A2 fill:#ffebee,stroke:#c62828
        style A3 fill:#ffebee,stroke:#c62828
        style A4 fill:#ffebee,stroke:#c62828
    end
    subgraph 2D Workgroup Mapping (wgm=8)
        B1["WG(0,0)"] --- B2["WG(0,1)"]
        B3["WG(1,0)"] --- B4["WG(1,1)"]
        B5["2D Tile Locality (Shared L2 Cache Lines)"]
        style B1 fill:#e8f5e9,stroke:#2e7d32
        style B2 fill:#e8f5e9,stroke:#2e7d32
        style B3 fill:#e8f5e9,stroke:#2e7d32
        style B4 fill:#e8f5e9,stroke:#2e7d32
        style B5 fill:#e3f2fd,stroke:#1565c0
    end
```

| Transpose Mode | Kernel Symbol | Time (ms) | Throughput (TFLOPS) | rocBLAS Baseline | % of rocBLAS | Scheduling Analysis |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **NN (Tail-Stall)** | `sgemm_nn_b256x128x16_sgl_vs2` | 0.758 ms | 22.68 TFLOPS | ~35.00 TFLOPS | 64.8% | 128 WGs on 104 CUs (80 CUs idle in tail wave) |
| **NN (Optimized)** | `sgemm_nn_b128x64x16_sgl_vs2_wgm8` | 0.528 ms | **32.52 TFLOPS** | ~35.00 TFLOPS | **92.9%** | 512 WGs (98.5% CU occupancy) + WGM=8 L2 reuse |
| **NT (Optimized)** | `sgemm_nt_b128x64x16_sgl_vs2_wgm8` | 0.544 ms | **31.58 TFLOPS** | ~35.00 TFLOPS | **90.2%** | 512 WGs (98.5% CU occupancy) + WGM=8 L2 reuse |
| **TN (Optimized)** | `sgemm_tn_b128x64x16_sgl_vs2_wgm8` | 0.579 ms | **29.69 TFLOPS** | ~35.00 TFLOPS | **84.8%** | 512 WGs (98.5% CU occupancy) + WGM=8 L2 reuse |
| **TT (Optimized)** | `sgemm_tt_b128x64x16_sgl_vs2_wgm8` | 0.563 ms | **30.51 TFLOPS** | ~35.00 TFLOPS | **87.2%** | 512 WGs (98.5% CU occupancy) + WGM=8 L2 reuse |