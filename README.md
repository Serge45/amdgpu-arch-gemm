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
To run the full test suite (83 unit tests verifying instruction-level emulation, register allocation, schedulers, DSL, and software SGEMM simulation):
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