# AMD GPU Architecture GEMM (amdgpu-arch-gemm)

An AMD GCN/CDNA assembly generator, virtual machine simulator, and runner driver for General Matrix Multiply (GEMM) kernels on AMD GPUs (supporting `gfx90a` and `gfx942` architectures).

This repository contains:
1. **GCN Assembly Generator** (Python): Generates raw AMD assembly (`.s`) and compiles it using LLM/ROCm tools to a Code Object (`.co`) and outputs a configuration `.toml` file.
2. **GCN Virtual Machine Simulator** (Python): Emulates execution of GCN assembly instructions in software. This allows you to verify SGEMM accuracy and code logic instruction-by-instruction **without requiring a physical AMD GPU**.
3. **C++ Runner Driver** (HIP / C++): Loads compiled Code Objects (`.co`) and benchmarks/validates them on AMD GPU hardware.
4. **Tuner** (Python): Performs a search over various GEMM tiling, wave, and MFMA parameters to find the best configuration.

---

## Directory Structure

```
amdgpu-arch-gemm/
├── generator/
│   ├── __init__.py
│   └── generator.py           # AST-like instruction generation & compilation logic
├── vm/
│   └── gcn_virtual_machine.py # Python-based instruction emulator/simulator
├── runner/
│   ├── Utils/                 # TOML and buffer helper headers for HIP
│   ├── generator_runner.cpp   # C++ driver to run & benchmark kernels on GPU
│   └── CMakeLists.txt         # Build definition for the runner
├── tuner/
│   └── sgemm_tuner.py         # Grid search tuner over GEMM configurations
├── test/
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
To run the test suite (which includes instruction-level emulations and verifying a complete $16 \times 16$ or $32 \times 32$ SGEMM kernel execution inside the simulator):
```bash
PYTHONPATH=. pytest
```
*No GPU is needed for this step! This makes developing and testing new instructions or logic changes extremely easy and fast.*

### 3. Compiling the Assembly on AMD ROCm Machines
To generate assembly for a default SGEMM shape and compile it to a Code Object:
```bash
# Output folder where .s, .o, and .co will be generated
mkdir -p out
PYTHONPATH=. python generator/generator.py --arch gfx90a:xnack- out
```
This produces:
* `out/generated_gemm.s` (raw assembly)
* `out/generated_gemm.o` (intermediate object file)
* `out/generated_gemm.co` (HSA Code Object loaded by HIP module loader)
* `generated_gemm.toml` (JSON-like configurations loaded by the runner driver)

### 4. Building the C++ Runner Driver
To compile the benchmark runner:
```bash
mkdir -p build && cd build
cmake ../runner
make
```
This builds `GeneratorRunner`.

### 5. Running and Benchmarking on Real Hardware
Run the compiled benchmark executable using:
```bash
./GeneratorRunner <path_to_co> <path_to_toml> M N K <num_warmup_runs> <num_benchmark_runs> <validation_flag>
```
Example:
```bash
./GeneratorRunner ../out/generated_gemm.co ../generated_gemm.toml 1024 1024 1024 10 100 1
```
* **validation_flag (`0` or `1`)**: If set to `1`, executes a reference GEMM on CPU and compares accuracy.

### 6. Tuning GEMM Configurations
Use the tuner to search over combinations of wave tiling, wave grouping, and MFMA instructions for a given matrix size:
```bash
PYTHONPATH=. python tuner/sgemm_tuner.py --m 1024 --n 1024 --k 1024 --bench ./build/GeneratorRunner --output-folder ./out
```

---

## Supported Architectures and Configurations
* **Architectures**: `gfx90a` (CDNA 2, e.g., MI200 series) and `gfx942` (CDNA 3, e.g., MI300 series).
* **Instruction Set**: Uses MFMA (Matrix Fused-Multiply Add) instructions like `v_mfma_f32_16x16x4f32` and `v_mfma_f32_32x32x2f32` for FP32 GEMM computation.
* **Layouts**: Standard Column-Major (NN) GEMM is supported.