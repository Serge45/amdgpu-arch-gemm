import subprocess
import csv
import os
import sys

KERNELS = [
    "hgemm_256x128x64_8w_wt4x4_base",
    "hgemm_256x128x64_8w_wt4x4_b128_disp",
    "hgemm_256x256x32_4w_wt8x8_sca_disp_vs2",
    "hgemm_256x256x32_4w_wt8x8_b128_disp_vs2",
    "hgemm_256x256x32_4w_wt8x8_sca_disp_vs3",
    "hgemm_256x256x32_4w_wt8x8_b128_disp_vs3",
    "hgemm_256x256x32_4w_wt8x8_b128_nodisp_vs3",
    "hgemm_256x128x64_4w_wt8x4_b128_disp_vs2",
]

PMC_COUNTERS = [
    "SQ_WAVE_CYCLES",
    "SQ_WAIT_ANY",
    "SQ_WAIT_INST_LDS",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    "SQ_INSTS_MFMA",
    "SQ_INSTS_LDS",
    "SQ_INSTS_VMEM",
    "SQ_LDS_BANK_CONFLICT",
]

def run_profile(dim: int, kernel_name: str, co_path: str, toml_path: str):
    pmc_file = f"pmc_{kernel_name}_{dim}.txt"
    out_dir = f"prof_{kernel_name}_{dim}"
    csv_file = f"{pmc_file.replace('.txt', '.csv')}"
    
    with open(pmc_file, "w") as f:
        f.write("pmc: " + ", ".join(PMC_COUNTERS) + "\n")
        
    cmd = [
        "rocprof",
        "-i", pmc_file,
        "-d", out_dir,
        "./build/GeneratorRunner",
        co_path,
        toml_path,
        str(dim), str(dim), str(dim),
        "1", "1", "0",
        kernel_name
    ]
    
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(f"Error running profile for {kernel_name} at {dim}: {res.stdout}")
        return None
        
    # Read generated csv
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found!")
        return None
        
    # Find the row corresponding to kernel_name (the last row is the warm run of the kernel)
    with open(csv_file, "r") as f:
        reader = list(csv.DictReader(f))
        for row in reversed(reader):
            if kernel_name in row.get("KernelName", ""):
                return row
    return None

def main():
    co_path = sys.argv[1] if len(sys.argv) > 1 else "out_wt8x8/mi300_wt8x8_gfx942.co"
    toml_path = sys.argv[2] if len(sys.argv) > 2 else "out_wt8x8/mi300_wt8x8_gfx942.toml"
    
    dims = [4096, 8192]
    
    for dim in dims:
        print(f"\n{'='*115}")
        print(f"PMC HARDWARE COUNTER PROFILING @ {dim}x{dim}x{dim} (FP16 HGEMM)")
        print(f"{'='*115}")
        
        results = []
        for kname in KERNELS:
            print(f"Profiling {kname} at {dim}...", flush=True)
            row = run_profile(dim, kname, co_path, toml_path)
            if row:
                results.append((kname, row))
                
        # Print summary table
        header = f"{'Kernel':<32} | {'WaveCyc(M)':<10} | {'WaitAny(M)':<10} | {'WaitLDS(M)':<10} | {'WaitLDS%':<8} | {'MFMA_Busy':<10} | {'LDS_BankConf':<12}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for kname, r in results:
            wave_cyc = int(r.get("SQ_WAVE_CYCLES", 0)) * 4 / 1e6
            wait_any = int(r.get("SQ_WAIT_ANY", 0)) * 4 / 1e6
            wait_lds = int(r.get("SQ_WAIT_INST_LDS", 0)) * 4 / 1e6
            mfma_busy = int(r.get("SQ_VALU_MFMA_BUSY_CYCLES", 0)) / 1e6
            bank_conf = int(r.get("SQ_LDS_BANK_CONFLICT", 0))
            
            wait_lds_pct = (wait_lds / wait_any * 100.0) if wait_any > 0 else 0.0
            
            print(f"{kname:<32} | {wave_cyc:10.2f} | {wait_any:10.2f} | {wait_lds:10.2f} | {wait_lds_pct:7.1f}% | {mfma_busy:10.2f} | {bank_conf:12,}")

if __name__ == "__main__":
    main()
