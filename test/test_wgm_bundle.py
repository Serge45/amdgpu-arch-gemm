from __future__ import annotations
import pytest
import numpy as np
from generator.dsl import GemmKernel, Tensor, GemmKernelBundle
from generator.atoms import MFMA_F32_32x32x2_F32, MFMA_F32_16x16x4_F32
from generator.target_spec import GFX90A
from generator.generator import DataType, GpuContext
from vm.gcn_virtual_machine import GcnVirtualMachine


def test_wgm_canonical_names():
    k1 = GemmKernel(target=GFX90A)
    k1.set_inputs(
        A=Tensor((2048, 2048), DataType.FP32),
        B=Tensor((2048, 2048), DataType.FP32),
        C=Tensor((2048, 2048), DataType.FP32),
    ).set_tiling(
        block_tile=(128, 64, 16),
        wave_group=(2, 2),
        wave_tiling=(2, 1),
    ).bind_atoms(mma=MFMA_F32_32x32x2_F32()).set_workgroup_mapping(1)

    assert "_wgm" not in k1.canonical_name

    k4 = GemmKernel(target=GFX90A)
    k4.set_inputs(
        A=Tensor((2048, 2048), DataType.FP32),
        B=Tensor((2048, 2048), DataType.FP32),
        C=Tensor((2048, 2048), DataType.FP32),
    ).set_tiling(
        block_tile=(128, 64, 16),
        wave_group=(2, 2),
        wave_tiling=(2, 1),
    ).bind_atoms(mma=MFMA_F32_32x32x2_F32()).set_workgroup_mapping(4)

    assert k4.canonical_name.endswith("_wgm4")


def test_wgm_bundle_assembly_no_label_collisions():
    bundle = GemmKernelBundle("wgm_bundle_test", target=GFX90A)

    for wgm in [1, 2, 4]:
        k = GemmKernel(target=GFX90A)
        k.set_inputs(
            A=Tensor((2048, 2048), DataType.FP32),
            B=Tensor((2048, 2048), DataType.FP32),
            C=Tensor((2048, 2048), DataType.FP32),
        ).set_tiling(
            block_tile=(128, 64, 16),
            wave_group=(2, 2),
            wave_tiling=(2, 1),
        ).bind_atoms(
            mma=MFMA_F32_32x32x2_F32()
        ).set_workgroup_mapping(wgm)
        bundle.add(k)

    # Generate single bundled assembly containing all 3 kernels
    bundle_asm = bundle.generate_assembly()
    assert ".amdhsa_kernel sgemm_nn_b128x64x16_wg2x2_wt2x1_mfma32x32x2_dbl_vs2" in bundle_asm
    assert ".amdhsa_kernel sgemm_nn_b128x64x16_wg2x2_wt2x1_mfma32x32x2_dbl_vs2_wgm2" in bundle_asm
    assert ".amdhsa_kernel sgemm_nn_b128x64x16_wg2x2_wt2x1_mfma32x32x2_dbl_vs2_wgm4" in bundle_asm

    # Check for code labels in bundle_asm before .amdgpu_metadata
    text_section = bundle_asm.split(".amdgpu_metadata")[0]
    lines = [
        line.strip()
        for line in text_section.splitlines()
        if line.strip().endswith(":") and not line.strip().startswith(".")
    ]
    seen = set()
    duplicates = set()
    for l in lines:
        if l in seen:
            duplicates.add(l)
        seen.add(l)
    assert not duplicates, f"Found duplicate labels in bundled assembly: {duplicates}"


def test_wgm_kernel_vm_simulation():
    # 16x16 GEMM with WGM = 2
    k = GemmKernel(target=GFX90A)
    k.set_inputs(
        A=Tensor((16, 16), DataType.FP32),
        B=Tensor((16, 16), DataType.FP32),
        C=Tensor((16, 16), DataType.FP32),
    ).set_tiling(
        block_tile=(16, 16, 4),
        wave_group=(1, 1),
        wave_tiling=(1, 1),
    ).bind_atoms(
        mma=MFMA_F32_16x16x4_F32()
    ).set_schedule(
        vmem_stages=1,
    ).set_workgroup_mapping(2)

    config, opt = k.to_gemm_solution_config()
    assert opt.wgm == 2
    assert config.wgm == 2
