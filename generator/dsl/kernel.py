from __future__ import annotations
from enum import Enum, auto
from typing import Tuple, List, Dict, Optional, Callable, Any, Union
import struct
import numpy as np

from generator.target_spec import TargetSpec, GFX90A
from generator.atoms import (
    MMAAtom,
    MFMA_F32_32x32x2_F32,
    MFMA_F32_16x16x4_F32,
    CopyAtom,
    BufferLoadAtom,
    DsWriteAtom,
    DsReadAtom,
)
from generator.layout import TiledMMA, TiledCopy, LdsPaddingSolver
from generator.scheduler import SchedulingPolicy
from generator.reg_allocator import RegisterAllocator
from generator.generator import (
    GpuContext,
    DataType,
    FunctionArgument,
    GemmSolutionConfig,
    GemmOptimizations,
    gemm,
    compile as compile_asm,
)
from vm.gcn_virtual_machine import GcnVirtualMachine


class LayoutType(Enum):
    COL_MAJOR = auto()
    ROW_MAJOR = auto()


class Tensor:
    """
    Represents a tensor operand with shape, datatype, and memory layout.
    """
    def __init__(
        self,
        shape: Tuple[Union[str, int], ...],
        dtype: DataType = DataType.FP32,
        layout: LayoutType = LayoutType.COL_MAJOR,
        name: str = "",
    ):
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self.name = name


class GemmKernel:
    """
    High-level declarative DSL for constructing and compiling optimized GEMM kernels.
    Combines hierarchical tiling (TiledMMA), hardware instructions (MMAAtom),
    register budgeting, and modulo software pipelining into a concise API.
    """
    def __init__(self, name: str = "gemm_kernel", target: TargetSpec = GFX90A):
        self.name = name
        self.target = target

        # Input tensors
        self.tensor_a: Optional[Tensor] = None
        self.tensor_b: Optional[Tensor] = None
        self.tensor_c: Optional[Tensor] = None
        self.tensor_d: Optional[Tensor] = None

        # Hierarchy and tiling
        self.block_tile: Tuple[int, int, int] = (128, 128, 16)
        self.wave_group: Tuple[int, int] = (2, 2)
        self.wave_tiling: Tuple[int, int] = (2, 2)
        self.depth_k: int = 16

        # Hardware atoms
        self.mma_atom: MMAAtom = MFMA_F32_32x32x2_F32()
        self.global_load_atom: CopyAtom = BufferLoadAtom(vector_dwords=4)
        self.lds_write_atom: CopyAtom = DsWriteAtom(vector_dwords=4)
        self.lds_read_atom: CopyAtom = DsReadAtom(vector_dwords=1)

        # Scheduling and register constraints
        self.vmem_stages: int = 2
        self.scheduling_policy: SchedulingPolicy = SchedulingPolicy.ROUNDROBIN
        self.max_vgpr_budget: int = target.max_vgpr

        # Custom epilogue function
        self._epilogue_fn: Optional[Callable] = None

    def set_inputs(
        self,
        A: Tensor,
        B: Tensor,
        C: Optional[Tensor] = None,
        D: Optional[Tensor] = None,
    ) -> GemmKernel:
        self.tensor_a = A
        self.tensor_b = B
        self.tensor_c = C
        self.tensor_d = D or C
        return self

    def set_tiling(
        self,
        block_tile: Tuple[int, int, int],
        wave_group: Tuple[int, int],
        wave_tiling: Tuple[int, int],
    ) -> GemmKernel:
        self.block_tile = block_tile
        self.wave_group = wave_group
        self.wave_tiling = wave_tiling
        self.depth_k = block_tile[2]
        return self

    def bind_atoms(
        self,
        mma: Optional[MMAAtom] = None,
        global_load: Optional[CopyAtom] = None,
        lds_write: Optional[CopyAtom] = None,
        lds_read: Optional[CopyAtom] = None,
    ) -> GemmKernel:
        if mma is not None:
            self.mma_atom = mma
        if global_load is not None:
            self.global_load_atom = global_load
        if lds_write is not None:
            self.lds_write_atom = lds_write
        if lds_read is not None:
            self.lds_read_atom = lds_read
        return self

    def set_schedule(
        self,
        vmem_stages: int = 2,
        scheduling_policy: SchedulingPolicy = SchedulingPolicy.ROUNDROBIN,
        max_vgpr_budget: int = 128,
    ) -> GemmKernel:
        self.vmem_stages = vmem_stages
        self.scheduling_policy = scheduling_policy
        self.max_vgpr_budget = max_vgpr_budget
        return self

    def epilogue(self, fn: Callable) -> Callable:
        """Decorator to register a custom epilogue operation."""
        self._epilogue_fn = fn
        return fn

    @property
    def tiled_mma(self) -> TiledMMA:
        return TiledMMA(
            atom=self.mma_atom,
            wave_group=self.wave_group,
            wave_tiling=self.wave_tiling,
            wavefront_size=self.target.wavefront_size,
            target=self.target,
        )

    def to_gemm_solution_config(self) -> Tuple[GemmSolutionConfig, GemmOptimizations]:
        """Lowers high-level DSL settings into low-level compiler configurations."""
        self.tiled_mma.validate(self.target)

        a_type = self.tensor_a.dtype if self.tensor_a else DataType.FP32
        b_type = self.tensor_b.dtype if self.tensor_b else DataType.FP32
        cd_type = self.tensor_c.dtype if self.tensor_c else DataType.FP32

        # AMDGPU GEMM native layout is column-major: trans is False for COL_MAJOR, True for ROW_MAJOR
        trans_a = self.tensor_a.layout == LayoutType.ROW_MAJOR if self.tensor_a else False
        trans_b = self.tensor_b.layout == LayoutType.ROW_MAJOR if self.tensor_b else False

        # Map DSL vmem_stages to backend vmem_stage:
        # vmem_stages=1: double-buffered LDS (2 partitions), unpipelined loop (plr=0)
        # vmem_stages=2: double-buffered LDS (2 partitions), pipelined loop (plr=1)
        # vmem_stages>=3: multi-buffered LDS (N partitions), pipelined loop (plr=1)
        backend_vmem_stage = max(1, self.vmem_stages - 1)

        config = GemmSolutionConfig(
            a_type=a_type,
            b_type=b_type,
            cd_type=cd_type,
            scalar_type=cd_type,
            mfma=self.mma_atom.shape,
            wave_group=self.wave_group,
            wave_tiling=self.wave_tiling,
            depth_k=self.depth_k,
            trans_a=trans_a,
            trans_b=trans_b,
            vmem_stage=backend_vmem_stage,
        )

        opt = GemmOptimizations(
            level=1 if self.vmem_stages >= 2 else 0,
            scheduling_policy=self.scheduling_policy,
        )
        opt.plr = 1 if self.vmem_stages >= 2 else 0
        opt.gw = 1

        return config, opt

    def get_function_arguments(self) -> List[FunctionArgument]:
        return [
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
        ]

    def generate_assembly(self) -> str:
        """Generates complete AMDGPU GCN assembly source code."""
        config, opt = self.to_gemm_solution_config()
        args = self.get_function_arguments()
        context = GpuContext()
        return gemm(context, self.name, f"{self.target.name}:xnack-", config, opt, args)

    def compile(self, output_folder: str = "out", arch: Optional[str] = None) -> int:
        """
        Compiles generated assembly into ELF object (.o) and Code Object (.co)
        using the ROCm clang++ toolchain.
        """
        import os
        os.makedirs(output_folder, exist_ok=True)
        arch_str = arch or f"{self.target.name}:xnack-"
        asm = self.generate_assembly()
        config, _ = self.to_gemm_solution_config()
        return compile_asm(self.name, asm, arch_str, output_folder, config)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Provides compile-time microarchitecture performance diagnostics."""
        config, _ = self.to_gemm_solution_config()
        pad_a, conf_a = LdsPaddingSolver.solve_pad_a(self.mma_atom, tile_m=config.tile_size[0])
        pad_b, conf_b = LdsPaddingSolver.solve_pad_b(self.mma_atom, depth_k=config.depth_k)

        alloc = RegisterAllocator(target=self.target, max_vgpr_budget=self.max_vgpr_budget)
        tiled_mma = self.tiled_mma

        return {
            "kernel_name": self.name,
            "target": self.target.name,
            "block_tile": tiled_mma.block_tile,
            "wave_tile": tiled_mma.wave_tile,
            "atom_tile": tiled_mma.atom_tile,
            "num_waves": tiled_mma.num_waves,
            "num_threads": tiled_mma.num_threads,
            "lds_usage_bytes": config.lds_usage_bytes,
            "lds_pad_a": pad_a,
            "lds_conflicts_a": conf_a,
            "lds_pad_b": pad_b,
            "lds_conflicts_b": conf_b,
            "agpr_per_thread": tiled_mma.num_acc_regs_per_thread,
            "max_vgpr_budget": self.max_vgpr_budget,
        }

    def emulate(
        self,
        a_np: np.ndarray,
        b_np: np.ndarray,
        c_np: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> np.ndarray:
        """
        Emulates kernel execution on CPU using GcnVirtualMachine.
        Returns the output matrix D computed by the simulated kernel.
        """
        config, opt = self.to_gemm_solution_config()
        args = self.get_function_arguments()
        context = GpuContext()
        gemm(context, self.name, f"{self.target.name}:xnack-", config, opt, args)

        m, k = a_np.shape
        k2, n = b_np.shape
        assert k == k2, f"Matrix dimension mismatch: A is {a_np.shape}, B is {b_np.shape}"

        if c_np is None:
            c_np = np.zeros((m, n), dtype=np.float32)

        vm = GcnVirtualMachine(
            num_total_sgpr=self.target.max_sgpr,
            num_total_vgpr=self.target.max_vgpr,
            wavefront_size=self.target.wavefront_size,
        )

        # Setup thread IDs for wave 0
        for i in range(self.target.wavefront_size):
            vm.v[0][i] = i

        # Initialize kernarg base address
        vm.s[0] = 0
        vm.s[1] = 0
        vm.s[2] = 0
        vm.s[3] = 0

        a_offset, a_size = 0, m * k * 4
        b_offset, b_size = a_size, n * k * 4
        c_offset, c_size = a_size + b_size, m * n * 4
        d_offset, d_size = a_size + b_size + c_size, m * n * 4

        # Write kernargs into scalar memory
        vm.smem.mem[:8] = int.to_bytes(a_offset, 8, "little")
        vm.smem.mem[8:16] = int.to_bytes(b_offset, 8, "little")
        vm.smem.mem[16:24] = int.to_bytes(c_offset, 8, "little")
        vm.smem.mem[24:32] = int.to_bytes(d_offset, 8, "little")
        vm.smem.mem[32:36] = int.to_bytes(m, 4, "little")
        vm.smem.mem[36:40] = int.to_bytes(n, 4, "little")
        vm.smem.mem[40:44] = int.to_bytes(k, 4, "little")
        vm.smem.mem[44:48] = int.to_bytes(m, 4, "little")   # lda
        vm.smem.mem[48:52] = int.to_bytes(k, 4, "little")   # ldb
        vm.smem.mem[52:56] = int.to_bytes(m, 4, "little")   # ldc
        vm.smem.mem[56:60] = int.to_bytes(m, 4, "little")   # ldd
        vm.smem.mem[60:64] = struct.pack("f", float(alpha))
        vm.smem.mem[64:68] = struct.pack("f", float(beta))
        vm.smem.mem[68:72] = int.to_bytes(1, 4, "little")   # numWorkgroupX
        vm.smem.mem[72:76] = int.to_bytes(1, 4, "little")   # numWorkgroupY

        # Write matrices into vector memory in column-major order
        vm.vmem.mem[a_offset:a_offset + a_size] = bytearray(a_np.astype(np.float32).tobytes(order="F"))
        vm.vmem.mem[b_offset:b_offset + b_size] = bytearray(b_np.astype(np.float32).tobytes(order="F"))
        vm.vmem.mem[c_offset:c_offset + c_size] = bytearray(c_np.astype(np.float32).tobytes(order="F"))

        # Execute on virtual machine
        vm.run(context)

        # Read back result matrix D in column-major order
        d_bytes = bytes(vm.vmem.mem[d_offset:d_offset + d_size])
        d_out = np.frombuffer(d_bytes, dtype=np.float32).reshape((m, n), order="F")

        # Apply custom epilogue if registered
        if self._epilogue_fn is not None:
            d_out = self._epilogue_fn(d_out)

        return d_out
