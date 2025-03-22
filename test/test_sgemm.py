from generator import generator as g


def test_basic_sgemm_gen():
    gemm_config = g.GemmSolutionConfig(
        g.DataType.FP32,
        g.DataType.FP32,
        g.DataType.FP32,
        g.DataType.FP32,
        (16, 16, 1, 4),
        (2, 2),
        (2, 2),
        16,
        False,
        False,
    )

    opt = g.GemmOptimizations(1)
    asm = g.gemm(
        None,
        "gemm",
        "gfx90a:xnack-",
        gemm_config,
        opt,
        [
            g.FunctionArgument("global_buffer", "a", None, 8),
            g.FunctionArgument("global_buffer", "b", None, 8),
            g.FunctionArgument("global_buffer", "c", None, 8),
            g.FunctionArgument("global_buffer", "d", None, 8),
            g.FunctionArgument("by_value", "m", None, 4),
            g.FunctionArgument("by_value", "n", None, 4),
            g.FunctionArgument("by_value", "k", None, 4),
            g.FunctionArgument("by_value", "lda", None, 4),
            g.FunctionArgument("by_value", "ldb", None, 4),
            g.FunctionArgument("by_value", "ldc", None, 4),
            g.FunctionArgument("by_value", "ldd", None, 4),
            g.FunctionArgument("by_value", "alpha", None, 4),
            g.FunctionArgument("by_value", "beta", None, 4),
            g.FunctionArgument("by_value", "numWorkgroupX", None, 4),
            g.FunctionArgument("by_value", "numWorkgroupY", None, 4),
        ],
    )

    assert len(asm)
