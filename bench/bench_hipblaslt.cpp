#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CHECK_HIP(cmd) do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_HIPBLASLT(cmd) do { \
    hipblasStatus_t status = cmd; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLASLt error: " << status << " at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

void bench_hipblaslt(int M, int N, int K, int warmup, int iters) {
    hipblasLtHandle_t handle;
    CHECK_HIPBLASLT(hipblasLtCreate(&handle));

    size_t sizeA = (size_t)M * K * sizeof(uint16_t);
    size_t sizeB = (size_t)K * N * sizeof(uint16_t);
    size_t sizeC = (size_t)M * N * sizeof(uint16_t);

    void *dA, *dB, *dC, *dD;
    CHECK_HIP(hipMalloc(&dA, sizeA));
    CHECK_HIP(hipMalloc(&dB, sizeB));
    CHECK_HIP(hipMalloc(&dC, sizeC));
    CHECK_HIP(hipMalloc(&dD, sizeC));

    CHECK_HIP(hipMemset(dA, 0x3c, sizeA));
    CHECK_HIP(hipMemset(dB, 0x3c, sizeB));
    CHECK_HIP(hipMemset(dC, 0, sizeC));
    CHECK_HIP(hipMemset(dD, 0, sizeC));

    hipblasLtMatmulDesc_t matmulDesc;
    CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t opA = HIPBLAS_OP_N;
    hipblasOperation_t opB = HIPBLAS_OP_N;
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16F, M, K, M));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16F, K, N, K));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16F, M, N, M));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_16F, M, N, M));

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&pref));
    size_t workspaceSize = 32 * 1024 * 1024;
    void *workspace;
    CHECK_HIP(hipMalloc(&workspace, workspaceSize));
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    const int requestSolutions = 16;
    hipblasLtMatmulHeuristicResult_t heuristicResults[requestSolutions];
    int returnedSolutions = 0;
    CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutA, layoutB, layoutC, layoutD, pref, requestSolutions, heuristicResults, &returnedSolutions));

    if (returnedSolutions == 0) {
        std::cerr << "No hipblasLt algorithm found!" << std::endl;
        return;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Find the best heuristic solution
    int bestIdx = 0;
    float bestMs = 1e9f;

    for (int s = 0; s < returnedSolutions; ++s) {
        if (heuristicResults[s].state != HIPBLAS_STATUS_SUCCESS) continue;

        // Warmup
        for (int i = 0; i < 5; ++i) {
            CHECK_HIPBLASLT(hipblasLtMatmul(handle, matmulDesc,
                                           &alpha, dA, layoutA,
                                           dB, layoutB,
                                           &beta, dC, layoutC,
                                           dD, layoutD,
                                           &heuristicResults[s].algo,
                                           workspace, workspaceSize, 0));
        }
        CHECK_HIP(hipDeviceSynchronize());

        hipEvent_t start, stop;
        CHECK_HIP(hipEventCreate(&start));
        CHECK_HIP(hipEventCreate(&stop));

        CHECK_HIP(hipEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            CHECK_HIPBLASLT(hipblasLtMatmul(handle, matmulDesc,
                                           &alpha, dA, layoutA,
                                           dB, layoutB,
                                           &beta, dC, layoutC,
                                           dD, layoutD,
                                           &heuristicResults[s].algo,
                                           workspace, workspaceSize, 0));
        }
        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
        ms /= 20;
        if (ms < bestMs) {
            bestMs = ms;
            bestIdx = s;
        }
        CHECK_HIP(hipEventDestroy(start));
        CHECK_HIP(hipEventDestroy(stop));
    }

    // Benchmark the champion solution with full warmup and iters
    for (int i = 0; i < warmup; ++i) {
        CHECK_HIPBLASLT(hipblasLtMatmul(handle, matmulDesc,
                                       &alpha, dA, layoutA,
                                       dB, layoutB,
                                       &beta, dC, layoutC,
                                       dD, layoutD,
                                       &heuristicResults[bestIdx].algo,
                                       workspace, workspaceSize, 0));
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_HIPBLASLT(hipblasLtMatmul(handle, matmulDesc,
                                       &alpha, dA, layoutA,
                                       dB, layoutB,
                                       &beta, dC, layoutC,
                                       dD, layoutD,
                                       &heuristicResults[bestIdx].algo,
                                       workspace, workspaceSize, 0));
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    ms /= iters;

    double tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12;

    std::cout << "[hipblasLtMatmul] M=" << M << " N=" << N << " K=" << K
              << " | Time: " << std::fixed << std::setprecision(4) << ms << " ms"
              << " | " << std::setprecision(2) << tflops << " TFLOPS" << std::endl;

    CHECK_HIP(hipFree(dA));
    CHECK_HIP(hipFree(dB));
    CHECK_HIP(hipFree(dC));
    CHECK_HIP(hipFree(dD));
    CHECK_HIP(hipFree(workspace));
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(layoutA));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(layoutB));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(layoutC));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(layoutD));
    CHECK_HIPBLASLT(hipblasLtMatmulDescDestroy(matmulDesc));
    CHECK_HIPBLASLT(hipblasLtDestroy(handle));
}

int main() {
    bench_hipblaslt(4096, 4096, 4096, 50, 200);
    bench_hipblaslt(8192, 8192, 8192, 50, 200);
    return 0;
}
