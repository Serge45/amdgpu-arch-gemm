#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
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

#define CHECK_HIPBLAS(cmd) do { \
    hipblasStatus_t status = cmd; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLAS error: " << status << " at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

void bench_hipblas(int M, int N, int K, int warmup, int iters) {
    hipblasHandle_t handle;
    CHECK_HIPBLAS(hipblasCreate(&handle));

    size_t sizeA = (size_t)M * K * sizeof(hipblasHalf);
    size_t sizeB = (size_t)K * N * sizeof(hipblasHalf);
    size_t sizeC = (size_t)M * N * sizeof(hipblasHalf);

    hipblasHalf *dA, *dB, *dC;
    CHECK_HIP(hipMalloc(&dA, sizeA));
    CHECK_HIP(hipMalloc(&dB, sizeB));
    CHECK_HIP(hipMalloc(&dC, sizeC));

    CHECK_HIP(hipMemset(dA, 0x3c, sizeA));
    CHECK_HIP(hipMemset(dB, 0x3c, sizeB));
    CHECK_HIP(hipMemset(dC, 0, sizeC));

    const hipblasHalf alpha = 1.0f;
    const hipblasHalf beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        CHECK_HIPBLAS(hipblasHgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                  M, N, K,
                                  &alpha,
                                  dA, M,
                                  dB, K,
                                  &beta,
                                  dC, M));
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_HIPBLAS(hipblasHgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                  M, N, K,
                                  &alpha,
                                  dA, M,
                                  dB, K,
                                  &beta,
                                  dC, M));
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    ms /= iters;

    double tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12;

    std::cout << "[hipblasHgemm] M=" << M << " N=" << N << " K=" << K
              << " | Time: " << std::fixed << std::setprecision(4) << ms << " ms"
              << " | " << std::setprecision(2) << tflops << " TFLOPS" << std::endl;

    CHECK_HIP(hipFree(dA));
    CHECK_HIP(hipFree(dB));
    CHECK_HIP(hipFree(dC));
    CHECK_HIPBLAS(hipblasDestroy(handle));
}

int main() {
    int deviceId = 0;
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, deviceId));
    std::cout << "Device: " << prop.name << " (Arch: " << prop.gcnArchName << ", CUs: " << prop.multiProcessorCount << ")" << std::endl;

    bench_hipblas(4096, 4096, 4096, 50, 200);
    bench_hipblas(8192, 8192, 8192, 50, 200);
    return 0;
}
