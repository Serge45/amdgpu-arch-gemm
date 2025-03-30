#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <numeric>
#include <hip/hip_runtime.h>
#include <hip/device_functions.h>
#include <hip/hip_ext.h>
#include "Utils/KernelArguments.hpp"
#include "Utils/Math.hpp"
#include "Utils/BufferUtils.hpp"
#include "Utils/toml.hpp"

void cpuGemm(
    const float *a, const float *b, const float *c, float *d,
    float alpha, float beta,
    std::uint32_t m, std::uint32_t n, std::uint32_t k
) {
    for (std::uint32_t i = 0; i < m; ++i) {
        for (std::uint32_t j = 0; j < n; ++j) {
            float acc{};

            for (std::uint32_t l = 0; l < k; ++l) {
                acc += a[i + l * m] * b[l + j * k];
            }

            const auto dstIdx = i + j * m;
            d[dstIdx] = beta * c[dstIdx] + alpha * acc;
        }
    }
}

template<size_t TileM, size_t TileN>
__global__ void naiveGemm(
    const float *a, const float *b, const float *c, float *d,
    float alpha, float beta,
    std::uint32_t m, std::uint32_t n, std::uint32_t k) {
    const auto blockRow = blockIdx.x * TileM;
    const auto blockCol = blockIdx.y * TileN;
    const auto blockOffset = blockCol * m + blockRow;
    const auto tId = threadIdx.x;
    const auto tRow = tId % TileM;
    const auto tCol = tId / TileM;
    float acc{};

    for (uint32_t i = 0; i < k; ++i) {
        acc += a[blockRow + tRow + m * i] * b[i + (tCol + blockCol) * k];
    }

    const uint64_t dstOffset = blockRow + tRow + (blockCol + tCol) * m;
    acc *= alpha;
    acc += beta * c[dstOffset];
    d[dstOffset] = acc;
}

void launchGpuGemm(
    const float *a, const float *b, const float *c, float *d,
    float alpha, float beta,
    std::uint32_t m, std::uint32_t n, std::uint32_t k) {
    constexpr size_t TileM = 16;
    constexpr size_t TileN = 16;
    const auto numWgM = (m / TileM) + !!(m % TileM);
    const auto numWgN = (n / TileN) + !!(n % TileN);
    naiveGemm<TileM, TileN><<<dim3(numWgM, numWgN, 1), 256>>>(a, b, c, d, alpha, beta, m, n, k);
}

hipError_t prepareASMKernel(const std::string &funcName, const std::string &coPath, hipModule_t *module, hipFunction_t *func) {
    auto err = hipModuleLoad(module, coPath.c_str());
    err = hipModuleGetFunction(func, *module, funcName.c_str());
    return err;
}

double gflops(uint32_t m, uint32_t n, uint32_t k, float durMs) {
    return 2.0 * m * n * k / durMs * 1e-6;
}

template<typename T>
float memBwGiB(size_t m, size_t n, size_t k, float timeMs) {
    constexpr size_t numBytes = sizeof(T);
    return (m * k + n * k + 2 * m * n) * numBytes / timeMs / 1024.f / 1024.f;
}

struct AsmKernelConfig {
    int aType;
    int bType;
    int cdType;
    int scalarType;
    std::tuple<int, int, int, int> mfma;
    std::tuple<int, int> waveGroup;
    std::tuple<int, int> waveTiling;
    int depthK;
    int wavefrontSize;
    int ldsUsageBytes;
    bool transA;
    bool transB;
};

AsmKernelConfig getAsmKernelConfig(const std::string &path) {
    auto rawData = toml::parse_file(path);
    AsmKernelConfig config;
    config.aType = **rawData.at("a_type").as<int64_t>();
    config.bType = **rawData.at("b_type").as<int64_t>();
    config.cdType = **rawData.at("cd_type").as<int64_t>();
    config.scalarType = **rawData.at("scalar_type").as<int64_t>();
    auto *rawMfma = rawData.at("mfma").as_array();
    std::get<0>(config.mfma) = **rawMfma->at(0).as<int64_t>();
    std::get<1>(config.mfma) = **rawMfma->at(1).as<int64_t>();
    std::get<2>(config.mfma) = **rawMfma->at(2).as<int64_t>();
    std::get<3>(config.mfma) = **rawMfma->at(3).as<int64_t>();
    auto *rawWaveGroup = rawData.at("wave_group").as_array();
    std::get<0>(config.waveGroup) = **rawWaveGroup->at(0).as<int64_t>();
    std::get<1>(config.waveGroup) = **rawWaveGroup->at(1).as<int64_t>();
    auto *rawWaveTiling = rawData.at("wave_tiling").as_array();
    std::get<0>(config.waveTiling) = **rawWaveTiling->at(0).as<int64_t>();
    std::get<1>(config.waveTiling) = **rawWaveTiling->at(1).as<int64_t>();
    config.depthK = **rawData.at("depth_k").as<int64_t>();
    config.transA = **rawData.at("trans_a").as_boolean();
    config.transB = **rawData.at("trans_b").as_boolean();
    config.wavefrontSize = **rawData.at("wavefront_size").as<int64_t>();
    config.ldsUsageBytes = **rawData.at("lds_usage_bytes").as<int64_t>();
    return config;
}

using AsmLaunchArgs = std::tuple<KernelArguments, int, int, int, int, int, int, int, int>;
 AsmLaunchArgs makeKernelArguments(const AsmKernelConfig &config, const float *a, const float *b, const float *c, float *d, float alpha, float beta, uint32_t m, uint32_t n, uint32_t k) {
    const auto mt0 = std::get<0>(config.mfma) * std::get<0>(config.waveGroup) * std::get<0>(config.waveTiling);
    const auto mt1 = std::get<1>(config.mfma) * std::get<1>(config.waveGroup) * std::get<1>(config.waveTiling);
    const auto numWorkgroups0 = m / mt0 + !!(m % mt0);
    const auto numWorkgroups1 = n / mt1 + !!(n % mt1);
    KernelArguments kArgs;
    kArgs.append(a);
    kArgs.append(b);
    kArgs.append(c);
    kArgs.append(d);
    kArgs.append(m);
    kArgs.append(n);
    kArgs.append(k);
    kArgs.append(m);
    kArgs.append(k);
    kArgs.append(m);
    kArgs.append(m);
    kArgs.append(alpha);
    kArgs.append(beta);
    kArgs.append<int32_t>(numWorkgroups0);
    kArgs.append<int32_t>(numWorkgroups1);
    kArgs.applyAlignment();
    const auto ldsUsageBytes = config.ldsUsageBytes;
    const auto numWaves = std::get<0>(config.waveGroup) * std::get<1>(config.waveGroup);
    const auto numWorkitems = numWaves * config.wavefrontSize;
    return {kArgs, mt0, mt1, config.depthK, numWorkgroups0, numWorkgroups1, ldsUsageBytes, numWaves, numWorkitems};
}

hipError_t launchASMKernel(hipFunction_t func, AsmKernelConfig &config, const float *a, const float *b, const float *c, float *d, float alpha, float beta, uint32_t m, uint32_t n, uint32_t k) {
    auto launchArgs = makeKernelArguments(config, a, b, c, d, alpha, beta, m, n, k);
    const auto ldsUsageBytes = std::get<6>(launchArgs);
    const auto numWaves = std::get<7>(launchArgs);
    const auto numWorkitems = std::get<8>(launchArgs);
    const auto numWorkgroups0 = std::get<4>(launchArgs);
    const auto numWorkgroups1 = std::get<5>(launchArgs);
    auto &kArgs = std::get<0>(launchArgs);
    std::size_t argSize = kArgs.size();
    void *args[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kArgs.buffer(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &argSize,
        HIP_LAUNCH_PARAM_END
    };
    return hipExtModuleLaunchKernel(func, numWorkgroups0 * numWorkitems, numWorkgroups1, 1, numWorkitems, 1, 1, ldsUsageBytes, nullptr, nullptr, args);
}

hipError_t launchASMKernel(hipFunction_t func, AsmLaunchArgs &launchArgs) {
    const auto ldsUsageBytes = std::get<6>(launchArgs);
    const auto numWaves = std::get<7>(launchArgs);
    const auto numWorkitems = std::get<8>(launchArgs);
    const auto numWorkgroups0 = std::get<4>(launchArgs);
    const auto numWorkgroups1 = std::get<5>(launchArgs);
    auto &kArgs = std::get<0>(launchArgs);
    std::size_t argSize = kArgs.size();
    void *args[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kArgs.buffer(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &argSize,
        HIP_LAUNCH_PARAM_END
    };
    return hipExtModuleLaunchKernel(func, numWorkgroups0 * numWorkitems, numWorkgroups1, 1, numWorkitems, 1, 1, 0, nullptr, nullptr, args);
}

int main(int argc, char **argv) {
    auto gemmConfig = getAsmKernelConfig(argv[2]);
    hipError_t err{};
    hipModule_t mod;
    hipFunction_t func;
    err = prepareASMKernel("generated_gemm", argv[1], &mod, &func);

    if (argc <= 8) {
        return -1;
    }

    const uint32_t m = std::atoi(argv[3]);
    const uint32_t n = std::atoi(argv[4]);
    const uint32_t k = std::atoi(argv[5]);
    std::vector<float> cpuA(m * k, 1);
    std::vector<float> cpuB(k * n, 1);
    std::vector<float> cpuC(m * n, 0);
    std::vector<float> cpuD(m * n, 1);
    // std::iota(begin(cpuA), end(cpuA), 0.f);
    // std::iota(begin(cpuB), end(cpuB), 0.f);
    // std::iota(begin(cpuC), end(cpuC), 0.f);
    // toIdentity(cpuA.data(), m, k);
    // toIdentity(cpuB.data(), k, n);
    // toIdentity(cpuC.data(), m, n);
    randomize(begin(cpuA), end(cpuA));
    randomize(begin(cpuB), end(cpuB));
    randomize(begin(cpuC), end(cpuC));
    float alpha{1.f};
    float beta{1.f};
    const uint32_t numRuns = std::atoi(argv[7]);
    const uint32_t numWarmupRuns = std::atoi(argv[6]);
    const bool validation = (std::atoi(argv[8]) != 0);

    if (validation) {
        auto cpuBeg = std::chrono::steady_clock::now();
        cpuGemm(cpuA.data(), cpuB.data(), cpuC.data(), cpuD.data(), alpha, beta, m, n, k);
        auto cpuEnd = std::chrono::steady_clock::now();
        std::cout << "cpuGemm func: " << std::chrono::duration<float, std::milli>(cpuEnd - cpuBeg).count() / numRuns << " ms\n";
    }

    float *gpuA{};
    float *gpuB{};
    float *gpuC{};
    float *gpuD{};
    err = hipMalloc(&gpuA, m * k * sizeof(float));
    err = hipMalloc(&gpuB, n * k * sizeof(float));
    err = hipMalloc(&gpuC, m * n * sizeof(float));
    err = hipMalloc(&gpuD, m * n * sizeof(float));
    err = hipMemcpyHtoD(gpuA, cpuA.data(), m * k * sizeof(float));
    err = hipMemcpyHtoD(gpuB, cpuB.data(), n * k * sizeof(float));
    err = hipMemcpyHtoD(gpuC, cpuC.data(), m * n * sizeof(float));
    hipEvent_t start, stop;
    err = hipEventCreate(&start);
    err = hipEventCreate(&stop);
    //warmup for HIP kernel
    for (uint32_t i = 0; i < numWarmupRuns; ++i) {
        launchGpuGemm(gpuA, gpuB, gpuC, gpuD, alpha, beta, m, n, k);
    }
    err = hipDeviceSynchronize();

    err = hipEventRecord(start);

    for (uint32_t i = 0; i < numRuns; ++i) {
        launchGpuGemm(gpuA, gpuB, gpuC, gpuD, alpha, beta, m, n, k);
    }

    err = hipEventRecord(stop);
    err = hipDeviceSynchronize();

    float dur{};
    err = hipEventElapsedTime(&dur, start, stop);
    std::cout << "HIP gemm: " << dur / numRuns << " ms\n"
              << "Gflops: " << gflops(m, n, k, dur / numRuns) << '\n'
              << "GiB/s: " << memBwGiB<float>(m, n, k, dur / numRuns) << '\n';

    (void)hipMemset(gpuD, 0, sizeof(float) * m * n);
    auto asmKernArgs = makeKernelArguments(gemmConfig, gpuA, gpuB, gpuC, gpuD, alpha, beta, m, n, k);
    //warmup
    for (uint32_t i = 0; i < numWarmupRuns; ++i) {
        (void)launchASMKernel(func, asmKernArgs);
    }
    err = hipDeviceSynchronize();
    err = hipEventRecord(start);
    for (uint32_t i = 0; i < numRuns; ++i) {
        (void)launchASMKernel(func, asmKernArgs);
    }
    err = hipEventRecord(stop);
    err = hipDeviceSynchronize();
    err = hipEventElapsedTime(&dur, start, stop);
    std::cout << "ASM gemm: " << dur / numRuns << " ms\n"
              << "Gflops: " << gflops(m, n, k, dur / numRuns) << '\n'
              << "GiB/s: " << memBwGiB<float>(m, n, k, dur / numRuns) << '\n';

    err = hipEventDestroy(start);
    err = hipEventDestroy(stop);
    size_t numMismatches{};

    if (validation) {
        std::vector<float> gpuResult(m * n, 0);
        err = hipMemcpyDtoH(gpuResult.data(), gpuD, m * n * sizeof(float));


        for (size_t i = 0; i < gpuResult.size(); ++i) {
            if (!almostEqual(gpuResult[i], cpuD[i], 1e-3f)) {
                if (numMismatches < 10) {
                    std::cout << "gpu & cpu results mismatched at index: " << i << '\n';
                    std::cout << gpuResult[i] << " != " << cpuD[i] << '\n';
                }
                ++numMismatches;
            }
        }

        std::cout << "# of mismatches: " << numMismatches << "/" << m * n << '\n';

        if (numMismatches) {
            // std::cout << "A:\n"
            // printMultiDim(cpuA.data(), m, k);
            // std::cout << "B:\n";
            // printMultiDim(cpuB.data(), k, n);
            // std::cout << "Ref:\n";
            // printMultiDim(cpuD.data(), m, n);
            // std::cout << "Actual:\n";
            // printMultiDim(gpuResult.data(), m, n);
        }
    }

    err = hipModuleUnload(mod);
    err = hipFree(gpuA);
    err = hipFree(gpuB);
    err = hipFree(gpuC);
    err = hipFree(gpuD);
    return numMismatches ? -1 : 0;
}