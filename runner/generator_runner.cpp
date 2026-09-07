#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <numeric>
#include <map>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/device_functions.h>
#include <hip/hip_ext.h>
#include "Utils/KernelArguments.hpp"
#include "Utils/Math.hpp"
#include "Utils/BufferUtils.hpp"
#include "Utils/toml.hpp"

template<typename TA, typename TB, typename TC, typename TD>
void cpuGemmTyped(
    const TA *a, const TB *b, const TC *c, TD *d,
    float alpha, float beta,
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    bool transA = false, bool transB = false
) {
    const uint32_t lda = transA ? k : m;
    const uint32_t ldb = transB ? n : k;
    for (std::uint32_t i = 0; i < m; ++i) {
        for (std::uint32_t j = 0; j < n; ++j) {
            float acc{};

            for (std::uint32_t l = 0; l < k; ++l) {
                const float aVal = static_cast<float>(transA ? a[l + i * lda] : a[i + l * lda]);
                const float bVal = static_cast<float>(transB ? b[j + l * ldb] : b[l + j * ldb]);
                acc += aVal * bVal;
            }

            const auto dstIdx = i + j * m;
            d[dstIdx] = static_cast<TD>(beta * static_cast<float>(c[dstIdx]) + alpha * acc);
        }
    }
}

void cpuGemm(
    const float *a, const float *b, const float *c, float *d,
    float alpha, float beta,
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    bool transA = false, bool transB = false
) {
    cpuGemmTyped<float, float, float, float>(a, b, c, d, alpha, beta, m, n, k, transA, transB);
}

template<size_t TileM, size_t TileN, typename TA, typename TB, typename TC, typename TD>
__global__ void naiveGemm(
    const TA *a, const TB *b, const TC *c, TD *d,
    float alpha, float beta,
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    bool transA, bool transB) {
    const auto blockRow = blockIdx.x * TileM;
    const auto blockCol = blockIdx.y * TileN;
    const auto tId = threadIdx.x;
    const auto tRow = tId % TileM;
    const auto tCol = tId / TileM;
    const auto row = blockRow + tRow;
    const auto col = blockCol + tCol;
    if (row >= m || col >= n) return;

    const uint32_t lda = transA ? k : m;
    const uint32_t ldb = transB ? n : k;
    float acc{};

    for (uint32_t l = 0; l < k; ++l) {
        const float aVal = static_cast<float>(transA ? a[l + row * lda] : a[row + l * lda]);
        const float bVal = static_cast<float>(transB ? b[col + l * ldb] : b[l + col * ldb]);
        acc += aVal * bVal;
    }

    const uint64_t dstOffset = row + col * m;
    acc *= alpha;
    acc += beta * static_cast<float>(c[dstOffset]);
    d[dstOffset] = static_cast<TD>(acc);
}

template<typename TA = float, typename TB = float, typename TC = float, typename TD = float>
void launchGpuGemm(
    const TA *a, const TB *b, const TC *c, TD *d,
    float alpha, float beta,
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    bool transA = false, bool transB = false) {
    constexpr size_t TileM = 16;
    constexpr size_t TileN = 16;
    const auto numWgM = (m / TileM) + !!(m % TileM);
    const auto numWgN = (n / TileN) + !!(n % TileN);
    dim3 grid(numWgM, numWgN, 1);
    dim3 block(TileM * TileN, 1, 1);
    naiveGemm<TileM, TileN><<<grid, block>>>(a, b, c, d, alpha, beta, m, n, k, transA, transB);
}

hipError_t prepareASMKernel(const std::string &funcName, const std::string &coPath, hipModule_t *module, hipFunction_t *func) {
    auto err = hipModuleLoad(module, coPath.c_str());
    err = hipModuleGetFunction(func, *module, funcName.c_str());
    return err;
}

double gflops(uint32_t m, uint32_t n, uint32_t k, float durMs) {
    return 2.0 * m * n * k / durMs * 1e-6;
}

template<typename TA = float, typename TB = float, typename TCD = float>
float memBwGiB(size_t m, size_t n, size_t k, float timeMs) {
    size_t numBytes = m * k * sizeof(TA) + n * k * sizeof(TB) + 2 * m * n * sizeof(TCD);
    return numBytes / timeMs / 1024.f / 1024.f;
}

struct AsmKernelConfig {
    std::string name;
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
    int wgm;
};

AsmKernelConfig parseKernelConfigFromTable(const toml::table &tbl, const std::string &defaultName = "") {
    AsmKernelConfig config;
    if (auto nameVal = tbl["name"].value<std::string>()) {
        config.name = *nameVal;
    } else {
        config.name = defaultName;
    }
    config.aType = tbl["a_type"].value_or(0);
    config.bType = tbl["b_type"].value_or(0);
    config.cdType = tbl["cd_type"].value_or(0);
    config.scalarType = tbl["scalar_type"].value_or(0);

    if (auto *rawMfma = tbl["mfma"].as_array()) {
        std::get<0>(config.mfma) = rawMfma->get(0)->value_or(32);
        std::get<1>(config.mfma) = rawMfma->get(1)->value_or(32);
        std::get<2>(config.mfma) = rawMfma->get(2)->value_or(1);
        std::get<3>(config.mfma) = rawMfma->get(3)->value_or(2);
    }
    if (auto *rawWaveGroup = tbl["wave_group"].as_array()) {
        std::get<0>(config.waveGroup) = rawWaveGroup->get(0)->value_or(2);
        std::get<1>(config.waveGroup) = rawWaveGroup->get(1)->value_or(2);
    }
    if (auto *rawWaveTiling = tbl["wave_tiling"].as_array()) {
        std::get<0>(config.waveTiling) = rawWaveTiling->get(0)->value_or(2);
        std::get<1>(config.waveTiling) = rawWaveTiling->get(1)->value_or(2);
    }
    config.depthK = tbl["depth_k"].value_or(16);
    config.transA = tbl["trans_a"].value_or(false);
    config.transB = tbl["trans_b"].value_or(false);
    config.wgm = tbl["wgm"].value_or(1);
    config.wavefrontSize = tbl["wavefront_size"].value_or(64);
    config.ldsUsageBytes = tbl["lds_usage_bytes"].value_or(0);
    return config;
}

std::map<std::string, AsmKernelConfig> getAsmKernelConfigs(const std::string &path) {
    auto rawData = toml::parse_file(path);
    std::map<std::string, AsmKernelConfig> configs;
    if (auto *kernelsTable = rawData["kernels"].as_table()) {
        for (const auto &[key, node] : *kernelsTable) {
            if (auto *tbl = node.as_table()) {
                std::string kName(key.str());
                configs[kName] = parseKernelConfigFromTable(*tbl, kName);
            }
        }
    } else {
        std::string kName = "generated_gemm";
        if (auto n = rawData["name"].value<std::string>()) {
            if (!n->empty()) kName = *n;
        }
        configs[kName] = parseKernelConfigFromTable(rawData, kName);
    }
    return configs;
}

AsmKernelConfig getAsmKernelConfig(const std::string &path) {
    auto configs = getAsmKernelConfigs(path);
    if (configs.empty()) {
        throw std::runtime_error("No kernel configurations found in TOML: " + path);
    }
    return configs.begin()->second;
}

using AsmLaunchArgs = std::tuple<KernelArguments, int, int, int, int, int, int, int, int>;
AsmLaunchArgs makeKernelArguments(const AsmKernelConfig &config, const void *a, const void *b, const void *c, void *d, float alpha, float beta, uint32_t m, uint32_t n, uint32_t k) {
    const auto mt0 = std::get<0>(config.mfma) * std::get<0>(config.waveGroup) * std::get<0>(config.waveTiling);
    const auto mt1 = std::get<1>(config.mfma) * std::get<1>(config.waveGroup) * std::get<1>(config.waveTiling);
    const auto numWorkgroups0 = m / mt0 + !!(m % mt0);
    const auto numWorkgroups1 = n / mt1 + !!(n % mt1);
    const auto lda = config.transA ? k : m;
    const auto ldb = config.transB ? n : k;
    const auto ldc = m;
    const auto ldd = m;
    KernelArguments kArgs;
    kArgs.append(a);
    kArgs.append(b);
    kArgs.append(c);
    kArgs.append(d);
    kArgs.append(m);
    kArgs.append(n);
    kArgs.append(k);
    kArgs.append(lda);
    kArgs.append(ldb);
    kArgs.append(ldc);
    kArgs.append(ldd);
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

hipError_t launchASMKernel(hipFunction_t func, AsmKernelConfig &config, const void *a, const void *b, const void *c, void *d, float alpha, float beta, uint32_t m, uint32_t n, uint32_t k) {
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
    if (argc <= 8) {
        std::cerr << "Usage: " << argv[0] << " <coPath> <tomlPath> <m> <n> <k> <warmupRuns> <numRuns> <validation> [kernel_name|--all]\n";
        return -1;
    }

    auto allConfigs = getAsmKernelConfigs(argv[2]);
    std::string targetKernel = (argc > 9) ? argv[9] : "";
    if (targetKernel.empty()) {
        if (allConfigs.size() > 1) {
            targetKernel = "--all";
        } else if (!allConfigs.empty()) {
            targetKernel = allConfigs.begin()->first;
        }
    }

    hipError_t err{};
    hipModule_t mod;
    err = hipModuleLoad(&mod, argv[1]);
    if (err != hipSuccess) {
        std::cerr << "Failed to load module: " << argv[1] << '\n';
        return -1;
    }

    const uint32_t m = std::atoi(argv[3]);
    const uint32_t n = std::atoi(argv[4]);
    const uint32_t k = std::atoi(argv[5]);
    std::vector<float> cpuA(m * k, 1);
    std::vector<float> cpuB(k * n, 1);
    std::vector<float> cpuC(m * n, 0);
    std::vector<float> cpuD(m * n, 1);
    randomize(begin(cpuA), end(cpuA));
    randomize(begin(cpuB), end(cpuB));
    randomize(begin(cpuC), end(cpuC));
    float alpha{1.f};
    float beta{1.f};
    const uint32_t numRuns = std::atoi(argv[7]);
    const uint32_t numWarmupRuns = std::atoi(argv[6]);
    const bool validation = (std::atoi(argv[8]) != 0);

    bool hasFp16 = false;
    for (const auto &[kName, cfg] : allConfigs) {
        if (cfg.aType == 1) hasFp16 = true;
    }

    std::vector<__half> cpuA_half;
    std::vector<__half> cpuB_half;
    if (hasFp16) {
        cpuA_half.resize(m * k);
        cpuB_half.resize(k * n);
        for (size_t i = 0; i < m * k; ++i) {
            cpuA_half[i] = __float2half(cpuA[i]);
        }
        for (size_t i = 0; i < k * n; ++i) {
            cpuB_half[i] = __float2half(cpuB[i]);
        }
    }

    void *gpuA{};
    void *gpuB{};
    float *gpuC{};
    float *gpuD{};
    err = hipMalloc(&gpuA, m * k * sizeof(float));
    err = hipMalloc(&gpuB, n * k * sizeof(float));
    err = hipMalloc(&gpuC, m * n * sizeof(float));
    err = hipMalloc(&gpuD, m * n * sizeof(float));
    if (hasFp16) {
        err = hipMemcpyHtoD(gpuA, cpuA_half.data(), m * k * sizeof(__half));
        err = hipMemcpyHtoD(gpuB, cpuB_half.data(), n * k * sizeof(__half));
    } else {
        err = hipMemcpyHtoD(gpuA, cpuA.data(), m * k * sizeof(float));
        err = hipMemcpyHtoD(gpuB, cpuB.data(), n * k * sizeof(float));
    }
    err = hipMemcpyHtoD(gpuC, cpuC.data(), m * n * sizeof(float));
    hipEvent_t start, stop;
    err = hipEventCreate(&start);
    err = hipEventCreate(&stop);

    // Warmup & benchmark for HIP naive kernel
    for (uint32_t i = 0; i < numWarmupRuns; ++i) {
        if (hasFp16) {
            launchGpuGemm(static_cast<const __half*>(gpuA), static_cast<const __half*>(gpuB), gpuC, gpuD, alpha, beta, m, n, k);
        } else {
            launchGpuGemm(static_cast<const float*>(gpuA), static_cast<const float*>(gpuB), gpuC, gpuD, alpha, beta, m, n, k);
        }
    }
    err = hipDeviceSynchronize();
    err = hipEventRecord(start);
    for (uint32_t i = 0; i < numRuns; ++i) {
        if (hasFp16) {
            launchGpuGemm(static_cast<const __half*>(gpuA), static_cast<const __half*>(gpuB), gpuC, gpuD, alpha, beta, m, n, k);
        } else {
            launchGpuGemm(static_cast<const float*>(gpuA), static_cast<const float*>(gpuB), gpuC, gpuD, alpha, beta, m, n, k);
        }
    }
    err = hipEventRecord(stop);
    err = hipDeviceSynchronize();
    float dur{};
    err = hipEventElapsedTime(&dur, start, stop);
    float naiveBw = hasFp16 ? memBwGiB<__half, __half, float>(m, n, k, dur / numRuns)
                            : memBwGiB<float, float, float>(m, n, k, dur / numRuns);
    std::cout << "HIP gemm: " << dur / numRuns << " ms\n"
              << "Gflops: " << gflops(m, n, k, dur / numRuns) << '\n'
              << "GiB/s: " << naiveBw << '\n';

    size_t numMismatches{};

    if (targetKernel == "--all") {
        std::string bestKernel;
        float minDur = std::numeric_limits<float>::max();
        double maxGflops = 0.0;

        for (auto &[kName, cfg] : allConfigs) {
            hipFunction_t func;
            err = hipModuleGetFunction(&func, mod, kName.c_str());
            if (err != hipSuccess) {
                err = hipModuleGetFunction(&func, mod, "generated_gemm");
                if (err != hipSuccess) {
                    std::cerr << "Failed to get function for: " << kName << '\n';
                    continue;
                }
            }

            (void)hipMemset(gpuD, 0, sizeof(float) * m * n);
            auto asmKernArgs = makeKernelArguments(cfg, gpuA, gpuB, gpuC, gpuD, alpha, beta, m, n, k);
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

            float kernelDur{};
            err = hipEventElapsedTime(&kernelDur, start, stop);
            float avgMs = kernelDur / numRuns;
            double gf = gflops(m, n, k, avgMs);
            std::cout << "[BATCH_BENCH] " << kName << " | " << avgMs << " ms | " << gf << " Gflops\n";

            if (validation) {
                if (cfg.aType == 1) {
                    cpuGemmTyped(cpuA_half.data(), cpuB_half.data(), cpuC.data(), cpuD.data(), alpha, beta, m, n, k, cfg.transA, cfg.transB);
                } else {
                    cpuGemmTyped(cpuA.data(), cpuB.data(), cpuC.data(), cpuD.data(), alpha, beta, m, n, k, cfg.transA, cfg.transB);
                }
                std::vector<float> gpuResult(m * n, 0);
                err = hipMemcpyDtoH(gpuResult.data(), gpuD, m * n * sizeof(float));
                float atol = (cfg.aType == 1) ? 1e-2f : 1e-3f;
                size_t kMismatches = 0;
                for (size_t i = 0; i < gpuResult.size(); ++i) {
                    if (!almostEqual(gpuResult[i], cpuD[i], atol)) {
                        ++kMismatches;
                    }
                }
                if (kMismatches > 0) {
                    std::cout << "  [VALIDATION FAILED] " << kName << ": " << kMismatches << " mismatches\n";
                    numMismatches += kMismatches;
                } else {
                    std::cout << "  [VALIDATION PASSED] " << kName << "\n";
                }
            }

            if (avgMs < minDur) {
                minDur = avgMs;
                maxGflops = gf;
                bestKernel = kName;
            }
        }
        std::cout << "\n[BEST] " << bestKernel << " | " << minDur << " ms | " << maxGflops << " Gflops\n";
    } else {
        // Single kernel execution
        AsmKernelConfig gemmConfig;
        if (allConfigs.count(targetKernel)) {
            gemmConfig = allConfigs[targetKernel];
        } else if (!allConfigs.empty()) {
            gemmConfig = allConfigs.begin()->second;
        }

        hipFunction_t func;
        err = hipModuleGetFunction(&func, mod, targetKernel.c_str());
        if (err != hipSuccess) {
            err = hipModuleGetFunction(&func, mod, "generated_gemm");
            if (err != hipSuccess) {
                std::cerr << "Failed to find kernel function: " << targetKernel << '\n';
                return -1;
            }
        }

        (void)hipMemset(gpuD, 0, sizeof(float) * m * n);
        auto asmKernArgs = makeKernelArguments(gemmConfig, gpuA, gpuB, gpuC, gpuD, alpha, beta, m, n, k);
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
        float bw = (gemmConfig.aType == 1) ? memBwGiB<__half, __half, float>(m, n, k, dur / numRuns)
                                           : memBwGiB<float, float, float>(m, n, k, dur / numRuns);
        std::cout << "ASM gemm: " << dur / numRuns << " ms\n"
                  << "Gflops: " << gflops(m, n, k, dur / numRuns) << '\n'
                  << "GiB/s: " << bw << '\n';

        if (validation) {
            if (gemmConfig.aType == 1) {
                cpuGemmTyped(cpuA_half.data(), cpuB_half.data(), cpuC.data(), cpuD.data(), alpha, beta, m, n, k, gemmConfig.transA, gemmConfig.transB);
            } else {
                cpuGemmTyped(cpuA.data(), cpuB.data(), cpuC.data(), cpuD.data(), alpha, beta, m, n, k, gemmConfig.transA, gemmConfig.transB);
            }
            std::vector<float> gpuResult(m * n, 0);
            err = hipMemcpyDtoH(gpuResult.data(), gpuD, m * n * sizeof(float));
            float atol = (gemmConfig.aType == 1) ? 1e-2f : 1e-3f;
            for (size_t i = 0; i < gpuResult.size(); ++i) {
                if (!almostEqual(gpuResult[i], cpuD[i], atol)) {
                    if (numMismatches < 10) {
                        std::cout << "gpu & cpu results mismatched at index: " << i << '\n';
                        std::cout << gpuResult[i] << " != " << cpuD[i] << '\n';
                    }
                    ++numMismatches;
                }
            }
            std::cout << "# of mismatches: " << numMismatches << "/" << m * n << '\n';
        }
    }

    err = hipEventDestroy(start);
    err = hipEventDestroy(stop);
    err = hipModuleUnload(mod);
    err = hipFree(gpuA);
    err = hipFree(gpuB);
    err = hipFree(gpuC);
    err = hipFree(gpuD);
    return numMismatches ? -1 : 0;
}