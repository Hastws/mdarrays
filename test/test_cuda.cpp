// 简单的 CUDA 功能测试
#include <iostream>
#include <cmath>
#include <chrono>

#ifdef USE_CUDA
#include "backend/cuda_common.h"
#include "backend/cuda_kernels.h"
#endif

#include "backend/simd_kernel.h"

using namespace Autoalg;

void test_cuda_gemm() {
#ifdef USE_CUDA
    std::cout << "=== CUDA GEMM Test ===" << std::endl;
    
    // 打印设备信息
    CUDA::PrintDeviceInfo();
    
    const size_t M = 512, K = 512, N = 512;
    const size_t size_A = M * K;
    const size_t size_B = K * N;
    const size_t size_C = M * N;
    
    // 分配 host 内存
    double* h_A = new double[size_A];
    double* h_B = new double[size_B];
    double* h_C_cpu = new double[size_C];
    double* h_C_gpu = new double[size_C];
    
    // 初始化数据
    for (size_t i = 0; i < size_A; ++i) h_A[i] = (double)(i % 100) / 100.0;
    for (size_t i = 0; i < size_B; ++i) h_B[i] = (double)(i % 100) / 100.0;
    
    // CPU 计算
    std::cout << "\nCPU GEMM..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    SIMD::gemm_naive(h_C_cpu, h_A, h_B, M, K, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    
    // 分配 device 内存
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(double)));
    
    // 复制数据到 GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(double), cudaMemcpyHostToDevice));
    
    // GPU 预热
    CUDA::gemm(d_C, d_A, d_B, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // GPU 计算
    std::cout << "\nGPU GEMM..." << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    CUDA::gemm(d_C, d_A, d_B, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    
    // 复制结果回 CPU
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C * sizeof(double), cudaMemcpyDeviceToHost));
    
    // 验证结果
    double max_diff = 0.0;
    for (size_t i = 0; i < size_C; ++i) {
        double diff = std::abs(h_C_cpu[i] - h_C_gpu[i]);
        max_diff = std::max(max_diff, diff);
    }
    std::cout << "\nMax difference between CPU and GPU: " << max_diff << std::endl;
    
    if (max_diff < 1e-6) {
        std::cout << "✓ Results match!" << std::endl;
    } else {
        std::cout << "✗ Results differ!" << std::endl;
    }
    
    std::cout << "\nSpeedup: " << cpu_time / gpu_time << "x" << std::endl;
    
    // 清理
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
#else
    std::cout << "CUDA not enabled. Compile with -DUSE_CUDA=ON" << std::endl;
#endif
}

void test_cuda_elementwise() {
#ifdef USE_CUDA
    std::cout << "\n=== CUDA Elementwise Test ===" << std::endl;
    
    const size_t N = 1000000;
    
    double* h_a = new double[N];
    double* h_b = new double[N];
    double* h_c_cpu = new double[N];
    double* h_c_gpu = new double[N];
    
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = (double)i / N;
        h_b[i] = (double)(N - i) / N;
    }
    
    // CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    SIMD::add(h_c_cpu, h_a, h_b, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU
    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));
    
    // 预热
    CUDA::add(d_c, d_a, d_b, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    CUDA::add(d_c, d_a, d_b, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // 验证
    double max_diff = 0.0;
    for (size_t i = 0; i < N; ++i) {
        max_diff = std::max(max_diff, std::abs(h_c_cpu[i] - h_c_gpu[i]));
    }
    
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "Max diff: " << max_diff << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
#endif
}

int main() {
    std::cout << "CUDA Backend Test" << std::endl;
    std::cout << "=================" << std::endl;
    
    test_cuda_gemm();
    test_cuda_elementwise();
    
    return 0;
}
