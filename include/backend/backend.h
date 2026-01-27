#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_BACKEND_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_BACKEND_H

#include "backend/device.h"
#include "backend/simd_kernel.h"

#ifdef USE_CUDA
#include "backend/cuda_common.h"
#include "backend/cuda_kernels.h"
#include "backend/cuda_storage.h"
#endif

namespace Autoalg {
namespace Backend {

//=============================================================================
// 统一的后端调度接口
// 根据当前设备自动选择 CPU/CUDA 实现
//=============================================================================

// 填充零
inline void FillZero(double* dst, size_t n, DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::fill_zero(dst, n);
        return;
    }
#endif
    SIMD::fill_zero(dst, n);
}

// 填充常数
inline void FillConst(double* dst, double val, size_t n, DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::fill_const(dst, val, n);
        return;
    }
#endif
    SIMD::fill_const(dst, val, n);
}

// 复制
inline void Copy(double* dst, const double* src, size_t n, DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::copy(dst, src, n);
        return;
    }
#endif
    SIMD::copy(dst, src, n);
}

// 加法
inline void Add(double* dst, const double* a, const double* b, size_t n, 
                DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::add(dst, a, b, n);
        return;
    }
#endif
    SIMD::add(dst, a, b, n);
}

// 原地加法
inline void AddInplace(double* dst, const double* src, size_t n,
                       DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::add_inplace(dst, src, n);
        return;
    }
#endif
    SIMD::add_inplace(dst, src, n);
}

// 减法
inline void Sub(double* dst, const double* a, const double* b, size_t n,
                DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::sub(dst, a, b, n);
        return;
    }
#endif
    SIMD::sub(dst, a, b, n);
}

// 逐元素乘法
inline void Mul(double* dst, const double* a, const double* b, size_t n,
                DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::mul(dst, a, b, n);
        return;
    }
#endif
    SIMD::mul(dst, a, b, n);
}

// 标量乘法
inline void Scale(double* dst, const double* a, double scalar, size_t n,
                  DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::scale(dst, a, scalar, n);
        return;
    }
#endif
    SIMD::scale(dst, a, scalar, n);
}

// 取负
inline void Negate(double* dst, const double* src, size_t n,
                   DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::negate(dst, src, n);
        return;
    }
#endif
    SIMD::negate(dst, src, n);
}

// ReLU
inline void ReLU(double* dst, const double* src, size_t n,
                 DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::relu(dst, src, n);
        return;
    }
#endif
    SIMD::relu(dst, src, n);
}

// ReLU 反向
inline void ReLUBackward(double* dst, const double* grad, const double* src, size_t n,
                         DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::relu_backward(dst, grad, src, n);
        return;
    }
#endif
    SIMD::relu_backward(dst, grad, src, n);
}

// 求和
inline double Sum(const double* src, size_t n,
                  DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        return CUDA::sum(src, n);
    }
#endif
    return SIMD::sum(src, n);
}

// 点积
inline double Dot(const double* a, const double* b, size_t n,
                  DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        return CUDA::dot(a, b, n);
    }
#endif
    return SIMD::dot(a, b, n);
}

// 矩阵乘法 C[M,N] = A[M,K] * B[K,N]
inline void Gemm(double* C, const double* A, const double* B,
                 size_t M, size_t K, size_t N,
                 DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::gemm(C, A, B, M, K, N);
        return;
    }
#endif
    SIMD::gemm_naive(C, A, B, M, K, N);
}

// 批量矩阵乘法
inline void BatchedGemm(double* C, const double* A, const double* B,
                        size_t batch, size_t M, size_t K, size_t N,
                        DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::batched_gemm(C, A, B, batch, M, K, N);
        return;
    }
#endif
    // CPU 回退：逐个 batch 计算
    for (size_t b = 0; b < batch; ++b) {
        SIMD::gemm_naive(C + b * M * N, 
                         A + b * M * K, 
                         B + b * K * N, 
                         M, K, N);
    }
}

// Softmax
inline void Softmax(double* dst, const double* src, size_t batch, size_t dim,
                    DeviceType device = Device::GetDevice()) {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) {
        CUDA::softmax(dst, src, batch, dim);
        return;
    }
#endif
    // CPU 实现
    for (size_t b = 0; b < batch; ++b) {
        const double* src_row = src + b * dim;
        double* dst_row = dst + b * dim;
        
        // 找最大值
        double max_val = src_row[0];
        for (size_t i = 1; i < dim; ++i) {
            max_val = std::max(max_val, src_row[i]);
        }
        
        // 计算 exp 和 sum
        double sum_exp = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            dst_row[i] = std::exp(src_row[i] - max_val);
            sum_exp += dst_row[i];
        }
        
        // 归一化
        for (size_t i = 0; i < dim; ++i) {
            dst_row[i] /= sum_exp;
        }
    }
}

}  // namespace Backend
}  // namespace Autoalg

#endif  // MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_BACKEND_H
