#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_CUDA_KERNELS_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_CUDA_KERNELS_H

#ifdef USE_CUDA

#include "backend/cuda_common.h"
#include "utils/base_config.h"

namespace Autoalg {
namespace CUDA {

//=============================================================================
// CUDA Kernel 函数声明
// 实现在 cuda_kernels.cu 中
//=============================================================================

// 内存操作
void fill_zero(double* dst, size_t n);
void fill_const(double* dst, double val, size_t n);
void copy(double* dst, const double* src, size_t n);

// 逐元素二元运算
void add(double* dst, const double* a, const double* b, size_t n);
void add_inplace(double* dst, const double* src, size_t n);
void sub(double* dst, const double* a, const double* b, size_t n);
void mul(double* dst, const double* a, const double* b, size_t n);
void scale(double* dst, const double* a, double scalar, size_t n);

// 逐元素一元运算
void negate(double* dst, const double* src, size_t n);
void relu(double* dst, const double* src, size_t n);
void relu_backward(double* dst, const double* grad, const double* src, size_t n);
void sigmoid(double* dst, const double* src, size_t n);
void exp_kernel(double* dst, const double* src, size_t n);

// 规约运算
double sum(const double* src, size_t n);
double dot(const double* a, const double* b, size_t n);
void row_sum(double* dst, const double* src, size_t rows, size_t cols);
void col_sum(double* dst, const double* src, size_t rows, size_t cols);

// 矩阵乘法
// C[M,N] = A[M,K] * B[K,N]
void gemm(double* C, const double* A, const double* B,
          size_t M, size_t K, size_t N,
          bool trans_a = false, bool trans_b = false,
          double alpha = 1.0, double beta = 0.0);

// 批量矩阵乘法
// C[batch, M, N] = A[batch, M, K] * B[batch, K, N]
void batched_gemm(double* C, const double* A, const double* B,
                  size_t batch, size_t M, size_t K, size_t N,
                  bool trans_a = false, bool trans_b = false);

// Softmax
void softmax(double* dst, const double* src, size_t batch, size_t dim);
void softmax_backward(double* dst, const double* grad, const double* output,
                      size_t batch, size_t dim);

// LogSoftmax
void log_softmax(double* dst, const double* src, size_t batch, size_t dim);

}  // namespace CUDA
}  // namespace Autoalg

#endif  // USE_CUDA

#endif  // MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_CUDA_KERNELS_H
