#ifdef USE_CUDA

#include "backend/cuda_common.h"
#include "backend/cuda_kernels.h"
#include <cub/cub.cuh>

namespace Autoalg {
namespace CUDA {

//=============================================================================
// 基本 Kernel
//=============================================================================

__global__ void fill_zero_kernel(double* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = 0.0;
    }
}

__global__ void fill_const_kernel(double* dst, double val, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = val;
    }
}

__global__ void copy_kernel(double* dst, const double* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

//=============================================================================
// 逐元素二元运算 Kernel
//=============================================================================

__global__ void add_kernel(double* dst, const double* a, const double* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a[idx] + b[idx];
    }
}

__global__ void add_inplace_kernel(double* dst, const double* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

__global__ void sub_kernel(double* dst, const double* a, const double* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_kernel(double* dst, const double* a, const double* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a[idx] * b[idx];
    }
}

__global__ void scale_kernel(double* dst, const double* a, double scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a[idx] * scalar;
    }
}

//=============================================================================
// 逐元素一元运算 Kernel
//=============================================================================

__global__ void negate_kernel(double* dst, const double* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = -src[idx];
    }
}

__global__ void relu_kernel(double* dst, const double* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx] > 0 ? src[idx] : 0;
    }
}

__global__ void relu_backward_kernel(double* dst, const double* grad, 
                                     const double* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx] > 0 ? grad[idx] : 0;
    }
}

__global__ void sigmoid_kernel(double* dst, const double* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = 1.0 / (1.0 + exp(-src[idx]));
    }
}

__global__ void exp_kernel_impl(double* dst, const double* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = exp(src[idx]);
    }
}

//=============================================================================
// 矩阵乘法 Kernel (使用 shared memory 优化)
//=============================================================================

__global__ void gemm_kernel(double* C, const double* A, const double* B,
                            size_t M, size_t K, size_t N) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];
    
    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0;
    
    for (size_t t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 加载 A 的 tile
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // 加载 B 的 tile
        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // 计算部分和
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void gemm_kernel_alpha_beta(double* C, const double* A, const double* B,
                                        size_t M, size_t K, size_t N,
                                        double alpha, double beta) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];
    
    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0;
    
    for (size_t t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

//=============================================================================
// 批量矩阵乘法 Kernel
//=============================================================================

__global__ void batched_gemm_kernel(double* C, const double* A, const double* B,
                                    size_t batch, size_t M, size_t K, size_t N) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];
    
    size_t b = blockIdx.z;
    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    const double* A_batch = A + b * M * K;
    const double* B_batch = B + b * K * N;
    double* C_batch = C + b * M * N;
    
    double sum = 0.0;
    
    for (size_t t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A_batch[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B_batch[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}

//=============================================================================
// Softmax Kernel
//=============================================================================

__global__ void softmax_kernel(double* dst, const double* src, 
                               size_t batch, size_t dim) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    
    const double* src_row = src + b * dim;
    double* dst_row = dst + b * dim;
    
    // 找最大值
    double max_val = src_row[0];
    for (size_t i = 1; i < dim; ++i) {
        max_val = max(max_val, src_row[i]);
    }
    
    // 计算 exp 和 sum
    double sum_exp = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dst_row[i] = exp(src_row[i] - max_val);
        sum_exp += dst_row[i];
    }
    
    // 归一化
    for (size_t i = 0; i < dim; ++i) {
        dst_row[i] /= sum_exp;
    }
}

__global__ void log_softmax_kernel(double* dst, const double* src,
                                   size_t batch, size_t dim) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    
    const double* src_row = src + b * dim;
    double* dst_row = dst + b * dim;
    
    // 找最大值
    double max_val = src_row[0];
    for (size_t i = 1; i < dim; ++i) {
        max_val = max(max_val, src_row[i]);
    }
    
    // 计算 log(sum(exp))
    double sum_exp = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum_exp += exp(src_row[i] - max_val);
    }
    double log_sum_exp = max_val + log(sum_exp);
    
    // log_softmax = x - log_sum_exp
    for (size_t i = 0; i < dim; ++i) {
        dst_row[i] = src_row[i] - log_sum_exp;
    }
}

//=============================================================================
// 规约 Kernel
//=============================================================================

__global__ void row_sum_kernel(double* dst, const double* src, 
                               size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    double sum = 0.0;
    for (size_t col = 0; col < cols; ++col) {
        sum += src[row * cols + col];
    }
    dst[row] = sum;
}

__global__ void col_sum_kernel(double* dst, const double* src,
                               size_t rows, size_t cols) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    
    double sum = 0.0;
    for (size_t row = 0; row < rows; ++row) {
        sum += src[row * cols + col];
    }
    dst[col] = sum;
}

//=============================================================================
// Host 调用接口
//=============================================================================

void fill_zero(double* dst, size_t n) {
    int grid = GetGridSize(n);
    fill_zero_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, n);
}

void fill_const(double* dst, double val, size_t n) {
    int grid = GetGridSize(n);
    fill_const_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, val, n);
}

void copy(double* dst, const double* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyDeviceToDevice));
}

void add(double* dst, const double* a, const double* b, size_t n) {
    int grid = GetGridSize(n);
    add_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, a, b, n);
}

void add_inplace(double* dst, const double* src, size_t n) {
    int grid = GetGridSize(n);
    add_inplace_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, src, n);
}

void sub(double* dst, const double* a, const double* b, size_t n) {
    int grid = GetGridSize(n);
    sub_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, a, b, n);
}

void mul(double* dst, const double* a, const double* b, size_t n) {
    int grid = GetGridSize(n);
    mul_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, a, b, n);
}

void scale(double* dst, const double* a, double scalar, size_t n) {
    int grid = GetGridSize(n);
    scale_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, a, scalar, n);
}

void negate(double* dst, const double* src, size_t n) {
    int grid = GetGridSize(n);
    negate_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, src, n);
}

void relu(double* dst, const double* src, size_t n) {
    int grid = GetGridSize(n);
    relu_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, src, n);
}

void relu_backward(double* dst, const double* grad, const double* src, size_t n) {
    int grid = GetGridSize(n);
    relu_backward_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, grad, src, n);
}

void sigmoid(double* dst, const double* src, size_t n) {
    int grid = GetGridSize(n);
    sigmoid_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, src, n);
}

void exp_kernel(double* dst, const double* src, size_t n) {
    int grid = GetGridSize(n);
    exp_kernel_impl<<<grid, BLOCK_SIZE_1D>>>(dst, src, n);
}

double sum(const double* src, size_t n) {
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, src, d_result, n);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceReduce::Sum(d_temp, temp_bytes, src, d_result, n);
    
    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_result));
    return result;
}

double dot(const double* a, const double* b, size_t n) {
    double* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(double)));
    
    // 先逐元素乘
    mul(d_temp, a, b, n);
    
    // 再求和
    double result = sum(d_temp, n);
    
    CUDA_CHECK(cudaFree(d_temp));
    return result;
}

void row_sum(double* dst, const double* src, size_t rows, size_t cols) {
    int grid = GetGridSize(rows);
    row_sum_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, src, rows, cols);
}

void col_sum(double* dst, const double* src, size_t rows, size_t cols) {
    int grid = GetGridSize(cols);
    col_sum_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, src, rows, cols);
}

void gemm(double* C, const double* A, const double* B,
          size_t M, size_t K, size_t N,
          bool trans_a, bool trans_b,
          double alpha, double beta) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    
    if (alpha == 1.0 && beta == 0.0) {
        gemm_kernel<<<grid, block>>>(C, A, B, M, K, N);
    } else {
        gemm_kernel_alpha_beta<<<grid, block>>>(C, A, B, M, K, N, alpha, beta);
    }
}

void batched_gemm(double* C, const double* A, const double* B,
                  size_t batch, size_t M, size_t K, size_t N,
                  bool trans_a, bool trans_b) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch);
    
    batched_gemm_kernel<<<grid, block>>>(C, A, B, batch, M, K, N);
}

void softmax(double* dst, const double* src, size_t batch, size_t dim) {
    int grid = GetGridSize(batch);
    softmax_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, src, batch, dim);
}

void log_softmax(double* dst, const double* src, size_t batch, size_t dim) {
    int grid = GetGridSize(batch);
    log_softmax_kernel<<<grid, BLOCK_SIZE_1D>>>(dst, src, batch, dim);
}

}  // namespace CUDA
}  // namespace Autoalg

#endif  // USE_CUDA
