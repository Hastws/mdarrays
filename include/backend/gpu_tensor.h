#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_GPU_TENSOR_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_GPU_TENSOR_H

#include <vector>
#include <iostream>
#include <initializer_list>
#include "backend/device.h"
#include "backend/gpu_storage.h"
#include "backend/simd_kernel.h"
#include "mdarray/shape.h"
#include "utils/base_config.h"

#ifdef USE_CUDA
#include "backend/cuda_kernels.h"
#endif

namespace Autoalg {

// GPU 友好的 Tensor 类
// 简化版本，专注于高效的 GPU 计算
class GpuTensor {
public:
    // 构造函数
    GpuTensor() : storage_(1, DeviceType::CPU), shape_({1}) {}
    
    explicit GpuTensor(const Shape& shape, DeviceType device = DeviceType::CPU)
        : storage_(shape.SpaceSize(), device), shape_(shape) {
        ComputeStride();
    }
    
    GpuTensor(const Shape& shape, BasicData value, DeviceType device = DeviceType::CPU)
        : storage_(shape.SpaceSize(), value, device), shape_(shape) {
        ComputeStride();
    }
    
    // 从 CPU 数据创建
    GpuTensor(const BasicData* data, const Shape& shape, DeviceType device = DeviceType::CPU)
        : storage_(data, shape.SpaceSize(), device), shape_(shape) {
        ComputeStride();
    }
    
    // 从 initializer_list 创建形状
    GpuTensor(std::initializer_list<Index> dims, DeviceType device = DeviceType::CPU)
        : GpuTensor(Shape(dims), device) {}
    
    // 拷贝和移动
    GpuTensor(const GpuTensor&) = default;
    GpuTensor(GpuTensor&&) = default;
    GpuTensor& operator=(const GpuTensor&) = default;
    GpuTensor& operator=(GpuTensor&&) = default;
    
    // 基本属性
    Index DimensionsSize() const { return shape_.DimensionsSize(); }
    Index Size(Index dim) const { return shape_[dim]; }
    const Shape& Size() const { return shape_; }
    Index TotalSize() const { return shape_.SpaceSize(); }
    Index Stride(Index dim) const { return stride_[dim]; }
    
    DeviceType Device() const { return storage_.Device(); }
    bool IsGpu() const { return storage_.IsGpu(); }
    bool IsCpu() const { return storage_.IsCpu(); }
    
    // 数据访问
    BasicData* Data() { return storage_.Data(); }
    const BasicData* Data() const { return storage_.Data(); }
    UnifiedStorage& Storage() { return storage_; }
    const UnifiedStorage& Storage() const { return storage_; }
    
    // 元素访问（主要用于 CPU）
    BasicData At(std::initializer_list<Index> indices) const {
        Index offset = ComputeOffset(indices);
        return storage_[offset];
    }
    
    void Set(std::initializer_list<Index> indices, BasicData value) {
        Index offset = ComputeOffset(indices);
        if (IsCpu()) {
            storage_[offset] = value;
        } else {
#ifdef USE_CUDA
            CUDA_CHECK(cudaMemcpy(storage_.Data() + offset, &value, sizeof(BasicData),
                                  cudaMemcpyHostToDevice));
#endif
        }
    }
    
    // 设备迁移
    GpuTensor To(DeviceType device) const {
        if (storage_.Device() == device) {
            return *this;
        }
        GpuTensor result(shape_, device);
        result.storage_.CopyFrom(storage_);
        return result;
    }
    
    GpuTensor Cpu() const { return To(DeviceType::CPU); }
    GpuTensor Cuda() const { return To(DeviceType::CUDA); }
    
    // 填充
    void Fill(BasicData value) { storage_.Fill(value); }
    void FillZero() { storage_.FillZero(); }
    
    // 拷贝数据到 CPU
    std::vector<BasicData> ToVector() const {
        std::vector<BasicData> result(TotalSize());
        storage_.CopyToHost(result.data(), TotalSize());
        return result;
    }
    
    //=========================================================================
    // 基本运算
    //=========================================================================
    
    // 逐元素加法
    GpuTensor operator+(const GpuTensor& other) const {
        GpuTensor result(shape_, Device());
        Add(result, *this, other);
        return result;
    }
    
    // 逐元素减法
    GpuTensor operator-(const GpuTensor& other) const {
        GpuTensor result(shape_, Device());
        Sub(result, *this, other);
        return result;
    }
    
    // 逐元素乘法
    GpuTensor operator*(const GpuTensor& other) const {
        GpuTensor result(shape_, Device());
        Mul(result, *this, other);
        return result;
    }
    
    // 标量乘法
    GpuTensor operator*(BasicData scalar) const {
        GpuTensor result(shape_, Device());
        Scale(result, *this, scalar);
        return result;
    }
    
    // 原地加法
    GpuTensor& operator+=(const GpuTensor& other) {
        AddInplace(*this, other);
        return *this;
    }
    
    //=========================================================================
    // 静态运算函数
    //=========================================================================
    
    // C = A + B
    static void Add(GpuTensor& C, const GpuTensor& A, const GpuTensor& B) {
        if (A.IsCpu()) {
            SIMD::add(C.Data(), A.Data(), B.Data(), A.TotalSize());
        } else {
#ifdef USE_CUDA
            CUDA::add(C.Data(), A.Data(), B.Data(), A.TotalSize());
#endif
        }
    }
    
    // A += B
    static void AddInplace(GpuTensor& A, const GpuTensor& B) {
        if (A.IsCpu()) {
            SIMD::add_inplace(A.Data(), B.Data(), A.TotalSize());
        } else {
#ifdef USE_CUDA
            CUDA::add_inplace(A.Data(), B.Data(), A.TotalSize());
#endif
        }
    }
    
    // C = A - B
    static void Sub(GpuTensor& C, const GpuTensor& A, const GpuTensor& B) {
        if (A.IsCpu()) {
            SIMD::sub(C.Data(), A.Data(), B.Data(), A.TotalSize());
        } else {
#ifdef USE_CUDA
            CUDA::sub(C.Data(), A.Data(), B.Data(), A.TotalSize());
#endif
        }
    }
    
    // C = A * B (逐元素)
    static void Mul(GpuTensor& C, const GpuTensor& A, const GpuTensor& B) {
        if (A.IsCpu()) {
            SIMD::mul(C.Data(), A.Data(), B.Data(), A.TotalSize());
        } else {
#ifdef USE_CUDA
            CUDA::mul(C.Data(), A.Data(), B.Data(), A.TotalSize());
#endif
        }
    }
    
    // C = A * scalar
    static void Scale(GpuTensor& C, const GpuTensor& A, BasicData scalar) {
        if (A.IsCpu()) {
            SIMD::scale(C.Data(), A.Data(), scalar, A.TotalSize());
        } else {
#ifdef USE_CUDA
            CUDA::scale(C.Data(), A.Data(), scalar, A.TotalSize());
#endif
        }
    }
    
    // C = ReLU(A)
    static void ReLU(GpuTensor& C, const GpuTensor& A) {
        if (A.IsCpu()) {
            SIMD::relu(C.Data(), A.Data(), A.TotalSize());
        } else {
#ifdef USE_CUDA
            CUDA::relu(C.Data(), A.Data(), A.TotalSize());
#endif
        }
    }
    
    // 矩阵乘法 C[M,N] = A[M,K] * B[K,N]
    static void Matmul(GpuTensor& C, const GpuTensor& A, const GpuTensor& B) {
        Index M = A.Size(0);
        Index K = A.Size(1);
        Index N = B.Size(1);
        
        if (A.IsCpu()) {
            SIMD::gemm_naive(C.Data(), A.Data(), B.Data(), M, K, N);
        } else {
#ifdef USE_CUDA
            CUDA::gemm(C.Data(), A.Data(), B.Data(), M, K, N);
#endif
        }
    }
    
    // 批量矩阵乘法 C[B,M,N] = A[B,M,K] * B[B,K,N]
    static void BatchedMatmul(GpuTensor& C, const GpuTensor& A, const GpuTensor& B) {
        Index batch = A.Size(0);
        Index M = A.Size(1);
        Index K = A.Size(2);
        Index N = B.Size(2);
        
        if (A.IsCpu()) {
            for (Index b = 0; b < batch; ++b) {
                SIMD::gemm_naive(C.Data() + b * M * N,
                                 A.Data() + b * M * K,
                                 B.Data() + b * K * N,
                                 M, K, N);
            }
        } else {
#ifdef USE_CUDA
            CUDA::batched_gemm(C.Data(), A.Data(), B.Data(), batch, M, K, N);
#endif
        }
    }
    
    // Softmax
    static void Softmax(GpuTensor& C, const GpuTensor& A) {
        Index batch = A.Size(0);
        Index dim = A.Size(1);
        
        if (A.IsCpu()) {
            for (Index b = 0; b < batch; ++b) {
                const BasicData* src = A.Data() + b * dim;
                BasicData* dst = C.Data() + b * dim;
                
                // 找最大值
                BasicData max_val = src[0];
                for (Index i = 1; i < dim; ++i) {
                    max_val = std::max(max_val, src[i]);
                }
                
                // exp 和 sum
                BasicData sum_exp = 0;
                for (Index i = 0; i < dim; ++i) {
                    dst[i] = std::exp(src[i] - max_val);
                    sum_exp += dst[i];
                }
                
                // 归一化
                for (Index i = 0; i < dim; ++i) {
                    dst[i] /= sum_exp;
                }
            }
        } else {
#ifdef USE_CUDA
            CUDA::softmax(C.Data(), A.Data(), batch, dim);
#endif
        }
    }
    
    // 求和
    static BasicData Sum(const GpuTensor& A) {
        if (A.IsCpu()) {
            return SIMD::sum(A.Data(), A.TotalSize());
        } else {
#ifdef USE_CUDA
            return CUDA::sum(A.Data(), A.TotalSize());
#else
            return 0;
#endif
        }
    }
    
    // 打印（调试用）
    friend std::ostream& operator<<(std::ostream& os, const GpuTensor& t) {
        auto vec = t.ToVector();
        os << "GpuTensor(shape=[";
        for (Index i = 0; i < t.DimensionsSize(); ++i) {
            if (i > 0) os << ",";
            os << t.Size(i);
        }
        os << "], device=" << (t.IsGpu() ? "cuda" : "cpu") << ")\n[";
        
        Index max_print = std::min((Index)10, t.TotalSize());
        for (Index i = 0; i < max_print; ++i) {
            if (i > 0) os << ", ";
            os << vec[i];
        }
        if (t.TotalSize() > max_print) os << ", ...";
        os << "]";
        return os;
    }

private:
    void ComputeStride() {
        stride_.resize(shape_.DimensionsSize());
        Index s = 1;
        for (int i = static_cast<int>(shape_.DimensionsSize()) - 1; i >= 0; --i) {
            stride_[i] = s;
            s *= shape_[i];
        }
    }
    
    Index ComputeOffset(std::initializer_list<Index> indices) const {
        Index offset = 0;
        Index i = 0;
        for (Index idx : indices) {
            offset += idx * stride_[i++];
        }
        return offset;
    }
    
    UnifiedStorage storage_;
    Shape shape_;
    std::vector<Index> stride_;
};

}  // namespace Autoalg

#endif  // MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_GPU_TENSOR_H
