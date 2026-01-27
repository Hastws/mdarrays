#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_CUDA_STORAGE_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_CUDA_STORAGE_H

#ifdef USE_CUDA

#include <memory>
#include "backend/cuda_common.h"
#include "backend/cuda_kernels.h"
#include "utils/base_config.h"

namespace Autoalg {

// CUDA 内存删除器
struct CudaDeleter {
    void operator()(void* ptr) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

// CUDA 存储类
class CudaStorage {
public:
    explicit CudaStorage(Index size) : size_(size), offset_(0) {
        double* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(double)));
        base_ptr_ = std::shared_ptr<double>(ptr, CudaDeleter());
        data_ptr_ = base_ptr_.get();
    }
    
    CudaStorage(const CudaStorage& other, Index offset)
        : size_(other.size_ - offset),
          offset_(other.offset_ + offset),
          base_ptr_(other.base_ptr_),
          data_ptr_(other.data_ptr_ + offset) {}
    
    CudaStorage(Index size, double value) : CudaStorage(size) {
        CUDA::fill_const(data_ptr_, value, size);
    }
    
    // 从 CPU 数据创建
    CudaStorage(const double* host_data, Index size) : CudaStorage(size) {
        CUDA_CHECK(cudaMemcpy(data_ptr_, host_data, size * sizeof(double),
                              cudaMemcpyHostToDevice));
    }
    
    CudaStorage(const CudaStorage&) = default;
    CudaStorage(CudaStorage&&) = default;
    ~CudaStorage() = default;
    
    // 获取设备指针
    double* Data() { return data_ptr_; }
    const double* Data() const { return data_ptr_; }
    
    Index Size() const { return size_; }
    Index Offset() const { return offset_; }
    
    // CPU <-> GPU 数据传输
    void CopyToHost(double* host_data) const {
        CUDA_CHECK(cudaMemcpy(host_data, data_ptr_, size_ * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
    
    void CopyFromHost(const double* host_data) {
        CUDA_CHECK(cudaMemcpy(data_ptr_, host_data, size_ * sizeof(double),
                              cudaMemcpyHostToDevice));
    }
    
    void CopyFromHost(const double* host_data, Index size) {
        CUDA_CHECK(cudaMemcpy(data_ptr_, host_data, size * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

private:
    Index size_;
    Index offset_;
    std::shared_ptr<double> base_ptr_;
    double* data_ptr_;
};

}  // namespace Autoalg

#endif  // USE_CUDA

#endif  // MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_CUDA_STORAGE_H
