#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_GPU_STORAGE_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_GPU_STORAGE_H

#include <memory>
#include <cstring>
#include "backend/device.h"
#include "backend/simd_kernel.h"
#include "utils/base_config.h"

#ifdef USE_CUDA
#include "backend/cuda_common.h"
#include "backend/cuda_kernels.h"
#endif

namespace Autoalg {

// GPU 内存删除器
struct GpuDeleter {
    void operator()(BasicData* ptr) {
#ifdef USE_CUDA
        if (ptr) {
            cudaFree(ptr);
        }
#endif
    }
};

// CPU 内存删除器
struct CpuDeleter {
    void operator()(BasicData* ptr) {
        if (ptr) {
            delete[] ptr;
        }
    }
};

// 统一存储类，支持 CPU 和 GPU
class UnifiedStorage {
public:
    // 创建指定设备上的存储
    explicit UnifiedStorage(Index size, DeviceType device = DeviceType::CPU)
        : size_(size), device_(device) {
        Allocate();
    }
    
    // 创建并填充常数
    UnifiedStorage(Index size, BasicData value, DeviceType device = DeviceType::CPU)
        : size_(size), device_(device) {
        Allocate();
        Fill(value);
    }
    
    // 从 CPU 数据创建
    UnifiedStorage(const BasicData* host_data, Index size, DeviceType device = DeviceType::CPU)
        : size_(size), device_(device) {
        Allocate();
        CopyFromHost(host_data, size);
    }
    
    // 视图构造（共享数据）
    UnifiedStorage(const UnifiedStorage& other, Index offset)
        : size_(other.size_ - offset),
          offset_(other.offset_ + offset),
          device_(other.device_),
          cpu_ptr_(other.cpu_ptr_),
          gpu_ptr_(other.gpu_ptr_) {
        if (device_ == DeviceType::CPU) {
            data_ptr_ = cpu_ptr_.get() + offset_ + other.offset_;
        } else {
#ifdef USE_CUDA
            data_ptr_ = gpu_ptr_.get() + offset_ + other.offset_;
#endif
        }
    }
    
    UnifiedStorage(const UnifiedStorage&) = default;
    UnifiedStorage(UnifiedStorage&&) = default;
    
    // 拷贝赋值运算符
    UnifiedStorage& operator=(const UnifiedStorage& other) {
        if (this != &other) {
            size_ = other.size_;
            offset_ = other.offset_;
            device_ = other.device_;
            cpu_ptr_ = other.cpu_ptr_;
            gpu_ptr_ = other.gpu_ptr_;
            data_ptr_ = other.data_ptr_;
        }
        return *this;
    }
    
    // 移动赋值运算符
    UnifiedStorage& operator=(UnifiedStorage&& other) {
        if (this != &other) {
            size_ = other.size_;
            offset_ = other.offset_;
            device_ = other.device_;
            cpu_ptr_ = std::move(other.cpu_ptr_);
            gpu_ptr_ = std::move(other.gpu_ptr_);
            data_ptr_ = other.data_ptr_;
            other.data_ptr_ = nullptr;
        }
        return *this;
    }
    
    ~UnifiedStorage() = default;
    
    // 获取数据指针
    BasicData* Data() { return data_ptr_; }
    const BasicData* Data() const { return data_ptr_; }
    
    Index Size() const { return size_; }
    Index Offset() const { return offset_; }
    DeviceType Device() const { return device_; }
    
    bool IsGpu() const { return device_ == DeviceType::CUDA; }
    bool IsCpu() const { return device_ == DeviceType::CPU; }
    
    // 元素访问（仅 CPU 有效）
    BasicData operator[](Index idx) const {
        if (device_ == DeviceType::CPU) {
            return data_ptr_[idx];
        }
#ifdef USE_CUDA
        BasicData val;
        CUDA_CHECK(cudaMemcpy(&val, data_ptr_ + idx, sizeof(BasicData), cudaMemcpyDeviceToHost));
        return val;
#else
        return 0;
#endif
    }
    
    BasicData& operator[](Index idx) {
        // 注意：GPU 上这个返回引用不安全，仅用于 CPU
        return data_ptr_[idx];
    }
    
    // 填充操作
    void Fill(BasicData value) {
        if (device_ == DeviceType::CPU) {
            SIMD::fill_const(data_ptr_, value, size_);
        } else {
#ifdef USE_CUDA
            CUDA::fill_const(data_ptr_, value, size_);
#endif
        }
    }
    
    void FillZero() {
        if (device_ == DeviceType::CPU) {
            SIMD::fill_zero(data_ptr_, size_);
        } else {
#ifdef USE_CUDA
            CUDA::fill_zero(data_ptr_, size_);
#endif
        }
    }
    
    // 数据传输
    void CopyFromHost(const BasicData* host_data, Index size) {
        if (device_ == DeviceType::CPU) {
            SIMD::copy(data_ptr_, host_data, size);
        } else {
#ifdef USE_CUDA
            CUDA_CHECK(cudaMemcpy(data_ptr_, host_data, size * sizeof(BasicData), 
                                  cudaMemcpyHostToDevice));
#endif
        }
    }
    
    void CopyToHost(BasicData* host_data, Index size) const {
        if (device_ == DeviceType::CPU) {
            SIMD::copy(host_data, data_ptr_, size);
        } else {
#ifdef USE_CUDA
            CUDA_CHECK(cudaMemcpy(host_data, data_ptr_, size * sizeof(BasicData),
                                  cudaMemcpyDeviceToHost));
#endif
        }
    }
    
    void CopyToHost(BasicData* host_data) const {
        CopyToHost(host_data, size_);
    }
    
    // 设备间拷贝
    void CopyFrom(const UnifiedStorage& other) {
        if (device_ == other.device_) {
            // 同设备拷贝
            if (device_ == DeviceType::CPU) {
                SIMD::copy(data_ptr_, other.data_ptr_, size_);
            } else {
#ifdef USE_CUDA
                CUDA::copy(data_ptr_, other.data_ptr_, size_);
#endif
            }
        } else {
            // 跨设备拷贝
#ifdef USE_CUDA
            if (device_ == DeviceType::CPU) {
                // GPU -> CPU
                CUDA_CHECK(cudaMemcpy(data_ptr_, other.data_ptr_, size_ * sizeof(BasicData),
                                      cudaMemcpyDeviceToHost));
            } else {
                // CPU -> GPU
                CUDA_CHECK(cudaMemcpy(data_ptr_, other.data_ptr_, size_ * sizeof(BasicData),
                                      cudaMemcpyHostToDevice));
            }
#endif
        }
    }
    
    // 迁移到指定设备
    UnifiedStorage To(DeviceType target_device) const {
        if (device_ == target_device) {
            return *this;
        }
        
        UnifiedStorage result(size_, target_device);
        result.CopyFrom(*this);
        return result;
    }
    
    UnifiedStorage ToCpu() const { return To(DeviceType::CPU); }
    UnifiedStorage ToGpu() const { return To(DeviceType::CUDA); }

private:
    void Allocate() {
        if (device_ == DeviceType::CPU) {
            BasicData* ptr = new BasicData[size_];
            cpu_ptr_ = std::shared_ptr<BasicData>(ptr, CpuDeleter());
            data_ptr_ = cpu_ptr_.get();
        } else {
#ifdef USE_CUDA
            BasicData* ptr;
            CUDA_CHECK(cudaMalloc(&ptr, size_ * sizeof(BasicData)));
            gpu_ptr_ = std::shared_ptr<BasicData>(ptr, GpuDeleter());
            data_ptr_ = gpu_ptr_.get();
#endif
        }
    }
    
    Index size_;
    Index offset_ = 0;
    DeviceType device_;
    
    std::shared_ptr<BasicData> cpu_ptr_;
    std::shared_ptr<BasicData> gpu_ptr_;
    BasicData* data_ptr_ = nullptr;
};

}  // namespace Autoalg

#endif  // MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_GPU_STORAGE_H
