#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_CUDA_COMMON_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_CUDA_COMMON_H

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace Autoalg {
namespace CUDA {

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(                                       \
                std::string("CUDA error at ") + __FILE__ + ":" +            \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err)); \
        }                                                                   \
    } while (0)

// 常用的 block 大小
constexpr int BLOCK_SIZE_1D = 256;
constexpr int BLOCK_SIZE_2D = 16;
constexpr int TILE_SIZE = 16;

// 计算 grid 大小
inline int GetGridSize(int n, int block_size = BLOCK_SIZE_1D) {
    return (n + block_size - 1) / block_size;
}

inline dim3 GetGridSize2D(int rows, int cols, int block_size = BLOCK_SIZE_2D) {
    return dim3((cols + block_size - 1) / block_size,
                (rows + block_size - 1) / block_size);
}

// 设备信息
inline void PrintDeviceInfo() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    printf("CUDA Device Count: %d\n", device_count);
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", 
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    }
}

// 设置 CUDA 设备
inline void SetDevice(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

// 同步
inline void Synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace CUDA
}  // namespace Autoalg

#endif  // USE_CUDA

#endif  // MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_CUDA_COMMON_H
