cmake_minimum_required(VERSION 3.5)
project(mdarray LANGUAGES CXX)

# 选项：是否启用 CUDA
option(USE_CUDA "Enable CUDA support" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -Wall -O3 -march=native")

# OpenMP 支持
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallel computation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found, using single-threaded mode")
endif()

# 检测 SIMD 支持
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2 supported, enabling SIMD optimizations")
else()
    message(WARNING "AVX2 not supported, SIMD optimizations disabled")
endif()

# CUDA 支持
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
    # 不使用全局 add_definitions，而是为每个目标单独设置
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++11")
endif()

include_directories(include)
include_directories(third_party)

FILE(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# CUDA 源文件
if(USE_CUDA)
    FILE(GLOB_RECURSE CUDA_SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cu)
endif()

message(${PROJECT_SOURCE_DIR})
# 数据路径不再硬编码，使用运行时自动检测/下载

# 库
if(USE_CUDA)
    cuda_add_library(cuda_kernels_lib STATIC ${CUDA_SRC_FILES})
    target_link_libraries(cuda_kernels_lib ${CUDA_LIBRARIES})
    target_compile_definitions(cuda_kernels_lib PRIVATE USE_CUDA)
endif()

add_library(multidimensional_arrays_lib STATIC ${SRC_FILES})
add_library(memory_pool_lib STATIC ${PROJECT_SOURCE_DIR}/third_party/memory_pool/allocator.h
        ${PROJECT_SOURCE_DIR}/third_party/memory_pool/allocator.cpp)

# 可执行文件
if(USE_CUDA)
    add_executable(mdarray test/test_base.cpp ${SRC_FILES})
    add_executable(mlp_mnist test/mlp_mnist.cpp ${SRC_FILES})
    add_executable(cnn_mnist test/cnn_mnist.cpp ${SRC_FILES})
    add_executable(cnn_cifar_10 test/cnn_cifar_10.cpp ${SRC_FILES})
    add_executable(transformer_mnist test/transformer_mnist.cpp ${SRC_FILES})
    add_executable(transformer_mnist_cuda test/transformer_mnist.cpp ${SRC_FILES})
    target_compile_definitions(transformer_mnist_cuda PRIVATE USE_CUDA)
else()
    add_executable(mdarray test/test_base.cpp ${SRC_FILES})
    add_executable(mlp_mnist test/mlp_mnist.cpp ${SRC_FILES})
    add_executable(cnn_mnist test/cnn_mnist.cpp ${SRC_FILES})
    add_executable(cnn_cifar_10 test/cnn_cifar_10.cpp ${SRC_FILES})
    add_executable(transformer_mnist test/transformer_mnist.cpp ${SRC_FILES})
endif()

target_link_libraries(multidimensional_arrays_lib memory_pool_lib)
if(OpenMP_CXX_FOUND)
    target_link_libraries(multidimensional_arrays_lib OpenMP::OpenMP_CXX)
endif()
if(USE_CUDA)
    target_link_libraries(multidimensional_arrays_lib cuda_kernels_lib ${CUDA_LIBRARIES})
endif()

target_link_libraries(mdarray multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(mlp_mnist multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(cnn_mnist multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(cnn_cifar_10 multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(transformer_mnist multidimensional_arrays_lib memory_pool_lib)
if(USE_CUDA)
    target_link_libraries(transformer_mnist_cuda multidimensional_arrays_lib memory_pool_lib cuda_kernels_lib ${CUDA_LIBRARIES})
    
    # CUDA 测试程序
    add_executable(test_cuda test/test_cuda.cpp)
    target_compile_definitions(test_cuda PRIVATE USE_CUDA)
    target_link_libraries(test_cuda multidimensional_arrays_lib memory_pool_lib cuda_kernels_lib ${CUDA_LIBRARIES})
    
    # GPU MLP MNIST (CUDA 版本)
    add_executable(mlp_mnist_gpu test/mlp_mnist_gpu.cpp ${SRC_FILES})
    target_compile_definitions(mlp_mnist_gpu PRIVATE USE_CUDA)
    target_link_libraries(mlp_mnist_gpu multidimensional_arrays_lib memory_pool_lib cuda_kernels_lib ${CUDA_LIBRARIES})
    
    # GPU MLP MNIST (CPU 版本 - 用于对比)
    add_executable(mlp_mnist_gpu_cpu test/mlp_mnist_gpu.cpp ${SRC_FILES})
    target_link_libraries(mlp_mnist_gpu_cpu multidimensional_arrays_lib memory_pool_lib)
endif()