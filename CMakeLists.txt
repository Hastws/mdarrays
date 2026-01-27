cmake_minimum_required(VERSION 3.5)
project(mdarray)

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

include_directories(include)
include_directories(third_party)

FILE(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

message(${PROJECT_SOURCE_DIR})
add_definitions(-DMLP_MNIST_TRAIN_IMAGES="${PROJECT_SOURCE_DIR}/data/mnist/train-images.idx3-ubyte")
add_definitions(-DMLP_MNIST_TRAIN_LABELS="${PROJECT_SOURCE_DIR}/data/mnist/train-labels.idx1-ubyte")
add_definitions(-DMLP_MNIST_TEST_IMAGES="${PROJECT_SOURCE_DIR}/data/mnist/t10k-images.idx3-ubyte")
add_definitions(-DMLP_MNIST_TEST_LABELS="${PROJECT_SOURCE_DIR}/data/mnist/t10k-labels.idx1-ubyte")
add_definitions(-DCIFAR_10_DATA_FOLDER="${PROJECT_SOURCE_DIR}/data/cifar_10")

add_library(multidimensional_arrays_lib STATIC ${SRC_FILES})
add_library(memory_pool_lib STATIC ${PROJECT_SOURCE_DIR}/third_party/memory_pool/allocator.h
        ${PROJECT_SOURCE_DIR}/third_party/memory_pool/allocator.cpp)

add_executable(mdarray test/test_base.cpp ${SRC_FILES})
add_executable(mlp_mnist test/mlp_mnist.cpp ${SRC_FILES})
add_executable(cnn_mnist test/cnn_mnist.cpp ${SRC_FILES})
add_executable(cnn_cifar_10 test/cnn_cifar_10.cpp ${SRC_FILES})
add_executable(transformer_mnist test/transformer_mnist.cpp ${SRC_FILES})

target_compile_definitions(cnn_cifar_10 PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

target_link_libraries(multidimensional_arrays_lib memory_pool_lib)
if(OpenMP_CXX_FOUND)
    target_link_libraries(multidimensional_arrays_lib OpenMP::OpenMP_CXX)
endif()

target_link_libraries(mdarray multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(mlp_mnist multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(cnn_mnist multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(cnn_cifar_10 multidimensional_arrays_lib memory_pool_lib)
target_link_libraries(transformer_mnist multidimensional_arrays_lib memory_pool_lib)