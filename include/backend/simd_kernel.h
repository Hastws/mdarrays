#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_SIMD_KERNEL_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_SIMD_KERNEL_H

#include <cmath>
#include <algorithm>
#include "utils/base_config.h"

// 检测 SIMD 支持
#if defined(__AVX2__) || defined(__AVX__)
#define USE_AVX
#include <immintrin.h>
#elif defined(__SSE4_1__) || defined(__SSE2__)
#define USE_SSE
#include <emmintrin.h>
#include <smmintrin.h>
#endif

namespace Autoalg {
namespace SIMD {

// AVX: 4 个 double 一组
constexpr size_t AVX_DOUBLE_WIDTH = 4;
// SSE: 2 个 double 一组
constexpr size_t SSE_DOUBLE_WIDTH = 2;

//=============================================================================
// 内存操作
//=============================================================================

// 快速填充零
inline void fill_zero(BasicData* dst, size_t n) {
#ifdef USE_AVX
    __m256d zero = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        _mm256_storeu_pd(dst + i, zero);
    }
    for (; i < n; ++i) {
        dst[i] = 0.0;
    }
#elif defined(USE_SSE)
    __m128d zero = _mm_setzero_pd();
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        _mm_storeu_pd(dst + i, zero);
    }
    for (; i < n; ++i) {
        dst[i] = 0.0;
    }
#else
    std::fill(dst, dst + n, 0.0);
#endif
}

// 快速填充常数
inline void fill_const(BasicData* dst, BasicData val, size_t n) {
#ifdef USE_AVX
    __m256d v = _mm256_set1_pd(val);
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        _mm256_storeu_pd(dst + i, v);
    }
    for (; i < n; ++i) {
        dst[i] = val;
    }
#elif defined(USE_SSE)
    __m128d v = _mm_set1_pd(val);
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        _mm_storeu_pd(dst + i, v);
    }
    for (; i < n; ++i) {
        dst[i] = val;
    }
#else
    std::fill(dst, dst + n, val);
#endif
}

// 快速拷贝
inline void copy(BasicData* dst, const BasicData* src, size_t n) {
#ifdef USE_AVX
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        __m256d v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, v);
    }
    for (; i < n; ++i) {
        dst[i] = src[i];
    }
#elif defined(USE_SSE)
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        __m128d v = _mm_loadu_pd(src + i);
        _mm_storeu_pd(dst + i, v);
    }
    for (; i < n; ++i) {
        dst[i] = src[i];
    }
#else
    std::copy(src, src + n, dst);
#endif
}

//=============================================================================
// 逐元素二元运算
//=============================================================================

// dst = a + b
inline void add(BasicData* dst, const BasicData* a, const BasicData* b, size_t n) {
#ifdef USE_AVX
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vsum = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(dst + i, vsum);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] + b[i];
    }
#elif defined(USE_SSE)
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d vsum = _mm_add_pd(va, vb);
        _mm_storeu_pd(dst + i, vsum);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] + b[i];
    }
#endif
}

// dst += src (in-place add)
inline void add_inplace(BasicData* dst, const BasicData* src, size_t n) {
#ifdef USE_AVX
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(dst + i);
        __m256d vb = _mm256_loadu_pd(src + i);
        __m256d vsum = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(dst + i, vsum);
    }
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
#elif defined(USE_SSE)
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(dst + i);
        __m128d vb = _mm_loadu_pd(src + i);
        __m128d vsum = _mm_add_pd(va, vb);
        _mm_storeu_pd(dst + i, vsum);
    }
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
#endif
}

// dst = a - b
inline void sub(BasicData* dst, const BasicData* a, const BasicData* b, size_t n) {
#ifdef USE_AVX
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vdiff = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(dst + i, vdiff);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] - b[i];
    }
#elif defined(USE_SSE)
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d vdiff = _mm_sub_pd(va, vb);
        _mm_storeu_pd(dst + i, vdiff);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] - b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] - b[i];
    }
#endif
}

// dst = a * b (element-wise)
inline void mul(BasicData* dst, const BasicData* a, const BasicData* b, size_t n) {
#ifdef USE_AVX
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vprod = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(dst + i, vprod);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
#elif defined(USE_SSE)
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d vprod = _mm_mul_pd(va, vb);
        _mm_storeu_pd(dst + i, vprod);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
#endif
}

// dst = a * scalar
inline void scale(BasicData* dst, const BasicData* a, BasicData scalar, size_t n) {
#ifdef USE_AVX
    __m256d vs = _mm256_set1_pd(scalar);
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vprod = _mm256_mul_pd(va, vs);
        _mm256_storeu_pd(dst + i, vprod);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] * scalar;
    }
#elif defined(USE_SSE)
    __m128d vs = _mm_set1_pd(scalar);
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vprod = _mm_mul_pd(va, vs);
        _mm_storeu_pd(dst + i, vprod);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] * scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] * scalar;
    }
#endif
}

//=============================================================================
// 逐元素一元运算
//=============================================================================

// dst = -src
inline void negate(BasicData* dst, const BasicData* src, size_t n) {
#ifdef USE_AVX
    __m256d zero = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        __m256d v = _mm256_loadu_pd(src + i);
        __m256d vneg = _mm256_sub_pd(zero, v);
        _mm256_storeu_pd(dst + i, vneg);
    }
    for (; i < n; ++i) {
        dst[i] = -src[i];
    }
#elif defined(USE_SSE)
    __m128d zero = _mm_setzero_pd();
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        __m128d v = _mm_loadu_pd(src + i);
        __m128d vneg = _mm_sub_pd(zero, v);
        _mm_storeu_pd(dst + i, vneg);
    }
    for (; i < n; ++i) {
        dst[i] = -src[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = -src[i];
    }
#endif
}

// dst = max(src, 0)  -- ReLU
inline void relu(BasicData* dst, const BasicData* src, size_t n) {
#ifdef USE_AVX
    __m256d zero = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        __m256d v = _mm256_loadu_pd(src + i);
        __m256d result = _mm256_max_pd(v, zero);
        _mm256_storeu_pd(dst + i, result);
    }
    for (; i < n; ++i) {
        dst[i] = std::max(src[i], 0.0);
    }
#elif defined(USE_SSE)
    __m128d zero = _mm_setzero_pd();
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        __m128d v = _mm_loadu_pd(src + i);
        __m128d result = _mm_max_pd(v, zero);
        _mm_storeu_pd(dst + i, result);
    }
    for (; i < n; ++i) {
        dst[i] = std::max(src[i], 0.0);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::max(src[i], 0.0);
    }
#endif
}

// ReLU backward: dst = (src > 0) ? grad : 0
inline void relu_backward(BasicData* dst, const BasicData* grad, 
                          const BasicData* src, size_t n) {
#ifdef USE_AVX
    __m256d zero = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        __m256d vs = _mm256_loadu_pd(src + i);
        __m256d vg = _mm256_loadu_pd(grad + i);
        // mask: src > 0
        __m256d mask = _mm256_cmp_pd(vs, zero, _CMP_GT_OQ);
        // result = mask ? grad : 0
        __m256d result = _mm256_and_pd(mask, vg);
        _mm256_storeu_pd(dst + i, result);
    }
    for (; i < n; ++i) {
        dst[i] = src[i] > 0.0 ? grad[i] : 0.0;
    }
#elif defined(USE_SSE)
    __m128d zero = _mm_setzero_pd();
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        __m128d vs = _mm_loadu_pd(src + i);
        __m128d vg = _mm_loadu_pd(grad + i);
        __m128d mask = _mm_cmpgt_pd(vs, zero);
        __m128d result = _mm_and_pd(mask, vg);
        _mm_storeu_pd(dst + i, result);
    }
    for (; i < n; ++i) {
        dst[i] = src[i] > 0.0 ? grad[i] : 0.0;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i] > 0.0 ? grad[i] : 0.0;
    }
#endif
}

//=============================================================================
// 规约运算
//=============================================================================

// sum of array
inline BasicData sum(const BasicData* src, size_t n) {
#ifdef USE_AVX
    __m256d vsum = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        __m256d v = _mm256_loadu_pd(src + i);
        vsum = _mm256_add_pd(vsum, v);
    }
    // horizontal sum
    __m128d vlow = _mm256_castpd256_pd128(vsum);
    __m128d vhigh = _mm256_extractf128_pd(vsum, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d shuf = _mm_shuffle_pd(vlow, vlow, 1);
    __m128d sums = _mm_add_sd(vlow, shuf);
    BasicData result = _mm_cvtsd_f64(sums);
    for (; i < n; ++i) {
        result += src[i];
    }
    return result;
#elif defined(USE_SSE)
    __m128d vsum = _mm_setzero_pd();
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        __m128d v = _mm_loadu_pd(src + i);
        vsum = _mm_add_pd(vsum, v);
    }
    __m128d shuf = _mm_shuffle_pd(vsum, vsum, 1);
    __m128d sums = _mm_add_sd(vsum, shuf);
    BasicData result = _mm_cvtsd_f64(sums);
    for (; i < n; ++i) {
        result += src[i];
    }
    return result;
#else
    BasicData result = 0.0;
    for (size_t i = 0; i < n; ++i) {
        result += src[i];
    }
    return result;
#endif
}

// dot product
inline BasicData dot(const BasicData* a, const BasicData* b, size_t n) {
#ifdef USE_AVX
    __m256d vsum = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + AVX_DOUBLE_WIDTH <= n; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        vsum = _mm256_fmadd_pd(va, vb, vsum);  // vsum += va * vb
    }
    // horizontal sum
    __m128d vlow = _mm256_castpd256_pd128(vsum);
    __m128d vhigh = _mm256_extractf128_pd(vsum, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d shuf = _mm_shuffle_pd(vlow, vlow, 1);
    __m128d sums = _mm_add_sd(vlow, shuf);
    BasicData result = _mm_cvtsd_f64(sums);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
#elif defined(USE_SSE)
    __m128d vsum = _mm_setzero_pd();
    size_t i = 0;
    for (; i + SSE_DOUBLE_WIDTH <= n; i += SSE_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d vprod = _mm_mul_pd(va, vb);
        vsum = _mm_add_pd(vsum, vprod);
    }
    __m128d shuf = _mm_shuffle_pd(vsum, vsum, 1);
    __m128d sums = _mm_add_sd(vsum, shuf);
    BasicData result = _mm_cvtsd_f64(sums);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
#else
    BasicData result = 0.0;
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
#endif
}

//=============================================================================
// 矩阵乘法 (简单版本，用于小矩阵)
// C[M,N] = A[M,K] * B[K,N]
//=============================================================================
inline void gemm_naive(BasicData* C, const BasicData* A, const BasicData* B,
                       size_t M, size_t K, size_t N) {
    // 先清零
    fill_zero(C, M * N);
    
    // 使用 ikj 顺序，对缓存更友好
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            BasicData a_ik = A[i * K + k];
#ifdef USE_AVX
            __m256d va = _mm256_set1_pd(a_ik);
            size_t j = 0;
            for (; j + AVX_DOUBLE_WIDTH <= N; j += AVX_DOUBLE_WIDTH) {
                __m256d vb = _mm256_loadu_pd(B + k * N + j);
                __m256d vc = _mm256_loadu_pd(C + i * N + j);
                vc = _mm256_fmadd_pd(va, vb, vc);
                _mm256_storeu_pd(C + i * N + j, vc);
            }
            for (; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
#else
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
#endif
        }
    }
}

}  // namespace SIMD
}  // namespace Autoalg

#endif  // MULTIDIMENSIONAL_ARRAYS_INCLUDE_BACKEND_SIMD_KERNEL_H
