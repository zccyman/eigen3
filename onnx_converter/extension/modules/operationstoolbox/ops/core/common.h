//
// Created by shengyuan.shen on 2023/4/3.
//

#ifndef C__COMMON_H
#define C__COMMON_H

#pragma once

#include <stdio.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <memory>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/ThreadPool"
#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>
#include <cudnn.h>
#include <cufft.h>
#endif

#ifdef WITH_CUDA
#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N)      \
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x; \
  if (id >= N)                                          \
    return;

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

// The number of cuda threads to use. Since work is assigned to SMs at the
// granularity of a block, 128 is chosen to allow utilizing more SMs for
// smaller input sizes.
// 1D grid
constexpr int CAFFE_CUDA_NUM_THREADS = 128;
// 2D grid
constexpr int CAFFE_CUDA_NUM_THREADS_2D_DIMX = 16;
constexpr int CAFFE_CUDA_NUM_THREADS_2D_DIMY = 16;

// The maximum number of blocks to use in the default kernel call. We set it to
// 4096 which would work for compute capability 2.x (where 65536 is the limit).
// This number is very carelessly chosen. Ideally, one would like to look at
// the hardware at runtime, and pick the number of blocks that makes most
// sense for the specific runtime environment. This is a todo item.
// 1D grid
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;
// 2D grid
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX = 128;
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY = 128;

constexpr int kCUDAGridDimMaxX = 2147483647;
constexpr int kCUDAGridDimMaxY = 65535;
constexpr int kCUDAGridDimMaxZ = 65535;

template <class INT, class INT2>
inline __host__ __device__ INT CeilDiv(INT a, INT2 b)  // ceil(a/b)
{
    return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);  // these size_t casts are necessary since b may be INT_MAX (for maxGridSize[])
}

struct GridDim {
    enum : CUDA_LONG {
        maxThreadsPerBlock = 256,  // max threads per block
        maxElementsPerThread = 4,  // max element processed per thread
    };
};
/**
 * @brief Compute the number of blocks needed to run N threads.
 */
inline int CAFFE_GET_BLOCKS(const int N) {
    return std::max(
            std::min(
                    (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
                    CAFFE_MAXIMUM_NUM_BLOCKS),
            // Use at least 1 block, since CUDA does not allow empty block
            1);
}

/**
 * @brief Compute the number of blocks needed to run N threads for a 2D grid
 */
inline dim3 CAFFE_GET_BLOCKS_2D(const int N, const int /* M */) {
    dim3 grid;
    // Not calling the 1D version for each dim to keep all constants as literals

    grid.x = std::max(
            std::min(
                    (N + CAFFE_CUDA_NUM_THREADS_2D_DIMX - 1) /
                    CAFFE_CUDA_NUM_THREADS_2D_DIMX,
                    CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX),
            // Use at least 1 block, since CUDA does not allow empty block
            1);

    grid.y = std::max(
            std::min(
                    (N + CAFFE_CUDA_NUM_THREADS_2D_DIMY - 1) /
                    CAFFE_CUDA_NUM_THREADS_2D_DIMY,
                    CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY),
            // Use at least 1 block, since CUDA does not allow empty block
            1);

    return grid;
}
#endif

//
// Macro to place variables at a specified alignment.
//

#if (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)
#define MLASCALL __stdcall
#else
#define MLASCALL
#endif

enum StorageOrder {
    UNKNOWN = 0,
    NHWC = 1,
    NCHW = 2,
};

/// Sum of a list of integers; accumulates into the int64_t datatype
template <
        typename C,
        typename std::enable_if<
                std::is_integral<typename C::value_type>::value,
                int>::type = 0>
inline int64_t sum_integers(const C& container) {
    // std::accumulate infers return type from `init` type, so if the `init` type
    // is not large enough to hold the result, computation can overflow. We use
    // `int64_t` here to avoid this.
    return std::accumulate(
            container.begin(), container.end(), static_cast<int64_t>(0));
}

/// Sum of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
        typename Iter,
        typename std::enable_if<
                std::is_integral<
                        typename std::iterator_traits<Iter>::value_type>::value,
                int>::type = 0>
inline int64_t sum_integers(Iter begin, Iter end) {
    // std::accumulate infers return type from `init` type, so if the `init` type
    // is not large enough to hold the result, computation can overflow. We use
    // `int64_t` here to avoid this.
    return std::accumulate(begin, end, static_cast<int64_t>(0));
}

/// Product of a list of integers; accumulates into the int64_t datatype
template <
        typename C,
        typename std::enable_if<
                std::is_integral<typename C::value_type>::value,
                int>::type = 0>
inline int64_t multiply_integers(const C& container) {
    // std::accumulate infers return type from `init` type, so if the `init` type
    // is not large enough to hold the result, computation can overflow. We use
    // `int64_t` here to avoid this.
    return std::accumulate(
            container.begin(),
            container.end(),
            static_cast<int64_t>(1),
            std::multiplies<int64_t>());
}

/// Product of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
        typename Iter,
        typename std::enable_if<
                std::is_integral<
                        typename std::iterator_traits<Iter>::value_type>::value,
                int>::type = 0>
inline int64_t multiply_integers(Iter begin, Iter end) {
    // std::accumulate infers return type from `init` type, so if the `init` type
    // is not large enough to hold the result, computation can overflow. We use
    // `int64_t` here to avoid this.
    return std::accumulate(
            begin, end, static_cast<int64_t>(1), std::multiplies<int64_t>());
}

#define TIMESINTELLI_DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define TIMESINTELLI_DISALLOW_ASSIGNMENT(TypeName) TypeName& operator=(const TypeName&) = delete

#define TIMESINTELLI_DISALLOW_COPY_AND_ASSIGNMENT(TypeName) \
  TIMESINTELLI_DISALLOW_COPY(TypeName);                     \
  TIMESINTELLI_DISALLOW_ASSIGNMENT(TypeName)

#define TIMESINTELLI_DISALLOW_MOVE(TypeName) \
  TypeName(TypeName&&) = delete;    \
  TypeName& operator=(TypeName&&) = delete

#define TIMESINTELLI_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName) \
  TIMESINTELLI_DISALLOW_COPY_AND_ASSIGNMENT(TypeName);           \
  TIMESINTELLI_DISALLOW_MOVE(TypeName)

#endif //C__COMMON_H
