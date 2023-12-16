//
// Created by shengyuan.shen on 2023/4/10.
//

#ifndef C__THREAD_UTILS_H
#define C__THREAD_UTILS_H

#include <sstream>
#include <iterator>

#ifdef _WIN32
#include <Windows.h>
#include <fstream>
#include <string>
#include <thread>
#include <process.h>
#include <fcntl.h>
#include <io.h>
#include <wil/Resource.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <ftw.h>
#include <string.h>
#include <thread>
#include <utility>  // for std::forward
#include <vector>
#include <assert.h>
#endif

#if defined(_M_AMD64)
#include <intrin.h>
#endif

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/ThreadPool"

std::pair<int, std::string> GetSystemError();

#ifdef _WIN32
// Define ORT_DLL_IMPORT if your program is dynamically linked to Ort.
// dllexport is not used, we use a .def file.
#ifdef ORT_DLL_IMPORT
#define ORT_EXPORT __declspec(dllimport)
#else
#define ORT_EXPORT
#endif
#define ORT_API_CALL _stdcall
#define ORT_MUST_USE_RESULT
#define ORTCHAR_T wchar_t
#else
// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define ORT_EXPORT __attribute__((visibility("default")))
#else
#define ORT_EXPORT
#endif
#define ORT_API_CALL
#define ORT_MUST_USE_RESULT __attribute__((warn_unused_result))
#define ORTCHAR_T char
#endif

#define ORT_TRY if (true)
#define ORT_CATCH(x) else if (false)
#define ORT_RETHROW

#define ORT_DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define ORT_DISALLOW_ASSIGNMENT(TypeName) TypeName& operator=(const TypeName&) = delete

#define ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName) \
  ORT_DISALLOW_COPY(TypeName);                     \
  ORT_DISALLOW_ASSIGNMENT(TypeName)

#define ORT_DISALLOW_MOVE(TypeName) \
  TypeName(TypeName&&) = delete;    \
  TypeName& operator=(TypeName&&) = delete

#define ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName) \
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName);           \
  ORT_DISALLOW_MOVE(TypeName)

#ifdef _WIN32
using CHAR_TYPE = wchar_t;
#else
using CHAR_TYPE = char;
#endif

#if defined(__x86_64__)
#define ORT_FALSE_SHARING_BYTES 128
#else
#define ORT_FALSE_SHARING_BYTES 64
#endif
#define ORT_ALIGN_TO_AVOID_FALSE_SHARING alignas(ORT_FALSE_SHARING_BYTES)

#ifdef _WIN32
#define ORT_UNUSED_PARAMETER(x) (x)
#else
#define ORT_UNUSED_PARAMETER(x) (void)(x)
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324) /* Padding added to LoopCounterShard, LoopCounter for alignment */
#endif

static constexpr int CACHE_LINE_BYTES = 64;
static constexpr unsigned MAX_SHARDS = 8;

static constexpr int TaskGranularityFactor = 4;

using TimePoint = std::chrono::high_resolution_clock::time_point;

struct TensorOpCost {
    double bytes_loaded;
    double bytes_stored;
    double compute_cycles;
};

enum class ThreadPoolType : uint8_t {
    INTRA_OP,
    INTER_OP
};

inline void SpinPause() {
#if defined(_M_AMD64) || defined(__x86_64__)
    _mm_pause();
#endif
}

inline long long TimeDiffMicroSeconds(TimePoint start_time) {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
}

inline long long TimeDiffMicroSeconds(TimePoint start_time, TimePoint end_time) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
}

#endif //C__THREAD_UTILS_H
