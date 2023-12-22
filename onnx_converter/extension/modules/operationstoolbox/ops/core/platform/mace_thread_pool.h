//
// Created by shengyuan.shen on 2023/4/6.
//

#ifndef C__MACE_THREAD_POOL_H
#define C__MACE_THREAD_POOL_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <mutex>
#include <thread>
#include <numeric>
#include <limits>
#include <atomic>
#include <functional>
#include <type_traits>
#include <condition_variable>

#include "timesintelli_mutex.h"
#include "environment.h"
#include "thread_utils.h"
#include "common.h"

/*
 * copy-wright: xiaomi
 * thread pool using std::thread, this method has not a lot of attribute of threadd, for example: pthread_attr_setstacksize
 *
 * */

//#define USE_STD_MUTEX

#define TUNUSED(var) (void)(var)

#define USE_STD_THREAD

#define ANDROID_THREAD_POOL

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
#include <pthread.h>
#endif

typedef int64_t index_t_;

constexpr int kThreadPoolSpinWaitTime = 2000000;  // ns
constexpr int kTileCountPerThread = 2;
constexpr int kMaxCostUsingSingleThread = 100;
constexpr int kMinCpuCoresForPerformance = 3;
constexpr int kMaxCpuCoresForPerformance = 5;

enum {
    kThreadPoolNone = 0,
    kThreadPoolInit = 1,
    kThreadPoolRun = 2,
    kThreadPoolShutdown = 4,
    kThreadPoolEventMask = 0x7fffffff
};

enum TimesIntelliStatus {
    TimesIntelli_SUCCESS = 0,
    TimesIntelli_INVALID_ARGS = 1,
    TimesIntelli_OUT_OF_RESOURCES = 2,
    TimesIntelli_UNSUPPORTED = 3,
    TimesIntelli_RUNTIME_ERROR = 4
};

enum CPUAffinityPolicy {
    AFFINITY_NONE = 0,
    AFFINITY_BIG_ONLY = 1,
    AFFINITY_LITTLE_ONLY = 2,
    AFFINITY_HIGH_PERFORMANCE = 3,
    AFFINITY_POWER_SAVE = 4,
};

template <typename Integer>
Integer RoundUp(Integer i, Integer factor) {
    return (i + factor - 1) / factor * factor;
}

template <typename Integer, uint32_t factor>
Integer RoundUpDiv(Integer i) {
    return (i + factor - 1) / factor;
}

// Partial specialization of function templates is not allowed
template <typename Integer>
Integer RoundUpDiv4(Integer i) {
    return (i + 3) >> 2;
}

template <typename Integer>
Integer RoundUpDiv8(Integer i) {
    return (i + 7) >> 3;
}

template <typename Integer>
Integer RoundUpDiv(Integer i, Integer factor) {
    return (i + factor - 1) / factor;
}

inline void SpinWaitUntil(const std::atomic<int> &variable,
                          const int value,
                          const int64_t spin_wait_max_time = -1) {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t k = 1; variable.load(std::memory_order_acquire) != value; ++k) {
        if (spin_wait_max_time > 0 && k % 1000 == 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            int64_t elapse =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                            end_time - start_time).count();
            if (elapse > spin_wait_max_time) {
                break;
            }
        }
    }
}

class CountDownLatch {
public:
    explicit CountDownLatch(int64_t spin_timeout)
            : spin_timeout_(spin_timeout), count_(0) {}
    CountDownLatch(int64_t spin_timeout, int count)
            : spin_timeout_(spin_timeout), count_(count) {}

    void Wait() {
        if (spin_timeout_ > 0) {
            SpinWaitUntil(count_, 0, spin_timeout_);
        }
        if (count_.load(std::memory_order_acquire) != 0) {
#ifdef USE_STD_MUTEX
            std::unique_lock<std::mutex> m(mutex_);
#else
            std::unique_lock<TimesIntelliMutex> m(mutex_);
#endif
            /*while (count_.load(std::memory_order_acquire) != 0) {
                cond_.wait(m);
            }*/
        }
    }

    void CountDown() {
        if (count_.fetch_sub(1, std::memory_order_release) == 1) {
#ifdef USE_STD_MUTEX
            std::unique_lock<std::mutex> m(mutex_);
#else
            std::unique_lock<TimesIntelliMutex> m(mutex_);
#endif
            cond_.notify_all();
        }
    }

    void Reset(int count) {
        count_.store(count, std::memory_order_release);
    }

    int count() const {
        return count_;
    }

private:
    int64_t spin_timeout_;
    std::atomic<int> count_;
#ifdef USE_STD_MUTEX
    std::mutex mutex_;
    std::condition_variable cond_;
#else
    TimesIntelliMutex mutex_;
    TimesIntelliCondVar cond_;
#endif
};

class MaceThreadPool {
public:
    MaceThreadPool(const int thread_count,
               const CPUAffinityPolicy affinity_policy);

    ~MaceThreadPool();

    void Init();

    void Run(const std::function<void(const index_t_)> &func,
             const int64_t iterations);

    void Compute1D(const std::function<void(index_t_ /* start */,
                                            index_t_ /* end */,
                                            index_t_ /* step */)> &func,
                   index_t_ start,
                   index_t_ end,
                   index_t_ step,
                   index_t_ tile_size = 0,
                   int cost_per_item = -1);

    void Compute2D(const std::function<void(index_t_ /* start */,
                                            index_t_ /* end */,
                                            index_t_ /* step */,
                                            index_t_ /* start */,
                                            index_t_ /* end */,
                                            index_t_ /* step */)> &func,
                   index_t_ start0,
                   index_t_ end0,
                   index_t_ step0,
                   index_t_ start1,
                   index_t_ end1,
                   index_t_ step1,
                   index_t_ tile_size0 = 0,
                   index_t_ tile_size1 = 0,
                   int cost_per_item = -1);

    void Compute3D(const std::function<void(index_t_ /* start */,
                                            index_t_ /* end */,
                                            index_t_ /* step */,
                                            index_t_ /* start */,
                                            index_t_ /* end */,
                                            index_t_ /* step */,
                                            index_t_ /* start */,
                                            index_t_ /* end */,
                                            index_t_ /* step */)> &func,
                   index_t_ start0,
                   index_t_ end0,
                   index_t_ step0,
                   index_t_ start1,
                   index_t_ end1,
                   index_t_ step1,
                   index_t_ start2,
                   index_t_ end2,
                   index_t_ step2,
                   index_t_ tile_size0 = 0,
                   index_t_ tile_size1 = 0,
                   index_t_ tile_size2 = 0,
                   int cost_per_item = -1);

private:
    void Destroy();
    void ThreadLoop(size_t tid);
    void ThreadRun(size_t tid);

    std::atomic<int> event_;
    CountDownLatch count_down_latch_;
#ifdef USE_STD_MUTEX
    std::mutex event_mutex_;
    std::condition_variable event_cond_;
    std::mutex run_mutex_;
#else
    TimesIntelliMutex event_mutex_;
    TimesIntelliCondVar event_cond_;
    TimesIntelliMutex run_mutex_;
#endif
    struct ThreadInfo {
        std::atomic<index_t_> range_start;
        std::atomic<index_t_> range_end;
        std::atomic<index_t_> range_len;
        uintptr_t func;
        std::vector<size_t> cpu_cores;
    };
    std::vector<ThreadInfo> thread_infos_;
#ifdef USE_STD_THREAD
    std::vector<std::thread> threads_;
#else
    std::vector<size_t> affinity;
    size_t stack_size;
#ifdef _WIN32
    std::vector<wil::unique_handle> threads_;
#else
    std::vector<pthread_t> threads_;
#endif // end for _WIN32
#endif
    std::vector<float> cpu_max_freqs_;

    int64_t default_tile_count_;
};

/*enum AcceleratorCachePolicy {
    ACCELERATOR_CACHE_NONE = 0,
    ACCELERATOR_CACHE_STORE = 1,
    ACCELERATOR_CACHE_LOAD = 2,
    APU_CACHE_LOAD_OR_STORE = 3,
};*/

static std::unique_ptr<MaceThreadPool>
CreateThreadPool(int num_threads, CPUAffinityPolicy policy) {
    if (num_threads == 1)
        return nullptr;

    return std::make_unique<MaceThreadPool>(num_threads, policy);
}

class Engine{
public:
    Engine():num_threads_(-1),
             cpu_affinity_policy_(CPUAffinityPolicy::AFFINITY_NONE){}

    TIMESINTELLI_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Engine);

    TimesIntelliStatus Init();

    TimesIntelliStatus SetCPUThreadPolicy(int num_threads, CPUAffinityPolicy policy);

    int32_t num_threads();

    CPUAffinityPolicy cpu_affinity_policy();

    /*TimesIntelliStatus SetGPUContext(std::shared_ptr<OpenclContext> context);*/

    /*TimesIntelliStatus SetAcceleratorCache(AcceleratorCachePolicy policy,
                                           const std::string &binary_file,
                                           const std::string &storage_file);*/
    ~Engine();


private:
    int32_t num_threads_;
    CPUAffinityPolicy cpu_affinity_policy_;
    MaceThreadPool* thread_pool_;
};

#endif //C__MACE_THREAD_POOL_H
