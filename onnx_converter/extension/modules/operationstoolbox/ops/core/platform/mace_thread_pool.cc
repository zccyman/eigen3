//
// Created by shengyuan.shen on 2023/4/6.
//

#include "mace_thread_pool.h"
#include "environment.h"
#include "environment.h"
#include <sstream>

struct CPUFreq {
    size_t core_id;
    float freq;
};

inline void SpinWait(const std::atomic<int> &variable, const int value, const int64_t spin_wait_max_time=-1) {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t k = 1; variable.load(std::memory_order_acquire) == value; ++k) {
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

static int GetCpuCoresForPerfomance(
        const std::vector<CPUFreq> &cpu_freqs,
        const std::function<bool(const float &x, const float &y)> &comp) {
    float total_freq = std::accumulate(cpu_freqs.begin(), cpu_freqs.end(), 0,
                                       [](float accum, CPUFreq cpu_freq) {
                                           return accum + cpu_freq.freq;
                                       });
    int64_t valid_cpu_nums = std::count_if(cpu_freqs.begin(), cpu_freqs.end(),
                                           [](CPUFreq cpu_freq) {
                                               return cpu_freq.freq != 0;
                                           });
    float avg_freq = total_freq / valid_cpu_nums;

    int cores_to_use = 0;
    for (auto cpu_info : cpu_freqs) {
        if ((comp(cpu_info.freq, avg_freq)
             && cores_to_use < kMaxCpuCoresForPerformance)
            || cores_to_use < kMinCpuCoresForPerformance) {
            ++cores_to_use;
        }
    }

    return cores_to_use;
}

static TimesIntelliStatus GetCPUCoresToUse(const std::vector<float> &cpu_max_freqs,
                            const CPUAffinityPolicy policy,
                            int *thread_count,
                            std::vector<size_t> *cores) {
    if (cpu_max_freqs.empty()) {
        *thread_count = 1;
        std::cout << "CPU core is empty\n";
        return TimesIntelli_RUNTIME_ERROR;
    }
    *thread_count = std::max(*thread_count, 0);
    const int cpu_count = static_cast<int>(cpu_max_freqs.size());
    if (*thread_count == 0 || *thread_count > cpu_count) {
        *thread_count = cpu_count;
    }

    if (policy != CPUAffinityPolicy::AFFINITY_NONE) {
        std::vector<CPUFreq> cpu_freq(cpu_max_freqs.size());
        for (size_t i = 0; i < cpu_max_freqs.size(); ++i) {
            cpu_freq[i].core_id = i;
            cpu_freq[i].freq = cpu_max_freqs[i];
        }
        if (policy == CPUAffinityPolicy::AFFINITY_POWER_SAVE ||
            policy == CPUAffinityPolicy::AFFINITY_LITTLE_ONLY) {
            std::sort(cpu_freq.begin(),
                      cpu_freq.end(),
                      [=](const CPUFreq &lhs, const CPUFreq &rhs) {
                          return lhs.freq < rhs.freq;
                      });
        } else if (policy == CPUAffinityPolicy::AFFINITY_HIGH_PERFORMANCE ||
                   policy == CPUAffinityPolicy::AFFINITY_BIG_ONLY) {
            for (size_t i = 0; i < cpu_max_freqs.size(); ++i) {
                if (cpu_max_freqs[i] == 0) {
                    std::cout << "CPU maybe isolated, don't set CPU affinity\n";
                    return TimesIntelli_SUCCESS;
                }
            }
            std::sort(cpu_freq.begin(),
                      cpu_freq.end(),
                      [](const CPUFreq &lhs, const CPUFreq &rhs) {
                          return lhs.freq > rhs.freq;
                      });
        }

        // decide num of cores to use
        int cores_to_use = 0;
        if (policy == CPUAffinityPolicy::AFFINITY_BIG_ONLY) {
            cores_to_use =
                    GetCpuCoresForPerfomance(cpu_freq, std::greater_equal<float>());
        } else if (policy == CPUAffinityPolicy::AFFINITY_LITTLE_ONLY) {
            cores_to_use =
                    GetCpuCoresForPerfomance(cpu_freq, std::less_equal<float>());
        } else {
            cores_to_use = *thread_count;
        }
        if (cores_to_use < 0){
            std::cout << "number of cores to use should > 0\n";
            exit(-1);
        }
        cores->resize(static_cast<size_t>(cores_to_use));
        for (int i = 0; i < cores_to_use; ++i) {
            std::cout << "Bind thread to core: " << cpu_freq[i].core_id
                    << " with freq "
                    << cpu_freq[i].freq << std::endl;
            (*cores)[i] = static_cast<int>(cpu_freq[i].core_id);
        }
        if (*thread_count == 0 || *thread_count > cores_to_use) {
            *thread_count = cores_to_use;
        }
    }

    return TimesIntelli_SUCCESS;
}


MaceThreadPool::MaceThreadPool(const int thread_count,
                       const CPUAffinityPolicy policy)
        : event_(kThreadPoolNone),
          count_down_latch_(kThreadPoolSpinWaitTime) {
    int thread_count_ = thread_count;
    GetCPUMaxFreq(cpu_max_freqs_);

    std::vector<size_t> cores_to_use;
    GetCPUCoresToUse(cpu_max_freqs_, policy, &thread_count_, &cores_to_use);
    if (thread_count_ <= 0){
        std::cout << "thread_count is zero\n";
        exit(-1);
    }
    std::cout << "Use " << thread_count_ << " threads\n";

    if (!cores_to_use.empty()) {
        if (SchedSetAffinity(cores_to_use) != 0) {
            std::cout << "Failed to sched_set_affinity\n";
            exit(-1);
        }
    }

    default_tile_count_ = thread_count_;
    if (thread_count_ > 1) {
        default_tile_count_ = thread_count_ * kTileCountPerThread;
    }
    if (default_tile_count_ <= 0){
        std::cout << "default tile count should > 0\n";
        exit(-1);
    }

#ifdef USE_STD_THREAD
    threads_ = std::vector<std::thread>(static_cast<size_t>(thread_count_));
#else
#ifdef _WIN32
    threads_ = std::vector<wil::unique_handle>(static_cast<wil::unique_handle>(thread_count_));
#else
    threads_ = std::vector<pthread_t>(static_cast<pthread_t>(thread_count_));
#endif
#endif
    thread_infos_ = std::vector<ThreadInfo>(static_cast<size_t>(thread_count_));
    for (auto &thread_info : thread_infos_) {
        thread_info.cpu_cores = cores_to_use;
    }
#ifndef USE_STD_THREAD
    stack_size = 0;
#ifndef _WIN32
    affinity = GetThreadAffinityMasks();
#endif
#endif
}


MaceThreadPool::~MaceThreadPool() {
    // Clear affinity of main thread
    if (!cpu_max_freqs_.empty()) {
        //std::cout << cpu_max_freqs_.size() <<  std::endl;
        std::vector<size_t> cores(cpu_max_freqs_.size());
        for (size_t i = 0; i < cores.size(); ++i) {
            cores[i] = i;
        }
        SchedSetAffinity(cores);
    }

    Destroy();
#ifdef _WIN32
    std::cout << "Windows Platform ThreadPool Destory!\n";
#else
    std::cout << "LinuxBase Platform ThreadPool Destory!\n";
#endif
}


void MaceThreadPool::Init() {
    std::cout << "Init thread pool";
    if (threads_.size() <= 1) {
        return;
    }
    count_down_latch_.Reset(static_cast<int>(threads_.size() - 1));
    event_ = kThreadPoolInit;
    for (size_t i = 1; i < threads_.size(); ++i) {
#ifdef USE_STD_THREAD
        threads_[i] = std::thread(&MaceThreadPool::ThreadLoop, this, i);
#else
#ifdef _WIN32
        unsigned threadID;
        threads_[i].reset(reinterpret_cast<HANDLE>(_beginthreadex(nullptr, stack_size, &MaceThreadPool::ThreadLoop
                                                    , 0, &threadID)));
#else
        pthread_attr_t attr;
        int s = pthread_attr_init(&attr);
        if (s != 0) {
            auto [err_no, err_msg] = GetSystemError();
            std::cout << "pthread_attr_init failed, error code: " << err_no << " error msg: " << err_msg << std::endl;
            exit(-1);
        }
        if (stack_size > 0) {
            s = pthread_attr_setstacksize(&attr, stack_size);
            if (s != 0) {
                auto [err_no, err_msg] = GetSystemError();
                std::cout << "pthread_attr_setstacksize failed, error code: " << err_no << " error msg: " << err_msg << std::endl;
                exit(-1);
            }
        }
        s = pthread_create(&threads_[i], &attr, &MaceThreadPool::ThreadLoop, this);
        if (s != 0) {
            auto [err_no, err_msg] = GetSystemError();
            std::cout << "pthread_create failed, error code: " << err_no << " error msg: " << err_msg << std::endl;
            exit(-1);
        }
#if !defined(__APPLE__) && !defined(__ANDROID__) && !defined(__wasm__)
        if (!affinity.empty()) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(affinity[index], &cpuset);
            s = pthread_setaffinity_np(threads_[i], sizeof(cpu_set_t), &cpuset);
            if (s != 0) {
                auto [err_no, err_msg] = GetSystemError();
                std::cout << "pthread_setaffinity_np failed, error code: " << err_no << " error msg: " << err_msg <<std::endl;
                exit(-1);
            }
        }
#endif
#endif // end for _WIN32
#endif // end for USE_STD_THREAD
    }
    count_down_latch_.Wait();
}


void MaceThreadPool::Run(const std::function<void(const index_t_)> &func,
                     const int64_t iterations) {
    const size_t thread_count = threads_.size();
    const index_t_ iters_per_thread = iterations / thread_count;
    const index_t_ remainder = iterations % thread_count;
    index_t_ iters_offset = 0;
#ifdef USE_STD_MUTEX
    std::unique_lock<std::mutex> run_lock(run_mutex_);
#else
    std::unique_lock<TimesIntelliMutex> run_lock(run_mutex_);
#endif

    for (size_t i = 0; i < thread_count; ++i) {
        int64_t range_len =
                iters_per_thread + (static_cast<int64_t>(i) < remainder);
        thread_infos_[i].range_start = iters_offset;
        thread_infos_[i].range_len = range_len;
        thread_infos_[i].range_end = iters_offset + range_len;
        thread_infos_[i].func = reinterpret_cast<uintptr_t>(&func);
        iters_offset = thread_infos_[i].range_end;
    }

    count_down_latch_.Reset(static_cast<int>(thread_count - 1));
    {
#ifdef USE_STD_MUTEX
        std::unique_lock<std::mutex> m(event_mutex_);
#else
        std::unique_lock<TimesIntelliMutex> m(event_mutex_);
#endif
        event_.store(kThreadPoolRun | ~(event_ | kThreadPoolEventMask),
                     std::memory_order::memory_order_release);
        event_cond_.notify_all();
    }

    ThreadRun(0);
    count_down_latch_.Wait();
}


void MaceThreadPool::Destroy() {
    std::cout << "Destroy thread pool";
    if (threads_.size() <= 1) {
        return;
    }
#ifdef USE_STD_MUTEX
    std::unique_lock<std::mutex> run_lock(run_mutex_);
#else
    std::unique_lock<TimesIntelliMutex> run_lock(run_mutex_);
#endif

    count_down_latch_.Wait();
    {
#ifdef USE_STD_MUTEX
        std::unique_lock<std::mutex> m(event_mutex_);
#else
        std::unique_lock<TimesIntelliMutex> m(event_mutex_);
#endif
        event_.store(kThreadPoolShutdown, std::memory_order::memory_order_release);
        event_cond_.notify_all();
    }

    for (size_t i = 1; i < threads_.size(); ++i) {
#ifdef USE_STD_THREAD
        if (threads_[i].joinable()) {
            threads_[i].join();
        } else {
            //std::cout << "Thread: " << threads_[i].get_id() << " not joinable" << std::endl;
        }
#else
#ifdef _WIN32
      DWORD waitStatus = WaitForSingleObject(hThread.get(), INFINITE);
      FAIL_FAST_LAST_ERROR_IF(waitStatus == WAIT_FAILED);
      int threadID = GetCurrentThreadId();
#else
      void* res;
#ifdef NDEBUG
      pthread_join(hThread, &res);
#else
      int ret = pthread_join(hThread, &res);
      assert(ret == 0);
#endif
#endif
#endif
    }
}

// Event is executed synchronously.

void MaceThreadPool::ThreadLoop(size_t tid) {
    if (!thread_infos_[tid].cpu_cores.empty()) {
        if (SchedSetAffinity(thread_infos_[tid].cpu_cores) != 0) {
            std::cout << "Failed to sched set affinity for tid: " << tid;
            exit(-1);
        }
    }

    int last_event = kThreadPoolNone;

    for (;;) {
        SpinWait(event_, last_event, kThreadPoolSpinWaitTime);
        if (event_.load(std::memory_order::memory_order_acquire) == last_event) {
#ifdef USE_STD_MUTEX
            std::unique_lock<std::mutex> m(event_mutex_);
#else
            std::unique_lock<TimesIntelliMutex> m(event_mutex_);
#endif
            while (event_ == last_event) {
                event_cond_.wait(m);
            }
        }

        int event = event_.load(std::memory_order::memory_order_acquire);
        switch (event & kThreadPoolEventMask) {
            case kThreadPoolInit: {
                count_down_latch_.CountDown();
                break;
            }

            case kThreadPoolRun: {
                ThreadRun(tid);
                count_down_latch_.CountDown();
                break;
            }

            case kThreadPoolShutdown: return;
            default: break;
        }

        last_event = event;
    }
}


void MaceThreadPool::ThreadRun(size_t tid) {
    ThreadInfo &thread_info = thread_infos_[tid];
    uintptr_t func_ptr = thread_info.func;
    const std::function<void(int64_t)> *func =
            reinterpret_cast<const std::function<void(int64_t)> *>(func_ptr);
    // do own work
    int64_t range_len;
    while ((range_len = thread_info.range_len) > 0) {
        if (thread_info.range_len.compare_exchange_strong(range_len,
                                                          range_len - 1)) {
            func->operator()(thread_info.range_start++);
        }
    }

    // steal other threads' work
    size_t thread_count = threads_.size();
    for (size_t t = (tid + 1) % thread_count; t != tid;
         t = (t + 1) % thread_count) {
        ThreadInfo &other_thread_info = thread_infos_[t];
        uintptr_t other_func_ptr = other_thread_info.func;
        const std::function<void(int64_t)> *other_func =
                reinterpret_cast<const std::function<void(int64_t)> *>(
                        other_func_ptr);
        while ((range_len = other_thread_info.range_len) > 0) {
            if (other_thread_info.range_len.compare_exchange_strong(range_len,
                                                                    range_len
                                                                    - 1)) {
                int64_t tail = other_thread_info.range_end--;
                other_func->operator()(tail - 1);
            }
        }
    }
}


void MaceThreadPool::Compute1D(const std::function<void(index_t_,
                                                index_t_,
                                                index_t_)> &func,
                           const index_t_ start,
                           const index_t_ end,
                           const index_t_ step,
                           index_t_ tile_size,
                           const int cost_per_item) {
    if (start >= end) {
        return;
    }

    const index_t_ items = 1 + (end - start - 1) / step;
    if (threads_.size() <= 1 || (cost_per_item >= 0
                                 && items * cost_per_item < kMaxCostUsingSingleThread)) {
        func(start, end, step);
        return;
    }

    if (tile_size == 0) {
        tile_size = 1 + (items - 1) / default_tile_count_;
    }

    const index_t_ step_tile_size = step * tile_size;
    const index_t_ tile_count = RoundUpDiv(items, tile_size);
    Run([=](index_t_ tile_idx) {
        const index_t_ tile_start = start + tile_idx * step_tile_size;
        const index_t_ tile_end = std::min(end, tile_start + step_tile_size);
        func(tile_start, tile_end, step);
    }, tile_count);
}


void MaceThreadPool::Compute2D(const std::function<void(const index_t_,
                                                    const index_t_,
                                                    const index_t_,
                                                    const index_t_,
                                                    const index_t_,
                                                    const index_t_)> &func,
                           const index_t_ start0,
                           const index_t_ end0,
                           const index_t_ step0,
                           const index_t_ start1,
                           const index_t_ end1,
                           const index_t_ step1,
                           index_t_ tile_size0,
                           index_t_ tile_size1,
                           const int cost_per_item) {
    if (start0 >= end0 || start1 >= end1) {
        return;
    }

    const index_t_ items0 = 1 + (end0 - start0 - 1) / step0;
    const index_t_ items1 = 1 + (end1 - start1 - 1) / step1;
    if (threads_.size() <= 1 || (cost_per_item >= 0
                                 && items0 * items1 * cost_per_item < kMaxCostUsingSingleThread)) {
        func(start0, end0, step0, start1, end1, step1);
        return;
    }

    if (tile_size0 == 0 || tile_size1 == 0) {
        if (items0 >= default_tile_count_) {
            tile_size0 = 1 + (items0 - 1) / default_tile_count_;
            tile_size1 = items1;
        } else {
            tile_size0 = 1;
            tile_size1 = 1 + (items1 * items0 - 1) / default_tile_count_;
        }
    }

    const index_t_ step_tile_size0 = step0 * tile_size0;
    const index_t_ step_tile_size1 = step1 * tile_size1;
    const index_t_ tile_count0 = RoundUpDiv(items0, tile_size0);
    const index_t_ tile_count1 = RoundUpDiv(items1, tile_size1);

    Run([=](index_t_ tile_idx) {
        const index_t_ tile_idx0 = tile_idx / tile_count1;
        const index_t_ tile_idx1 = tile_idx - tile_idx0 * tile_count1;
        const index_t_ tile_start0 = start0 + tile_idx0 * step_tile_size0;
        const index_t_ tile_end0 = std::min(end0, tile_start0 + step_tile_size0);
        const index_t_ tile_start1 = start1 + tile_idx1 * step_tile_size1;
        const index_t_ tile_end1 = std::min(end1, tile_start1 + step_tile_size1);
        func(tile_start0, tile_end0, step0, tile_start1, tile_end1, step1);
    }, tile_count0 * tile_count1);
}


void MaceThreadPool::Compute3D(const std::function<void(const index_t_,
                                                    const index_t_,
                                                    const index_t_,
                                                    const index_t_,
                                                    const index_t_,
                                                    const index_t_,
                                                    const index_t_,
                                                    const index_t_,
                                                    const index_t_)> &func,
                           const index_t_ start0,
                           const index_t_ end0,
                           const index_t_ step0,
                           const index_t_ start1,
                           const index_t_ end1,
                           const index_t_ step1,
                           const index_t_ start2,
                           const index_t_ end2,
                           const index_t_ step2,
                           index_t_ tile_size0,
                           index_t_ tile_size1,
                           index_t_ tile_size2,
                           const int cost_per_item) {
    if (start0 >= end0 || start1 >= end1 || start2 >= end2) {
        return;
    }

    const index_t_ items0 = 1 + (end0 - start0 - 1) / step0;
    const index_t_ items1 = 1 + (end1 - start1 - 1) / step1;
    const index_t_ items2 = 1 + (end2 - start2 - 1) / step2;
    if (threads_.size() <= 1 || (cost_per_item >= 0
                                 && items0 * items1 * items2 * cost_per_item
                                    < kMaxCostUsingSingleThread)) {
        func(start0, end0, step0, start1, end1, step1, start2, end2, step2);
        return;
    }

    if (tile_size0 == 0 || tile_size1 == 0 || tile_size2 == 0) {
        if (items0 >= default_tile_count_) {
            tile_size0 = 1 + (items0 - 1) / default_tile_count_;
            tile_size1 = items1;
            tile_size2 = items2;
        } else {
            tile_size0 = 1;
            const index_t_ items01 = items1 * items0;
            if (items01 >= default_tile_count_) {
                tile_size1 = 1 + (items01 - 1) / default_tile_count_;
                tile_size2 = items2;
            } else {
                tile_size1 = 1;
                tile_size2 = 1 + (items01 * items2 - 1) / default_tile_count_;
            }
        }
    }

    const index_t_ step_tile_size0 = step0 * tile_size0;
    const index_t_ step_tile_size1 = step1 * tile_size1;
    const index_t_ step_tile_size2 = step2 * tile_size2;
    const index_t_ tile_count0 = RoundUpDiv(items0, tile_size0);
    const index_t_ tile_count1 = RoundUpDiv(items1, tile_size1);
    const index_t_ tile_count2 = RoundUpDiv(items2, tile_size2);
    const index_t_ tile_count12 = tile_count1 * tile_count2;

    Run([=](index_t_ tile_idx) {
        const index_t_ tile_idx0 = tile_idx / tile_count12;
        const index_t_ tile_idx12 = tile_idx - tile_idx0 * tile_count12;
        const index_t_ tile_idx1 = tile_idx12 / tile_count2;
        const index_t_ tile_idx2 = tile_idx12 - tile_idx1 * tile_count2;
        const index_t_ tile_start0 = start0 + tile_idx0 * step_tile_size0;
        const index_t_ tile_end0 = std::min(end0, tile_start0 + step_tile_size0);
        const index_t_ tile_start1 = start1 + tile_idx1 * step_tile_size1;
        const index_t_ tile_end1 = std::min(end1, tile_start1 + step_tile_size1);
        const index_t_ tile_start2 = start2 + tile_idx2 * step_tile_size2;
        const index_t_ tile_end2 = std::min(end2, tile_start2 + step_tile_size2);
        func(tile_start0,
             tile_end0,
             step0,
             tile_start1,
             tile_end1,
             step1,
             tile_start2,
             tile_end2,
             step2);
    }, tile_count0 * tile_count12);
}

TimesIntelliStatus Engine::Init(){
    thread_pool_ = new MaceThreadPool(this->num_threads(), this->cpu_affinity_policy());
    return TimesIntelli_SUCCESS;
}

TimesIntelliStatus Engine::SetCPUThreadPolicy(int num_threads,
                                             CPUAffinityPolicy policy){
    num_threads_ = num_threads;
    cpu_affinity_policy_ = policy;
    return TimesIntelli_SUCCESS;
}

int32_t Engine::num_threads(){
    return this->num_threads_;
}

CPUAffinityPolicy Engine::cpu_affinity_policy(){
    return this->cpu_affinity_policy_;
}

Engine::~Engine(){
    if (thread_pool_){
        delete thread_pool_;
        thread_pool_ = nullptr;
    }
}