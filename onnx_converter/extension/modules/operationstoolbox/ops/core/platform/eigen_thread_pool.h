//
// Created by shengyuan.shen on 2023/4/10.
//

#ifndef C__EIGEN_THREAD_POOL_H
#define C__EIGEN_THREAD_POOL_H

/*
 * copy-wright: microsoft software
 * thread pool using platform thread, this method has a lot of attribute of threadd, for example: pthread_attr_setstacksize,
 * but we will written different code for operation system
 *
 * */
#include <iostream>
#include <memory>
#include <string>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/ThreadPool"

#include "environment.h"
#include "thread_utils.h"
#include "timesintelli_mutex.h"

enum class StealAttemptKind {
    TRY_ONE,
    TRY_ALL,
};

enum class PushResult {
    REJECTED,
    ACCEPTED_IDLE,
    ACCEPTED_BUSY
};

struct OrtThreadPoolParams {
    //0: Use default setting. (All the physical cores or half of the logical cores)
    //1: Don't create thread pool
    //n: Create a thread pool with n threads.
    int thread_pool_size = 0;
    //If it is true and thread_pool_size = 0, populate the thread affinity information in OrtThreadOptions.
    //Otherwise if the thread_options has affinity information, we'll use it and set it.
    //In the other case, don't set affinity
    bool auto_set_affinity = false;
    //If it is true, the thread pool will spin a while after the queue became empty.
    bool allow_spinning = true;
    //It it is non-negative, thread pool will split a task by a decreasing block size
    //of remaining_of_total_iterations / (num_of_threads * dynamic_block_base_)
    int dynamic_block_base_ = 0;

    unsigned int stack_size = 0;
    //Index is thread id, value is processor ID
    //If the vector is empty, no explict affinity binding
    size_t* affinity_vec = nullptr;
    size_t affinity_vec_len = 0;
    const ORTCHAR_T* name = nullptr;

    // Set or unset denormal as zero
    bool set_denormal_as_zero = false;

    // members to manage custom threads
    /*OrtCustomCreateThreadFn custom_create_thread_fn = nullptr;
    void* custom_thread_creation_options = nullptr;
    OrtCustomJoinThreadFn custom_join_thread_fn = nullptr;*/
    void* custom_create_thread_fn = nullptr;
    void* custom_thread_creation_options = nullptr;
    void* custom_join_thread_fn = nullptr;
};

struct alignas(CACHE_LINE_BYTES) LoopCounterShard {
    ::std::atomic<uint64_t> _next{0};
    uint64_t _end{0};
};

static_assert(sizeof(LoopCounterShard) == CACHE_LINE_BYTES, "Expected loop counter shards to match cache-line size");

class alignas(CACHE_LINE_BYTES) LoopCounter {
public:
    LoopCounter(uint64_t num_iterations,
                uint64_t d_of_p,
                uint64_t block_size = 1) : _num_shards(GetNumShards(num_iterations,
                                                                    d_of_p,
                                                                    block_size)) {
        // Divide the iteration space between the shards.  If the iteration
        // space does not divide evenly into shards of multiples of
        // block_size then the final shard is left uneven.

        auto num_blocks = num_iterations / block_size;
        auto blocks_per_shard = num_blocks / _num_shards;
        auto iterations_per_shard = blocks_per_shard * block_size;

        for (uint64_t shard = 0; shard < _num_shards; shard++) {
            // Initialize with a relaxed store; synchronization with worker
            // threads is provided via the thread pool
            _shards[shard]._next.store(shard * iterations_per_shard,
                                       ::std::memory_order_relaxed);

            bool is_last_shard = (shard == _num_shards - 1);
            _shards[shard]._end = is_last_shard ? num_iterations : ((shard + 1) * iterations_per_shard);
        }
    }

    // Allocate each thread to a home shard, from which it starts
    // claiming iterations.
    //
    // We use the worker ID provided by the thread pool as the basis of
    // this allocation.  Doing so promotes locality between successive
    // loops: the worker that runs a given iteration in one loop will
    // tend to run the same iterations in the next loop.  This helps
    // operators with a series of short loops, such as GRU.

    unsigned GetHomeShard(unsigned idx) const {
        return idx % _num_shards;
    }

    // Attempt to claim iterations from the sharded counter.  The function either
    // returns true, along with a block of exactly block_size iterations, or it returns false
    // if all of the iterations have been claimed.
    bool ClaimIterations(unsigned my_home_shard,
                         unsigned& my_shard,
                         uint64_t& my_start,
                         uint64_t& my_end,
                         uint64_t block_size) {
        do {
            if (_shards[my_shard]._next < _shards[my_shard]._end) {
                // Appears to be work in the current shard, try to claim with atomic fetch-and-add
                uint64_t temp_start = _shards[my_shard]._next.fetch_add(block_size);
                if (temp_start < _shards[my_shard]._end) {
                    my_start = temp_start;
                    my_end = std::min(_shards[my_shard]._end, temp_start + block_size);
                    return true;
                }
            }
            // Work in the current shard is exhausted, move to the next shard, until
            // we are back at the home shard.
            my_shard = (my_shard + 1) % _num_shards;
        } while (my_shard != my_home_shard);
        return false;
    }

private:
    // Derive the number of shards to use for a given loop.  We require
    // at least one block of work per shard, and subject to the
    // constraints:
    //
    // - We use no more than MAX_SHARDS (limiting the amount of space needed
    //   for the LoopCounter, and work needed to confirm that all shards have been
    //   completed at the end of a loop).
    //
    // - The number of shards is <= the number of threads (d_of_p).
    //   Hence, at low thread counts, each of N threads will get its own
    //   shard representing 1/N of the work.
    constexpr static unsigned GetNumShards(uint64_t num_iterations,
                                           uint64_t d_of_p,
                                           uint64_t block_size) {
        unsigned num_shards = 0;
        auto num_blocks = num_iterations / block_size;
        if (num_blocks == 0) {
            num_shards = 1;
        } else if (num_blocks < MAX_SHARDS) {
            num_shards = static_cast<unsigned>(num_blocks);
        } else {
            num_shards = MAX_SHARDS;
        }
        if (num_shards > d_of_p) {
            num_shards = static_cast<unsigned>(d_of_p);
        }
        return num_shards;
    }

    alignas(CACHE_LINE_BYTES) LoopCounterShard _shards[MAX_SHARDS];
    const unsigned _num_shards;
};

// Parameters that are required to create a set of threads for a thread pool
struct OrtThreadOptions {
    // Stack size for a new thread. If it is 0, the operating system uses the same value as the stack that's specified for
    // the main thread, which is usually set in the main executable(not controlled by onnxruntime.dll).
    unsigned int stack_size = 0;

    // Thread affinity means a thread can only run on the logical processors that the thread is allowed to run on.
    // If the vector is not empty, set the affinity of each thread to just one CPU.
    // Index is thread index, value is CPU ID, starting from zero. For example, the first thread in the pool will be bound
    // to the logical processor with id of affinity[0]. If the vector is empty, the thread can run on all the processors
    // its process can run on. NOTE: When hyperthreading is enabled, for example, on a 4 cores 8 physical threads CPU,
    // processor group [0,1,2,3] may only contain half of the physical cores.
    std::vector<size_t> affinity;

    // Set or unset denormal as zero.
    bool set_denormal_as_zero = false;
    /*
     * custom thread function interface
     * */
    void* custom_create_thread_fn = nullptr;
    void* custom_thread_creation_options = nullptr;
    void* custom_join_thread_fn = nullptr;
    int dynamic_block_base_ = 0;
};

class OrtEnvThread {
public:
    virtual ~OrtEnvThread() = default;

protected:
    void* custom_create_thread_fn = nullptr;
    void* custom_thread_creation_options = nullptr;
    void* custom_join_thread_fn = nullptr;
    void* custom_thread_handle = nullptr;
};

#ifdef _WIN32

class WindowsThread : public OrtEnvThread{
 private:
  struct Param {
    const ORTCHAR_T* name_prefix;
    int index;
    unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param);
    Eigen::ThreadPoolInterface* param;
    const OrtThreadOptions& thread_options;
    Param(const ORTCHAR_T* name_prefix1,
          int index1,
          unsigned (*start_address1)(int id, Eigen::ThreadPoolInterface* param),
          Eigen::ThreadPoolInterface* param1,
          const OrtThreadOptions& thread_options1) : name_prefix(name_prefix1), index(index1), start_address(start_address1), param(param1), thread_options(thread_options1) {}
  };

 public:
  WindowsThread(const ORTCHAR_T* name_prefix, int index,
                unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param), Eigen::ThreadPoolInterface* param,
                const OrtThreadOptions& thread_options) {
    std::unique_ptr<Param> local_param = std::make_unique<Param>(name_prefix, index, start_address, param, thread_options);

    hThread.reset(reinterpret_cast<HANDLE>(_beginthreadex(nullptr, thread_options.stack_size, ThreadMain,
                                                        local_param.release(), 0,
                                                        &threadID)));
  }

  ~WindowsThread() override {
      DWORD waitStatus = WaitForSingleObject(hThread.get(), INFINITE);
      FAIL_FAST_LAST_ERROR_IF(waitStatus == WAIT_FAILED);
  }

 private:
  typedef HRESULT(WINAPI* SetThreadDescriptionFunc)(HANDLE hThread, PCWSTR lpThreadDescription);

#pragma warning(push)
#pragma warning(disable : 6387)
  static unsigned __stdcall ThreadMain(void* param) {
    std::unique_ptr<Param> p((Param*)param);
    // TODO: should I try to use SetThreadSelectedCpuSets?
    if (!p->thread_options.affinity.empty())
      SetThreadAffinityMask(GetCurrentThread(), p->thread_options.affinity[p->index]);
#if WINVER >= _WIN32_WINNT_WIN10
    constexpr SetThreadDescriptionFunc pSetThrDesc = SetThreadDescription;
#elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
    HMODULE kernelModule = GetModuleHandle(TEXT("kernel32.dll"));
    // kernel32.dll is always loaded
    assert(kernelModule != nullptr);
    auto pSetThrDesc =
        (SetThreadDescriptionFunc)GetProcAddress(kernelModule, "SetThreadDescription");
#else
    constexpr SetThreadDescriptionFunc pSetThrDesc = nullptr;
#endif
    if (pSetThrDesc != nullptr) {
      const ORTCHAR_T* name_prefix =
          (p->name_prefix == nullptr || wcslen(p->name_prefix) == 0) ? L"onnxruntime" : p->name_prefix;
      std::wostringstream oss;
      oss << name_prefix << "-" << p->index;
      // Ignore the error
      (void)pSetThrDesc(GetCurrentThread(), oss.str().c_str());
    }
    unsigned ret = 0;
    ORT_TRY {
      ret = p->start_address(p->index, p->param);
    }
    ORT_CATCH(const std::exception&) {
      p->param->Cancel();
      ret = 1;
    }
    return ret;
  }
#pragma warning(pop)

  static void __stdcall CustomThreadMain(void* param) {
    std::unique_ptr<Param> p((Param*)param);
    ORT_TRY {
      p->start_address(p->index, p->param);
    }
    ORT_CATCH(const std::exception&) {
      p->param->Cancel();
    }
  }
  unsigned threadID = 0;
  wil::unique_handle hThread;
};

#else

class PosixThread : public OrtEnvThread{
private:
    struct Param {
        const ORTCHAR_T* name_prefix;
        int index;
        unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param);
        Eigen::ThreadPoolInterface* param;
        const OrtThreadOptions& thread_options;
    };

public:
    PosixThread(const ORTCHAR_T* name_prefix, int index,
                unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param), Eigen::ThreadPoolInterface* param,
                const OrtThreadOptions& thread_options) {
        pthread_attr_t attr;
        int s = pthread_attr_init(&attr);
        if (s != 0) {
            auto error_pair = GetSystemError();
            auto err_no = error_pair.first;
            auto err_msg =error_pair.second;
            std::cout << "pthread_attr_init failed, error code: " << err_no << " error msg: " << err_msg << std::endl;
            exit(-1);
        }
        if (thread_options.stack_size > 0) {
            s = pthread_attr_setstacksize(&attr, thread_options.stack_size);
            if (s != 0) {
                auto error_pair = GetSystemError();
                auto err_no = error_pair.first;
                auto err_msg =error_pair.second;
                std::cout << "pthread_attr_setstacksize failed, error code: " << err_no << " error msg: " << err_msg << std::endl;
                exit(-1);
            }
        }
        s = pthread_create(&hThread, &attr, ThreadMain,
                           new Param{name_prefix, index, start_address, param, thread_options});
        if (s != 0) {
            auto error_pair = GetSystemError();
            auto err_no = error_pair.first;
            auto err_msg =error_pair.second;
            std::cout << "pthread_create failed, error code: " << err_no << " error msg: " << err_msg << std::endl;
            exit(-1);
        }
#if !defined(__APPLE__) && !defined(__ANDROID__) && !defined(__wasm__)
        if (!thread_options.affinity.empty()) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(thread_options.affinity[index], &cpuset);
            s = pthread_setaffinity_np(hThread, sizeof(cpu_set_t), &cpuset);
            if (s != 0) {
                auto error_pair = GetSystemError();
                auto err_no = error_pair.first;
                auto err_msg =error_pair.second;
                std::cout << "pthread_setaffinity_np failed, error code: " << err_no << " error msg: " << err_msg <<std::endl;
                exit(-1);
            }
        }
#endif
    }

    ~PosixThread() override {

        void* res;
#ifdef NDEBUG
        pthread_join(hThread, &res);
#else
        int ret = pthread_join(hThread, &res);
        assert(ret == 0);
#endif

    }

private:
    static void* ThreadMain(void* param) {
        std::unique_ptr<Param> p((Param*)param);
        ORT_TRY {
            // Ignore the returned value for now
            p->start_address(p->index, p->param);
        }
        ORT_CATCH(const std::exception&) {
            //ignore any exceptions
        }
        return nullptr;
    }
    static void CustomThreadMain(void* param) {
        ThreadMain(param);
    }
    pthread_t hThread;
};

#endif // endif with win32 class of thread for Eigen::ThreadPoolInterface

struct PaddingToAvoidFalseSharing {
    char padding[ORT_FALSE_SHARING_BYTES];
};

/* Usage:
1. In executor, call Start() before profiling and Stop() to get profiled numbers;
2. Inside thread pool, call LogStart() before interested section and LogEnd... after to log elapsed time;
3. To extend, just add more events in enum Event before "All", and update GetEventName(...) accordingly;
4. Note LogStart must pair with either LogEnd or LogEndAndStart, otherwise ORT_ENFORCE will fail;
5. OrtThreadPoolProfiler is thread-safe.
*/

class OrtThreadPoolLoop;

class OrtThreadPoolProfiler {
public:
    enum ThreadPoolEvent {
        DISTRIBUTION = 0,
        DISTRIBUTION_ENQUEUE,
        RUN,
        WAIT,
        WAIT_REVOKE,
        MAX_EVENT
    };
    OrtThreadPoolProfiler(int num_threads, const CHAR_TYPE* threal_pool_name);
    ~OrtThreadPoolProfiler();
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrtThreadPoolProfiler);
    using Clock = std::chrono::high_resolution_clock;
    void Start();                  //called by executor to start profiling
    std::string Stop();            //called by executor to stop profiling and return collected numbers
    void LogStart();               //called in main thread to record the starting time point
    void LogEnd(ThreadPoolEvent);  //called in main thread to calculate and save the time elapsed from last start point
    void LogEndAndStart(ThreadPoolEvent);
    void LogStartAndCoreAndBlock(std::ptrdiff_t block_size);
    void LogCoreAndBlock(std::ptrdiff_t block_size);  //called in main thread to log core and block size for task breakdown
    void LogThreadId(int thread_idx);                 //called in child thread to log its id
    void LogRun(int thread_idx);                      //called in child thread to log num of run
    std::string DumpChildThreadStat();                //return all child statitics collected so far

private:
    static const char* GetEventName(ThreadPoolEvent);
    struct MainThreadStat {
        uint64_t events_[MAX_EVENT] = {};
        int32_t core_ = -1;
        std::vector<std::ptrdiff_t> blocks_;  //block size determined by cost model
        std::vector<TimePoint> points_;
        void LogCore();
        void LogBlockSize(std::ptrdiff_t block_size);
        void LogStart();
        void LogEnd(ThreadPoolEvent);
        void LogEndAndStart(ThreadPoolEvent);
        std::string Reset();
    };
    bool enabled_ = false;
    MainThreadStat& GetMainThreadStat();  //return thread local stat
    int num_threads_;
    struct ChildThreadStat {
        std::thread::id thread_id_;
        uint64_t num_run_ = 0;
        TimePoint last_logged_point_ = Clock::now();
        int32_t core_ = -1;                   //core that the child thread is running on
        PaddingToAvoidFalseSharing padding_;  //to prevent false sharing
    };
    std::vector<ChildThreadStat> child_thread_stats_;
    std::string thread_pool_name_;
};

class OrtThreadPoolParallelSection;

// Extended Eigen thread pool interface, avoiding the need to modify
// the ThreadPoolInterface.h header from the external Eigen
// repository.

class OrtExtendedThreadPoolInterface : public Eigen::ThreadPoolInterface {
public:
    // Start/end a parallel section, within which calls to
    // RunInParallelSection may be made.  Parallel sections are
    // non-nesting.
    virtual std::unique_ptr<OrtThreadPoolParallelSection, void (*)(OrtThreadPoolParallelSection*)> AllocateParallelSection() = 0;
    virtual void StartParallelSection(OrtThreadPoolParallelSection& ps) = 0;
    virtual void EndParallelSection(OrtThreadPoolParallelSection& ps) = 0;

    // Run fn with up to n degree-of-parallelism enlisting the thread
    // pool for help.  The degree-of-parallelism includes the caller,
    // and so if n==1 then the function will run directly in the caller.
    //
    // The fork-join synchronization is handled in the thread pool, and
    // so any state captured by fn() is safe from concurrent access once
    // RunInParallelSection returns.
    //
    // The parameter idx provides a loop-local thread ID in the range
    // [0,k) where k<=n.
    virtual void RunInParallelSection(OrtThreadPoolParallelSection& ps,
                                      std::function<void(unsigned idx)> fn,
                                      unsigned n, std::ptrdiff_t block_size) = 0;

    // Special case alternative to RunInParallelSection for use without
    // an existing parallel section.  Ideally we would use a single
    // iplemenation and a stack-allocated OrtThreadPoolParallelSection.
    //
    // However, on the BM_ThreadPoolParallelFor microbenchmark I saw
    // ~20% overhead on the resulting single-loop parallel sections.
    // There are some additional costs (~5%) for additional invocations
    // through lambda functions on loop entry.  Most significantly, on
    // loop exit, we incurred ~15% cost by no longer being able to
    // overlap clean-up of unused Task objects in EndParallelSection
    // with waiting for loop iterations to complete.
    //
    // [ Note that this 20% overhead is more than paid for when we have
    // two loops execute in series in a parallel section. ]
    virtual void RunInParallel(std::function<void(unsigned idx)> fn,
                               unsigned n, std::ptrdiff_t block_size) = 0;
    virtual void StartProfiling() = 0;
    virtual std::string StopProfiling() = 0;
};

class OrtThreadPoolParallelSection {
public:
    // State accessed only by the main thread
    // --------------------------------------

    // Tasks successfully submitted to the work queues.  This sets the
    // maximum degree of parallelism that the section will support.
    std::vector<std::pair<int, unsigned>> tasks;

    // Number of tasks revoked (i.e., removed from the queues prior to
    // execution).  We count this at various points, and omit waiting
    // for them at the end of a loop.
    unsigned tasks_revoked{0};

    // Current degree of parallelism, including work in the main thread
    // and in the dispatcher.
    unsigned current_dop{0};

    // State shared between the main thread and worker threads
    // -------------------------------------------------------

    // Flag to signal termination of the parallel section
    std::atomic<bool> active{false};

    // Count of the number of tasks that completed normally.  Other
    // tasks may be running currently, or may be present in work queues,
    // or may have been removed from the queues by
    // OrtRunQueue::RevokeWithTag.
    PaddingToAvoidFalseSharing padding_1;
    std::atomic<unsigned> tasks_finished{0};
    PaddingToAvoidFalseSharing padding_2;

    // If non-null, the current loop that tasks should be executing.  We
    // need to be careful on access to the contents of current_loop
    // because it can be stack allocated on the thread entering the
    // loop:
    //
    // - Readers increment workers_in_loop and then read current_loop
    //
    // - Writers wishing to deallocate *current_loop must first clear
    //   current_loop and then wait for workers_in_loop==0
    std::atomic<OrtThreadPoolLoop*> current_loop{nullptr};
    std::atomic<unsigned> workers_in_loop{0};

    // Members to track asynchronous dispatching
    int dispatch_q_idx = -1;      // index of thread that dispatch work to all other threads
    unsigned dispatch_w_idx = 0;  // index of enqueued work
    std::atomic<bool> dispatch_started{false};
    std::atomic<bool> dispatch_done{false};
    std::atomic<bool> work_done{false};
};

class OrtThreadPoolLoop {
public:
    OrtThreadPoolLoop(std::function<void(unsigned)> f, unsigned t) : fn(std::move(f)), threads_needed(t) {
    }

    const std::function<void(unsigned)> fn;
    const unsigned threads_needed;

private:
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrtThreadPoolLoop);
};

template <typename Work, typename Tag, unsigned kSize>
class OrtRunQueue {
public:
    OrtRunQueue() : front_(0), back_(0) {
        // require power-of-two for fast masking
        assert((kSize & (kSize - 1)) == 0);
        assert(kSize > 2);            // why would you do this?
        assert(kSize <= (64 << 10));  // leave enough space for counter
        for (unsigned i = 0; i < kSize; i++) array_[i].state.store(ElemState::kEmpty, std::memory_order_relaxed);
    }

    ~OrtRunQueue() {
        assert(Size() == 0);
    }

    // PopFront removes and returns the first element in the queue.
    // If the queue was empty returns default-constructed Work.
    Work PopFront() {
        unsigned front;
        Elem* e;
        ElemState s;

        // Drain revoked items from the front of the queue.  CAS to busy to synchronize with
        // any attempt to take the same item from the back of the queue.
        do {
            front = front_.load(std::memory_order_relaxed);
            e = &array_[(front - 1) & kMask];
            s = e->state.load(std::memory_order_relaxed);
            if (s == ElemState::kRevoked &&
                e->state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire)) {
                e->state.store(ElemState::kEmpty, std::memory_order_release);
                front = ((front - 1) & kMask2) | (front & ~kMask2);
                front_.store(front, std::memory_order_relaxed);
            }
        } while (s == ElemState::kRevoked);

        // Attempt to take next item.  State kEmpty shows the queue is empty, kBusy shows
        // the work is in progress on the item at the front of the queue.
        if (s != ElemState::kReady ||
            !e->state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire))
            return Work();
        Work w = std::move(e->w);
        e->tag = Tag();
        e->state.store(ElemState::kEmpty, std::memory_order_release);
        front = ((front - 1) & kMask2) | (front & ~kMask2);
        front_.store(front, std::memory_order_relaxed);
        return w;
    }

    // PushBack adds w at the end of the queue.
    // If queue is full returns w, otherwise returns default-constructed Work.
    Work PushBack(Work w) {
        std::unique_lock<TimesIntelliMutex> lock(mutex_);
        unsigned back = back_.load(std::memory_order_relaxed);
        Elem& e = array_[(back - 1) & kMask];
        ElemState s = e.state.load(std::memory_order_relaxed);
        if (s != ElemState::kEmpty ||
            !e.state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire))
            return w;
        back = ((back - 1) & kMask2) | (back & ~kMask2);
        back_.store(back, std::memory_order_relaxed);
        e.w = std::move(w);
        e.tag = Tag();
        e.state.store(ElemState::kReady, std::memory_order_release);
        return Work();
    }

    // PushBackWithTag adds w at the end of the queue.  The tag value can be used on a
    // subsequent call to RevokeWithTag to remove the item from the queue in combination
    // with w_idx.  Typically the tag will be a per-thread ID to distinguish work
    // submitted from different threads.
    PushResult PushBackWithTag(Work w, Tag tag, unsigned& w_idx) {
        std::unique_lock<TimesIntelliMutex> lock(mutex_);
        unsigned back = back_.load(std::memory_order_relaxed);
        w_idx = (back - 1) & kMask;
        Elem& e = array_[w_idx];
        ElemState s = e.state.load(std::memory_order_relaxed);
        if (s != ElemState::kEmpty ||
            !e.state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire))
            return PushResult::REJECTED; /* Not enqueued */
        bool was_ready = (((back ^ (front_.load(std::memory_order_relaxed))) & kMask) == 0);
        back = ((back - 1) & kMask2) | (back & ~kMask2);
        back_.store(back, std::memory_order_relaxed);
        e.w = std::move(w);
        e.tag = tag;
        e.state.store(ElemState::kReady, std::memory_order_release);
        return was_ready ? PushResult::ACCEPTED_IDLE : PushResult::ACCEPTED_BUSY; /* Enqueued */
    }

    // PopBack removes and returns the last elements in the queue.
    Work PopBack() {
        if (Empty())
            return Work();
        std::unique_lock<TimesIntelliMutex> lock(mutex_);
        unsigned back;
        Elem* e;
        ElemState s;

        // Drain revoked items from the back of the queue.  CAS to busy to synchronize with
        // any attempt to take the same item from the front of the queue.
        do {
            back = back_.load(std::memory_order_relaxed);
            e = &array_[back & kMask];
            s = e->state.load(std::memory_order_relaxed);
            if (s == ElemState::kRevoked &&
                e->state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire)) {
                e->state.store(ElemState::kEmpty, std::memory_order_release);
                back_.store(back + 1 + (kSize << 1), std::memory_order_relaxed);
            }
        } while (s == ElemState::kRevoked);

        if (s != ElemState::kReady ||
            !e->state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire))
            return Work();
        Work w = std::move(e->w);
        e->tag = Tag();
        e->state.store(ElemState::kEmpty, std::memory_order_release);
        back_.store(back + 1 + (kSize << 1), std::memory_order_relaxed);
        return w;
    }

    // RevokeItem removes a work item from the queue.  Items are identified positionally,
    // and so a tag is used to detect whether the same position is occupied by a
    // different work item at the time of removal.  RevokeWithTags lets threads offer work
    // for parallel execution, and then revoke the offer prior to the work executing (for
    // instance if the thread itself completes all of the work).  Revoking the work
    // lets the thread deallocate state that might otherwise have been captured by the work item
    // and accessed by it.
    //
    // Return true iff the item is successfully revoked.  If the item is not revoked then
    // the caller must assume that it may still execute, for instance because it
    // has been pop'd from the queue concurrent with the revocation request.

    bool RevokeWithTag(Tag tag, unsigned w_idx) {
        bool revoked = false;
        std::unique_lock<TimesIntelliMutex> lock(mutex_);
        Elem& e = array_[w_idx];
        ElemState s = e.state.load(std::memory_order_relaxed);

        // We have acquired a lock on the queue, synchronizing with
        // operations aside from the PopFront fast-path.  Synchronize with
        // that by attempting the same kReady->kBusy transition via CAS.

        if (s == ElemState::kReady &&
            e.state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire)) {
            if (e.tag == tag) {
                unsigned back = back_.load(std::memory_order_relaxed);
                unsigned back_idx = back & kMask;
                if (back_idx != w_idx) {
                    // Item is not at the back of the queue, mark it in-place as revoked
                    e.tag = Tag();
                    e.w = Work();
                    e.state.store(ElemState::kRevoked, std::memory_order_release);
                    revoked = true;
                } else {
                    // Item being removed as still at the back; shift the back pointer over it,
                    // and bump the version number.
                    e.tag = Tag();
                    e.w = Work();
                    e.state.store(ElemState::kEmpty, std::memory_order_release);
                    back_.store(back + 1 + (kSize << 1), std::memory_order_relaxed);
                    revoked = true;
                }
            } else {
                // Tag mismatch, i.e. work queue slot re-used
                e.state.store(ElemState::kReady, std::memory_order_release);
            }
        }
        return revoked;
    }

    // Size returns current queue size.
    // Can be called by any thread at any time.
    unsigned Size() const {
        return SizeOrNotEmpty<true>();
    }

    // Empty tests whether container is empty.
    // Can be called by any thread at any time.
    bool Empty() const {
        return SizeOrNotEmpty<false>() == 0;
    }

private:
    static const unsigned kMask = kSize - 1;
    static const unsigned kMask2 = (kSize << 1) - 1;

    enum class ElemState : uint8_t {
        kEmpty,
        kBusy,
        kReady,
        kRevoked,
    };

    // Updates to an element are bracketed by a std::memory_order_acquire
    // load from the state, and a std::memory_order_release store.  Accesses
    // to the front/back indices for the work queue use relaxed semantics,
    // with the state of the elements being authoritative.
    //
    // TODO: Revisit whether there is a significant benefit for the current
    // workloads in the complexity here.
    struct Elem {
        std::atomic<ElemState> state;
        Tag tag;
        Work w;
    };

    TimesIntelliMutex mutex_;

    // Low log(kSize) + 1 bits in front_ and back_ contain rolling index of
    // front/back, respectively. The remaining bits contain modification counters
    // that are incremented on Push operations. This allows us to (1) distinguish
    // between empty and full conditions (if we would use log(kSize) bits for
    // position, these conditions would be indistinguishable); (2) obtain
    // consistent snapshot of front_/back_ for Size operation using the
    // modification counters.
    ORT_ALIGN_TO_AVOID_FALSE_SHARING std::atomic<unsigned> front_;
    ORT_ALIGN_TO_AVOID_FALSE_SHARING std::atomic<unsigned> back_;
    ORT_ALIGN_TO_AVOID_FALSE_SHARING Elem array_[kSize];

    // SizeOrNotEmpty returns current queue size; if NeedSizeEstimate is false,
    // only whether the size is 0 is guaranteed to be correct.
    // Can be called by any thread at any time.
    template <bool NeedSizeEstimate>
    unsigned SizeOrNotEmpty() const {
        // Emptiness plays critical role in thread pool blocking. So we go to great
        // effort to not produce false positives (claim non-empty queue as empty).
        unsigned front = front_.load(std::memory_order_acquire);
        for (;;) {
            // Capture a consistent snapshot of front/tail.
            unsigned back = back_.load(std::memory_order_acquire);
            unsigned front1 = front_.load(std::memory_order_relaxed);
            if (front != front1) {
                front = front1;
                std::atomic_thread_fence(std::memory_order_acquire);
                continue;
            }
            if (NeedSizeEstimate) {
                return CalculateSize(front, back);
            }
            // This value will be 0 if the queue is empty, and undefined otherwise.
            unsigned maybe_zero = ((front ^ back) & kMask2);
            // Queue size estimate must agree with maybe zero check on the queue
            // empty/non-empty state.
            eigen_assert((CalculateSize(front, back) == 0) == (maybe_zero == 0));
            return maybe_zero;
        }
    }

    EIGEN_ALWAYS_INLINE
    unsigned CalculateSize(unsigned front, unsigned back) const {
        int size = (front & kMask2) - (back & kMask2);
        // Fix overflow.
        if (size < 0)
            size += 2 * kSize;
        // Order of modification in push/pop is crafted to make the queue look
        // larger than it is during concurrent modifications. E.g. push can
        // increment size before the corresponding pop has decremented it.
        // So the computed size can be up to kSize + 1, fix it.
        if (size > static_cast<int>(kSize))
            size = kSize;
        return static_cast<unsigned>(size);
    }

    OrtRunQueue(const OrtRunQueue&) = delete;
    void operator=(const OrtRunQueue&) = delete;
};

static std::atomic<uint32_t> next_tag{1};

OrtEnvThread* CreateThread(const ORTCHAR_T* name_prefix, int index,
                           unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param),
                           Eigen::ThreadPoolInterface* param, const OrtThreadOptions& thread_options);

class OrtThreadPoolTempl : public OrtExtendedThreadPoolInterface {
private:
    struct PerThread;

    static unsigned WorkerLoop(int id, Eigen::ThreadPoolInterface* param) {
        // unsafe downcast
        OrtThreadPoolTempl* this_ptr = (OrtThreadPoolTempl*)param;
        this_ptr->WorkerLoop(id);
        return 0;
    }

    OrtThreadPoolProfiler profiler_;

public:
    void StartProfiling() override {
        profiler_.Start();
    }

    std::string StopProfiling() override {
        return profiler_.Stop();
    }

    struct Tag {
        constexpr Tag() : v_(0) {
        }

        Tag(uint32_t v) : v_(v) {
        }

        // Allocate a new tag to use to identify work items from a given
        // thread in a parallel section.  Ideally, threads will have
        // unique tags, but re-use is not incorrect if the counter wraps
        // (for intsance, if a long-running workload is calling into ORT
        // from a fresh thread for each request).  We must not re-use the
        // default tag 0 which is used to identify work items added via
        // Schedule as opposed to requests for help in parallel sections.

        static Tag GetNext() {
            Tag t = Tag(next_tag++);
            if (t.v_ == 0) {
                t = Tag(next_tag++);
            }
            return t;
        }

        uint32_t Get() const {
            return v_;
        }

        bool operator==(Tag& other) const {
            return v_ == other.v_;
        }

        uint32_t v_ = 0;
    };

    typedef std::function<void()> Task;
    typedef OrtRunQueue<Task, Tag, 1024> Queue;

    OrtThreadPoolTempl(const CHAR_TYPE* name, int num_threads, bool allow_spinning,
                       const OrtThreadOptions& thread_options)
            : profiler_(num_threads, name),
              num_threads_(num_threads),
              allow_spinning_(allow_spinning),
              //set_denormal_as_zero_(thread_options.set_denormal_as_zero),
              worker_data_(num_threads),
              all_coprimes_(num_threads),
              blocked_(0),
              done_(false) {
        // Calculate coprimes of all numbers [1, num_threads].
        // Coprimes are used for random walks over all threads in Steal
        // and NonEmptyQueueIndex. Iteration is based on the fact that if we take
        // a random starting thread index t and calculate num_threads - 1 subsequent
        // indices as (t + coprime) % num_threads, we will cover all threads without
        // repetitions (effectively getting a presudo-random permutation of thread
        // indices).
        for (auto i = 1u; i <= num_threads_; ++i) {
            all_coprimes_.emplace_back(Eigen::MaxSizeVector<unsigned>{i});
            ComputeCoprimes(i, &all_coprimes_.back());
        }

        worker_data_.resize(num_threads_);
        for (auto i = 0u; i < num_threads_; i++) {
            worker_data_[i].thread.reset(CreateThread(name, i, WorkerLoop, this, thread_options));
        }
    }

    ~OrtThreadPoolTempl() override {
        done_ = true;

        // Now if all threads block without work, they will start exiting.
        // But note that threads can continue to work arbitrary long,
        // block, submit new work, unblock and otherwise live full life.
        WakeAllWorkersForExit();
        // Join threads explicitly (by destroying) to avoid destruction order within
        // this class.
        for (size_t i = 0; i < worker_data_.size(); ++i) worker_data_[i].thread.reset();
    }

    // Run fn().  Ordinarily, the function will be added to the thread pool and executed
    // by a worker thread.  If the thread pool rejects the work then fn() will instead
    // execute synchronously during Schedule(fn).  Currently the thread pool will only
    // reject work if the queue of pending work is full.

    void Schedule(std::function<void()> fn) override {
        PerThread* pt = GetPerThread();
        int q_idx = Rand(&pt->rand) % num_threads_;
        WorkerData& td = worker_data_[q_idx];
        Queue& q = td.queue;
        fn = q.PushBack(std::move(fn));
        if (!fn) {
            // The queue accepted the work; ensure that the thread will pick it up
            td.EnsureAwake();
        } else {
            // Run the work directly if the queue rejected the work
            fn();
        }
    }

    //......................................................................
    //
    // Parallel sections
    // -----------------
    //
    // Allocate a new OrtThreadPoolParallelSection, owned by the returned
    // unique_ptr.  The explicit deleter avoids the Eigen-specific
    // definition of OrtThreadPoolParallelSection needing to be avilable in
    // threadpool.h where the user-facing parallel section API is defined.
    //GSL_SUPPRESS(r .11)
    std::unique_ptr<OrtThreadPoolParallelSection, void (*)(OrtThreadPoolParallelSection*)> AllocateParallelSection() override {
        return std::unique_ptr<OrtThreadPoolParallelSection, void (*)(OrtThreadPoolParallelSection*) > (new OrtThreadPoolParallelSection,
                [](OrtThreadPoolParallelSection* tps) {
                    delete tps;
                });
    }

    // Start a parallel section, using a caller-provided
    // OrtThreadPoolParallelSection for maintaining the per-section state.
    // Starting a parallel section is just book-keeping; threads are
    // "summoned" to help with the parallel section once it enters
    // parallel loops.  The threads are then retained until the end of the
    // section, being re-used over subsequent loops.

    void StartParallelSectionInternal(PerThread& pt,
                                      OrtThreadPoolParallelSection& ps) {
        assert((!pt.leading_par_section) && "Nested parallelism not supported");
        assert((!ps.active) && "Starting parallel section, but active already");
        pt.leading_par_section = true;
        if (!pt.tag.Get()) {
            pt.tag = Tag::GetNext();
        }
        ps.dispatch_q_idx = -1;
        ps.dispatch_started = false;
        ps.dispatch_done = false;
        ps.work_done = false;
        ps.tasks_revoked = 0;
        ps.current_dop = 1;
        ps.active = true;
    }

    void StartParallelSection(OrtThreadPoolParallelSection& ps) override {
        PerThread* pt = GetPerThread();
        StartParallelSectionInternal(*pt, ps);
    }

    // End a parallel section, waiting for all worker threads to exit from
    // section.  Hence, on return, the OrtThreadPoolParallelSection object
    // can be dealloacted.
    void EndParallelSectionInternal(PerThread& pt,
                                    OrtThreadPoolParallelSection& ps) {
        assert((pt.leading_par_section) && "Ending parallel section, but none started");
        assert((ps.active) && "Ending parallel section, but not active");
        pt.leading_par_section = false;

        // Notify workers to exit from the section
        ps.active = false;

        // First, attempt to revoke the dispatch task.  If we succeed then
        // we know we revoked _something_ pushed for the current loop.  That
        // may be the dispatch task itself, or it may be a task pushed by
        // the dispatch task.  Those cases are distinguished by whether or
        // not the dispatch task itself has started -- if it has not started
        // then it cannot have pushed tasks.
        if (ps.dispatch_q_idx != -1) {
            Queue& q = worker_data_[ps.dispatch_q_idx].queue;
            if (q.RevokeWithTag(pt.tag, ps.dispatch_w_idx)) {
                if (!ps.dispatch_started.load(std::memory_order_acquire)) {
                    // We successfully revoked a task, and saw the dispatch task
                    // not started.  Hence we know we revoked the dispatch task.
                    // This should be the common case.
                    ps.dispatch_q_idx = -1;
                } else {
                    // We successfully revoked a task, but saw the dispatch task
                    // had started.  Hence we know we revoked one of the _new_
                    // tasks created by the dispatcher (not the dispatcher
                    // itself).  This should be the rare case, but can occur if
                    // one of the tasks created by the dispatcher occupies the
                    // exact same slot in a work queue that the dispatcher used.
                    ps.tasks_revoked++;
                }
            }
        }

        // Second, if we failed to revoke the dispatch task, wait for it to
        // finish dispatch work.  This avoids new tasks being started
        // concurrently with us attempting to end the parallel section.
        if (ps.dispatch_q_idx != -1) {
            while (!ps.dispatch_done.load(std::memory_order_acquire)) {
                SpinPause();
            }
        }

        // Now we know that dispatch is finshed, we synchronize with the
        // tasks that were created (if any) for the parallel section.  We
        // revoke tasks still in queues, and then wait for any that are
        // still running.
        profiler_.LogStart();
        unsigned tasks_started = static_cast<unsigned>(ps.tasks.size());
        while (!ps.tasks.empty()) {
            const auto& item = ps.tasks.back();
            Queue& q = worker_data_[item.first].queue;
            if (q.RevokeWithTag(pt.tag, item.second)) {
                ps.tasks_revoked++;
            }
            ps.tasks.pop_back();
        }
        profiler_.LogEnd(OrtThreadPoolProfiler::WAIT_REVOKE);

        // Wait for the dispatch task's own work...
        if (ps.dispatch_q_idx > -1) {
            while (!ps.work_done.load(std::memory_order_acquire)) {
                SpinPause();
            }
        }

        // ...and wait for any other tasks not revoked to finish their work
        auto tasks_to_wait_for = tasks_started - ps.tasks_revoked;
        while (ps.tasks_finished < tasks_to_wait_for) {
            SpinPause();
        }

        // Clear status to allow the OrtThreadPoolParallelSection to be
        // re-used.
        ps.tasks_finished = 0;
    }

    void EndParallelSection(OrtThreadPoolParallelSection& ps) override {
        PerThread* pt = GetPerThread();
        EndParallelSectionInternal(*pt, ps);
    }

    //----------------------------------------------------------------------
    //
    // Preferred workers
    // -----------------
    //
    // Initialize the set of hints for preferred worker threads we will
    // use.  We do this once, covering the maximum num_threads_ items,
    // in order to avoid resizing preferred_workers concurrent with
    // access from worker threads.
    //
    // For simplicity we initialize with hints round-robin among the
    // workers.  For simple workloads with 1 main thread this means we
    // will distribute work across the pool of workers.  For workers
    // with multiple main threads it attempts to balance the load.
    //
    // These hints are just used as a starting point, and are updated by
    // the worker thread that actually claims an item (e.g., if an item
    // initially assigned to thread T1 is stolen and executed by T2,
    // then T2 is assigned at the new preferred worker).
    //
    // Note that the hints are held in the _main_ thread that submits
    // work to the pool.  We assume that a thread is primarily
    // submitting work to just one pool, but allow for the pool to
    // change over time.  Hence we allow the hints vector to grow over
    // time.
    //
    // A note on terminology used in the variable names here:
    //
    // dop - degree of parallelism, as seen by the user.  For instance
    //       dop=4 means 4 threads in total: 1 main thread that enters the
    //       loop, plus 1 dispatcher thread, plus 2 additional worker
    //       threads.
    //
    // par_idx - a thread's index within the loop, in the range [0,dop).
    //
    // num_threads_ - the number of worker threads in the thread pool.  A
    //       loop with dop=4 will be common on a pool with 3 threads
    //       (given that the main thread will also participate).
    //
    // q_idx - a worker queue index, in the range [0,num_threads_).
    //
    // preferred_workers - this maps from par_idx values to q_idx.  Hence,
    //        with dop=4 the vector will have length 4, and will identify
    //        which of the workers (0,1,2) should run tasks for the loop.
    //        Note that mapping from par_idx values means that only slots
    //        [1,dop) are actually used in preferred_workers.
    //
    // Here are three examples, all assuming a machine with 4 h/w threads,
    // and ORT configured to use dop=4.
    //
    // * First, suppose that a single job is running a series of loops.
    //   Its main thread enters a parallel loop.  Initially, let's assume
    //   its preferred worker array is [_,0,1,2], writing "_" for the
    //   unusued element for the par_idx=0 work that the main thread will
    //   run.
    //
    //   The main thread schedules the dispatcher task onto worker 0.
    //
    //   The dispatcher task schedules worker tasks onto workers 1 and 2.
    //
    //   The tasks all execute, without any work stealing, on the threads
    //   they were scheduled on.  The preferred worker array remains
    //   [_,0,1,2].
    //
    // * Next, assume we have the same job, and for whatever reason the
    //   preferred workers were initially [_,0,0,0].
    //
    //   The main thread schedules the dispatcher onto worker 0.
    //
    //   This dispatcher task runs on worker 0, and pushes the worker
    //   tasks back onto worker 0's queue.
    //
    //   Workers 1 and 2 are idle, and steal tasks from worker 0.  As the
    //   tasks run, they update the preferred_workers array to record the
    //   workers that execute them.
    //
    //   After the loop, the preferred worker array may now be [_,0,2,1]
    //   or [_,0,1,2], reflecting the fact that the work has got
    //   re-distributed.  The next loop will start out by distributing the
    //   work to those same workers.
    //
    // * Finally, let's assume we have two jobs running on two main
    //   threads, and we are now using DoP=2 in the loops, and have 2
    //   workers in the thread pool (so the machine is not
    //   over-subscribed).
    //
    //   Each main thread has its own preferred_workers, and
    //   let's say initially these are both [_,0].
    //
    //   Here, with DoP=2, each main thread will just dispatch a single
    //   task immediately (there is no need for asynchrony with only one
    //   task to generate).
    //
    //   Initially both main threads will submit these tasks to worker 0.
    //
    //   Once worker 1 steals one of these tasks, the task will update its
    //   preferred worker to be 1.
    //
    //   From that point onwards, the two main threads will dispatch tasks
    //   to separate workers, avoiding the need for further work stealing.

    void InitializePreferredWorkers(std::vector<int>& preferred_workers) {
        static std::atomic<unsigned> next_worker{0};

        // preferred_workers[0] isn't supposed to be used, so initializing it with -1 to:
        // a) fault if inappropriately accessed
        // b) avoid wasting next_worker value
        if (preferred_workers.empty()) {
            preferred_workers.push_back(-1);
        }

        // preferred_workers maps from a par_idx to a q_idx, hence we
        // initialize slots in the range [0,num_threads_]
        while (preferred_workers.size() <= num_threads_) {
            preferred_workers.push_back(next_worker++ % num_threads_);
        }
    }

    // Update the preferred worker for par_idx to be the calling thread

    void UpdatePreferredWorker(std::vector<int>& preferred_workers,
                               unsigned par_idx) {
        unsigned ran_on_idx = GetPerThread()->thread_id;
        assert(ran_on_idx < num_threads_);
        assert(par_idx < preferred_workers.size());
        preferred_workers[par_idx] = ran_on_idx;
    }

    // Schedule [par_idx_start,par_idx_end) across the preferred workers

    void ScheduleOnPreferredWorkers(PerThread& pt,
                                    OrtThreadPoolParallelSection& ps,
                                    std::vector<int>& preferred_workers,
                                    unsigned par_idx_start,
                                    unsigned par_idx_end,
                                    std::function<void(unsigned)> worker_fn) {
        for (auto par_idx = par_idx_start; par_idx < par_idx_end; ++par_idx) {
            // Look up hint for par_idx.  Note that the hints may have been
            // recorded from a prior thread pool with a different number of
            // threads, hence we must cap at num_threads_.
            assert(par_idx < preferred_workers.size());
            unsigned q_idx = preferred_workers[par_idx] % num_threads_;
            assert(q_idx < num_threads_);
            WorkerData& td = worker_data_[q_idx];
            Queue& q = td.queue;
            unsigned w_idx;

            // Attempt to enqueue the task
            auto push_status = q.PushBackWithTag([worker_fn, par_idx, &preferred_workers, &ps, this]() {
                                                     // Record the worker thread that actually runs this task.
                                                     // This will form the preferred worker for the next loop.
                                                     UpdatePreferredWorker(preferred_workers, par_idx);
                                                     worker_fn(par_idx);
                                                     ps.tasks_finished++;
                                                 },
                                                 pt.tag, w_idx);

            // Queue accepted the task; wake the thread that owns the queue.
            // In addition, if the queue was non-empty, attempt to wake
            // another thread (which may then steal the task).
            if (push_status == PushResult::ACCEPTED_IDLE || push_status == PushResult::ACCEPTED_BUSY) {
                ps.tasks.push_back({q_idx, w_idx});
                td.EnsureAwake();
                if (push_status == PushResult::ACCEPTED_BUSY) {
                    worker_data_[Rand(&pt.rand) % num_threads_].EnsureAwake();
                }
            }
        }
    }

    //......................................................................
    //
    // Parallel loops
    // --------------
    //
    // Ensure that the OrtThreadPoolParallelSection has sufficient workers to
    // execute a loop with degree of parallelism n.  We track the number
    // of workers already avaiable to the parallel section, prior to
    // submitting tasks to the work queues to make up the total.
    //
    // Each worker will call in to worker_fn(idx) with a per-worker thread
    // ID.  Note there are different levels of indirection here:
    //
    // - In a single-loop parallel section, worker_fn will directly
    //   execute the threadpool.cc code that implements the parallel loop.
    //
    // - In a multi-loop parallel section, worker_fn is an intermediate
    //   function that is long-lived (i.e., that lasts until the end of
    //   the parallel section, as opposed to just a single loop's
    //   duration).
    //
    // For ordinary parallel sections, RunInParallelInternal dispatch
    // tasks to a number of workers asynchronously.  A worker thread will
    // be selected as the dispatcher that distributes tasks.  This removes
    // the O(n) work off the critical path of starting the first loop
    // iteration, helping maintain good performance on very short loops.
    //
    // See the note on terminology above for the use of variable names
    // here.

    void RunInParallelInternal(PerThread& pt,
                               OrtThreadPoolParallelSection& ps,
                               unsigned new_dop,
                               bool dispatch_async,
                               std::function<void(unsigned)> worker_fn) {
        // Ensure that the vector of preferred workers is sufficient for the
        // size of the loop we are entering.  We do this before dispatching
        // tasks for the loop in order to avoid any races between changes to
        // the size of the vector and recording the locations that tasks run
        // in as they complete.
        assert(new_dop <= (unsigned)(num_threads_ + 1));
        std::vector<int>& preferred_workers = pt.preferred_workers;
        InitializePreferredWorkers(preferred_workers);

        // current_dop is the degree of parallelism via any workers already
        // participating in the current parallel section.  Usually, for
        // single-loop parallel sections, current_dop=1.
        unsigned current_dop = ps.current_dop;

        if (current_dop < new_dop) {
            unsigned extra_needed = new_dop - current_dop;

            // Attempt to summon additional workers asynchronously if we
            // need more than one.  Otherwise, we fall back to simple
            // synchronous scheduling.
            if (dispatch_async && extra_needed > 1) {
                assert(current_dop == 1);

                // Task for dispatching work asynchronously.
                Task dispatch_task = [current_dop, new_dop, worker_fn, &preferred_workers, &ps, &pt, this]() {
                    // Record that dispatch work has started.  This must occur
                    // prior to scheduling tasks, in order to synchronize with
                    // EndParallelSectionInternal.  [ If EndParallelSection
                    // revoked a task, and then sees distpatch_started=false, then
                    // it knows that it revoked the dispatcher.  Conversely, if it
                    // revokes a task, and then sees dispatch_started=true, then
                    // it knows it revoked a worker task. ]
                    ps.dispatch_started.store(true, std::memory_order_seq_cst);

                    // Schedule tasks par_idx=[current_dop+1,new_dop)
                    ScheduleOnPreferredWorkers(pt, ps, preferred_workers, current_dop + 1, new_dop, worker_fn);
                    ps.dispatch_done.store(true, std::memory_order_release);

                    // Record the worker thread that actually runs this task.
                    // This will form the preferred worker for the next loop.
                    UpdatePreferredWorker(preferred_workers, current_dop);

                    // Run dispatcher task's own work, par_idx=current_dop
                    worker_fn(current_dop);

                    // Dispatcher's work complete
                    ps.work_done.store(true, std::memory_order_release);
                };

                profiler_.LogStart();
                ps.dispatch_q_idx = preferred_workers[current_dop] % num_threads_;
                WorkerData& dispatch_td = worker_data_[ps.dispatch_q_idx];
                Queue& dispatch_que = dispatch_td.queue;

                // assign dispatch task to selected dispatcher
                auto push_status = dispatch_que.PushBackWithTag(dispatch_task, pt.tag, ps.dispatch_w_idx);
                // Queue accepted the task; wake the thread that owns the queue.
                // In addition, if the queue was non-empty, attempt to wake
                // another thread (which may then steal the task).
                if (push_status == PushResult::ACCEPTED_IDLE || push_status == PushResult::ACCEPTED_BUSY) {
                    dispatch_td.EnsureAwake();
                    if (push_status == PushResult::ACCEPTED_BUSY) {
                        worker_data_[Rand(&pt.rand) % num_threads_].EnsureAwake();
                    }
                } else {
                    ps.dispatch_q_idx = -1;  // failed to enqueue dispatch_task
                }
                profiler_.LogEnd(OrtThreadPoolProfiler::DISTRIBUTION_ENQUEUE);
            } else {
                // Synchronous dispatch
                ScheduleOnPreferredWorkers(pt, ps, preferred_workers, current_dop, new_dop, std::move(worker_fn));
            }
            ps.current_dop = new_dop;
        }
    }

    // Run a single parallel loop in an existing parallel section.  This
    // maps directly onto SummonWorkers to create sufficient worker
    // threads for the desired degree of parallelism, followed by
    // dispatching the loop to those workers.
    void RunInParallelSection(OrtThreadPoolParallelSection& ps,
                              std::function<void(unsigned idx)> fn,
                              unsigned n,
                              std::ptrdiff_t block_size) override {
        if (n > num_threads_ + 1) {
            std::cout << "More work items than threads\n";
            exit(-1);
            // n = num_threads_ + 1;
        }
        profiler_.LogStartAndCoreAndBlock(block_size);
        PerThread* pt = GetPerThread();
        assert(pt->leading_par_section && "RunInParallel, but not in parallel section");
        assert((n > 1) && "Trivial parallel section; should be avoided by caller");

        // Publish the work to any existing workers in the parallel
        // section, and ensure it is visible to any new threads created
        // below.
        assert((!ps.current_loop) && "RunInParallelSection, but loop already active");
        OrtThreadPoolLoop loop{std::move(fn), n};
        ps.current_loop = &loop;

        // Increase the worker count if needed.  Each worker will pick up
        // loops to execute from the current parallel section.
        std::function<void(unsigned)> worker_fn = [&ps](unsigned par_idx) {
            while (ps.active) {
                if (ps.current_loop.load() == nullptr) {
                    SpinPause();
                } else {
                    ps.workers_in_loop++;
                    OrtThreadPoolLoop* work_item = ps.current_loop;
                    if (work_item && par_idx < work_item->threads_needed) {
                        work_item->fn(par_idx);
                    }
                    ps.workers_in_loop--;
                }
            }
        };
        RunInParallelInternal(*pt, ps, n, false, std::move(worker_fn));
        assert(ps.dispatch_q_idx == -1);
        profiler_.LogEndAndStart(OrtThreadPoolProfiler::DISTRIBUTION);

        // Run work in the main thread
        loop.fn(0);
        profiler_.LogEndAndStart(OrtThreadPoolProfiler::RUN);

        // Wait for workers to exit the loop
        ps.current_loop = 0;
        while (ps.workers_in_loop) {
            SpinPause();
        }
        profiler_.LogEnd(OrtThreadPoolProfiler::WAIT);
    }

    // Run a single parallel loop _without_ a parallel section.  This is a
    // special case of RunInParallelSection, avoiding code paths for
    // handing off multiple loops to the pool of workers.
    // For main thread:
    //  1. select a dispatcher and do job distribution;
    //  2. run fn(0);
    //  3, wait for all;
    // For dispatcher:
    //  1. distribute jobs to all other threads;
    //  2. run fn(...) itself.
    // For all other threads:
    //  1. run fn(...);
    void RunInParallel(std::function<void(unsigned idx)> fn, unsigned n, std::ptrdiff_t block_size) override {
        if (n > num_threads_ + 1) {
            std::cout << "More work items than threads\n";
            exit(-1);
        }
        profiler_.LogStartAndCoreAndBlock(block_size);
        PerThread* pt = GetPerThread();
        OrtThreadPoolParallelSection ps;
        StartParallelSectionInternal(*pt, ps);
        RunInParallelInternal(*pt, ps, n, true, fn);  // select dispatcher and do job distribution;
        profiler_.LogEndAndStart(OrtThreadPoolProfiler::DISTRIBUTION);
        fn(0);  // run fn(0)
        profiler_.LogEndAndStart(OrtThreadPoolProfiler::RUN);
        EndParallelSectionInternal(*pt, ps);  // wait for all
        profiler_.LogEnd(OrtThreadPoolProfiler::WAIT);
    }

    int NumThreads() const final {
        return num_threads_;
    }

    int CurrentThreadId() const final {
        const PerThread* pt = const_cast<OrtThreadPoolTempl*>(this)->GetPerThread();
        if (pt->pool == this) {
            return pt->thread_id;
        }
        return -1;
    }

    void EnableSpinning() {
        spin_loop_status_ = SpinLoopStatus::kBusy;
    }

    void DisableSpinning() {
        spin_loop_status_ = SpinLoopStatus::kIdle;
    }

private:
    void ComputeCoprimes(int N, Eigen::MaxSizeVector<unsigned>* coprimes) {
        for (int i = 1; i <= N; i++) {
            unsigned a = i;
            unsigned b = N;
            // If GCD(a, b) == 1, then a and b are coprimes.
            while (b != 0) {
                unsigned tmp = a;
                a = b;
                b = tmp % b;
            }
            if (a == 1) {
                coprimes->push_back(i);
            }
        }
    }

    typedef OrtEnvThread Thread;
    struct WorkerData;

    // PerThread objects are allocated in thread-local storage and
    // allocated on the thread's first call to GetPerThread.  PerThread
    // objects are allocated for all threads that submit work to the
    // thread pool, in addition to threads within the pool.
    //
    // In contrast, the WorkerData objects are allocated only for the
    // threads in the pool, and their lifetime is managed along with the
    // pool.

    struct PerThread {
        PerThread() : pool(nullptr) {
        }
        OrtThreadPoolTempl* pool;            // Parent pool, or null for normal threads.
        bool initialized{false};          // Non-trivial initialization ran (e.g. for RNG)
        uint64_t rand{0};                 // Random generator state.
        int thread_id{-1};                // Worker thread index in pool.
        Tag tag{};                        // Work item tag used to identify this thread.
        bool leading_par_section{false};  // Leading a parallel section (used only for asserts)

        // When this thread is entering a parallel section, it will
        // initially push work to this set of workers.  The aim is to
        // retain cache state within the workers, and to reduce the number
        // of times that the work-stealing code paths are used for
        // rebalancing.
        std::vector<int> preferred_workers;
        PaddingToAvoidFalseSharing padding_2;
    };

    struct WorkerData {
        WorkerData() : thread(), queue() {
        }
        std::unique_ptr<Thread> thread;
        Queue queue;

        // Each thread has a status, available read-only without locking, and protected
        // by the mutex field below for updates.  The status is used for three
        // purposes:
        //
        // 1. To identify threads that are good candidates to push work to.
        //    We prefer to push work to threads that are actively spinning (no need
        //    for an OS wake-up, and no need for current work to finish).  After that, we
        //    prefer to push work to threads that are blocked (no need to wait for the
        //    current work to finish).
        //
        // 2. To identify threads that are good candidates to steal work from.  We
        //    prefer to steal work from threads that are active outside the worker loop.
        //    This avoids "snatching" new work away from a thread that has just been
        //    given it but not yet noticed.
        //
        // 3. When pushing work to a thread, we use the status read-only to identify
        //    when we need to wake the thread.  This read-only check avoids the
        //    need for mutex / condvar operations in the case where the thread pool
        //    remains busy.

        enum class ThreadStatus : uint8_t {
            Spinning,  // Spinning in the work loop, and other cases (initialization) where
            // the thread will soon be in the loop
            Active,    // Running user code, not waiting for work
            Blocking,  // In the process of blocking; may no longer notice work pushed to it
            Blocked,   // Blocked on cv
            Waking,    // Not yet back in the worker loop, but wake-up notification sent
        };

        ThreadStatus GetStatus() const {
            return status;
        }

        // State transitions, called from other threads

        // We employ mutex for synchronizing on Blocked/Waking state (EnsureAwake/SeBlocked)
        // to wakeup the thread in the event it goes to sleep. Because thread status
        // is an atomic member the lock is not necessary to update it.
        // Thus, we do not obtain the mutex when we set Active/Spinning state for the thread.
        // While manipulating under the mutex, we employ relaxed semantics so the compiler is not restricted
        // any further.
        void EnsureAwake() {
            ThreadStatus seen = GetStatus();
            if (seen == ThreadStatus::Blocking ||
                seen == ThreadStatus::Blocked) {
                std::unique_lock<TimesIntelliMutex> lk(mutex);
                // Blocking state exists only transiently during the SetBlock() method
                // while holding the lock.  We may observe it at the start of this
                // function, but after acquiring the lock then the target thread
                // will either be blocked or not.
                seen = status.load(std::memory_order_relaxed);
                assert(seen != ThreadStatus::Blocking);
                if (seen == ThreadStatus::Blocked) {
                    status.store(ThreadStatus::Waking, std::memory_order_relaxed);
                    lk.unlock();
                    cv.notify_one();
                }
            }
        }

        // State transitions, called only from the thread itself
        // The lock is only used in the synchronization between EnsureAwake and SetBlocked,
        // while the Active vs Spinning states are just used as a hint for work stealing
        // (prefer to steal from a thread that is actively running a task, rather than stealing from
        // a thread that is spinning and likely to pick up the task itself).
        void SetActive() {
            status = ThreadStatus::Active;
        }

        void SetSpinning() {
            status = ThreadStatus::Spinning;
        }

        void SetBlocked(std::function<bool()> should_block,
                        std::function<void()> post_block) {
            std::unique_lock<TimesIntelliMutex> lk(mutex);
            assert(GetStatus() == ThreadStatus::Spinning);
            status.store(ThreadStatus::Blocking, std::memory_order_relaxed);
            if (should_block()) {
                status.store(ThreadStatus::Blocked, std::memory_order_relaxed);
                do {
                    cv.wait(lk);
                } while (status.load(std::memory_order_relaxed) == ThreadStatus::Blocked);
                post_block();
            }
            status.store(ThreadStatus::Spinning, std::memory_order_relaxed);
        }

    private:
        std::atomic<ThreadStatus> status{ThreadStatus::Spinning};
        TimesIntelliMutex mutex;
        TimesIntelliCondVar cv;
    };

    const unsigned num_threads_;
    const bool allow_spinning_;
    //const bool set_denormal_as_zero_;
    Eigen::MaxSizeVector<WorkerData> worker_data_;
    Eigen::MaxSizeVector<Eigen::MaxSizeVector<unsigned>> all_coprimes_;
    std::atomic<unsigned> blocked_;  // Count of blocked workers, used as a termination condition
    std::atomic<bool> done_;

    // SpinLoopStatus indicates whether the main worker spinning (inner) loop should exit immediately when there is
    // no work available (kIdle) or whether it should follow the configured spin-then-block policy (kBusy).
    // This lets the ORT session layer hint to the thread pool that it should stop spinning in between
    // requests.
    enum class SpinLoopStatus {
        kIdle,
        kBusy
    };

    // Default is no control over spinning
    std::atomic<SpinLoopStatus> spin_loop_status_{SpinLoopStatus::kBusy};

    // Wake any blocked workers so that they can cleanly exit WorkerLoop().  For
    // a clean exit, each thread will observe (1) done_ set, indicating that the
    // destructor has been called, (2) all threads blocked, and (3) no
    // items in the work queues.

    void WakeAllWorkersForExit() {
        for (auto& td : worker_data_) {
            td.EnsureAwake();
        }
    }

    // Main worker thread loop.
    void WorkerLoop(int thread_id) {
        PerThread* pt = GetPerThread();
        WorkerData& td = worker_data_[thread_id];
        Queue& q = td.queue;
        bool should_exit = false;
        pt->pool = this;
        pt->thread_id = thread_id;

        assert(td.GetStatus() == WorkerData::ThreadStatus::Spinning);

        constexpr int log2_spin = 20;
        const int spin_count = allow_spinning_ ? (1ull << log2_spin) : 0;
        const int steal_count = spin_count / 100;

        //SetDenormalAsZero(set_denormal_as_zero_);
        profiler_.LogThreadId(thread_id);

        while (!should_exit) {
            Task t = q.PopFront();
            if (!t) {
                // Spin waiting for work.
                for (int i = 0; i < spin_count && !done_; i++) {
                    if (((i + 1) % steal_count == 0)) {
                        t = Steal(StealAttemptKind::TRY_ONE);
                    } else {
                        t = q.PopFront();
                    }
                    if (t) break;

                    if (spin_loop_status_.load(std::memory_order_relaxed) == SpinLoopStatus::kIdle) {
                        break;
                    }
                    SpinPause();
                }

                // Attempt to block
                if (!t) {
                    td.SetBlocked(  // Pre-block test
                            [&]() -> bool {
                                bool should_block = true;
                                // Check whether work was pushed to us while attempting to block.  We make
                                // this test while holding the per-thread status lock, and after setting
                                // our status to ThreadStatus::Blocking.
                                //
                                // This synchronizes with ThreadPool::Schedule which pushes work to the queue
                                // and then tests for ThreadStatus::Blocking/Blocked (via EnsureAwake):
                                //
                                // Main thread:                    Worker:
                                //   #1 Push work                   #A Set status blocking
                                //   #2 Read worker status          #B Check queue
                                //   #3 Wake if blocking/blocked
                                //
                                // If #A is before #2 then main sees worker blocked and wakes
                                //
                                // If #A if after #2 then #B will see #1, and we abandon blocking
                                assert(!t);
                                t = q.PopFront();
                                if (t) {
                                    should_block = false;
                                }

                                // No work pushed to us, continue attempting to block.  The remaining
                                // test  is to synchronize with termination requests.  If we are
                                // shutting down and all worker threads blocked without work, that's
                                // we are done.
                                if (should_block) {
                                    blocked_++;
                                    if (done_ && blocked_ == num_threads_) {
                                        should_block = false;
                                        // Almost done, but need to re-check queues.
                                        // Consider that all queues are empty and all worker threads are preempted
                                        // right after incrementing blocked_ above. Now a free-standing thread
                                        // submits work and calls destructor (which sets done_). If we don't
                                        // re-check queues, we will exit leaving the work unexecuted.
                                        if (NonEmptyQueueIndex() != -1) {
                                            // Note: we must not pop from queues before we decrement blocked_,
                                            // otherwise the following scenario is possible. Consider that instead
                                            // of checking for emptiness we popped the only element from queues.
                                            // Now other worker threads can start exiting, which is bad if the
                                            // work item submits other work. So we just check emptiness here,
                                            // which ensures that all worker threads exit at the same time.
                                            blocked_--;
                                        } else {
                                            should_exit = true;
                                        }
                                    }
                                }
                                return should_block;
                            },
                            // Post-block update (executed only if we blocked)
                            [&]() {
                                blocked_--;
                            });
                    // Thread just unblocked.  Unless we picked up work while
                    // blocking, or are exiting, then either work was pushed to
                    // us, or it was pushed to an overloaded queue
                    if (!t) t = q.PopFront();
                    if (!t) t = Steal(StealAttemptKind::TRY_ALL);
                }
            }

            if (t) {
                td.SetActive();
                t();
                profiler_.LogRun(thread_id);
                td.SetSpinning();
            }
        }

        // Whichever thread(s) observe the termination conditions are responsible for waking
        // any other threads that have remained blocked.
        if (should_exit) {
            WakeAllWorkersForExit();
        }
    }

    // Steal tries to steal work from other worker threads in a
    // best-effort manner.  We steal only from threads that are running
    // in user code (ThreadStatus::Active).  The intuition behind this
    // is that the thread is busy with other work, and we will avoid
    // "snatching" work from a thread which is just about to notice the
    // work itself.

    Task Steal(StealAttemptKind steal_kind) {
        PerThread* pt = GetPerThread();
        unsigned size = num_threads_;
        unsigned num_attempts = (steal_kind == StealAttemptKind::TRY_ALL) ? size : 1;
        unsigned r = Rand(&pt->rand);
        unsigned inc = all_coprimes_[size - 1][r % all_coprimes_[size - 1].size()];
        unsigned victim = r % size;

        for (unsigned i = 0; i < num_attempts; i++) {
            assert(victim < size);
            if (worker_data_[victim].GetStatus() == WorkerData::ThreadStatus::Active) {
                Task t = worker_data_[victim].queue.PopBack();
                if (t) {
                    return t;
                }
            }
            victim += inc;
            if (victim >= size) {
                victim -= size;
            }
        }

        return Task();
    }

    int NonEmptyQueueIndex() {
        PerThread* pt = GetPerThread();
        const unsigned size = static_cast<unsigned>(worker_data_.size());
        unsigned r = Rand(&pt->rand);
        unsigned inc = all_coprimes_[size - 1][r % all_coprimes_[size - 1].size()];
        unsigned victim = r % size;
        for (unsigned i = 0; i < size; i++) {
            if (!worker_data_[victim].queue.Empty()) {
                return victim;
            }
            victim += inc;
            if (victim >= size) {
                victim -= size;
            }
        }
        return -1;
    }

    static EIGEN_STRONG_INLINE uint64_t GlobalThreadIdHash() {
        return std::hash<std::thread::id>()(std::this_thread::get_id());
    }

    static EIGEN_STRONG_INLINE PerThread* GetPerThread() {
        static thread_local PerThread per_thread_;
        PerThread* pt = &per_thread_;
        if (!pt->initialized) {
            pt->rand = GlobalThreadIdHash();
            pt->initialized = true;
        }
        return pt;
    }

    static EIGEN_STRONG_INLINE unsigned Rand(uint64_t* state) {
        uint64_t current = *state;
        // Update the internal state
        *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
        // Generate the random output (using the PCG-XSH-RS scheme)
        return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
    }
};

//class Env;

class OrtThreadPool {
public:
#ifdef _WIN32
    using NAME_CHAR_TYPE = wchar_t;
#else
    using NAME_CHAR_TYPE = char;
#endif
    // Constructs a pool for running with with "degree_of_parallelism" threads with
    // specified "name". env->StartThread() is used to create individual threads
    // with the given OrtThreadOptions. If "low_latency_hint" is true the thread pool
    // implementation may use it as a hint that lower latency is preferred at the
    // cost of higher CPU usage, e.g. by letting one or more idle threads spin
    // wait. Conversely, if the threadpool is used to schedule high-latency
    // operations like I/O the hint should be set to false.
    //
    // REQUIRES: degree_of_parallelism > 0
    OrtThreadPool(const OrtThreadOptions& thread_options,
               const NAME_CHAR_TYPE* name,
               int degree_of_parallelism,
               bool low_latency_hint,
               bool force_hybrid = false);

    // Waits until all scheduled work has finished and then destroy the
    // set of threads.
    ~OrtThreadPool();

    // Start and end a multi-loop parallel section.  Parallel loops can
    // be executed directly (without using this API), but entering a
    // parallel section allows the runtime system to amortize loop
    // entry/exit costs over multiple loops, and allows it to promote
    // affinity between corresponding iterations of different loops.
    //
    // Multi-loop sections would typically be used in cases where a
    // series of loops executes without much code in between them, and
    // where it is impractical to refactor code into a single loop.  For
    // instance:
    //
    // {
    //   onnxruntime::concurrency::ThreadPoool::ParallelSection ps(tp);
    //   for (int x = 0; x < seq_len; x++) {
    //     TrySimpleParallelFor(tp, 16, [&]() { ... });
    //   }
    // }
    //
    // The parallel section is entered via the constructor of
    // ThreadPool::ParallelSection, and exited via the destructor.
    // Currently, thread-local state is used to track whether or not the
    // current thread is inside a parallel section.  In contrast to
    // handling parallel section objects explicitly in user code, this
    // approach allows code such as MLAS to operate with/without the use
    // of parallel sections.
    //
    // Parallel sections are only implemented with the Eigen threadpool.
    // They have no effect when using OpenMP.
    //
    // Parallel sections may not be nested, and may not be used inside
    // parallel loops.

    class ParallelSection {
    public:
        explicit ParallelSection(OrtThreadPool *tp);
        ~ParallelSection();

    private:
        friend class OrtThreadPool;

        // Owning reference for the underlying OrtThreadPoolParallelSection
        // which implements the thread management.  We use an explicit
        // deleter here so that the definition of
        // OrtThreadPoolParallelSection does not need to be available at this
        // point to avoid a dependence on the Eigen headers.
        std::unique_ptr<OrtThreadPoolParallelSection, void(*)(OrtThreadPoolParallelSection*)>
                ps_{nullptr, [](OrtThreadPoolParallelSection*){}};
        OrtThreadPool *tp_;
        ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ParallelSection);

        // Non-owning reference to the current thread's paralel section
        // (or nullptr outside parallel sections).
        static thread_local ParallelSection *current_parallel_section;
        static_assert(std::is_trivially_destructible<decltype(current_parallel_section)>::value,
                      "Per-thread state should be trivially destructible");
    };

    // The below API allows to disable spinning
    // This is used to support real-time scenarios where
    // spinning between relatively infrequent requests
    // contributes to high CPU usage while not processing anything.
    void EnableSpinning();

    void DisableSpinning();

    // Schedules fn() for execution in the pool of threads.  The function may run
    // synchronously if it cannot be enqueued.  This will occur if the thread pool's
    // degree-of-parallelism is 1, but it may also occur for implementation-dependent
    // reasons such as if queues used for buffering work are full.
    static void Schedule(OrtThreadPool* tp,
                         std::function<void()> fn) {
        if (tp) {
            tp->Schedule(fn);
        } else {
            fn();
        }
    }

    // ParallelFor shards the "total" units of work assuming each unit of work
    // having roughly "cost_per_unit" cost, in cycles. Each unit of work is
    // indexed 0, 1, ..., total - 1. Each shard contains 1 or more units of work
    // and the total cost of each shard is roughly the same.
    //
    // "cost_per_unit" is an estimate of the number of CPU cycles (or nanoseconds
    // if not CPU-bound) to complete a unit of work. Overestimating creates too
    // many shards and CPU time will be dominated by per-shard overhead, such as
    // Context creation. Underestimating may not fully make use of the specified
    // parallelism, and may also cause inefficiencies due to load balancing
    // issues and stragglers.

    static void TryParallelFor(OrtThreadPool* tp, std::ptrdiff_t total, double cost_per_unit,
            const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn) {
        TryParallelFor(tp, total, TensorOpCost{0, 0, static_cast<double>(cost_per_unit)}, fn);
    }

    static void TryParallelFor(OrtThreadPool* tp, std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
            const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn);

    // Directly schedule the 'total' tasks to the underlying threadpool, without
    // cutting them by halves

    inline static void TrySimpleParallelFor(OrtThreadPool* tp, std::ptrdiff_t total,
                                            const std::function<void(std::ptrdiff_t)>& fn) {
        if (tp != nullptr) {
            tp->SimpleParallelFor(total, fn);
        } else {
            for (std::ptrdiff_t i = 0; i < total; ++i) {
                // In many cases, fn can be inlined here.
                fn(i);
            }
        }
    }

    /**
     * Tries to call the given function in parallel, with calls split into (num_batches) batches.
     *\param num_batches If it is zero, it will be replaced to the value of DegreeOfParallelism().
     *\param fn A std::function or STL style functor with signature of "void f(std::ptrdiff_t);"
     * Pitfall: Caller should cap `num_batches` to a reasonable value based on the cost of `fn` and the value of `total`.
     *For example, if fn is as simple as: int sum=0; fn = [&](int i){sum +=i;} and `total` is 100, then num_batches should
     *be just 1.
     *
     * ```
     **/
    template <typename F>
    inline static void TryBatchParallelFor(OrtThreadPool* tp, std::ptrdiff_t total, F&& fn, std::ptrdiff_t num_batches) {
        if (tp == nullptr) {
            for (std::ptrdiff_t i = 0; i < total; ++i) {
                // In many cases, fn can be inlined here.
                fn(i);
            }
            return;
        }
        if (total <= 0)
            return;

        if (total == 1) {
            fn(0);
            return;
        }

        if (num_batches <= 0) {
            num_batches = std::min<std::ptrdiff_t>(total, DegreeOfParallelism(tp));
        }

        if (num_batches <= 1) {
            for (int i = 0; i < total; i++) {
                fn(i);
            }
            return;
        }

        tp->SimpleParallelFor(num_batches, [&](std::ptrdiff_t batch_index) {
            auto work = PartitionWork(batch_index, num_batches, total);
            for (std::ptrdiff_t i = work.start; i < work.end; i++) {
                fn(i);
            }
        });
    }

    struct WorkInfo {
        std::ptrdiff_t start{0};
        std::ptrdiff_t end{0};
    };

    /** Calculate the start and end offsets for a batch.
        @remarks Based on MlasPartitionWork
    */
    constexpr static WorkInfo PartitionWork(std::ptrdiff_t batch_idx, std::ptrdiff_t num_batches, std::ptrdiff_t total_work) {
        const std::ptrdiff_t work_per_batch = total_work / num_batches;
        const std::ptrdiff_t work_per_batch_extra = total_work % num_batches;

        WorkInfo info;
        if (batch_idx < work_per_batch_extra) {
            info.start = (work_per_batch + 1) * batch_idx;
            info.end = info.start + work_per_batch + 1;
        } else {
            info.start = work_per_batch * batch_idx + work_per_batch_extra;
            info.end = info.start + work_per_batch;
        }

        return info;
    }

    //......................................................................
    //
    // The following static methods take into account whether OpenMP is
    // enabled/disabled, and if the thread pool pointer is nullptr
    // during sequential execution.

    // Provide a hint to the caller for whether or not to parallelize
    // work.  This lets a caller switch to a sequential version of an
    // algorithm rather than using calls via the ParallelFor functions.

    static bool ShouldParallelize(const OrtThreadPool* tp);

    // Return the degree of parallelism that code should assume when using the thread pool.
    // It decouples the degree of parallelism for use with the thread pool from
    // the implementation choice of whether this matches the number of threads created in
    // the pool.
    //
    // Currently, a loop with degree-of-parallelism N is supported by a pool of N-1 threads
    // working in combination with the thread initiating the loop.
    static int DegreeOfParallelism(const OrtThreadPool* tp);

    ORT_DISALLOW_COPY_AND_ASSIGNMENT(OrtThreadPool);

    // StartProfiling and StopProfiling are not to be consumed as public-facing API
    static void StartProfiling(OrtThreadPool* tp);
    static std::string StopProfiling(OrtThreadPool* tp);

private:
    friend class LoopCounter;

    // Returns the number of threads created in the pool.  This may be different from the
    // value returned by DegreeOfParallelism to code using the pool.
    int NumThreads() const;

    // Returns current thread id between 0 and NumThreads() - 1, if called from a
    // thread in the pool. Returns -1 otherwise.
    int CurrentThreadId() const;

    // Run fn with up to n degree-of-parallelism enlisting the thread pool for
    // help.  The degree-of-parallelism includes the caller, and so if n==1
    // then the function will run directly in the caller.  The fork-join
    // synchronization is handled in the thread pool, and so any state captured
    // by fn() is safe from concurrent access once RunWithHelp returns.
    void RunInParallel(std::function<void(unsigned idx)> fn, unsigned n, std::ptrdiff_t block_size);

    // Divides the work represented by the range [0, total) into k shards.
    // Calls fn(i*block_size, (i+1)*block_size) from the ith shard (0 <= i < k).
    // Each shard may be executed on a different thread in parallel, depending on
    // the number of threads available in the pool.
    // When (i+1)*block_size > total, fn(i*block_size, total) is called instead.
    // Requires 0 < block_size <= total.
    void ParallelForFixedBlockSizeScheduling(std::ptrdiff_t total, std::ptrdiff_t block_size,
                                             const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn);

    // Return whether or not the calling thread should run a loop of
    // num_iterations divided in chunks of block_size in parallel.  If not,
    // the caller should run the loop sequentially.
    bool ShouldParallelizeLoop(const std::ptrdiff_t num_iterations,
                               const std::ptrdiff_t block_size = 1) const;

    // Internal (non-static) parallel loop methods.  Unlike the public static methods,
    // these will not handle the cases of OpenMP builds. or builds without a threadpool.
    void ParallelFor(std::ptrdiff_t total, double cost_per_unit,
            const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn);

    void ParallelFor(std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
            const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn);

    void SimpleParallelFor(std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn);

    void Schedule(std::function<void()> fn);

    void StartProfiling();

    std::string StopProfiling();

    OrtThreadOptions thread_options_;

    // If a thread pool is created with degree_of_parallelism != 1 then an underlying
    // EigenThreadPool is used to create OS threads and handle work distribution to them.
    // If degree_of_parallelism == 1 then underlying_threadpool_ is left as nullptr
    // and parallel work is run directly by the caller.
    OrtExtendedThreadPoolInterface* underlying_threadpool_ = nullptr;

    // If used, underlying_threadpool_ is instantiated and owned by the ThreadPool.
    std::unique_ptr<OrtThreadPoolTempl> extended_eigen_threadpool_;

    // Force the thread pool to run in hybrid mode on a normal cpu.
    bool force_hybrid_ = false;
};

static std::unique_ptr<OrtThreadPool>
CreateThreadPoolHelper(OrtThreadPoolParams options);

std::unique_ptr<OrtThreadPool>
CreateThreadPool(OrtThreadPoolParams options);

OrtThreadPool *
OrtCreateThreadPool(OrtThreadPoolParams options);

#endif //C__EIGEN_THREAD_POOL_H
