//
// Created by shengyuan.shen on 2023/4/10.
//

#include "eigen_thread_pool.h"

OrtThreadPoolProfiler::OrtThreadPoolProfiler(int num_threads, const CHAR_TYPE* thread_pool_name) : num_threads_(num_threads) {
    child_thread_stats_.assign(num_threads, {});
    if (thread_pool_name) {
#ifdef _WIN32
        thread_pool_name_ = ToUTF8String(thread_pool_name);
#else
        thread_pool_name_ = thread_pool_name;
#endif
    } else {
        thread_pool_name_ = "unnamed_thread_pool";
    }
}

OrtThreadPoolProfiler::~OrtThreadPoolProfiler() {
    enabled_ = false;
}

void OrtThreadPoolProfiler::Start() {
    enabled_ = true;
}

OrtThreadPoolProfiler::MainThreadStat& OrtThreadPoolProfiler::GetMainThreadStat() {
    static thread_local std::unique_ptr<MainThreadStat> stat;
    if (!stat) {
        stat = std::make_unique<MainThreadStat>();
    }
    return *stat;
}

std::string OrtThreadPoolProfiler::Stop() {
    if(!enabled_){
        std::cout << "Profiler not started yet\n";
        exit(-1);
    }
    std::ostringstream ss;
    ss << "{\"main_thread\": {"
       << "\"thread_pool_name\": \""
       << thread_pool_name_ << "\", "
       << GetMainThreadStat().Reset()
       << "}, \"sub_threads\": {"
       << DumpChildThreadStat()
       << "}}";
    return ss.str();
}

void OrtThreadPoolProfiler::LogStartAndCoreAndBlock(std::ptrdiff_t block_size) {
    if (enabled_) {
        MainThreadStat& stat = GetMainThreadStat();
        stat.LogCore();
        stat.LogBlockSize(block_size);
        stat.LogStart();
    }
}

void OrtThreadPoolProfiler::LogCoreAndBlock(std::ptrdiff_t block_size) {
    if (enabled_) {
        MainThreadStat& stat = GetMainThreadStat();
        stat.LogCore();
        stat.LogBlockSize(block_size);
    }
}

void OrtThreadPoolProfiler::LogStart() {
    if (enabled_) {
        GetMainThreadStat().LogStart();
    }
}

void OrtThreadPoolProfiler::LogEnd(ThreadPoolEvent evt) {
    if (enabled_) {
        GetMainThreadStat().LogEnd(evt);
    }
}

void OrtThreadPoolProfiler::LogEndAndStart(ThreadPoolEvent evt) {
    if (enabled_) {
        GetMainThreadStat().LogEndAndStart(evt);
    }
}

void OrtThreadPoolProfiler::MainThreadStat::LogCore() {
#ifdef _WIN32
    core_ = GetCurrentProcessorNumber();
#elif defined(__APPLE__)
    #if defined(__x86_64__) || defined(__i386__)
  uint32_t CPUInfo[4];
  __cpuid_count(1, 0, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  if ((CPUInfo[3] & (1 << 9)) != 0) {
    core_ = (unsigned)CPUInfo[1] >> 24;
  }
#endif
#elif defined(__wasm__)
    core_ = emscripten_num_logical_cores();
#else
    core_ = sched_getcpu();
#endif
}

void OrtThreadPoolProfiler::MainThreadStat::LogBlockSize(std::ptrdiff_t block_size) {
    blocks_.emplace_back(block_size);
}

void OrtThreadPoolProfiler::MainThreadStat::LogStart() {
    points_.emplace_back(Clock::now());
}

void OrtThreadPoolProfiler::MainThreadStat::LogEnd(ThreadPoolEvent evt) {
    if(points_.empty()){
        std::cout << "LogStart must pair with LogEnd\n";
        exit(-1);
    }
    events_[evt] += TimeDiffMicroSeconds(points_.back(), Clock::now());
    points_.pop_back();
}

void OrtThreadPoolProfiler::MainThreadStat::LogEndAndStart(ThreadPoolEvent evt) {
    if(points_.empty()){
        std::cout << "LogStart must pair with LogEnd\n";
        exit(-1);
    }
    events_[evt] += TimeDiffMicroSeconds(points_.back(), Clock::now());
    points_.back() = Clock::now();
}

std::string OrtThreadPoolProfiler::MainThreadStat::Reset() {
    if(!points_.empty()){
        std::cout << "LogStart must pair with LogEnd\n";
        exit(-1);
    }
    std::stringstream ss;
    ss << "\"thread_id\": \"" << std::this_thread::get_id() << "\", \"block_size\": [";
    if (!blocks_.empty()) {
        std::copy(blocks_.begin(), blocks_.end() - 1, std::ostream_iterator<std::ptrdiff_t>(ss, ", "));
        ss << blocks_.back();
        blocks_.clear();
    }
    ss << "], \"core\": " << core_ << ", ";
    for (int i = 0; i < MAX_EVENT; ++i) {
        ss << "\"" << OrtThreadPoolProfiler::GetEventName(static_cast<ThreadPoolEvent>(i))
           << "\": " << events_[i] << ((i == MAX_EVENT - 1) ? std::string{} : ", ");
    }
    memset(events_, 0, sizeof(uint64_t) * MAX_EVENT);
    return ss.str();
}

const char* OrtThreadPoolProfiler::GetEventName(ThreadPoolEvent event) {
    switch (event) {
        case DISTRIBUTION:
            return "Distribution";
        case DISTRIBUTION_ENQUEUE:
            return "DistributionEnqueue";
        case RUN:
            return "Run";
        case WAIT:
            return "Wait";
        case WAIT_REVOKE:
            return "WaitRevoke";
        default:
            return "UnknownEvent";
    }
}

void OrtThreadPoolProfiler::LogThreadId(int thread_idx) {
    child_thread_stats_[thread_idx].thread_id_ = std::this_thread::get_id();
}

void OrtThreadPoolProfiler::LogRun(int thread_idx) {
    if (enabled_) {
        child_thread_stats_[thread_idx].num_run_++;
        auto now = Clock::now();
        if (child_thread_stats_[thread_idx].core_ < 0 ||
            TimeDiffMicroSeconds(child_thread_stats_[thread_idx].last_logged_point_, now) > 10000) {
#ifdef _WIN32
            child_thread_stats_[thread_idx].core_ = GetCurrentProcessorNumber();
#elif defined(__APPLE__)
            #if defined(__x86_64__) || defined(__i386__)
      uint32_t CPUInfo[4];
      __cpuid_count(1, 0, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
      if ((CPUInfo[3] & (1 << 9)) != 0) {
        child_thread_stats_[thread_idx].core_ = (unsigned)CPUInfo[1] >> 24;
      }
#endif
#elif defined(__wasm__)
            child_thread_stats_[thread_idx].core_ = emscripten_num_logical_cores();
#else
            child_thread_stats_[thread_idx].core_ = sched_getcpu();
#endif
            child_thread_stats_[thread_idx].last_logged_point_ = now;
        }
    }
}

std::string OrtThreadPoolProfiler::DumpChildThreadStat() {
    std::stringstream ss;
    for (int i = 0; i < num_threads_; ++i) {
        ss << "\"" << child_thread_stats_[i].thread_id_ << "\": {"
           << "\"num_run\": " << child_thread_stats_[i].num_run_ << ", "
           << "\"core\": " << child_thread_stats_[i].core_ << "}"
           << (i == num_threads_ - 1 ? "" : ",");
    }
    return ss.str();
}

static std::unique_ptr<OrtThreadPool>
CreateThreadPoolHelper(OrtThreadPoolParams options) {
    if (options.thread_pool_size == 1)
        return nullptr;
    std::vector<size_t> cpu_list;
    OrtThreadOptions to;
    if (options.affinity_vec_len != 0) {
        to.affinity.assign(options.affinity_vec, options.affinity_vec + options.affinity_vec_len);
    }
    if (options.thread_pool_size <= 0) {  // default
        cpu_list = GetThreadAffinityMasks();
        if (cpu_list.empty() || cpu_list.size() == 1)
            return nullptr;
        options.thread_pool_size = static_cast<int>(cpu_list.size());
        if (options.auto_set_affinity)
            to.affinity = cpu_list;
    }
    to.set_denormal_as_zero = options.set_denormal_as_zero;

    // set custom thread management members
    to.custom_create_thread_fn = options.custom_create_thread_fn;
    to.custom_thread_creation_options = options.custom_thread_creation_options;
    to.custom_join_thread_fn = options.custom_join_thread_fn;
    to.dynamic_block_base_ = options.dynamic_block_base_;
    if (to.custom_create_thread_fn) {
        std::cout << "custom join thread function not set\n";
        exit(-1);
    }

    return std::make_unique<OrtThreadPool>(to, options.name, options.thread_pool_size,
                                        options.allow_spinning);
}

OrtThreadPool *
OrtCreateThreadPool(OrtThreadPoolParams options){
    if (options.thread_pool_size == 1)
        return nullptr;
    std::vector<size_t> cpu_list;
    OrtThreadOptions to;
    if (options.affinity_vec_len != 0) {
        to.affinity.assign(options.affinity_vec, options.affinity_vec + options.affinity_vec_len);
    }
    if (options.thread_pool_size <= 0) {  // default
        cpu_list = GetThreadAffinityMasks();
        if (cpu_list.empty() || cpu_list.size() == 1)
            return nullptr;
        options.thread_pool_size = static_cast<int>(cpu_list.size());
        if (options.auto_set_affinity)
            to.affinity = cpu_list;
    }
    to.set_denormal_as_zero = options.set_denormal_as_zero;

    // set custom thread management members
    to.custom_create_thread_fn = options.custom_create_thread_fn;
    to.custom_thread_creation_options = options.custom_thread_creation_options;
    to.custom_join_thread_fn = options.custom_join_thread_fn;
    to.dynamic_block_base_ = options.dynamic_block_base_;
    if (to.custom_create_thread_fn) {
        std::cout << "custom join thread function not set\n";
        exit(-1);
    }

    return new OrtThreadPool(to, options.name, options.thread_pool_size,
                                           options.allow_spinning);
}

std::unique_ptr<OrtThreadPool>
CreateThreadPool(OrtThreadPoolParams options) {
    // If openmp is enabled we don't want to create any additional threadpools for sequential execution.
    // However, parallel execution relies on
    return CreateThreadPoolHelper(options);
}

OrtThreadPool::OrtThreadPool(const OrtThreadOptions& thread_options,
                       const NAME_CHAR_TYPE* name,
                       int degree_of_parallelism,
                       bool low_latency_hint,
                       bool force_hybrid)
        : thread_options_(thread_options), force_hybrid_(force_hybrid) {
    // In the current implementation, a thread pool with degree_of_parallelism==1 uses
    // the caller as one of the threads for executing work.  Hence we only create
    // additional thread(s) for degree_of_parallelism>=2.
    assert(degree_of_parallelism >= 1);
    if (degree_of_parallelism >= 2) {
        int threads_to_create = degree_of_parallelism - 1;

        if (!thread_options_.affinity.empty()) {
            // Remove first affinity element as designated for the caller thread
            thread_options_.affinity.erase(thread_options_.affinity.begin());
            assert(thread_options_.affinity.size() >= size_t(threads_to_create));
        }

        extended_eigen_threadpool_ =
                std::make_unique<OrtThreadPoolTempl>(name,
                                                     threads_to_create,
                                                     low_latency_hint,
                                                     thread_options_);
        underlying_threadpool_ = extended_eigen_threadpool_.get();
    }
}

OrtThreadPool::~OrtThreadPool() = default;

// Base case for parallel loops, running iterations 0..total, divided into blocks
// of block_size iterations, and calling into a function that takes a start..end
// range of indices to run.
void OrtThreadPool::ParallelForFixedBlockSizeScheduling(const std::ptrdiff_t total,
                                                     const std::ptrdiff_t block_size,
                                                     const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn) {
    if (total <= 0)
        return;

    if (total <= block_size) {
        fn(0, total);
        return;
    }

    auto d_of_p = DegreeOfParallelism(this);
    if (thread_options_.dynamic_block_base_ <= 0) {
        // Split the work across threads in the pool.  Each work item will run a loop claiming iterations,
        // hence we need at most one for each thread, even if the number of blocks of iterations is larger.
        auto num_blocks = total / block_size;
        auto num_threads_inc_main = NumThreads() + 1;
        int num_work_items = static_cast<int>(std::min(static_cast<std::ptrdiff_t>(num_threads_inc_main), num_blocks));
        assert(num_work_items > 0);

        LoopCounter lc(total, d_of_p, block_size);
        std::function<void(unsigned)> run_work = [&](unsigned idx) {
            unsigned my_home_shard = lc.GetHomeShard(idx);
            unsigned my_shard = my_home_shard;
            uint64_t my_iter_start, my_iter_end;
            while (lc.ClaimIterations(my_home_shard, my_shard, my_iter_start, my_iter_end, block_size)) {
                fn(static_cast<std::ptrdiff_t>(my_iter_start),
                   static_cast<std::ptrdiff_t>(my_iter_end));
            }
        };
        // Run the work in the thread pool (and in the current thread).  Synchronization with helping
        // threads is handled within RunInParallel, hence we can deallocate lc and other state captured by
        // run_work.
        RunInParallel(run_work, num_work_items, block_size);
    } else {
        int num_of_blocks = d_of_p * thread_options_.dynamic_block_base_;
        std::ptrdiff_t base_block_size = static_cast<std::ptrdiff_t>(std::max(1LL, std::llroundl(static_cast<long double>(total) / num_of_blocks)));
        alignas(CACHE_LINE_BYTES) std::atomic<std::ptrdiff_t> left{total};
        LoopCounter lc(total, d_of_p, base_block_size);
        std::function<void(unsigned)> run_work = [&](unsigned idx) {
            std::ptrdiff_t b = base_block_size;
            unsigned my_home_shard = lc.GetHomeShard(idx);
            unsigned my_shard = my_home_shard;
            uint64_t my_iter_start, my_iter_end;
            while (lc.ClaimIterations(my_home_shard, my_shard, my_iter_start, my_iter_end, b)) {
                fn(static_cast<std::ptrdiff_t>(my_iter_start),
                   static_cast<std::ptrdiff_t>(my_iter_end));
                auto todo = left.fetch_sub(static_cast<std::ptrdiff_t>(my_iter_end - my_iter_start), std::memory_order_relaxed);
                if (b > 1) {
                    b = static_cast<std::ptrdiff_t>(std::max(1LL, std::llroundl(static_cast<long double>(todo) / num_of_blocks)));
                }
            }
        };
        // Distribute task among all threads in the pool, reduce number of work items if
        // num_of_blocks is smaller than number of threads.
        RunInParallel(run_work, std::min(NumThreads() + 1, num_of_blocks), base_block_size);
    }
}

void OrtThreadPool::SimpleParallelFor(std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
    ParallelForFixedBlockSizeScheduling(total, 1, [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        for (std::ptrdiff_t idx = first; idx < last; idx++) {
            fn(idx);
        }
    });
}

void OrtThreadPool::Schedule(std::function<void()> fn) {
    if (underlying_threadpool_) {
        underlying_threadpool_->Schedule(std::move(fn));
    } else {
        fn();
    }
}

void OrtThreadPool::StartProfiling() {
    if (underlying_threadpool_) {
        underlying_threadpool_->StartProfiling();
    }
}

std::string OrtThreadPool::StopProfiling() {
    if (underlying_threadpool_) {
        return underlying_threadpool_->StopProfiling();
    } else {
        return {};
    }
}

thread_local OrtThreadPool::ParallelSection* OrtThreadPool::ParallelSection::current_parallel_section{nullptr};

OrtThreadPool::ParallelSection::ParallelSection(OrtThreadPool* tp) {
    if(current_parallel_section){
        std::cout << "Nested parallelism not supported\n";
        exit(-1);
    }
    if (ps_.get()){
        std::cout << "ps_ inner pointer is empty!\n";
        exit(-1);
    }
    tp_ = tp;
    if (tp && tp->underlying_threadpool_) {
        ps_ = tp->underlying_threadpool_->AllocateParallelSection();
        tp_->underlying_threadpool_->StartParallelSection(*ps_.get());
        current_parallel_section = this;
    }
}

OrtThreadPool::ParallelSection::~ParallelSection() {
    if (current_parallel_section) {
        tp_->underlying_threadpool_->EndParallelSection(*ps_.get());
        ps_.reset();
        current_parallel_section = nullptr;
    }
}

void OrtThreadPool::RunInParallel(std::function<void(unsigned idx)> fn, unsigned n, std::ptrdiff_t block_size) {
    if (underlying_threadpool_) {
        if (OrtThreadPool::ParallelSection::current_parallel_section) {
            underlying_threadpool_->RunInParallelSection(*(OrtThreadPool::ParallelSection::current_parallel_section->ps_.get()),
                                                         std::move(fn),
                                                         n, block_size);
        } else {
            underlying_threadpool_->RunInParallel(std::move(fn),
                                                  n, block_size);
        }
    } else {
        fn(0);
    }
}

bool OrtThreadPool::ShouldParallelizeLoop(const std::ptrdiff_t num_iterations,
                                       const std::ptrdiff_t block_size) const {
    // Do not parallelize trivial loops, with only a single block of work
    if (block_size <= 0 || num_iterations <= block_size) {
        return false;
    }

    // Do not parallelize loops with only a single thread available.  If the
    // caller is outside the current pool (ID == -1) then we parallelize
    // if the pool has any threads.  If the caller is inside the current pool
    // (ID != -1) then we require at least one additional thread in the pool.
    if ((CurrentThreadId() == -1 && NumThreads() == 0) ||
        (CurrentThreadId() != -1 && NumThreads() == 1)) {
        return false;
    }

    return true;
}

using CostModel = Eigen::TensorCostModel<Eigen::ThreadPoolDevice>;

// Calculates block size based on (1) the iteration cost and (2) parallel
// efficiency. We want blocks to be not too small to mitigate parallelization
// overheads; not too large to mitigate tail effect and potential load
// imbalance and we also want number of blocks to be evenly dividable across
// threads.
static ptrdiff_t CalculateParallelForBlock(const ptrdiff_t n, const Eigen::TensorOpCost& cost,
                                           std::function<ptrdiff_t(ptrdiff_t)> block_align, int num_threads) {
    const double block_size_f = 1.0 / CostModel::taskSize(1, cost);
    constexpr ptrdiff_t max_oversharding_factor = 4;
    ptrdiff_t block_size = Eigen::numext::mini(
            n,
            Eigen::numext::maxi<ptrdiff_t>(Eigen::divup<ptrdiff_t>(n, max_oversharding_factor * num_threads), static_cast<ptrdiff_t>(block_size_f)));
    const ptrdiff_t max_block_size = Eigen::numext::mini(n, 2 * block_size);

    if (block_align) {
        ptrdiff_t new_block_size = block_align(block_size);
        assert(new_block_size >= block_size);
        block_size = Eigen::numext::mini(n, new_block_size);
    }

    ptrdiff_t block_count = Eigen::divup(n, block_size);

    // Calculate parallel efficiency as fraction of total CPU time used for
    // computations:
    double max_efficiency =
            static_cast<double>(block_count) / (Eigen::divup<ptrdiff_t>(block_count, num_threads) * num_threads);

    // Now try to increase block size up to max_block_size as long as it
    // doesn't decrease parallel efficiency.
    for (ptrdiff_t prev_block_count = block_count; max_efficiency < 1.0 && prev_block_count > 1;) {
        // This is the next block size that divides size into a smaller number
        // of blocks than the current block_size.
        ptrdiff_t coarser_block_size = Eigen::divup(n, prev_block_count - 1);
        if (block_align) {
            ptrdiff_t new_block_size = block_align(coarser_block_size);
            assert(new_block_size >= coarser_block_size);
            coarser_block_size = Eigen::numext::mini(n, new_block_size);
        }
        if (coarser_block_size > max_block_size) {
            break;  // Reached max block size. Stop.
        }
        // Recalculate parallel efficiency.
        const ptrdiff_t coarser_block_count = Eigen::divup(n, coarser_block_size);
        assert(coarser_block_count < prev_block_count);
        prev_block_count = coarser_block_count;
        const double coarser_efficiency =
                static_cast<double>(coarser_block_count) / (Eigen::divup<ptrdiff_t>(coarser_block_count, num_threads) * num_threads);
        if (coarser_efficiency + 0.01 >= max_efficiency) {
            // Taking it.
            block_size = coarser_block_size;
            if (max_efficiency < coarser_efficiency) {
                max_efficiency = coarser_efficiency;
            }
        }
    }

    return block_size;
}

void OrtThreadPool::ParallelFor(std::ptrdiff_t n, const TensorOpCost& c,
        const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& f) {
    if(n < 0){
        std::cout << "ParallelFor n must gt 0\n";
        exit(-1);
    }
    Eigen::TensorOpCost cost{c.bytes_loaded, c.bytes_stored, c.compute_cycles};
    auto d_of_p = DegreeOfParallelism(this);
    // Compute small problems directly in the caller thread.
    if ((!ShouldParallelizeLoop(n)) || CostModel::numThreads(static_cast<double>(n), cost, d_of_p) == 1) {
        f(0, n);
        return;
    }

    ptrdiff_t block = CalculateParallelForBlock(n, cost, nullptr, d_of_p);
    ParallelForFixedBlockSizeScheduling(n, block, f);
}

void OrtThreadPool::ParallelFor(std::ptrdiff_t total, double cost_per_unit,
        const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn) {
ParallelFor(total, TensorOpCost{0, 0, static_cast<double>(cost_per_unit)}, fn);
}

bool OrtThreadPool::ShouldParallelize(const OrtThreadPool* tp) {
    return (DegreeOfParallelism(tp) != 1);
}

int OrtThreadPool::DegreeOfParallelism(const OrtThreadPool* tp) {
    // When not using OpenMP, we parallelise over the N threads created by the pool
    // tp, plus 1 for the thread entering a loop.
    if (tp) {
        if (tp->force_hybrid_) {
            return ((tp->NumThreads() + 1)) * TaskGranularityFactor;
        } else {
            return ((tp->NumThreads() + 1));
        }
    } else {
        return 1;
    }
}

void OrtThreadPool::StartProfiling(OrtThreadPool* tp) {
    if (tp) {
        tp->StartProfiling();
    }
}

std::string OrtThreadPool::StopProfiling(OrtThreadPool* tp) {
    if (tp) {
        return tp->StopProfiling();
    } else {
        return {};
    }
}

void OrtThreadPool::EnableSpinning() {
    if (extended_eigen_threadpool_) {
        extended_eigen_threadpool_->EnableSpinning();
    }
}

void OrtThreadPool::DisableSpinning() {
    if (extended_eigen_threadpool_) {
        extended_eigen_threadpool_->DisableSpinning();
    }
}

// Return the number of threads created by the pool.
int OrtThreadPool::NumThreads() const {
    if (underlying_threadpool_) {
        return underlying_threadpool_->NumThreads();
    } else {
        return 0;
    }
}

// Return ID of the current thread within this pool.  Returns -1 for a thread outside the
// current pool.
int OrtThreadPool::CurrentThreadId() const {
    if (underlying_threadpool_) {
        return underlying_threadpool_->CurrentThreadId();
    } else {
        return -1;
    }
}

void OrtThreadPool::TryParallelFor(OrtThreadPool* tp, std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
        const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn) {
    if (tp == nullptr) {
        fn(0, total);
        return;
    }
    tp->ParallelFor(total, cost_per_unit, fn);
}

OrtEnvThread* CreateThread(const ORTCHAR_T* name_prefix, int index,
                           unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param),
                           Eigen::ThreadPoolInterface* param, const OrtThreadOptions& thread_options) {
#ifdef _WIN32
    return new WindowsThread(name_prefix, index, start_address, param, thread_options);
#else
    return new PosixThread(name_prefix, index, start_address, param, thread_options);
#endif
}