//
// Created by shengyuan.shen on 2023/4/12.
//

#include "context_engine.h"

BaseEngine::BaseEngine(int num_thread, CPUAffinityPolicy policy)
        :num_threads_(num_thread), cpu_affinity_policy_(policy){
#ifdef USE_EIGEN_THREAD_POOL
    OrtThreadPoolParams options;
    options.thread_pool_size = num_threads_;
    thread_pool_ = OrtCreateThreadPool(options);
#else
    thread_pool_ = new MaceThreadPool(num_threads_, policy);
#endif
}