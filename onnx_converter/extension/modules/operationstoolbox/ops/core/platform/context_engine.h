//
// Created by shengyuan.shen on 2023/4/12.
//

#ifndef C__CONTEXT_ENGINE_H
#define C__CONTEXT_ENGINE_H

#include "common.h"
#include "eigen_thread_pool.h"
#include "mace_thread_pool.h"

#include <iostream>
#include <memory>

class BaseEngine{
public:
    BaseEngine(int num_thread=-1, CPUAffinityPolicy policy=CPUAffinityPolicy::AFFINITY_NONE);

    TIMESINTELLI_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BaseEngine);

    ~BaseEngine(){
        delete thread_pool_;
    };

#ifdef USE_EIGEN_THREAD_POOL
    OrtThreadPool* get_pool(){
        return thread_pool_;
    }
#else
    MaceThreadPool* get_pool(){
        return thread_pool_;
    }
#endif


private:
    int32_t num_threads_;
    CPUAffinityPolicy cpu_affinity_policy_;

#ifdef USE_EIGEN_THREAD_POOL
    OrtThreadPool *thread_pool_;
#else
    MaceThreadPool *thread_pool_;
#endif

};


#endif //C__CONTEXT_ENGINE_H
