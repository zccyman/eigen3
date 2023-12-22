#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <algorithm>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <atomic>
#include <condition_variable>  // NOLINT(build/c++11)
#include <mutex>               // NOLINT(build/c++11)
#include <thread>              // NOLINT(build/c++11)
#include <vector>
#include <iostream>

typedef int index_t;

template <typename T>
T *cpu_splice(std::vector<int> &in_shape,
    std::vector<int> &context_,
    std::vector<int> &forward_indexs,
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape,
    T *input_data,  //T *output_data,
    int const_dim_);

template <typename T>
T *gpu_splice(std::vector<int> &in_shape,
    std::vector<int> &context_,
    std::vector<int> &forward_indexs,
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape,
    T *input, /*T *output,*/
    int const_dim_ = 0);