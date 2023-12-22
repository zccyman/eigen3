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
#include "eigen3/Eigen/Dense"


#define USE_THREAD_POOL 0

template <typename T>
void fast_fc(const T* ma, const T* mb, const T* mc, uint32_t K, uint32_t M, uint32_t N, T* result);

