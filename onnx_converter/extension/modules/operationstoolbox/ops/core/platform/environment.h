//
// Created by shengyuan.shen on 2023/4/7.
//

#ifndef C__ENVIRONMENT_H
#define C__ENVIRONMENT_H

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
#include <string>
#include <thread>
#include <utility>  // for std::forward
#include <assert.h>
#endif

#include <vector>
#include <numeric>
#include <atomic>

std::vector<size_t> GetThreadAffinityMasks();

int GetCPUMaxFreq(std::vector<float> &max_freqs);

int SchedSetAffinity(const std::vector<size_t> &cpu_ids);

#endif //C__ENVIRONMENT_H
