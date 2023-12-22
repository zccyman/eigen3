//
// Created by shengyuan.shen on 2023/4/7.
//

#ifdef _WIN32

#else
#include <execinfo.h>
#include <stdint.h>
//#include <sys/sysctl.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sstream>
#include <fstream>
#include <cstddef>
#include <string>
#include <vector>
#include <errno.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>
#endif

#include <iostream>
#include <stdio.h>
#include "environment.h"

#ifdef _WIN32

LONGLONG GetFrequency(DWORD sleepTime)    //获取CPU主频
{

    DWORD low1 = 0, high1 = 0, low2 = 0, high2 = 0;

    LARGE_INTEGER fq, st, ed;

    /*在定时前应该先调用QueryPerformanceFrequency()函数获得机器内部计时器的时钟频率。接着在

    需要严格计时的事件发生前和发生之后分别调用QueryPerformanceCounter()，利用两次获得的技术

    之差和时钟的频率，就可以计算出时间经历的精确时间。*/

    ::QueryPerformanceFrequency(&fq);    //精确计时（返回硬件支持的高精度计数器的频率）

    ::QueryPerformanceCounter(&st);    //获得起始时间

    __asm {    //获得当前CPU的时间数

        rdtsc

        mov low1, eax

        mov high1, edx

    }

    ::Sleep(sleepTime);    //将线程挂起片刻

    ::QueryPerformanceCounter(&ed);    //获得结束时间

    __asm {

        rdtsc    //读取CPU的时间戳计数器

        mov low2, eax

        mov high2, edx

    }
    //将CPU得时间周期数转化成64位整数

    LONGLONG begin = (LONGLONG)high1 << 32 | low1;

    LONGLONG end = (LONGLONG)high2 << 32 | low2;

    //将两次获得的CPU时间周期数除以间隔时间，即得到CPU的频率

    //由于windows的Sleep函数有大约15毫秒的误差，故以windows的精确计时为准

    return (end - begin)*fq.QuadPart / (ed.QuadPart - st.QuadPart);

}

std::vector<size_t> GetThreadAffinityMasks() {
    auto generate_vector_of_n = [](int n) {
      std::vector<size_t> ret(n);
      std::iota(ret.begin(), ret.end(), 0);
      return ret;
    };
    // Indeed 64 should be enough. However, it's harmless to have a little more.
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
    DWORD returnLength = sizeof(buffer);
    if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
      return generate_vector_of_n(std::thread::hardware_concurrency());
    }
    std::vector<size_t> ret;
    int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    for (int i = 0; i != count; ++i) {
      if (buffer[i].Relationship == RelationProcessorCore) {
        ret.push_back(buffer[i].ProcessorMask);
      }
    }
    if (ret.empty()){
      return generate_vector_of_n(std::thread::hardware_concurrency());
    }
    //std::cout << "CPU cores: " << ret.size() << std::endl;
    return ret;
  }

  int GetCPUMaxFreq(std::vector<float> &max_freqs){
    std::vector<size_t> cpu_cores = GetThreadAffinityMasks();
    int cpu_count = cpu_cores.size();
    float freq = GetFrequency(1000);
    max_freqs.resize(cpu_count){freq};
    retuen 0;
}

int SchedSetAffinity(const std::vector<size_t> &cpu_ids){
    return 0;
}

#else

std::vector<size_t> GetThreadAffinityMasks(){
    std::vector<size_t> ret(std::thread::hardware_concurrency());
    std::iota(ret.begin(), ret.end(), 0);
    return ret;
}

int GetCPUCount() {
    int cpu_count = 0;
    std::string cpu_sys_conf = "/proc/cpuinfo";
    std::ifstream f(cpu_sys_conf);
    if (!f.is_open()) {
        std::cout << "failed to open " << cpu_sys_conf;
        return -1;
    }
    std::string line;
    const std::string processor_key = "processor";
    while (std::getline(f, line)) {
        if (line.size() >= processor_key.size()
            && line.compare(0, processor_key.size(), processor_key) == 0) {
            ++cpu_count;
        }
    }
    if (f.bad()) {
        std::cout << "failed to read " << cpu_sys_conf;
        exit(-1);
    }
    if (!f.eof()) {
        std::cout << "failed to read end of " << cpu_sys_conf;
        exit(-1);
    }
    f.close();
    //std::cout << "CPU cores: " << cpu_count << std::endl;
    return cpu_count;
}

int GetCPUMaxFreq(std::vector<float> &max_freqs){
    int cpu_count = GetCPUCount();
    if (cpu_count < 0) {
        std::cout << "not enough cpu core has been used!\n";
        exit(-1);
    }
    for (int cpu_id = 0; cpu_id < cpu_count; ++cpu_id) {
        std::string cpuinfo_max_freq_sys_conf =
                std::string("/sys/devices/system/cpu/cpu") + std::to_string(cpu_id) + "/cpufreq/cpuinfo_max_freq";
        std::ifstream f(cpuinfo_max_freq_sys_conf);
        if (!f.is_open()) {
            std::cout << "failed to open " << cpuinfo_max_freq_sys_conf;
            exit(-1);
        }
        std::string line;
        if (std::getline(f, line)) {
            float freq = strtof(line.c_str(), nullptr);
            max_freqs.push_back(freq);
        }
        if (f.bad()) {
            std::cout << "failed to read " << cpuinfo_max_freq_sys_conf;
            exit(-1);
        }
        f.close();
    }

    /*std::cout << "CPU freq: ";
    for (auto item : max_freqs){
        std::cout << item << " \n";
    }*/
    std::cout << std::endl;

    return 0;
}

int SchedSetAffinity(const std::vector<size_t> &cpu_ids) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (auto cpu_id : cpu_ids) {
        CPU_SET(cpu_id, &mask);
    }

    pid_t pid = syscall(SYS_gettid);
    int err = sched_setaffinity(pid, sizeof(mask), &mask);
    if (err) {
        std::cout << "SchedSetAffinity failed: " << strerror(errno);
        return -1;
    }

    return 0;
}

#endif

