#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <memory>
#include <sys/syscall.h>
#include <string.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <execinfo.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <algorithm>

// typedef unsigned int uint32_t;
// typedef unsigned long long uint64_t;

namespace utils {
// inline void MakeStringInternal(std::stringstream& /*ss*/) {}

template <typename T>
inline void MakeStringInternal(std::stringstream &ss, const T &t)
{
    ss << t;
}

template <typename T, typename... Args>
inline void MakeStringInternal(
    std::stringstream &ss, const T &t, const Args &...args)
{
    MakeStringInternal(ss, t);
    MakeStringInternal(ss, args...);
}

class StringFormatter {
public:
    static std::string Table(const std::string &title,
        const std::vector<std::string> &header,
        const std::vector<std::vector<std::string>> &data);
};

template <typename... Args>
std::string MakeString(const Args &...args)
{
    std::stringstream ss;
    MakeStringInternal(ss, args...);
    return ss.str();
}

template <typename T>
std::string MakeListString(const T *args, size_t size)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < size; ++i) {
        ss << args[i];
        if (i < size - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

template <typename T>
std::string MakeString(const std::vector<T> &args)
{
    return MakeListString(args.data(), args.size());
}

// Specializations for already-a-string types.
template <>
inline std::string MakeString(const std::string &str)
{
    return str;
}

inline std::string MakeString(const char *c_str)
{
    return std::string(c_str);
}

inline std::string ToLower(const std::string &src)
{
    std::string dest(src);
    std::transform(src.begin(), src.end(), dest.begin(), ::tolower);
    return dest;
}

inline std::string ToUpper(const std::string &src)
{
    std::string dest(src);
    std::transform(src.begin(), src.end(), dest.begin(), ::toupper);
    return dest;
}

inline int ORTGetCPUCount()
{
    int cpu_count = 0;
    std::string cpu_sys_conf = "/proc/cpuinfo";
    std::ifstream f(cpu_sys_conf);
    if (!f.is_open()) {
        std::cout << "failed to open " << cpu_sys_conf << std::endl;
        return -1;
    }
    std::string line;
    const std::string processor_key = "processor";
    while (std::getline(f, line)) {
        if (line.size() >= processor_key.size() &&
            line.compare(0, processor_key.size(), processor_key) == 0) {
            ++cpu_count;
        }
    }
    if (f.bad()) {
        std::cout << "failed to read " << cpu_sys_conf << std::endl;
        exit(-1);
    }
    if (!f.eof()) {
        std::cout << "failed to read end of " << cpu_sys_conf << std::endl;
        exit(-1);
    }
    f.close();
    std::cout << "CPU cores: " << cpu_count << std::endl;
    return cpu_count;
}

template <typename T>
T &&CheckNotNull(const char *file, int line, const char *exprtext, T &&t)
{
    if (t == nullptr) {
        std::cout << std::string(exprtext) << std::endl;
    }
    return std::forward<T>(t);
}

#define MACE_CHECK_NOTNULL(val) \
    CheckNotNull(__FILE__, __LINE__, "'" #val "' Must not be NULL", (val))

// #define MACE_UNUSED(var) (void)(var)

inline int64_t NowMicros()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}
inline std::vector<std::string> GetBackTraceUnsafe(int max_steps)
{
    std::vector<void *> buffer(max_steps, 0);
    int steps = backtrace(buffer.data(), max_steps);

    std::vector<std::string> bt;
    char **symbols = backtrace_symbols(buffer.data(), steps);
    if (symbols != nullptr) {
        for (int i = 0; i < steps; i++) {
            bt.push_back(symbols[i]);
        }
    }
    return bt;
}
};  // namespace utils

namespace env {
class Env {
public:
    virtual int64_t NowMicros() = 0;
    virtual uint32_t CalculateCRC32(const unsigned char *p, uint64_t n);
    virtual bool CheckArrayCRC32(const unsigned char *data, uint64_t len);
    virtual int AdviseFree(void *addr, size_t length);
    virtual int GetCPUMaxFreq(std::vector<float> *max_freqs);
    virtual int SchedSetAffinity(const std::vector<size_t> &cpu_ids);
    // Return the current backtrace, will allocate memory inside the call
    // which may fail
    virtual std::vector<std::string> GetBackTraceUnsafe(int max_steps) = 0;

    static Env *Default();
};
class LinuxBaseEnv : public Env {
public:
    int64_t NowMicros() override;
    int AdviseFree(void *addr, size_t length) override;
    int GetCPUMaxFreq(std::vector<float> *max_freqs) override;
    // FileSystem *GetFileSystem() override;
    int SchedSetAffinity(const std::vector<size_t> &cpu_ids) override;

    // protected:
    // PosixFileSystem posix_file_system_;
};
class LinuxEnv : public LinuxBaseEnv {
public:
    int SchedSetAffinity(const std::vector<size_t> &cpu_ids) override;
    // LogWriter *GetLogWriter() override;
    std::vector<std::string> GetBackTraceUnsafe(int max_steps) override;

    // private:
    // LogWriter log_writer_;
};
// class WindowsEnv : public Env {
//     public:
//         int64_t NowMicros() override;
//         // FileSystem *GetFileSystem() override;
//         // LogWriter *GetLogWriter() override;
//         std::vector<std::string> GetBackTraceUnsafe(int max_steps) override;

//         // private:
//             // LogWriter log_writer_;
//             // WindowsFileSystem windows_file_system_;
// };
};  // namespace env
