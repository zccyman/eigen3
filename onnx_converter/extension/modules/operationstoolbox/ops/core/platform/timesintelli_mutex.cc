//
// Created by shengyuan.shen on 2023/4/6.
//

#include "timesintelli_mutex.h"
#include <iostream>

#ifdef _WIN32

template <class _Predicate>
void TimesIntelliCondVar::wait(std::unique_lock<TimesIntelliMutex>& __lk, _Predicate __pred) {
  while (!__pred()) wait(__lk);
}

template <class Rep, class Period>
std::cv_status TimesIntelliCondVar::wait_for(std::unique_lock<TimesIntelliMutex>& cond_mutex,
                                    const std::chrono::duration<Rep, Period>& rel_time) {
  // TODO: is it possible to use nsync_from_time_point_ ?
  using namespace std::chrono;
  if (rel_time <= duration<Rep, Period>::zero())
    return std::cv_status::timeout;
  using SystemTimePointFloat = time_point<system_clock, duration<long double, std::nano> >;
  using SystemTimePoint = time_point<system_clock, nanoseconds>;
  SystemTimePointFloat max_time = SystemTimePoint::max();
  steady_clock::time_point steady_now = steady_clock::now();
  system_clock::time_point system_now = system_clock::now();
  if (max_time - rel_time > system_now) {
    nanoseconds remain = duration_cast<nanoseconds>(rel_time);
    if (remain < rel_time)
      ++remain;
    timed_wait_impl(cond_mutex, system_now + remain);
  } else
    timed_wait_impl(cond_mutex, SystemTimePoint::max());
  return steady_clock::now() - steady_now < rel_time ? std::cv_status::no_timeout : std::cv_status::timeout;
}

#else

template <class _Predicate>
void TimesIntelliCondVar::wait(std::unique_lock<TimesIntelliMutex>& __lk, _Predicate __pred) {
    while (!__pred()) wait(__lk);
}

template <class Rep, class Period>
std::cv_status TimesIntelliCondVar::wait_for(std::unique_lock<TimesIntelliMutex>& cond_mutex,
                                             const std::chrono::duration<Rep, Period>& rel_time) {
    // TODO: is it possible to use nsync_from_time_point_ ?
    using namespace std::chrono;
    if (rel_time <= duration<Rep, Period>::zero())
        return std::cv_status::timeout;
    using SystemTimePointFloat = time_point<system_clock, duration<long double, std::nano> >;
    using SystemTimePoint = time_point<system_clock, nanoseconds>;
    SystemTimePointFloat max_time = SystemTimePoint::max();
    steady_clock::time_point steady_now = steady_clock::now();
    system_clock::time_point system_now = system_clock::now();
    if (max_time - rel_time > system_now) {
        nanoseconds remain = duration_cast<nanoseconds>(rel_time);
        if (remain < rel_time)
            ++remain;
        timed_wait_impl(cond_mutex, system_now + remain);
    } else
        timed_wait_impl(cond_mutex, SystemTimePoint::max());
    return steady_clock::now() - steady_now < rel_time ? std::cv_status::no_timeout : std::cv_status::timeout;
}

#endif

void TimesIntelliCondVar::timed_wait_impl(std::unique_lock<TimesIntelliMutex>& lk,
                                 std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> tp) {
    using namespace std::chrono;
#ifndef NDEBUG
    if (!lk.owns_lock()){
        std::cout <<"condition_variable::timed wait: mutex not locked\n";
        exit(-1);
    }

#endif
    nanoseconds d = tp.time_since_epoch();
    timespec abs_deadline;
    seconds s = duration_cast<seconds>(d);
    using ts_sec = decltype(abs_deadline.tv_sec);
    constexpr ts_sec ts_sec_max = std::numeric_limits<ts_sec>::max();
    if (s.count() < ts_sec_max) {
        abs_deadline.tv_sec = static_cast<ts_sec>(s.count());
        abs_deadline.tv_nsec = static_cast<decltype(abs_deadline.tv_nsec)>((d - s).count());
    } else {
        abs_deadline.tv_sec = ts_sec_max;
        abs_deadline.tv_nsec = 999999999;
    }
    nsync::nsync_cv_wait_with_deadline(&native_cv_object, lk.mutex()->native_handle(), abs_deadline, nullptr);
}

void TimesIntelliCondVar::wait(std::unique_lock<TimesIntelliMutex>& lk) {
#ifndef NDEBUG
    if (!lk.owns_lock()) {
        std::cout << "OrtCondVar wait failed: mutex not locked\n";
        exit(-1);
    }
#endif
    nsync::nsync_cv_wait(&native_cv_object, lk.mutex()->native_handle());
}