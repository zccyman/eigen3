//
// Created by shengyuan.shen on 2023/4/6.
//

#ifndef C__TIMESINTELLI_MUTEX_H
#define C__TIMESINTELLI_MUTEX_H


// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#define _WIN32

#pragma once
#ifdef _WIN32
#include <Windows.h>
#include <mutex>
#include <nsync_mu.h>

// Q: Why TimesIntelliMutex is better than std::mutex
// A: TimesIntelliMutex supports static initialization but std::mutex doesn't. Static initialization helps us prevent the "static
// initialization order problem".

// Q: Why std::mutex can't make it?
// A: VC runtime has to support Windows XP at ABI level. But we don't have such requirement.

// Q: Is TimesIntelliMutex faster than std::mutex?
// A: Sure

class TimesIntelliMutex {
 private:
  SRWLOCK data_ = SRWLOCK_INIT;

 public:
  constexpr TimesIntelliMutex() = default;
  // SRW locks do not need to be explicitly destroyed.
  ~TimesIntelliMutex() = default;
  TimesIntelliMutex(const TimesIntelliMutex&) = delete;
  TimesIntelliMutex& operator=(const TimesIntelliMutex&) = delete;
  void lock() { AcquireSRWLockExclusive(native_handle()); }
  bool try_lock() noexcept { return TryAcquireSRWLockExclusive(native_handle()) == TRUE; }
  void unlock() noexcept { ReleaseSRWLockExclusive(native_handle()); }
  using native_handle_type = SRWLOCK*;

  __forceinline native_handle_type native_handle() { return &data_; }
};

class TimesIntelliCondVar {
  CONDITION_VARIABLE native_cv_object = CONDITION_VARIABLE_INIT;

 public:
  constexpr TimesIntelliCondVar() noexcept = default;
  ~TimesIntelliCondVar() = default;

  TimesIntelliCondVar(const TimesIntelliCondVar&) = delete;
  TimesIntelliCondVar& operator=(const TimesIntelliCondVar&) = delete;

  void notify_one() noexcept { WakeConditionVariable(&native_cv_object); }
  void notify_all() noexcept { WakeAllConditionVariable(&native_cv_object); }

  void wait(std::unique_lock<TimesIntelliMutex>& lk) {
    if (SleepConditionVariableSRW(&native_cv_object, lk.mutex()->native_handle(), INFINITE, 0) != TRUE) {
      std::terminate();
    }
  }
  template <class _Predicate>
  void wait(std::unique_lock<TimesIntelliMutex>& __lk, _Predicate __pred);

  /**
   * returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise, the method returns
   * cv_status::no_timeout.
   * @param cond_mutex A unique_lock<TimesIntelliMutex> object.
   * @param rel_time A chrono::duration object that specifies the amount of time before the thread wakes up.
   * @return returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise, the method returns
   * cv_status::no_timeout
   */
  template <class Rep, class Period>
  std::cv_status wait_for(std::unique_lock<TimesIntelliMutex>& cond_mutex, const std::chrono::duration<Rep, Period>& rel_time);
  using native_handle_type = CONDITION_VARIABLE*;

  native_handle_type native_handle() { return &native_cv_object; }

 private:
  void timed_wait_impl(std::unique_lock<TimesIntelliMutex>& __lk,
                       std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>);
};

#else
#include "nsync.h"
#include <mutex>               //for unique_lock
#include <condition_variable>  //for cv_status

class TimesIntelliMutex {

    nsync::nsync_mu data_ = NSYNC_MU_INIT;

public:
    constexpr TimesIntelliMutex() = default;
    ~TimesIntelliMutex() = default;
    TimesIntelliMutex(const TimesIntelliMutex&) = delete;
    TimesIntelliMutex& operator=(const TimesIntelliMutex&) = delete;

    void lock() { nsync::nsync_mu_lock(&data_); }
    bool try_lock() noexcept { return nsync::nsync_mu_trylock(&data_) == 0; }
    void unlock() noexcept { nsync::nsync_mu_unlock(&data_); }

    using native_handle_type = nsync::nsync_mu*;
    native_handle_type native_handle() { return &data_; }
};

class TimesIntelliCondVar {
    nsync::nsync_cv native_cv_object = NSYNC_CV_INIT;

public:
    constexpr TimesIntelliCondVar() noexcept = default;

    ~TimesIntelliCondVar() = default;
    TimesIntelliCondVar(const TimesIntelliCondVar&) = delete;
    TimesIntelliCondVar& operator=(const TimesIntelliCondVar&) = delete;

    void notify_one() noexcept { nsync::nsync_cv_signal(&native_cv_object); }
    void notify_all() noexcept { nsync::nsync_cv_broadcast(&native_cv_object); }

    void wait(std::unique_lock<TimesIntelliMutex>& lk);

    template <class _Predicate>
    void wait(std::unique_lock<TimesIntelliMutex>& __lk, _Predicate __pred);

    /**
     * returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise, the method returns
     * cv_status::no_timeout.
     * @param cond_mutex A unique_lock<TimesIntelliMutex> object.
     * @param rel_time A chrono::duration object that specifies the amount of time before the thread wakes up.
     * @return returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise, the method returns
     * cv_status::no_timeout
     */
    template <class Rep, class Period>
    std::cv_status wait_for(std::unique_lock<TimesIntelliMutex>& cond_mutex, const std::chrono::duration<Rep, Period>& rel_time);
    using native_handle_type = nsync::nsync_cv*;
    native_handle_type native_handle() { return &native_cv_object; }

private:
    void timed_wait_impl(std::unique_lock<TimesIntelliMutex>& __lk,
                         std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>);
};

#endif



#endif //C__TIMESINTELLI_MUTEX_H
