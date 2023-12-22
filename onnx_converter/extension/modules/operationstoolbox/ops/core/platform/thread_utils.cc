//
// Created by shengyuan.shen on 2023/4/10.
//

#include "thread_utils.h"

#ifdef _WIN32

#else
std::pair<int, std::string> GetSystemError() {
    auto e = errno;
    char buf[1024];
    const char* msg = "";
    if (e > 0) {
#if defined(__GLIBC__) && defined(_GNU_SOURCE) && !defined(__ANDROID__)
        msg = strerror_r(e, buf, sizeof(buf));
#else
        // for Mac OS X and Android lower than API 23
        if (strerror_r(e, buf, sizeof(buf)) != 0) {
            buf[0] = '\0';
        }
        msg = buf;
#endif
    }

    return std::make_pair(e, msg);
}
#endif
