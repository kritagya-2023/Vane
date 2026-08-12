// Minimal dependency-free check harness.
#pragma once

#include <cmath>
#include <cstdio>
#include <string>

namespace vt {

inline int g_checks = 0;
inline int g_failed = 0;
inline std::string g_section;

inline void section(const char* s) {
    g_section = s;
    std::printf("\n-- %s\n", s);
}

inline void check(bool cond, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!cond) {
        ++g_failed;
        std::printf("  FAIL  %s:%d  %s\n", file, line, expr);
    }
}

inline void check_msg(bool cond, const std::string& msg, const char* file, int line) {
    ++g_checks;
    if (!cond) {
        ++g_failed;
        std::printf("  FAIL  %s:%d  %s\n", file, line, msg.c_str());
    }
}

inline bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

inline int report(const char* suite) {
    std::printf("\n%s: %d checks, %d failed\n", suite, g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}

}  // namespace vt

#define CHECK(cond)        ::vt::check((cond), #cond, __FILE__, __LINE__)
#define CHECK_MSG(cond, m) ::vt::check_msg((cond), (m), __FILE__, __LINE__)
