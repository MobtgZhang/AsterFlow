#pragma once

#include <stdexcept>
#include <string>

namespace asterflow {
namespace native {
namespace detail {

inline void af_check(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(msg ? msg : "AsterFlow native runtime check failed");
  }
}

inline void af_check(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

} // namespace detail
} // namespace native
} // namespace asterflow
