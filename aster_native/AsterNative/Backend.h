#pragma once

#include <cstdint>

namespace asterflow {
namespace native {

/** 逻辑执行后端（具体 kernel 由各后端子目录注册）。 */
enum class Backend : std::uint8_t {
  CPU = 0,
  CUDA,
  HIP,
  MPS,
  XPU,
  Vulkan,
  Meta,
  Count,
};

} // namespace native
} // namespace asterflow
