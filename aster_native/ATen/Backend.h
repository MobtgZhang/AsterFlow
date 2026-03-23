#pragma once

#include <cstdint>

namespace asterflow {
namespace aten {

/**
 * 逻辑后端（与 PyTorch ATen Backend 概念对齐；具体 runtime 由扩展注册）。
 */
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

} // namespace aten
} // namespace asterflow
