#pragma once

#include <cstdint>

namespace asterflow {
namespace native {

/**
 * 调度与内核分桶的主键（按设备类划分）；与 `DeviceType` 的枚举顺序一致，便于互转。
 * 日后若增加「自动微分」「编译路径」等维度，可改为位集或二级查表，而不必改动设备序。
 */
enum class DispatchKey : std::uint8_t {
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
