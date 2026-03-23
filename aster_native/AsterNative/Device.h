#pragma once

#include <AsterNative/Backend.h>
#include <cstdint>

namespace asterflow {
namespace native {

enum class DeviceType : std::int8_t {
  CPU = 0,
  CUDA,
  HIP,
  MPS,
  XPU,
  Vulkan,
  Meta,
};

struct Device {
  DeviceType type{DeviceType::CPU};
  std::int32_t index{0};

  constexpr bool is_cpu() const noexcept { return type == DeviceType::CPU; }
  constexpr bool is_cuda() const noexcept { return type == DeviceType::CUDA; }
};

Backend device_type_to_backend(DeviceType t) noexcept;

} // namespace native
} // namespace asterflow
