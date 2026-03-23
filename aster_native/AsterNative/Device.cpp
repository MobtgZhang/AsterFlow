#include <AsterNative/Device.h>

namespace asterflow {
namespace native {

Backend device_type_to_backend(DeviceType t) noexcept {
  switch (t) {
  case DeviceType::CPU:
    return Backend::CPU;
  case DeviceType::CUDA:
    return Backend::CUDA;
  case DeviceType::HIP:
    return Backend::HIP;
  case DeviceType::MPS:
    return Backend::MPS;
  case DeviceType::XPU:
    return Backend::XPU;
  case DeviceType::Vulkan:
    return Backend::Vulkan;
  case DeviceType::Meta:
    return Backend::Meta;
  default:
    return Backend::CPU;
  }
}

} // namespace native
} // namespace asterflow
