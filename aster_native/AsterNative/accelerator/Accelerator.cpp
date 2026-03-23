#include <AsterNative/accelerator/Accelerator.h>
#include <unordered_map>

namespace asterflow {
namespace native {
namespace accelerator {

namespace {

std::unordered_map<DeviceType, DeviceProbe>& probes() {
  static std::unordered_map<DeviceType, DeviceProbe> m;
  return m;
}

} // namespace

void register_device_probe(DeviceType t, DeviceProbe probe) {
  probes()[t] = std::move(probe);
}

bool is_available(Device dev) {
  if (dev.is_cpu()) {
    return true;
  }
  auto it = probes().find(static_cast<DeviceType>(dev.type));
  if (it == probes().end() || !it->second) {
    return false;
  }
  return it->second(dev);
}

} // namespace accelerator
} // namespace native
} // namespace asterflow
