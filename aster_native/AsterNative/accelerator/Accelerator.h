#pragma once

#include <AsterNative/Device.h>
#include <functional>

namespace asterflow {
namespace native {
namespace accelerator {

using DeviceProbe = std::function<bool(Device)>;

void register_device_probe(DeviceType t, DeviceProbe probe);
bool is_available(Device dev);

} // namespace accelerator
} // namespace native
} // namespace asterflow
