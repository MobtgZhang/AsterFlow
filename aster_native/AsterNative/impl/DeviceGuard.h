#pragma once

#include <AsterNative/Context.h>

namespace asterflow {
namespace native {
namespace impl {

/**
 * 在作用域内临时切换进程默认设备，退出时恢复（测试与多段初始化常用）。
 */
class DeviceGuard {
public:
  explicit DeviceGuard(Device next) : previous_(Context::global().default_device()) {
    Context::global().set_default_device(next);
  }

  ~DeviceGuard() { Context::global().set_default_device(previous_); }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
  Device previous_;
};

} // namespace impl
} // namespace native
} // namespace asterflow
