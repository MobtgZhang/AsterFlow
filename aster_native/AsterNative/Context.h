#pragma once

#include <AsterNative/Device.h>

namespace asterflow {
namespace native {

/** 进程级运行时上下文（默认设备、日后可扩展线程池与第三方库句柄等）。 */
class Context {
public:
  static Context& global();

  Device default_device() const;
  void set_default_device(Device d);

private:
  Context() = default;
  Device default_device_{DeviceType::CPU, 0};
};

} // namespace native
} // namespace asterflow
