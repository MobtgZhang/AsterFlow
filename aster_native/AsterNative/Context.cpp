#include <AsterNative/Context.h>

namespace asterflow {
namespace native {

Context& Context::global() {
  static Context inst;
  return inst;
}

Device Context::default_device() const { return default_device_; }

void Context::set_default_device(Device d) { default_device_ = d; }

} // namespace native
} // namespace asterflow
