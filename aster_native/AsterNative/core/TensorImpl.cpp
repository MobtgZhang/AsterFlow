#include <AsterNative/core/TensorImpl.h>
#include <numeric>

namespace asterflow {
namespace native {

TensorImpl::TensorImpl(std::vector<std::int64_t> shape, ScalarType dtype, Device dev)
    : sizes_(std::move(shape)), dtype_(dtype), device_(dev) {}

std::size_t TensorImpl::numel() const noexcept {
  if (sizes_.empty()) {
    return 0;
  }
  return static_cast<std::size_t>(std::accumulate(
      sizes_.begin(), sizes_.end(), std::int64_t{1}, std::multiplies<>{}));
}

} // namespace native
} // namespace asterflow
