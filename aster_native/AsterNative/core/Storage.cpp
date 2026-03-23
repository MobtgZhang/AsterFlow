#include <AsterNative/core/Storage.h>
#include <AsterNative/detail/Exception.h>

namespace asterflow {
namespace native {

Storage::Storage(std::size_t num_elements, ScalarType dtype)
    : numel_(num_elements), dtype_(dtype) {
  const std::size_t esz = scalar_type_byte_size(dtype_);
  detail::af_check(esz > 0 || numel_ == 0, "Storage: unsupported scalar type");
  const std::size_t n = numel_ * esz;
  bytes_.resize(n);
}

std::size_t Storage::nbytes() const noexcept {
  return bytes_.size();
}

std::shared_ptr<Storage> Storage::allocate(std::size_t num_elements, ScalarType dtype) {
  return std::make_shared<Storage>(num_elements, dtype);
}

} // namespace native
} // namespace asterflow
