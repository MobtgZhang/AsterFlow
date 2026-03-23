#include <AsterNative/core/TensorImpl.h>
#include <AsterNative/detail/Exception.h>
#include <functional>
#include <numeric>

namespace asterflow {
namespace native {

std::vector<std::int64_t> TensorImpl::make_contiguous_strides(const std::vector<std::int64_t>& sizes) {
  std::vector<std::int64_t> st(sizes.size());
  std::int64_t s = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    const auto ui = static_cast<std::size_t>(i);
    st[ui] = s;
    s *= sizes[ui];
  }
  return st;
}

TensorImpl::TensorImpl(std::vector<std::int64_t> shape, ScalarType dtype, Device dev)
    : sizes_(std::move(shape))
    , strides_(make_contiguous_strides(sizes_))
    , storage_offset_(0)
    , dtype_(dtype)
    , device_(dev)
    , layout_(Layout::Strided)
    , requires_grad_(false) {
  const std::size_t n = numel();
  if (!is_meta() && n > 0) {
    detail::af_check(scalar_type_byte_size(dtype_) > 0, "TensorImpl: unsupported scalar type for storage");
  }
  if (layout_ == Layout::Strided && !is_meta() && n > 0) {
    storage_ = Storage::allocate(n, dtype_);
  }
}

TensorImpl::TensorImpl(std::vector<std::int64_t> shape, const TensorOptions& opts)
    : sizes_(std::move(shape))
    , strides_(make_contiguous_strides(sizes_))
    , storage_offset_(0)
    , dtype_(opts.dtype)
    , device_(opts.placement)
    , layout_(opts.mem_layout)
    , requires_grad_(opts.requires_grad) {
  const std::size_t n = numel();
  if (!is_meta() && n > 0) {
    detail::af_check(scalar_type_byte_size(dtype_) > 0, "TensorImpl: unsupported scalar type for storage");
  }
  if (layout_ == Layout::Strided && !is_meta() && n > 0) {
    storage_ = Storage::allocate(n, dtype_);
  }
}

std::size_t TensorImpl::numel() const noexcept {
  if (sizes_.empty()) {
    return 0;
  }
  return static_cast<std::size_t>(std::accumulate(
      sizes_.begin(), sizes_.end(), std::int64_t{1}, std::multiplies<>{}));
}

bool TensorImpl::is_meta() const noexcept { return device_.type == DeviceType::Meta; }

bool TensorImpl::is_contiguous() const noexcept {
  if (layout_ != Layout::Strided) {
    return false;
  }
  if (storage_offset_ != 0) {
    return false;
  }
  return strides_ == make_contiguous_strides(sizes_);
}

} // namespace native
} // namespace asterflow
