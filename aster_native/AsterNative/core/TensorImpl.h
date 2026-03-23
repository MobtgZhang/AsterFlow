#pragma once

#include <AsterNative/Device.h>
#include <AsterNative/core/Layout.h>
#include <AsterNative/core/ScalarType.h>
#include <AsterNative/core/Storage.h>
#include <AsterNative/core/TensorOptions.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace asterflow {
namespace native {

class TensorImpl {
public:
  TensorImpl() = default;
  explicit TensorImpl(std::vector<std::int64_t> shape, ScalarType dtype, Device dev);
  explicit TensorImpl(std::vector<std::int64_t> shape, const TensorOptions& opts);

  const std::vector<std::int64_t>& sizes() const noexcept { return sizes_; }
  const std::vector<std::int64_t>& strides() const noexcept { return strides_; }
  std::int64_t storage_offset() const noexcept { return storage_offset_; }

  ScalarType scalar_type() const noexcept { return dtype_; }
  Device device() const noexcept { return device_; }
  Layout layout() const noexcept { return layout_; }
  bool requires_grad() const noexcept { return requires_grad_; }

  std::size_t numel() const noexcept;
  bool is_meta() const noexcept;
  bool is_contiguous() const noexcept;

  const Storage* storage() const noexcept { return storage_.get(); }
  std::shared_ptr<Storage> storage_ptr() const noexcept { return storage_; }

private:
  static std::vector<std::int64_t> make_contiguous_strides(const std::vector<std::int64_t>& sizes);

  std::vector<std::int64_t> sizes_{};
  std::vector<std::int64_t> strides_{};
  std::int64_t storage_offset_{0};
  ScalarType dtype_{ScalarType::Float};
  Device device_{};
  Layout layout_{Layout::Strided};
  bool requires_grad_{false};
  std::shared_ptr<Storage> storage_{};
};

} // namespace native
} // namespace asterflow
