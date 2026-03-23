#pragma once

#include <AsterNative/core/ScalarType.h>
#include <AsterNative/Device.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace asterflow {
namespace native {

/**
 * 张量实现占位：形状、标量类型、设备；storage / strides 等待与分配器对接后补齐。
 */
class TensorImpl {
public:
  TensorImpl() = default;
  explicit TensorImpl(std::vector<std::int64_t> shape, ScalarType dtype, Device dev);

  const std::vector<std::int64_t>& sizes() const noexcept { return sizes_; }
  ScalarType scalar_type() const noexcept { return dtype_; }
  Device device() const noexcept { return device_; }
  std::size_t numel() const noexcept;

private:
  std::vector<std::int64_t> sizes_{};
  ScalarType dtype_{ScalarType::Float};
  Device device_{};
};

} // namespace native
} // namespace asterflow
