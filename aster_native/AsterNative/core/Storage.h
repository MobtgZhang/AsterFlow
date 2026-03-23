#pragma once

#include <AsterNative/core/ScalarType.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace asterflow {
namespace native {

/**
 * 与形状解耦的一维字节缓冲区，供多个张量视图共享（shared ownership）。
 * 等价于「一块连续 device 内存」的宿主侧占位模型；真实设备分配器接入后可替换内部表示。
 */
class Storage {
public:
  Storage() = default;
  Storage(std::size_t num_elements, ScalarType dtype);

  ScalarType scalar_type() const noexcept { return dtype_; }
  std::size_t numel() const noexcept { return numel_; }
  std::size_t nbytes() const noexcept;

  const void* data() const noexcept { return bytes_.empty() ? nullptr : bytes_.data(); }
  void* mutable_data() noexcept { return bytes_.empty() ? nullptr : bytes_.data(); }

  static std::shared_ptr<Storage> allocate(std::size_t num_elements, ScalarType dtype);

private:
  std::size_t numel_{0};
  ScalarType dtype_{ScalarType::Undefined};
  std::vector<std::byte> bytes_{};
};

} // namespace native
} // namespace asterflow
