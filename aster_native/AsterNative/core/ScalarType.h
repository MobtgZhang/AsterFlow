#pragma once

#include <cstddef>
#include <cstdint>

namespace asterflow {
namespace native {

enum class ScalarType : std::uint8_t {
  Undefined = 0,
  Float,
  Double,
  Half,
  BFloat16,
  Int,
  Long,
  Bool,
  Count,
};

/** 稠密存储中单个元素的字节数；未知或未实现类型返回 0。 */
constexpr std::size_t scalar_type_byte_size(ScalarType t) noexcept {
  switch (t) {
  case ScalarType::Bool:
    return sizeof(bool);
  case ScalarType::Half:
  case ScalarType::BFloat16:
    return 2;
  case ScalarType::Float:
    return sizeof(float);
  case ScalarType::Double:
    return sizeof(double);
  case ScalarType::Int:
    return sizeof(int);
  case ScalarType::Long:
    return sizeof(std::int64_t);
  default:
    return 0;
  }
}

} // namespace native
} // namespace asterflow
