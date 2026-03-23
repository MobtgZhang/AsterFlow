#pragma once

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

} // namespace native
} // namespace asterflow
