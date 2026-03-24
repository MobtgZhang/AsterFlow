#pragma once

#include <cstdint>

namespace asterflow {
namespace native {
namespace cpu {

void add_f32(const float* a, const float* b, float* y, std::int64_t n);

void relu_f32(const float* x, float* y, std::int64_t n);

void matmul_f32_colmajor(const float* A, const float* B, float* C, std::int64_t m, std::int64_t k, std::int64_t n);

} // namespace cpu
} // namespace native
} // namespace asterflow
