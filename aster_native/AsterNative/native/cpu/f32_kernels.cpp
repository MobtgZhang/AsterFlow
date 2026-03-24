#include <AsterNative/native/cpu/f32_kernels.hpp>

#include <cstdint>

namespace asterflow {
namespace native {
namespace cpu {

void add_f32(const float* a, const float* b, float* y, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    y[i] = a[i] + b[i];
  }
}

void relu_f32(const float* x, float* y, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    const float v = x[i];
    y[i] = v > 0.f ? v : 0.f;
  }
}

/** C = A * B，列主；A: m×k，B: k×n，C: m×n */
void matmul_f32_colmajor(const float* A, const float* B, float* C, std::int64_t m, std::int64_t k, std::int64_t n) {
  for (std::int64_t j = 0; j < n; ++j) {
    for (std::int64_t i = 0; i < m; ++i) {
      float s = 0.f;
      for (std::int64_t t = 0; t < k; ++t) {
        s += A[i + t * m] * B[t + j * k];
      }
      C[i + j * m] = s;
    }
  }
}

} // namespace cpu
} // namespace native
} // namespace asterflow
