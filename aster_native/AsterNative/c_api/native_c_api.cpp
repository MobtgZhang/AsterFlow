#include <asterflow_native.h>

#include <AsterNative/native/cpu/f32_kernels.hpp>
#include <AsterNative/ops/OpRegistry.h>

#include <mutex>

namespace {

std::once_flag g_register_once;

void register_builtin_kernels() {
  auto& reg = asterflow::native::ops::OpRegistry::instance();
  reg.register_kernel("add_f32", reinterpret_cast<void*>(asterflow::native::cpu::add_f32));
  reg.register_kernel("relu_f32", reinterpret_cast<void*>(asterflow::native::cpu::relu_f32));
  reg.register_kernel("matmul_f32_colmajor", reinterpret_cast<void*>(asterflow::native::cpu::matmul_f32_colmajor));
}

} // namespace

extern "C" {

int32_t af_native_version(void) {
  std::call_once(g_register_once, register_builtin_kernels);
  return static_cast<int32_t>(AF_NATIVE_ABI_VERSION);
}

void af_native_add_f32(const float* a, const float* b, float* y, int64_t n) {
  std::call_once(g_register_once, register_builtin_kernels);
  asterflow::native::cpu::add_f32(a, b, y, n);
}

void af_native_relu_f32(const float* x, float* y, int64_t n) {
  std::call_once(g_register_once, register_builtin_kernels);
  asterflow::native::cpu::relu_f32(x, y, n);
}

void af_native_matmul_f32_colmajor(
    const float* A,
    const float* B,
    float* C,
    int64_t m,
    int64_t k,
    int64_t n) {
  std::call_once(g_register_once, register_builtin_kernels);
  asterflow::native::cpu::matmul_f32_colmajor(A, B, C, m, k, n);
}

} // extern "C"
