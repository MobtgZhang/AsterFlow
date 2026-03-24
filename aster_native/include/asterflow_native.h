#ifndef ASTERFLOW_NATIVE_H
#define ASTERFLOW_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** ABI 版本，与实现递增保持同步。 */
#define AF_NATIVE_ABI_VERSION 1

/**
 * 列主（column-major）布局，与 Julia / libasterflow 一致：
 * m×k 矩阵 A 中元素 (i, t) 位于 A[i + t * m]。
 */
int32_t af_native_version(void);

void af_native_add_f32(const float* a, const float* b, float* y, int64_t n);

void af_native_relu_f32(const float* x, float* y, int64_t n);

void af_native_matmul_f32_colmajor(
    const float* A,
    const float* B,
    float* C,
    int64_t m,
    int64_t k,
    int64_t n);

#ifdef __cplusplus
}
#endif

#endif
