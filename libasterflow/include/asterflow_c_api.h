#ifndef ASTERFLOW_C_API_H
#define ASTERFLOW_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t af_version(void);
void* af_alloc_bytes(size_t nbytes);
void af_free(void* p);

void af_matmul_f32_colmajor(
    const float* A,
    const float* B,
    float* C,
    int64_t m,
    int64_t k,
    int64_t n
);

void af_relu_f32(const float* x, float* y, int64_t n);

void af_add_f32(const float* a, const float* b, float* y, int64_t n);

#ifdef __cplusplus
}
#endif

#endif
