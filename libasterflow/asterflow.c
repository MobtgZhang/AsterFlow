#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* 列主序 (column-major): A[m x k], B[k x n], C[m x n] */
void af_matmul_f32_colmajor(
    const float* A,
    const float* B,
    float* C,
    int64_t m,
    int64_t k,
    int64_t n
) {
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < m; i++) {
            float s = 0.f;
            for (int64_t t = 0; t < k; t++) {
                float a_it = A[i + t * m];
                float b_tj = B[t + j * k];
                s += a_it * b_tj;
            }
            C[i + j * m] = s;
        }
    }
}

void* af_alloc_bytes(size_t nbytes) {
    return malloc(nbytes);
}

void af_free(void* p) {
    if (p) {
        free(p);
    }
}

int32_t af_version(void) {
    return 1;
}

void af_relu_f32(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v > 0.f ? v : 0.f;
    }
}

void af_add_f32(const float* a, const float* b, float* y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = a[i] + b[i];
    }
}
