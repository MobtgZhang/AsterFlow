#include <catch2/catch_test_macros.hpp>

#include <asterflow_native.h>

#include <cmath>
#include <cstring>

TEST_CASE("af_native_version") {
  REQUIRE(af_native_version() >= 1);
}

TEST_CASE("af_native_add_f32") {
  float a[] = {1.f, 2.f, 3.f};
  float b[] = {4.f, 5.f, 6.f};
  float y[3];
  af_native_add_f32(a, b, y, 3);
  REQUIRE(y[0] == 5.f);
  REQUIRE(y[1] == 7.f);
  REQUIRE(y[2] == 9.f);
}

TEST_CASE("af_native_relu_f32") {
  float x[] = {-1.f, 0.f, 2.f};
  float y[3];
  af_native_relu_f32(x, y, 3);
  REQUIRE(y[0] == 0.f);
  REQUIRE(y[1] == 0.f);
  REQUIRE(y[2] == 2.f);
}

TEST_CASE("af_native_matmul_f32_colmajor") {
  /* A 2x2 identity, B arbitrary, C = B */
  float A[] = {1.f, 0.f, 0.f, 1.f};
  float B[] = {1.f, 2.f, 3.f, 4.f};
  float C[4];
  std::memset(C, 0, sizeof(C));
  af_native_matmul_f32_colmajor(A, B, C, 2, 2, 2);
  for (int i = 0; i < 4; ++i) {
    REQUIRE(std::abs(C[i] - B[i]) < 1e-5f);
  }
}
