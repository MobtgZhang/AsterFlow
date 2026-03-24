# AsterFlow 原生张量运行时（目录布局参考常见张量库的分层方式：core / native / 各设备后端）。
cmake_minimum_required(VERSION 3.18)
project(asterflow_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ASTERFLOW_NATIVE_BUILD_TESTS "Build C++ unit tests (Catch2)" ON)
option(ASTERFLOW_NATIVE_ENABLE_CUDA "Compile CUDA kernels (not implemented)" OFF)

if(ASTERFLOW_NATIVE_ENABLE_CUDA)
  message(WARNING "ASTERFLOW_NATIVE_ENABLE_CUDA: no CUDA sources wired yet; option reserved.")
endif()

add_subdirectory(AsterNative)

if(ASTERFLOW_NATIVE_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
  )
  FetchContent_MakeAvailable(Catch2)
  add_executable(af_native_tests tests/test_native_kernels.cpp)
  target_link_libraries(af_native_tests PRIVATE Catch2::Catch2WithMain asterflow_native)
  target_include_directories(af_native_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  add_test(NAME af_native_tests COMMAND af_native_tests)
endif()
