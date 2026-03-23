# AsterFlow 原生张量运行时（目录布局参考常见张量库的分层方式：core / native / 各设备后端）。
# 当前为骨架与占位实现；后续在此接入真实算子与 CUDA/HIP 等 runtime。
cmake_minimum_required(VERSION 3.18)
project(asterflow_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(AsterNative)
