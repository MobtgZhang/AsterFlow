#pragma once

namespace asterflow {
namespace native {
namespace cuda {

/**
 * CUDA 设备上下文占位（.cu / driver API 接入前仅占位）。
 * 编译未启用 CUDA 时仍可提供类型与空实现头，便于上层统一 include。
 */
struct CudaDeviceState {
  int device_id{0};
};

} // namespace cuda
} // namespace native
} // namespace asterflow
