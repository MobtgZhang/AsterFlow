#pragma once

#include <AsterNative/Device.h>
#include <AsterNative/core/Layout.h>
#include <AsterNative/core/ScalarType.h>

namespace asterflow {
namespace native {

/**
 * 构造张量时的公共选项（dtype、设备、布局、是否参与自动微分等）。
 * 与「一次性把元数据打成一个结构体」的常见做法一致，减少构造函数组合爆炸。
 */
struct TensorOptions {
  ScalarType dtype{ScalarType::Float};
  Device placement{DeviceType::CPU, 0};
  Layout mem_layout{Layout::Strided};
  bool requires_grad{false};

  TensorOptions& scalar_type(ScalarType t) noexcept {
    dtype = t;
    return *this;
  }

  TensorOptions& device(Device d) noexcept {
    placement = d;
    return *this;
  }

  TensorOptions& layout(Layout l) noexcept {
    mem_layout = l;
    return *this;
  }

  TensorOptions& requires_gradient(bool v = true) noexcept {
    requires_grad = v;
    return *this;
  }
};

} // namespace native
} // namespace asterflow
