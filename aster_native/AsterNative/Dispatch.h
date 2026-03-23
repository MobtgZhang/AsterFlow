#pragma once

#include <AsterNative/Device.h>
#include <AsterNative/core/DispatchKey.h>

namespace asterflow {
namespace native {
namespace dispatch {

/** 由 `DeviceType` 得到用于查表的主调度键（与 `device_type_to_backend` 结果一致）。 */
constexpr DispatchKey key_for_device(DeviceType t) noexcept {
  const auto o = static_cast<std::uint8_t>(t);
  if (o >= static_cast<std::uint8_t>(DispatchKey::Count)) {
    return DispatchKey::CPU;
  }
  return static_cast<DispatchKey>(o);
}

/**
 * 占位：正式实现时可在此集中定义
 * - 算子名 → 多后端内核表的解析顺序；
 * - 或代码生成产生的注册入口。
 * 宏非必需；优先使用 constexpr 与小型模板以减少与特定工具链的耦合。
 */
template <typename KernelTable>
struct DispatchStub {
  // 实现期：KernelTable 提供 lookup(op, DispatchKey) 等接口。
};

} // namespace dispatch
} // namespace native
} // namespace asterflow
