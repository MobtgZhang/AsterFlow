#pragma once

/**
 * 算子分发占位：按 dtype / device 选择内核；实现阶段可在此接入注册表或代码生成。
 */
namespace asterflow {
namespace native {
namespace dispatch {

// 预留：例如 REGISTER_KERNEL(op, device, dtype, fn)

} // namespace dispatch
} // namespace native
} // namespace asterflow
