#pragma once

namespace asterflow {
namespace native {
namespace cpu {

/** 占位：日后用于线程亲和 / 并行区域等与 CPU 执行相关的 RAII。 */
struct CPUGuard {
  CPUGuard() = default;
};

} // namespace cpu
} // namespace native
} // namespace asterflow
