#pragma once

#include <string>

namespace asterflow {
namespace native {
namespace ops {

using KernelId = std::string;

/**
 * 算子注册表占位：按 (算子名, 后端) 绑定可调用对象；具体签名在实现期细化。
 */
class OpRegistry {
public:
  static OpRegistry& instance();

  // void register_fn(KernelId id, Backend b, ...);

private:
  OpRegistry() = default;
};

inline OpRegistry& OpRegistry::instance() {
  static OpRegistry r;
  return r;
}

} // namespace ops
} // namespace native
} // namespace asterflow
