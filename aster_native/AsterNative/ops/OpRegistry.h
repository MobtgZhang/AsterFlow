#pragma once

#include <AsterNative/core/DispatchKey.h>
#include <mutex>
#include <string>
#include <unordered_map>

namespace asterflow {
namespace native {
namespace ops {

/**
 * 算子注册表：按字符串 id 绑定内核指针（Phase1 为 CPU f32；后续可扩展 (id, DispatchKey)）。
 */
class OpRegistry {
public:
  static OpRegistry& instance();

  void register_kernel(const std::string& id, void* fn) {
    std::lock_guard<std::mutex> lock(mu_);
    table_[id] = fn;
  }

  void* get_kernel(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(id);
    return it == table_.end() ? nullptr : it->second;
  }

private:
  OpRegistry() = default;

  mutable std::mutex mu_;
  std::unordered_map<std::string, void*> table_{};
};

inline OpRegistry& OpRegistry::instance() {
  static OpRegistry r;
  return r;
}

} // namespace ops
} // namespace native
} // namespace asterflow
