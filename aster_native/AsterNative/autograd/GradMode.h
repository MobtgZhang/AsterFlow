#pragma once

namespace asterflow {
namespace native {
namespace autograd {

/**
 * 全局梯度模式开关（线程局部）。正式接入 autograd 时，前向可在 `NoGradGuard` 下跳过图构建。
 */
class GradMode {
public:
  static bool is_enabled() noexcept { return enabled(); }

  static void set_enabled(bool on) noexcept { enabled() = on; }

private:
  static bool& enabled() noexcept {
    thread_local bool v = true;
    return v;
  }
};

class NoGradGuard {
public:
  NoGradGuard() : prev_(GradMode::is_enabled()) { GradMode::set_enabled(false); }

  ~NoGradGuard() { GradMode::set_enabled(prev_); }

  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
  bool prev_;
};

} // namespace autograd
} // namespace native
} // namespace asterflow
