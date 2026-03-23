#pragma once

namespace asterflow {
namespace native {

/** 张量内存布局（当前仅支持稠密步长布局；稀疏等可在实现期扩展）。 */
enum class Layout : std::uint8_t {
  Strided = 0,
  SparseCSR, // 占位：尚无存储格式实现
};

} // namespace native
} // namespace asterflow
