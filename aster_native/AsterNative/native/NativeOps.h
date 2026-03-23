#pragma once

/**
 * 原生算子声明区占位（逐算子 .cpp / 融合 kernel 可挂在此树之下）。
 * 建议分组：pointwise、reduction、blas、nn 等，按目录拆分。
 */
namespace asterflow {
namespace native {

// 例如：void add_f32(...);  — 实现阶段再落地

} // namespace native
} // namespace asterflow
