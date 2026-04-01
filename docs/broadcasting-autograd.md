# Broadcasting 与反向传播规范（AsterFlow）

## 前向

- 二元运算（如 `add`）在形状可广播时，前向结果形状为逐维 `max` 对齐后的形状（与 NumPy / PyTorch 一致）。
- 当前实现中 **2D 显式广播** 可通过 `expand_tensor` / `broadcast_to_tensor`（stride=0 的 expand 视图）表达。

## 反向

- 若输入 `a` 形状为 `target_sz`，输出梯度 `grad_out` 形状为广播后形状，则输入 `a` 的梯度为：将 `grad_out` 在 `a` 的尺寸为 1 的维度上 **求和规约** 到 `size(a)`。
- 实现入口：`sum_grad_to_shape`（[`src/ops_helpers.jl`](../src/ops_helpers.jl)）、`ExpandBackward`（[`src/graph.jl`](../src/graph.jl)）。

## 示例

- `x: (4,1)` 与 `y: (1,3)` 相加得 `z: (4,3)`。若 `∂L/∂z` 全 1，则  
  - `∂L/∂x` 应对列求和，得到 `(4,1)`，每行之和为 3。  
  - `∂L/∂y` 应对行求和，得到 `(1,3)`，每列之和为 4。

## 测试

- 见 `test/runtests.jl` 中「广播加法反传」及「非连续 / broadcasting」相关 `@testset`。
