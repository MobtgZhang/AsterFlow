# Autograd：view、reshape、stride 与 contiguous 语义

本文固定 AsterFlow 张量与 PyTorch 风格 autograd 在**视图与内存布局**上的行为，供实现与测试对齐。实现入口见 [`src/tensor.jl`](../src/tensor.jl)、[`src/layout.jl`](../src/layout.jl)、[`src/graph.jl`](../src/graph.jl)。

## 1. 存储模型

- 张量由一维 `storage`、`size`、`strides`（列主，与 Julia `Array` / BLAS 一致）、`offset`（元素偏移）描述。
- **视图**与**基张量**共享同一段 `storage`（及共享的 autograd **版本计数** `version_ref`，见下文 inplace 规范）。

## 2. view 与 reshape

- **`view_tensor`**：仅改变 `size` / `strides` / `offset`，**不复制**数据；新张量与父张量共享 `storage` 与 `version_ref`。
- **`reshape_tensor`**（带 autograd）：
  - 若当前张量**连续**（`is_contiguous`），可在同一 `storage` 上改形状，仍共享 `version_ref`。
  - 若**非连续**，实现会先 `contiguous`（拷贝到新连续缓冲），得到**新** `storage` 与**新** `version_ref`。
- **无 autograd 的 `_reshape_storage` / `_permute_storage`**：供反向内部使用；调用方需保证语义正确。

## 3. permute 与转置

- **`permute_tensor`**：交换维度，通常产生 **strided 非连续**视图，共享原 `storage` 与 `version_ref`。
- 反向：对输出梯度做逆置换写回输入，见 `PermuteBackward`。

## 4. contiguous

- **`is_contiguous`**：`strides` 等于 `column_major_strides(size)` 且 `offset == 0`。
- **`contiguous(t)`**：
  - 已连续则返回 `t`（同一对象）。
  - 否则分配新连续缓冲并拷贝；新张量使用**新** `storage` 与**新** `version_ref`（与视图链脱钩）。

## 5. 反向传播中的视图梯度

- 对共享 `storage` 的多个逻辑张量，梯度应**累加**到各自在 storage 中的位置（通过 `accumulate_grad!` 与各类 `*Backward`）。
- **reshape**：`ReshapeBackward` 将输出梯度 reshape 回前向输入形状写回。
- **permute**：`PermuteBackward` 对梯度做逆置换。
- **expand / 广播（2D）**：`ExpandBackward` 将梯度在广播维度上 **sum-reduce** 回输入形状（见 `docs/broadcasting-autograd.md` 或测试用例）。

## 6. inplace 与版本计数

- 任意可能修改 `storage` 内容的操作（如 `setindex_tensor!`）会递增共享的 `version_ref`。
- 前向构建 `grad_fn` 时**快照**各输入张量的版本；`backward` 执行对应 `apply_backward!` 时若版本已变，**报错**，避免静默错误梯度。
- 在 `grad_enabled()` 下对 `requires_grad=true` 的张量做逐元赋值仍**禁止**（见 `setindex_tensor!`）；版本递增在 `no_grad` 下修改数据时同样生效，以便检测「先前向、再 inplace、再 backward」的错误顺序。

## 7. 与后端算子的关系

- CPU/CUDA 等 kernel 应把 **非连续**输入视为合法 strided 张量，或通过 `contiguous` 再计算；具体见各算子契约（[`docs/op-contract.md`](op-contract.md)）。
