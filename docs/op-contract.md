# 算子注册契约与内核清单

## `register_op!(op::Symbol, backend::Symbol, fn)`

- **registry 键**：`(op, backend)`，其中 `backend` 来自 `device_backend(dev)`（如 `:cpu`、`:cuda`）。
- **调用方**：`dispatch_op(op, dev, args...; kwargs...)`（见 `src/dispatch.jl`），可选 **fallback 链**（见 `DISPATCH_FALLBACK_CHAIN`）。
- **内核 `fn` 约定**：
  - 返回新的 `Tensor`（或约定好的输出），**不**修改输入 storage（除非文档明确为 inplace 内核）。
  - 输入张量须与 `dev` 一致；**不**自动跨设备搬运（与 PyTorch 一致：设备不匹配应报错）。
  - `kwargs` 仅传递注册时约定的键（如 `dims=` 用于 `sum`/`mean`）。
- **与 autograd 边界**：`src/ops.jl` 在 `grad_enabled()` 下为输出张量设置 `requires_grad` 与 `grad_fn`；内核本身**不**应依赖 autograd 状态。

## 执行模式键（Dispatch execution mode）

- `ASTERFLOW_EXECUTION_MODE`：`Ref{Symbol}`，取 `:eager`（默认）或 `:debug`（可挂日志/断言）。
- 预留 `:compiled`：与 `compile/` IR 路径对接，当前不修改算子语义。

## dtype × device 支持矩阵（摘要）

| eltype | CPU | CUDA 扩展 |
|--------|-----|-----------|
| Float32 | 全路径主推荐 | 主推荐（CuArray 逐元 / matmul） |
| Float64 | 支持 | 依赖 CUDA 与注册实现 |
| Float16 / BFloat16 | 部分 / AMP | 渐进支持，配合 `autocast` |

未覆盖组合应在实现处 `error("...")`，避免静默错误结果。

## 已注册算子清单（代码真相源）

运行时调用 `AsterFlow.registered_ops_report()` 返回按后端分组的 `op` 列表（见 `src/dispatch.jl`）。
