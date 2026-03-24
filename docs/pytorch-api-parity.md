# AsterFlow 与 PyTorch API 对照与语义说明

本文冻结「用户层应对齐什么、刻意不同什么」，与 [design-principles.md](design-principles.md) §9 的语义风险一致。测试用例见 `test/runtests.jl` 中对应 `@testset`。

## 一页摘要

| 概念 | PyTorch | AsterFlow | 备注 |
|------|---------|-----------|------|
| 张量布局 | stride + storage | `Tensor`：`storage` + `size` + `strides` + `offset`，列主 | 与 Julia `Array` / BLAS 一致 |
| 设备 | `torch.device` | `device("cuda:0")`、`CPUDevice()`、`cuda_device(0)` | 字符串与构造器并存 |
| 自动微分 | 动态图、`backward` | `backward`、`grad_fn` 链 | 非标量 `backward` 需传入 `gradient=` |
| 模块 | `nn.Module` | `Module`、`Sequential`、`Linear`… | 可调用 `struct` |
| 优化器 | `torch.optim` | `SGD`、`AdamW`…（`optimise/` 拼写） | `AdamW` 支持 `AdamWGroup` 参数组 |
| 权重 | `state_dict` | `state_dict`、`load_state_dict!` | `pytorch_compat` 处理 Linear 权重转置 |
| 编译 | `torch.compile` | `@trace`、`trace_graph`、`IRGraph`、stub codegen | 演进中 |

## Linear 权重布局（重要）

- **AsterFlow / 本仓库**：`Linear` 的 `weight` 形状为 `(in_features, out_features)`。
- **PyTorch `nn.Linear`**：`weight` 为 `(out_features, in_features)`。
- **迁移**：`load_state_dict!(m, d; pytorch_compat=true)` 会对二维 `weight` 做转置匹配。

## 语义风险与策略（设计冻结）

### 1. view / reshape / stride 与 backward

- `permute_tensor`、`reshape_tensor` 在 `grad_enabled()` 且输入 `requires_grad` 时会挂接 `PermuteBackward` / `ReshapeBackward`。
- 内部反向传播中的几何变换使用无跟踪路径（`no_grad` 包裹），避免图中混入辅助节点。

### 2. inplace 与 `requires_grad`

- 对 `requires_grad==true` 的张量，在 `grad_enabled()==true` 时 **`setindex_tensor!` 直接报错**（硬禁止策略）。
- 版本计数字段保留为演进点；当前不实现完整 PyTorch 式 version bump 检测。

### 3. broadcast 反传

- CPU `add`/`sub`/`mul` 等允许 NumPy 风格广播（基于 `to_array` 上 `.+` 等）。
- `AddBackward`/`SubBackward`/`MulBackward` 等对输出梯度做 `sum_to_shape` 缩回输入形状。

### 4. 混合精度（AMP）

- 见 [amp.jl 模块说明](../src/amp.jl)：`autocast`、`GradScaler` 为可组合上下文，算子白名单可扩展。

### 5. 异步与生命周期

- CUDA 上默认依赖 `CUDA.jl` 默认流；多流与显式同步见 [accelerator-streams.md](accelerator-streams.md)。

## dtype / 设备组合

首版主路径：**Float32** 在 CPU 与 CUDA（若加载扩展）上完整；Float64 多在 CPU；Float16 依赖 AMP 与后端能力。不支持组合在算子入口尽量抛出明确 `error` 信息。

## 与 Flux / Zygote

- 主 AD 路径为 **PyTorch 式 tape**，与 Zygote 源码变换不同。混用建议见 [flux-zygote-interop.md](flux-zygote-interop.md)。
