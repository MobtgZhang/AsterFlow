# API 参考（摘要）

完整自动文档可在后续接入 [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) 后由 docstring 生成；此处列出**公开入口**与常用模式。

## 张量

| 符号 | 说明 |
|------|------|
| `tensor(array; device=CPUDevice(), requires_grad=false)` | 从 `Array` 构造张量 |
| `to_array(t)` | 物化为 CPU `Array` |
| `device(t)`, `size(t)`, `numel(t)`, `is_contiguous(t)`, `contiguous(t)` | 元数据与布局 |
| `reshape_tensor`, `permute_tensor`, `view_tensor`, `expand_tensor` | 视图 / 变形（见 [`autograd-view-semantics.md`](autograd-view-semantics.md)） |
| `tensor_version(t)` | autograd 用存储版本号（共享 storage 的视图相同） |
| `detach_tensor`, `requires_grad!`, `zero_grad!`, `grad(t)` | 图与梯度缓冲 |
| `setindex_tensor!` | inplace 逐元写入（`grad` 开启且 `requires_grad` 时禁止） |

## 算子与调度

| 符号 | 说明 |
|------|------|
| `add`, `sub`, `mul`, `matmul`, `relu_tensor`, `softmax_rows`, … | 见 `src/ops.jl` |
| `register_op!(op, backend, fn; dtype=nothing)` | `dtype` 指定 `eltype` 专用实现，`nothing` 为通配 |
| `register_dispatch_fallback!(:cuda, [:cpu])` | 已在 `__init__` 注册；主后端无内核时尝试 fallback |
| `dispatch_op`, `registered_ops_report` | 调度与 introspection |

## 自动微分

| 符号 | 说明 |
|------|------|
| `backward(loss; retain_graph=false, gradient=nothing)` | 非标量须提供 `gradient=` |
| `no_grad() do ... end` | 禁用梯度记录 |
| `grad_enabled()` | 当前是否记录图 |

## 模块与优化器

| 符号 | 说明 |
|------|------|
| `Linear`, `Sequential`, `ReLU`, `LayerNorm`, `TransformerBlock`, … | `src/layers/`、`src/transformer/` |
| `params(m)`, `train!`, `evalmode!` | 参数与模式 |
| `SGD`, `AdamW`, `Adam`, …、`step!(opt)` | `src/optimise/` |

## 设备

| 符号 | 说明 |
|------|------|
| `CPUDevice()`, `cuda_device(i)`, `AcceleratorDevice(:rocm, i)`, … | `src/devices.jl` |
| `to_device(t, dev)`, `module_to_device!(m, dev)` | 迁移 |

## 编译（占位）

| 符号 | 说明 |
|------|------|
| `@trace`, `IRGraph`, `graph_to_json`, `codegen_stub_cache` | 见 `src/compile/` |

## 扩展包

- `AsterFlowCUDAExt`、`AsterFlowROCMExt`：在加载 `CUDA` / `AMDGPU` 时注册加速器算子（见 `Project.toml` `[extensions]`）。
