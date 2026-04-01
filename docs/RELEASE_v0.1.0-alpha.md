# Release v0.1.0-alpha（清单）

> 建议在 GitHub 上创建 **Pre-release** 时使用本文案要点。

## 已具备（摘要）

- Julia 包：`Tensor`、eager 算子、`backward`、常用 `layers` / `optimise` / `Transformer` 子集。
- Dispatcher：按设备 fallback（如 `:cuda` → `:cpu`）、按 `eltype` 的 dtype 注册键。
- Autograd：视图语义文档、inplace 版本检测、broadcast / reshape / permute 反传测试。
- 可选：`libasterflow`、`aster_native`、CUDA/ROCm 扩展。
- CI：Julia 多版本测试、`aster_native` CMake/ctest。

## 已知限制

- 编译路径以 IR/桩为主，非生产级 codegen。
- DDP、FlashAttention、完整 AMP 为路线图项（见 [`roadmap-performance.md`](roadmap-performance.md)）。
- 部分 dtype×设备组合会 `error` 而非静默降级（见 [`op-contract.md`](op-contract.md)）。

## 安装与验证

```julia
using Pkg
Pkg.add(url="https://github.com/MobtgZhang/AsterFlow", rev="main")  # 发版后替换为版本号
using Pkg; Pkg.test("AsterFlow")
```

本地开发：`Pkg.develop(path="...")`，见 [`build-and-install.md`](build-and-install.md)。
