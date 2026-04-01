# CUDA / 加速器测试说明

## CI（GitHub Actions）

默认 runner **无 GPU**，因此 CI 仅执行 **CPU** 上的 `Pkg.test()`。加速器相关逻辑在测试中通过 `try using CUDA` + `isavailable(dev)` **有则测、无则跳过**（见 `test/runtests.jl` 中 `accelerator device` 用例）。

## 本地端到端（E2E）

在装有驱动与 `CUDA.jl` 的机器上：

```julia
using Pkg
Pkg.test("AsterFlow")
```

`accelerator device` 用例会在可用时覆盖 **设备张量 `add`、小 MLP 的 forward/backward、优化器一步** 等路径，作为轻量 E2E 回归。

## 与 CPU 数值对齐（可选手测）

对同一随机种子与网络，可将 `logits`、`loss` 在 CPU 与 CUDA 上各跑一遍，比较 `to_array` 结果容差（本仓库未在 CI 中强制双跑，以免无 GPU 环境失败）。
