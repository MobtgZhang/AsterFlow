# AsterFlow 文档

本目录提供**设计原则**、**架构**与**构建 / 安装**说明，便于对齐实现与上手开发。

| 文档 | 内容 |
|------|------|
| [设计原则与架构](design-principles.md) | 分层、MVP 路线、Dispatcher / Autograd、与 PyTorch 2.x 的对应、语义风险、仓库模块布局 |
| [构建与安装](build-and-install.md) | Julia 包、`libasterflow` 与 `aster_native` 模块、环境变量、测试与可选加速器扩展 |
| [快速上手](tutorial-getting-started.md) | 安装、最小训练循环、设备迁移、延伸阅读 |
| [PyTorch API 对照](pytorch-api-parity.md) | API 表、Linear 权重布局、语义风险冻结 |
| [算子注册契约](op-contract.md) | `register_op!` 约定、dtype×device、已注册算子报告 API |
| [加速器与流](accelerator-streams.md) | CUDA 异步、同步点、ROCm/NPU 注意 |
| [DDP 设计备忘](ddp-design.md) | 分布式数据并行方向与占位模块 |
| [ONNX / HuggingFace 互操作](interop-onnx-hf.md) | 权重与导出工作流 |
| [Flux / Zygote 关系](flux-zygote-interop.md) | 与 Flux AD 的边界说明 |

仓库根目录另有 [README.md](../README.md)，汇总目录结构与快速开始。
