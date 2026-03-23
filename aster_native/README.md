# aster_native 模块

**职责**：原生张量运行时的 **C++ 骨架**（命名空间 `asterflow::native`，目录 `AsterNative/`），在分层与概念上参考常见 eager 框架的 C++ 侧组织方式（张量实现、设备、调度键、存储与可选算子树），供后续接入真实算子与多后端。

**目录结构（概要）**：

| 路径 | 作用 |
|------|------|
| `AsterNative/core/` | `TensorImpl`、`Storage`、`ScalarType`、`Layout`、`TensorOptions`、`DispatchKey` |
| `AsterNative/impl/` | 实现细节辅助（如 `DeviceGuard`） |
| `AsterNative/native/` | 逐算子实现占位（可再分子目录） |
| `AsterNative/accelerator/` | 设备可用性探测注册 |
| `AsterNative/ops/` | 算子注册表占位 |
| `AsterNative/cpu/`、`cuda/` 等 | 各后端占位头文件 |

**构建**（在本目录下）：

```bash
cmake -S . -B build && cmake --build build
```

产物为静态库 `af_native_tensor`（路径形如 `build/AsterNative/libaf_native_tensor.a`，以生成树为准）。

详见仓库根目录 [docs/build-and-install.md](../docs/build-and-install.md)。
