# aster_native 模块

**职责**：原生张量运行时的 **C++ 骨架**（命名空间与目录 `AsterNative/`），目标与 PyTorch 侧 `TensorImpl`、设备、调度等概念对齐，供后续接入真实算子与多后端。

**内容**：

- `AsterNative/` — 源码与 CMake 子项目，静态库目标 `af_native_tensor`。
- `ATen/` — 与 ATen 概念对齐的**参考性头文件**，与当前 Julia 包无直接链接。

**构建**（在本目录下）：

```bash
cmake -S . -B build && cmake --build build
```

详见仓库根目录 [docs/build-and-install.md](../docs/build-and-install.md)。
