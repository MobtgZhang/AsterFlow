# aster_native 模块

**职责**：AsterFlow 的 **C++ 原生张量运行时**（命名空间 `asterflow::native`），与 Julia 包通过 **可选共享库 `libasterflow_native`**（`ASTERFLOW_NATIVE_LIB`）对接；默认不构建也不影响 `using AsterFlow`。

## Phase 1（当前目标）

- **独立可测**：CMake 产出 `libasterflow_native.so`（Linux）及可选 C++ 单测，不依赖 Julia。
- **列主（column-major）**：与 Julia `Array`、`libasterflow` 的 `af_matmul_f32_colmajor` 一致——矩阵元素 `(i, j)`（0-based）在 `m×n` 列主存储中下标为 `i + j * m`。
- **CPU Float32 内核**：`add`（逐元素）、`relu`、`matmul`（朴素三重循环；后续可接 OpenBLAS）。
- **Julia 开关**：设置 `ASTERFLOW_USE_NATIVE_CPP=1` 且成功加载 `libasterflow_native` 时，[src/native.jl](../src/native.jl) 中部分算子可走 C++ 路径（见 [docs/build-and-install.md](../docs/build-and-install.md)）。

## 与 libasterflow 的关系

| 组件 | 语言 | 用途 |
|------|------|------|
| `libasterflow` | C | 轻量 `af_*`，已由 Julia 广泛使用 |
| `libasterflow_native` | C++17 | `TensorImpl` / `Storage` 骨架 + 可演进算子注册表 + C ABI |

长期可将两库在 ABI 层合并；当前 **双库并存**，通过不同环境变量分别定位。

## 目录结构

| 路径 | 作用 |
|------|------|
| `include/asterflow_native.h` | 稳定 C ABI（`extern "C"` 声明） |
| `AsterNative/core/` | `TensorImpl`、`Storage`、`ScalarType`、`Layout`、`DispatchKey` |
| `AsterNative/native/cpu/` | CPU Float32 内核实现 |
| `AsterNative/c_api/` | C ABI 导出实现 |
| `AsterNative/ops/` | `OpRegistry`（内核注册表） |
| `AsterNative/accelerator/` | 设备探测占位 |
| `tests/` | Catch2 单测 |

## 构建

```bash
cd aster_native
cmake -S . -B build -DASTERFLOW_NATIVE_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

共享库路径示例：`build/AsterNative/libasterflow_native.so`（具体以生成树为准）。

静态库 `libaf_native_tensor.a` 仍可用于仅链接 C++ 的目标。

## 路线图（Phase 2+）

- **Half / BFloat16**：在 `ScalarType` 已有枚举，内核与 C ABI 再扩展。
- **CMake `ENABLE_CUDA=ON`**：仅当显式开启时编译 `cuda/` 下设备代码；默认 **OFF**，适配无 GPU 或极小显存环境。
- **C++ autograd**：与 `autograd/GradMode.h` 对齐的记录式反传，仅在需要「全 C++ 训练」时推进；Julia 主路径仍以 tape AD 为主。
