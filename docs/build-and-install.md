# 构建与安装

本文说明如何安装 **Julia 包 AsterFlow**、如何构建 **`libasterflow` 模块**（C 共享库），以及如何构建 **`aster_native` 模块**（C++ 骨架）。**当前默认开发路径以 Julia 为主**；远期 Python + CUDA 完整栈可在设计文档中单独规划。

---

## 1. 依赖概览

| 组件 | 要求 |
|------|------|
| **Julia** | 1.9+（见仓库根 `Project.toml` 中 `[compat] julia`） |
| **标准库** | `LinearAlgebra`、`Libdl`、`TOML`（已在 `Project.toml` 声明） |
| **可选：CUDA 加速** | 安装 `CUDA.jl`（兼容版本见 `[compat] CUDA`）；加载后会激活包扩展 `AsterFlowCUDAExt` |
| **可选：AMD ROCm** | 安装 `AMDGPU.jl`（兼容版本见 `[compat] AMDGPU`）；激活 `AsterFlowROCMExt` |
| **构建 libasterflow** | CMake ≥ 3.18、C11 编译器 |
| **构建 aster_native** | CMake ≥ 3.18、C++17 编译器 |

---

## 2. 安装 Julia 包（开发模式）

Git 仓库根目录即 **Julia 包根**（根目录含 `Project.toml`、`src/`、`ext/`、`test/`）。

在 Julia 中：

```julia
using Pkg
Pkg.develop(path="/绝对路径/到/本仓库")
```

或在仓库根目录启动 Julia 时：

```julia
Pkg.develop(path=".")
```

之后在任意环境中 `using AsterFlow` 即可。首次 `using` 会执行 `__init__`，尝试定位 `libasterflow`（见下文）。

---

## 3. 构建可选动态库（`libasterflow` 模块）

源码位于 **`libasterflow/`**（`asterflow.c` + `CMakeLists.txt`）。默认产物名为 **`libasterflow.so`**（Linux）。

```bash
cd libasterflow
cmake -S . -B build
cmake --build build
```

成功后动态库通常在：

`libasterflow/build/libasterflow.so`

Julia 侧按以下顺序查找（见 `src/libasterflow.jl`）：

1. 环境变量 **`ASTERFLOW_LIB`**：指向 `.so` / `.dylib` 的完整路径（最高优先级）。  
2. 仓库根下 **`libasterflow/build/libasterflow.so`**。  
3. **`deps/libasterflow.so`**（若手动拷贝到包根下的 `deps/`）。

未找到库时，部分算子会回退到纯 Julia 实现，包仍可使用。

---

## 4. 构建 C++ 骨架（`aster_native` 模块）

源码位于 **`aster_native/AsterNative/`**，由 **`aster_native/CMakeLists.txt`** 聚合（`project(asterflow_native LANGUAGES CXX)`，`CXX_STANDARD 17`）。

典型流程：

```bash
cd aster_native
cmake -S . -B build
cmake --build build
```

静态库示例路径：`aster_native/build/AsterNative/libaf_native_tensor.a`（具体以生成树为准）。

该目标与 Julia 包的 **可选** `ccall` 路径独立；当前 Julia 主要对接 **`libasterflow` 模块** 的共享库。长期可将两模块在 ABI 层合并演进。

---

## 5. 环境变量

| 变量 | 作用 |
|------|------|
| **`ASTERFLOW_LIB`** | 显式指定 `libasterflow` 动态库路径 |
| **`ASTERFLOW_DEVICE`** | 从环境解析默认设备，例如 `cuda:0`（推荐） |
| **`ASTERFLOW_ACCELERATOR`** | 与上一项兼容的别名 |

设备解析逻辑见 `src/devices.jl`。测试或脚本中也可用 `first_available_accelerator` 等 API 选择首个可用加速器。

---

## 6. 运行测试

在已 `develop` 包的前提下：

```julia
using Pkg
Pkg.test("AsterFlow")
```

或在包目录下：

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

（在仓库根目录执行时，`--project=.` 指向包根，即仓库根目录。）

测试覆盖 CPU 语义、MLP 反传、IR 往返、TinyGPT 单步等；若已安装 CUDA 等扩展，会 exercise 相应后端。

---

## 7. 与完整「PyTorch 式」原生栈的差异

完整愿景包括 **`project(... LANGUAGES CXX CUDA)`、cuBLAS/cuDNN、pybind11、Python 包** 等。本仓库**当前** CMake 为：

- **`libasterflow/`**：仅 **C** 共享库（非 CUDA 工程）。  
- **`aster_native/`**：**C++17** 骨架，尚未接入 CUDA 语言与 Python 绑定。

后续若合并为单一顶层 CMake 或增加 CUDA arch，应同步更新本文档与仓库根 `README.md`。

---

## 8. 可选加速器扩展小结

- **CUDA**：安装 `CUDA` 包后，Julia 加载扩展注册 `:cuda` 后端算子与 runtime。  
- **ROCm**：安装 `AMDGPU` 包后，扩展注册 `:rocm` 相关路径。  
- **Ascend / Rockchip**：`ext/` 下占位扩展在包 `__init__` 中注册；具体算子与硬件依赖以各厂商工具链为准，需按需配置环境。

若扩展未安装或未加载，核心功能仍可在 CPU 与 `libasterflow`（若存在）路径下运行。
