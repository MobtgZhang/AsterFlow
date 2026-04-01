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

## 4. 构建 C++ 原生运行时（`aster_native` 模块）

源码位于 **`aster_native/AsterNative/`**，由 **`aster_native/CMakeLists.txt`** 聚合（`project(asterflow_native LANGUAGES CXX)`，`CXX_STANDARD 17`）。

典型流程（含可选 C++ 单测，推荐 CI 与本地校验时开启）：

```bash
cd aster_native
cmake -S . -B build -DASTERFLOW_NATIVE_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

产物（路径以生成树为准）：

| 产物 | 示例路径（Linux） |
|------|-------------------|
| **共享库** `libasterflow_native.so` | `aster_native/build/AsterNative/libasterflow_native.so` |
| 静态库 `libaf_native_tensor.a` | `aster_native/build/AsterNative/libaf_native_tensor.a` |

稳定 C 头文件：`aster_native/include/asterflow_native.h`（`af_native_version`、`af_native_matmul_f32_colmajor` 等）。

Julia 侧按以下顺序查找共享库（见 `src/libasterflow_native.jl`）：

1. 环境变量 **`ASTERFLOW_NATIVE_LIB`**：指向 `.so` / `.dylib` / `.dll` 的绝对路径（最高优先级）。  
2. 仓库根下 **`aster_native/build/AsterNative/libasterflow_native.so`**（macOS 为 `.dylib`，Windows 为 `asterflow_native.dll`）。

未找到库时，算子仍走 Julia / `libasterflow` 既有路径，包可正常使用。

### 4.1 使用 C++ CPU 内核（可选）

设置 **`ASTERFLOW_USE_NATIVE_CPP=1`** 且成功加载 `libasterflow_native` 时，`Float32` 的 **`add` / `relu` / `matmul`** 会优先调用 `aster_native` 导出函数（见 `src/native.jl`）。默认不设置该变量，行为与仅使用 `libasterflow` 时一致。

`libasterflow` 与 `libasterflow_native` 由不同环境变量定位，可并存；详见 [`aster_native/README.md`](../aster_native/README.md)。

### 4.2 `libasterflow` 与 `aster_native` 职责边界（约定）

| 组件 | 角色 |
|------|------|
| **`libasterflow`** | 瘦 **C ABI** 共享库：稳定 `ccall` 符号、少量 FP32 核（如 `af_matmul` / `af_add`）；便于无 C++ 运行时的环境。 |
| **`aster_native`（`libasterflow_native`）** | **C++17** 运行时骨架：与 CMake 测试、现代工具链对齐；在 **`ASTERFLOW_USE_NATIVE_CPP=1`** 且库可用时，对 **Float32** 的 `add` / `relu` / `matmul` 可走 `an_native_*`（见 `src/native.jl`）。 |
| **Julia 主路径** | 未启用或未命中上述条件时，**以 Julia 实现的 CPU 内核为准**；两套 native 库为**可选加速**，不应被文档暗示为互斥的两种「产品」。 |

**CI 建议**：矩阵中分别跑「不设置 `ASTERFLOW_USE_NATIVE_CPP`」与「`ASTERFLOW_USE_NATIVE_CPP=1` 且已构建 `aster_native`」以捕捉双路径行为差异。

---

## 5. 环境变量

| 变量 | 作用 |
|------|------|
| **`ASTERFLOW_LIB`** | 显式指定 `libasterflow` 动态库路径 |
| **`ASTERFLOW_NATIVE_LIB`** | 显式指定 `libasterflow_native`（`aster_native` 共享库）路径 |
| **`ASTERFLOW_USE_NATIVE_CPP`** | 设为 `1` 时，在已加载 `libasterflow_native` 的前提下，`Float32` 的 `add` / `relu` / `matmul` 可走 C++ 路径 |
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
- **`aster_native/`**：**C++17** CPU Float32 内核 + 可选共享库；CUDA 语言与 Python 绑定仍为远期项（见 [`aster_native/README.md`](../aster_native/README.md)）。

后续若合并为单一顶层 CMake 或增加 CUDA arch，应同步更新本文档与仓库根 `README.md`。

---

## 8. 可选加速器扩展小结

- **CUDA**：安装 `CUDA` 包后，Julia 加载扩展注册 `:cuda` 后端算子与 runtime。  
- **ROCm**：安装 `AMDGPU` 包后，扩展注册 `:rocm` 相关路径。  
- **Ascend / Rockchip**：`ext/` 下占位扩展在包 `__init__` 中注册；具体算子与硬件依赖以各厂商工具链为准，需按需配置环境。

若扩展未安装或未加载，核心功能仍可在 CPU 与 `libasterflow`（若存在）路径下运行。
