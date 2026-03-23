# AsterFlow

Julia 前端的深度学习运行时骨架：**张量、eager 自动微分、按设备分发的算子注册**，以及可选的 **C ABI 动态库** 与 **原生张量运行时（C++ 骨架）**。设计目标是在语义与分层上对齐 PyTorch 2.x 的 eager + 可扩展后端思路，体量按 MVP 迭代。

## 仓库布局（按模块）

| 目录 | 模块 |
|------|------|
| [`Project.toml`](Project.toml)、[`src/`](src/)、[`ext/`](ext/)、[`test/`](test/) | **Julia 包**（与 [Flux.jl](https://github.com/FluxML/Flux.jl) 等一致：包根在仓库根）：用户 API、dispatcher、autograd、`layers` / `optimise` / `transformer`、`compile` 占位、`src/data` 数据加载等。 |
| [`libasterflow/`](libasterflow/) | **C ABI 运行时**：`libasterflow` 共享库，与 Julia `ccall` 对接（可选加速路径）。 |
| [`aster_native/`](aster_native/) | **原生张量运行时（骨架）**：`AsterNative` C++17 子工程、CMake 目标 `af_native_tensor`（`TensorImpl` / `Storage` / 调度与设备占位等）。 |
| [`docs/`](docs/) | 设计原则、架构、构建与安装。 |
| [`examples/`](examples/) | 示例脚本。 |

顶层同时放 **Julia 包清单** 与 **C/C++ 子工程**；Julia 侧以 `Project.toml` 为包根，不再套一层 `julia/`。

## 快速开始

```julia
using Pkg
Pkg.develop(path=".")   # 在仓库根目录下，或传入本仓库的绝对路径
using AsterFlow
```

可选：构建 `libasterflow` 模块：

```bash
cd libasterflow && cmake -S . -B build && cmake --build build
```

默认会在 `libasterflow/build/libasterflow.so` 被自动探测；也可设置 `ASTERFLOW_LIB` 指向该文件。

运行示例（在仓库根目录）：

```bash
julia --project=. examples/tiny_mlp_train.jl
```

## 文档

- [文档索引](docs/README.md)
- [设计原则与架构](docs/design-principles.md)
- [构建与安装](docs/build-and-install.md)

## 许可证

见 [LICENSE](LICENSE)。
