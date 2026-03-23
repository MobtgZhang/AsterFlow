# libasterflow 模块

**职责**：对外提供稳定的 **C ABI**（`asterflow_c_api.h`），供 Julia 包通过 `ccall` 调用部分算子（如列主 `matmul`、激活等）。未构建或未加载时，Julia 侧回退到纯 Julia 实现。

**内容**：`asterflow.c`、`CMakeLists.txt`、`include/`。

**构建**（在本目录下）：

```bash
cmake -S . -B build && cmake --build build
```

产物一般为 `build/libasterflow.so`（Linux）。详见仓库根目录 [docs/build-and-install.md](../docs/build-and-install.md)。
