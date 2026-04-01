# 设计原则与架构

本文整合 `ideas/idea1.md` 与 `ideas/source-design-pytorch2-cpp-cuda.md` 中的核心结论，并与本仓库**当前实现**对齐。

---

## 1. 路线判断：先小后大

不建议首版就做「完整 PyTorch + CUDA + Transformers + 编译器优化」的大一统形态。更稳妥的路径是：

1. **先做可用内核**：Julia 前端 +（可选）C/C++ 后端 + 最小 autograd + eager 执行 + Transformer 子集，跑通训练与推理。
2. **再逐步叠加**：图优化与融合、分布式、混合精度、FlashAttention / fused optimizer、AOT/JIT、外部权重格式等。

---

## 2. 四层职责（你在「造什么」）

| 层次 | 职责 |
|------|------|
| **用户 / API 层** | 类似 PyTorch 的 `Tensor`、`Module`、`Optimizer`、设备迁移语法糖 |
| **张量与执行层** | `Tensor` 元数据、device/dtype/shape/stride、eager、kernel 调度、内存与流（后期） |
| **Autograd 层** | 动态反向图、`backward`、梯度累积、自定义算子反传、view / inplace 语义 |
| **性能层** | CUDA / BLAS / DNN、融合算子、编译缓存、多卡（后期） |

---

## 3. 推荐总体拆分：前端 / Runtime / 后端

- **Julia 前端**：用户 API、`Module` / 参数遍历、形状推导、计算图记录（trace）、调度入口。
- **核心 Runtime**（可在 Julia 先落地）：张量元数据、op registry、eager 执行、autograd  tape、dispatcher、内存策略。
- **C++ / CUDA（或 C ABI）后端**：重型 kernel、库封装、分配器与流、profiling；Julia 通过 **稳定 C ABI**（`ccall`）调用，避免过早绑定复杂 C++ 互操作。

**关键决策**：Julia 负责表达、图与开发体验；高性能热点下沉到 native 层。项目定位更接近 **「Julia 绑定的 DL runtime」**，而非全栈纯 Julia 学术原型。

---

## 4. 与 PyTorch 2.x 的对应（设计对标）

| PyTorch 2.x | AsterFlow 设计落点 |
|-------------|-------------------|
| `Tensor` / `TensorImpl` | 元数据 + `Storage`；Julia 侧 `Tensor`，C++ 侧 `AsterNative` 骨架 |
| Dispatcher / `DispatchKey` | 按设备后端符号注册与查表；并支持 **dtype** 键与 **fallback 链**（如 `:cuda` 未实现某算子时尝试 `:cpu`） |
| Autograd | 张量级动态图：`Node` 子类型 + `backward` 拓扑 |
| `torch.compile` | Julia 侧 `compile/`：IR、序列化、占位 codegen 与磁盘缓存；长期可对齐全图编译 |
| 自定义算子 | `register_op!` + 各后端实现；扩展包注册加速器算子 |

**原则**：语义与分层模仿 PyTorch 2.x，实现体量按 MVP 裁剪；编译器先做**子图 + 融合 + 缓存**，不追求首版等价完整 Inductor。

---

## 5. 总体架构（概念图）

```
┌─────────────────────────────────────────┐
│  Julia API（Tensor, nn, optim, compile） │
└───────────────────┬─────────────────────┘
                    │ ccall / 稳定 C ABI（可选 libasterflow）
┌───────────────────▼─────────────────────┐
│  Runtime：Tensor · Dispatcher · Autograd │
└───────────────────┬─────────────────────┘
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     CPU 实现   加速器扩展    编译占位（IR / stub）
```

- **Eager**：API → `dispatch_op` → 已注册 kernel。
- **Compile（演进中）**：trace → IR →（未来）Pass / CUDA 生成 → 缓存。

---

## 6. Dispatcher 是扩展中枢

每个算子不应写死单一实现，而应经统一注册与按 **设备后端** 分发，便于后续增加 CPU、CUDA、融合、debug、编译路径等实现，而 API 层仍是一个 `matmul` / `add`。

---

## 7. Autograd：优先张量级反向模式

首版采用 **PyTorch 风格动态图**：前向记录输入、输出与 `grad_fn`，反向按拓扑应用规则。这与底层 C++/CUDA 自定义算子、混合精度、checkpointing、后期融合更契合；不必一上来走语言级 source-to-source AD（如 Zygote）作为主路径。

---

## 8. MVP 与开发顺序（摘要）

**MVP 宜包含**：`Tensor`（data/shape/stride/dtype/device/requires_grad/grad）、基础二元运算与 `matmul`、reshape/view/contiguous、常用激活与归一、`nn.Module` 风格、SGD / AdamW、最小 Transformer 块、单卡 eager。

**分阶段（与 idea 文档一致，可裁剪）**：

1. Tensor + CPU + autograd + 简单层与 SGD  
2. 加速器后端 + matmul / 逐元 + AdamW + LayerNorm / Softmax  
3. Attention / TransformerBlock + 小规模训练与 checkpoint  
4. 混合精度、FlashAttention、KV cache 等  
5. 图捕获、融合、多卡  

**编译路径**：先 eager 语义与 autograd 稳定，再对热点子图做 trace / fusion，避免首版全局静态图。

---

## 9. 语义与一致性风险（必须在设计评审中固定）

以下比「写一个 CUDA kernel」更容易拖垮正确性，需单独规范：

1. **view / reshape / stride / contiguous** 与 backward 的视图梯度规则。  
2. **inplace** 与 `requires_grad`、版本计数冲突。  
3. **broadcasting** 在反传与融合中的显式化。  
4. **混合精度**：主权重、grad scaling、autocast 策略。  
5. **异步执行**：流、同步点、tensor 生命周期。

---

## 10. 本仓库目录与实现对应

| 层级 | 路径 |
|------|------|
| Julia 包根 | 仓库根（`Project.toml`、`src/`、`ext/`、`test/`） |
| 聚合入口 | `src/AsterFlow.jl` |
| 数据加载（Dataset / DataLoader） | `src/data/dataset.jl` |
| Dispatcher | `src/dispatch.jl` |
| Autograd | `src/graph.jl`、`src/ops.jl`、`src/autograd.jl` |
| 设备与后端符号 | `src/devices.jl`、`src/accelerator_dispatch.jl` |
| C ABI 模块（可选） | `libasterflow/` → `build/libasterflow.so`；头文件 `libasterflow/include/asterflow_c_api.h` |
| 原生张量骨架模块 | `aster_native/AsterNative/`，顶层 `aster_native/CMakeLists.txt`，目标 `af_native_tensor` |
| CUDA / ROCm 包扩展 | `ext/AsterFlowCUDAExt.jl`、`AsterFlowROCMExt.jl`（weakdeps） |
| 其他加速器占位注册 | `ext/AsterFlowHuaweiAscendExt.jl`、`AsterFlowRockchipNPUExt.jl` |
| 编译占位 | `src/compile/` |
| 测试 | `test/runtests.jl` |

---

## 11. API 风格

- **借鉴 PyTorch**：`Tensor`、`Module`、`requires_grad` / `no_grad`、设备语义、`compile` 方向。  
- **发挥 Julia 优势**：多重派发、可调用 `struct`、宏与类型参数，避免机械复制 Python 语法。

---

## 12. 参考生态（思路层面）

PyTorch（eager + dispatcher + autograd）、JAX/XLA（trace + compile）、Flux、CUDA.jl、Transformers.jl 等。
