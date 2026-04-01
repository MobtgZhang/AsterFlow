# 编译路径 IR：概念设计（与实现渐进对齐）

`src/compile/` 当前提供 **IR 数据结构、trace 占位、序列化与融合桩**。本文固定**概念层**约定，避免未来 IR 与 autograd tape 形状冲突导致大规模重构。

## 1. 目标

- **短期**：eager + 动态 autograd 为真源；IR 用于调试、缓存键、文档化子图。
- **中期**：对热点子图 trace → IR → Pass（如常量折叠、算子融合）→ 后端桩或手写 kernel。
- **长期**：可选与全图编译、AOT 路径对接（非 MVP 承诺）。

## 2. 节点与值

- **IRValue**：逻辑张量槽位（含静态形状 / dtype 字符串等元数据，见 `IRGraph` 内定义）。
- **IRNode**：一次计算，`IROpKind` 枚举算子种类（如 `IR_Add`、`IR_MatMul`、`IR_ReLU`）；输入为 `IRValue` id 列表，输出为新的 `IRValue` id。
- **IRGraph**：持有 inputs、nodes、outputs；可序列化为 JSON（`graph_to_json` / `graph_from_json`）。

## 3. 与 autograd tape 的关系

| 概念 | autograd（`graph.jl`） | compile IR |
|------|------------------------|------------|
| 粒度 | 张量 + `Node` 子类型 | 子图级 `IRNode` |
| 生命周期 | 一次 `backward` 后可丢弃 | 可磁盘缓存、跨进程 |
| 语义 | 精确梯度规则 | 可先仅支持无状态逐元 / matmul 子集 |

**原则**：IR **不替代** autograd；融合 kernel 若改变数值，须在 eager 侧有等价定义或明确文档非 bitwise 一致。

## 4. Pass 接口（约定）

未来 Pass 建议采用纯函数风格：

- 输入：`IRGraph`
- 输出：新的 `IRGraph` 或就地改写 + 变更日志
- 示例 Pass：死代码消除、线性+ReLU 融合（已有 `fuse_linear_relu_chain!` 占位）、常量折叠。

具体函数签名可在首次实现 Pass 时在 `src/compile/` 落地并链接本文。

## 5. 代码入口

| 文件 | 角色 |
|------|------|
| [`src/compile/ir.jl`](../src/compile/ir.jl) | `IRGraph`、`IRNode`、构造 API |
| [`src/compile/trace.jl`](../src/compile/trace.jl) | 前向捕获 |
| [`src/compile/fusion.jl`](../src/compile/fusion.jl) | 融合规则 |
| [`src/compile/codegen_stub.jl`](../src/compile/codegen_stub.jl) | 占位代码生成与缓存路径 |
