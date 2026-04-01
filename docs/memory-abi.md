# C ABI 与 Julia GC：内存所有权约定

本文描述 AsterFlow 在 **Julia 拥有的张量缓冲** 与 **C/C++ 动态库** 之间传递数据时的所有权与生命周期规则，避免 double-free 与「Julia 已回收、native 仍悬挂指针」类错误。

## 1. 默认原则：数据由 Julia 侧拥有

- **`Tensor` 的 `storage`** 类型为 `AbstractVector{T}`（CPU 上通常为 `Vector{T}`；CUDA 上为 `CuArray` 等）。
- **`libasterflow` / `libasterflow_native` 的 `ccall`** 在 MVP 中采用 **调用期间有效** 的指针视图：由 Julia 数组保证底层数据在 `ccall` 执行时不被移动或回收；**不在 C 侧 `free` Julia 提供的指针**。
- C 侧若需长期持有缓冲，必须在文档与 API 中显式约定「由谁分配、谁释放」，当前仓库**未**将张量所有权转移给 C 库；扩展若引入池化分配器，应单独设计 `af_retain` / `af_release` 式引用计数或 arena。

## 2. `ccall` 与 GC 安全

- 从 `Vector` 取 `pointer(...)` 调用 C 时，应持有对**源数组**的 Julia 引用直至 C 返回；本包在 `libasterflow.jl` / `libasterflow_native.jl` 中通过让 `Tensor` 参与调用栈来隐式钉住对象。
- **不应**把裸指针存入仅存在于 C 全局区的结构而无同步机制，除非有显式同步点与生命周期文档。

## 3. 加速器缓冲

- `CuArray` / `ROCArray` 由对应 Julia 包管理设备内存；向自定义 CUDA kernel 传递时，生命周期规则与 **CUDA.jl / AMDGPU.jl** 一致：在 kernel 完成前保持对数组的引用，异步流场景见 [`accelerator-streams.md`](accelerator-streams.md)。

## 4. 与 `tensor` / `to_array` 的关系

- **`to_array`**：将张量物化为 CPU `Array`（可能触发设备到主机拷贝）；结果归 Julia GC。
- **`tensor(...)`**：新建拥有独立或可共享 `storage` 的 `Tensor`；与 C 互操作时仍以 Julia 侧 `Tensor` 为真源。

## 5. 实现索引

| 组件 | 路径 |
|------|------|
| C ABI 探测与调用 | [`src/libasterflow.jl`](../src/libasterflow.jl) |
| `aster_native` 绑定 | [`src/libasterflow_native.jl`](../src/libasterflow_native.jl) |
| 张量定义 | [`src/tensor.jl`](../src/tensor.jl) |
