# 加速器异步执行与流（CUDA）

## 当前行为

- **NVIDIA**：`AsterFlowCUDAExt` 使用 `CUDA.jl` 与 `CuArray`；逐元与 `*`（matmul）等在默认 CUDA 流上异步执行，与主机同步发生在：
  - `to_array` / `Array(...)` 拉回主机；
  - 部分需要标量观测的测试路径（应尽量避免 `@allowscalar` 热循环）。
- **默认策略**：不在 AsterFlow 层显式创建多流；与 PyTorch 默认单流语义接近。

## 推荐实践

- 训练循环中批量使用 GPU 张量，减少 CPU↔GPU 往返。
- 性能剖析使用 `CUDA.@time` / Nsight；若未来引入多流，需在文档中固定「参数 all-reduce 与算子 kernel 的流归属」。

## ROCm / NPU

- `AsterFlowROCMExt` 等与 CUDA 对称注册；未安装对应包时 `isavailable` 为 `false`，算子不得静默回退到错误设备。
