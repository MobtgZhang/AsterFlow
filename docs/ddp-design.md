# 分布式数据并行（DDP）设计备忘

## 目标语义（与 PyTorch DDP 对齐方向）

- 每张卡持有完整模型副本；前向各自计算；反向得到本地梯度。
- 使用 **all-reduce** 对梯度求平均（或 sum 后再缩放）。
- **bucket**：按参数分组打包通信，与计算重叠（overlap）。

## Julia 侧可选后端

| 方案 | 优点 | 缺点 |
|------|------|------|
| `Distributed.jl` + 自定义 allreduce | 标准库 | 需自管进程拓扑与 CUDA 设备绑定 |
| MPI.jl（如 Open MPI） | 高性能集群常见 | 部署重 |
| NCCL 等（经 JLL） | GPU 友好 | 绑定与版本管理复杂 |

## 仓库内占位

- 模块 `AsterFlow.DistributedStub`（`src/distributed/ddp_stub.jl`）：提供 `ddp_barrier!`、`ddp_allreduce_mean_grads!` 的 **单机空实现**，便于后续替换为真实通信。

## 里程碑建议

1. 单机多卡数据并行 + 正确性测试（与单卡 loss 对齐）。
2. 再引入 bucket 与异步通信重叠。
