# AsterFlow 快速上手

## 安装

见 [build-and-install.md](build-and-install.md)：在仓库根 `Pkg.develop(path=".")`，可选构建 `libasterflow` 与加载 `CUDA` 扩展。

## 最小训练循环

```julia
using AsterFlow

m = train!(Sequential(Linear(4, 8), ReLU(), Linear(8, 1)))
x = tensor(randn(Float32, 16, 4))
y = tensor(randn(Float32, 16, 1))
opt = AdamW(params(m); lr = 1f-3)

for _ in 1:10
    yhat = m(x)
    loss = mse_loss(yhat, y)
    zero_grad!.(params(m))
    backward(loss)
    step!(opt)
end
```

## 设备迁移

```julia
dev = cuda_device(0)
m_cpu = train!(Linear(3, 2))
module_to_device!(m_cpu, dev)   # 或先建层时 `device=dev`
```

截断计算图：`detach_tensor(t)`（对应 PyTorch 的 `detach()`，避免与 `Base.detach(::Cmd)` 同名）。

## 自定义算子注册

见仓库 `examples/custom_op_register.jl`。

## 测试与基准

- 测试：`Pkg.test("AsterFlow")`
- 微基准：`benchmark/micro.jl`（需本地 `using BenchmarkTools`）

## 延伸阅读

- [设计原则](design-principles.md)
- [PyTorch API 对照](pytorch-api-parity.md)
- [算子契约](op-contract.md)
