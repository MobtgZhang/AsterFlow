# 与 Flux.jl / Zygote.jl 的关系

## 设计选择

- **AsterFlow** 主路径为 **PyTorch 风格的动态图 tape**（`Tensor` + `grad_fn` + `backward`），便于与自定义 C/CUDA 算子、设备分发、未来子图编译同一套语义。
- **Flux** 传统上以 **Zygote**（或 Enzyme 等）做语言级 AD，与「显式 tape + 算子注册」模型不同。

## 混用建议

- **不推荐**在同一前向中混用 Zygote 对 AsterFlow `Tensor` 求导（缺少链式规则与性能路径）。
- **可行**：
  - 推理：Flux 模型权重导出为数组 / safetensors，再 `load_state_dict!` 到 AsterFlow `Module`。
  - 训练：选定一侧为主框架，另一侧仅作数据或后处理。

## 组织借鉴

- 目录布局（`layers/`、`optimise/`、`data/`）借鉴 Flux 习惯；**不**表示 AD 实现相同。
