# 互操作：ONNX、Hugging Face 权重

## 当前能力

- **Safetensors / BSON / PyTorch state_dict（经 Python 脚本）**：见 `src/loading.jl` 与 [build-and-install.md](build-and-install.md)。
- **键名**：`load_state_dict!` 支持 `strict`、`pytorch_compat`；`expected_state_keys` 可预检。

## Hugging Face

- **权重文件**：优先使用 `safetensors` 导出为 F32（BF16 需外部转换，见 `load_safetensors` 报错提示）。
- **命名映射**：各模型架构不同；推荐在加载脚本中构造 `Dict{String,Array}` 后调用 `load_state_dict!`，或对 `state_dict` 键做一层重命名表（可在项目层实现，不强制进核心包）。

## ONNX

- **导入**：未内置 ONNX Runtime；可选路径为外部工具将 ONNX 转为 `state_dict` 或 numpy 再 `tensor(...)`。
- **导出**：可将 `state_dict(m)` 交给 Python `onnx` 生态由用户脚本完成；核心包保持可选依赖为零增加。

## 建议工作流

1. HF 下载 `model.safetensors` → `load_safetensors` → 键名映射 → `load_state_dict!`。
2. 需 ONNX 推理时，在部署环境用专用推理引擎，与训练包解耦。
