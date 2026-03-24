# AsterFlow 轻量示例

在**仓库根目录**执行（`--project=.` 指向包根）：

```bash
julia --project=. examples/mnist_tiny_mlp.jl
julia --project=. examples/speech_frame_classifier.jl
julia --project=. examples/char_lm_tiny.jl
```

## 资源与设备

- **默认 CPU**：适合约 **3GB 显存**的笔记本；不假设已安装 CUDA。
- **可选 GPU**：若已安装并能使用 CUDA 后端，可设置 `ASTERFLOW_EXAMPLE_CUDA=1` 再运行；小显存仍可能 OOM，请先以 CPU 验证。

## 脚本说明

| 文件 | 内容 |
|------|------|
| `mnist_tiny_mlp.jl` | 合成 `(N,784)` 特征 + 10 类，小 MLP + `cross_entropy_loss`。可选：设置 `ASTERFLOW_MNIST_DIR` 指向含 `train-images-idx3-ubyte` 与 `train-labels-idx1-ubyte` 的目录，加载真实 MNIST 子集。 |
| `speech_frame_classifier.jl` | 合成 `(B,T,F)` 帧特征与帧级标签，`Linear`+`ReLU` 帧级分类占位（非真实语音识别）。 |
| `char_lm_tiny.jl` | 使用同目录下 `corpus_tiny.txt`，`TinyGPT` 字符级训练若干 epoch。 |

## 与 `aster_native`（可选）

构建 `aster_native` 共享库并设置 `ASTERFLOW_USE_NATIVE_CPP=1` 时，部分 `Float32` 算子可走 C++ 路径；不设置则与未构建该库时行为一致。详见仓库根目录 [`docs/build-and-install.md`](../docs/build-and-install.md) 与 [`aster_native/README.md`](../aster_native/README.md)。
