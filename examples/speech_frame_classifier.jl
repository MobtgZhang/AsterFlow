#!/usr/bin/env julia
# 在仓库根目录执行：
#   julia --project=. examples/speech_frame_classifier.jl
#
# 教学占位：合成 (B,T,F) 帧特征 + 帧级类别标签，不做真实 ASR。
# 真实 pipeline 需 Kaldi、Whisper 等；默认 CPU，T≤32、F≤40、B≤16。

using AsterFlow
using Random

function example_device()
    if get(ENV, "ASTERFLOW_EXAMPLE_CUDA", "") == "1" && isavailable(cuda_device(0))
        return cuda_device(0)
    end
    return CPUDevice()
end

function main()
    Random.seed!(7)
    dev = example_device()
    B, T, F = 16, 32, 40
    n_classes = 12
    ## 合成帧特征：弱结构 + 噪声，便于脚本跑通
    rng = Random.default_rng()
    frames = zeros(Float32, B, T, F)
    for b in 1:B, t in 1:T, f in 1:F
        frames[b, t, f] =
            Float32(0.1 * sin(0.3 * t + 0.07 * f + b)) + Float32(0.05 * randn(rng))
    end
    Xflat = reshape(frames, B * T, F)
    labels = [rand(rng, 1:n_classes) for _ in 1:(B * T)]
    x = tensor(Xflat; device = dev, requires_grad = false)
    model = train!(
        Sequential(
            Linear(F, 24; device = dev),
            ReLU(),
            Linear(24, n_classes; device = dev),
        ),
    )
    opt = AdamW(params(model), lr = 5f-3)
    for it in 1:25
        logits = model(x)
        loss = cross_entropy_loss(logits, labels)
        zero_grad!.(params(model))
        backward(loss)
        step!(opt)
        if it % 5 == 0
            println("iter $it loss = ", to_array(loss)[1])
        end
    end
    println("speech_frame_classifier.jl 完成 (device=$dev)")
end

main()
