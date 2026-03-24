#!/usr/bin/env julia
# 在仓库根目录执行：
#   julia --project=. examples/char_lm_tiny.jl
#
# 字符级小语言模型：极短语料 + TinyGPT；默认 CPU，vocab/序列长度/宽度均很小。

using AsterFlow
using Random

function example_device()
    if get(ENV, "ASTERFLOW_EXAMPLE_CUDA", "") == "1" && isavailable(cuda_device(0))
        return cuda_device(0)
    end
    return CPUDevice()
end

function load_corpus()
    p = joinpath(@__DIR__, "corpus_tiny.txt")
    isfile(p) || error("缺少语料: $p")
    text = read(p, String)
    isempty(strip(text)) && error("语料为空")
    return text
end

function build_vocab(text::AbstractString)
    chars = sort(unique(collect(text)))
    length(chars) <= 128 || error("本样例假设 vocab≤128，请缩短语料")
    stoi = Dict{Char,Int}(c => i for (i, c) in enumerate(chars))
    return chars, stoi
end

function main()
    Random.seed!(11)
    dev = example_device()
    text = load_corpus()
    chars, stoi = build_vocab(text)
    codepoints = collect(text)
    vocab = length(chars)
    d_model = 32
    ff_dim = 48
    seq_len = min(16, length(codepoints) - 1)
    seq_len >= 4 || error("语料太短，无法训练")
    model = train!(TinyGPT(vocab, d_model, ff_dim; device = dev))
    opt = AdamW(params(model), lr = 2f-3)
    chunk = codepoints[1:(seq_len + 1)]
    idx = reshape([stoi[c] for c in chunk[1:seq_len]], 1, seq_len)
    targets = [stoi[chunk[i + 1]] for i in 1:seq_len]
    for ep in 1:5
        logits = model(idx)
        loss = cross_entropy_loss(logits, targets)
        zero_grad!.(params(model))
        backward(loss)
        step!(opt)
        println("epoch $ep loss = ", to_array(loss)[1])
    end
    println("char_lm_tiny.jl 完成 (device=$dev, vocab=$vocab)")
end

main()
