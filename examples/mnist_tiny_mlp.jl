#!/usr/bin/env julia
# 在仓库根目录执行：
#   julia --project=. examples/mnist_tiny_mlp.jl
#
# 默认 CPU、小批量、小隐藏层（适合约 3GB 显存笔记本；GPU 仅可选）。
# 可选真实 MNIST：设置 ASTERFLOW_MNIST_DIR 指向含官方 idx 文件的目录：
#   train-images-idx3-ubyte、train-labels-idx1-ubyte

using AsterFlow
using Random

function example_device()
    if get(ENV, "ASTERFLOW_EXAMPLE_CUDA", "") == "1" && isavailable(cuda_device(0))
        return cuda_device(0)
    end
    return CPUDevice()
end

function load_mnist_train_optional(max_samples::Int)
    dir = get(ENV, "ASTERFLOW_MNIST_DIR", "")
    isempty(dir) && return nothing
    isdir(dir) || error("ASTERFLOW_MNIST_DIR 不是目录: $dir")
    ip = joinpath(dir, "train-images-idx3-ubyte")
    lp = joinpath(dir, "train-labels-idx1-ubyte")
    isfile(ip) && isfile(lp) || error("MNIST 目录缺少 train-images-idx3-ubyte 或 train-labels-idx1-ubyte")
    X = open(ip) do io
        magic = ntoh(read(io, UInt32))
        magic == UInt32(2051) || error("train-images: magic 应为 2051")
        n = Int(ntoh(read(io, UInt32)))
        rows = Int(ntoh(read(io, UInt32)))
        cols = Int(ntoh(read(io, UInt32)))
        rows == 28 && cols == 28 || error("期望 28×28 MNIST")
        n_use = min(n, max_samples)
        raw = read(io, n_use * rows * cols)
        M = zeros(Float32, n_use, 784)
        @inbounds for k in 1:n_use, r in 1:784
            M[k, r] = Float32(raw[(k - 1) * 784 + r]) / Float32(255)
        end
        M
    end
    y = open(lp) do io
        magic = ntoh(read(io, UInt32))
        magic == UInt32(2049) || error("train-labels: magic 应为 2049")
        n = Int(ntoh(read(io, UInt32)))
        n_use = min(n, size(X, 1))
        lb = read(io, n_use)
        [Int(lb[i]) + 1 for i in 1:n_use]
    end
    return (tensor(X; device = CPUDevice(), requires_grad = false), y)
end

function synthetic_mnist_like(rng::AbstractRNG, n::Int)
    X = randn(rng, Float32, n, 784) .* Float32(0.2)
    y = [rand(rng, 1:10) for _ in 1:n]
    return (tensor(X; device = CPUDevice(), requires_grad = false), y)
end

function main()
    Random.seed!(42)
    dev = example_device()
    batch = 32
    hidden = 32
    loaded = load_mnist_train_optional(2048)
    if loaded === nothing
        X_cpu, y_cpu = synthetic_mnist_like(Random.default_rng(), 512)
        println("使用合成数据（设置 ASTERFLOW_MNIST_DIR 可加载真实 MNIST idx）")
    else
        X_cpu, y_cpu = loaded
        println("使用 MNIST 子集: n=$(size(to_array(X_cpu), 1))")
    end
    X_mat = to_array(X_cpu)
    model = train!(
        Sequential(
            Linear(784, hidden; device = dev),
            ReLU(),
            Linear(hidden, 10; device = dev),
        ),
    )
    opt = AdamW(params(model), lr = 3f-3)
    n = size(X_mat, 1)
    for it in 1:30
        r = min(batch, n)
        mask = randperm(n)[1:r]
        xb = tensor(X_mat[mask, :]; device = dev, requires_grad = false)
        yb = y_cpu[mask]
        logits = model(xb)
        loss = cross_entropy_loss(logits, yb)
        zero_grad!.(params(model))
        backward(loss)
        step!(opt)
        if it % 10 == 0
            println("iter $it loss = ", to_array(loss)[1])
        end
    end
    println("mnist_tiny_mlp.jl 完成 (device=$dev)")
end

main()
