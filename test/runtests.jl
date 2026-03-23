using Test
using AsterFlow

@testset "device string + isavailable (PyTorch-style)" begin
    @test device("cpu") isa CPUDevice
    @test isavailable(device("cpu"))
    @test device("CUDA") == AcceleratorDevice(:cuda, 0)
    @test device("cuda:1") == AcceleratorDevice(:cuda, 1)
    @test isavailable(device("cuda")) isa Bool
    @test runtime_available(device("cpu"))  # 别名
end

@testset "tensor view / contiguous" begin
    t = tensor([1.0 2.0; 3.0 4.0])
    @test is_contiguous(t)
    p = permute_tensor(t, (2, 1))
    @test !is_contiguous(p)
    c = contiguous(p)
    @test is_contiguous(c)
    @test size(c) == (2, 2)
end

@testset "dispatcher add matmul" begin
    a = tensor(ones(Float32, 2, 3))
    b = tensor(ones(Float32, 2, 3))
    c = add(a, b)
    @test all(to_array(c) .== 2)
    x = tensor(Float32[1 2; 3 4])
    y = tensor(Float32[1 0; 0 1])
    z = matmul(x, y)
    @test to_array(z) ≈ Float32[1 2; 3 4]
end

@testset "autograd mlp step" begin
    m = Sequential(Linear(4, 8), ReLU(), Linear(8, 1))
    x = tensor(randn(Float32, 16, 4))
    y = tensor(randn(Float32, 16, 1))
    yhat = m(x)
    loss = mse_loss(yhat, y)
    backward(loss)
    opt = SGD(params(m), lr = 1f-2)
    step!(opt)
    @test numel(loss) == 1
end

@testset "libasterflow stub optional" begin
    v = libasterflow_version()
    @test v == -1 || v >= 1
    a = tensor(Float32[1 2; 3 4])
    b = tensor(Float32[1 0; 0 1])
    out = af_matmul_nograd(a, b)
    @test size(out) == (2, 2)
end

@testset "weights: safetensors + BSON + load_state_dict!" begin
    m = Sequential(Linear(3, 4), ReLU(), Linear(4, 2))
    d = state_dict(m)
    p = tempname() * ".safetensors"
    save_safetensors(p, d)
    d2 = load_safetensors(p)
    @test Set(keys(d2)) == Set(keys(d))
    for k in keys(d)
        @test eltype(d2[k]) == Float32
        @test d2[k] ≈ d[k]
    end
    rm(p; force = true)

    bpath = tempname() * ".bson"
    save_weights_bson(bpath, d)
    d3 = load_weights_bson(bpath)
    @test Set(keys(d3)) == Set(keys(d))
    for k in keys(d)
        @test Array{Float32}(d3[k]) ≈ d[k]
    end
    rm(bpath; force = true)

    m2 = Sequential(Linear(3, 4), ReLU(), Linear(4, 2))
    load_state_dict!(m2, d2)
    @test state_dict(m2)["0.weight"] ≈ d["0.weight"]

    ## PyTorch `nn.Linear` 的 weight 为 (out, in)，AsterFlow 为 (in, out)
    m3 = Linear(3, 4)
    w_pt = randn(Float32, 4, 3)
    b_pt = randn(Float32, 4)
    load_state_dict!(m3, Dict("weight" => w_pt, "bias" => b_pt); pytorch_compat = true)
    @test to_array(m3.weight) ≈ collect(permutedims(w_pt, (2, 1)))
    @test to_array(m3.bias) ≈ b_pt
end

@testset "compile IR roundtrip" begin
    g = IRGraph()
    i1 = ir_new_input!(g, "Float32", [2, 2])
    i2 = ir_new_input!(g, "Float32", [2, 2])
    outs = ir_append_node!(g, IR_MatMul, [i1, i2], [[2, 2]])
    ir_set_outputs!(g, outs)
    js = graph_to_json(g)
    g2 = graph_from_json(js)
    @test length(g2.nodes) == 1
    stub = codegen_stub_cache(g2)
    @test isfile(stub.spec_path)
    @test isfile(stub.cuda_stub_path)
end

@testset "TinyGPT one step" begin
    model = TinyGPT(32, 16, 32)
    idx = ones(Int, 1, 8)
    logits = model(idx)
    @test size(logits) == (8, 32)
    tgt = tensor(randn(Float32, 8, 32))
    loss = mse_loss(logits, tgt)
    backward(loss)
    opt = AdamW(params(model), lr = 1f-3)
    step!(opt)
end

@testset "optimizers CPU" begin
    m = Sequential(Linear(3, 4), ReLU(), Linear(4, 1))
    x = tensor(randn(Float32, 8, 3))
    y = tensor(randn(Float32, 8, 1))
    for (Opt, kws) in (
        (Adam, (; lr = 5f-4)),
        (AdamW, (; lr = 5f-4)),
        (RMSprop, (; lr = 5f-4)),
        (Adagrad, (; lr = 1f-2)),
        (Adadelta, (;)),
        (Adamax, (; lr = 5f-4)),
        (RAdam, (; lr = 5f-4)),
        (AdaFactor, (; lr = 1f-3)),
        (Lion, (; lr = 1f-4)),
        (Sophia, (; lr = 1f-3, rho = 1f1)),
    )
        train!(m)
        yhat = m(x)
        loss = mse_loss(yhat, y)
        zero_grad!.(params(m))
        backward(loss)
        step!(Opt(params(m); kws...))
    end
    train!(m)
    yhat = m(x)
    loss = mse_loss(yhat, y)
    zero_grad!.(params(m))
    backward(loss)
    step!(Lookahead(Adam(params(m); lr = 5f-4); k = 2, alpha = 0.5f0))
    m2 = train!(Sequential(Linear(3, 2)))
    x2 = tensor(randn(Float32, 4, 3))
    loss2 = mse_loss(m2(x2), tensor(randn(Float32, 4, 2)))
    backward(loss2)
    step!(SGD(params(m2); lr = 1f-2, momentum = 9f-1))
    @test true
end

@testset "cross entropy, LayerNorm, DataLoader" begin
    m = train!(Sequential(Linear(4, 3), LayerNorm(3), ReLU(), Linear(3, 2)))
    x = tensor(randn(Float32, 8, 4))
    yhat = m(x)
    tgt = [1, 2, 1, 2, 1, 2, 1, 2]
    loss = cross_entropy_loss(yhat, tgt)
    zero_grad!.(params(m))
    backward(loss)
    @test numel(loss) == 1
    ds = TensorDataset(x, tensor(randn(Float32, 8, 2)))
    @test length(ds) == 8
    batches = collect(DataLoader(ds; batchsize = 4, shuffle = true))
    @test length(batches) == 2
    @test size(batches[1][1]) == (4, 4)
    train_ds, val_ds = random_split(ds, [6, 2])
    @test length(train_ds) == 6 && length(val_ds) == 2
end

@testset "accelerator device (vendor-agnostic)" begin
    # 不硬编码 CUDA：由扩展注册 :cuda / :rocm 等 probe；此处仅加载 Julia 生态中已有的后端
    try
        @eval using CUDA
    catch
    end
    env_dev = accelerator_from_env()
    dev = env_dev !== nothing ? env_dev : first_available_accelerator([:cuda, :rocm, :npu])
    if dev !== nothing && isavailable(dev)
        @test dev isa AcceleratorDevice
        @test device_backend(dev) isa Symbol
        x = dev_ones(Float32, (2, 2), dev)
        y = dev_ones(Float32, (2, 2), dev)
        c = add(x, y)
        @test eltype(c) == Float32
        m = train!(Sequential(Linear(4, 8; device = dev), ReLU(), Linear(8, 1; device = dev)))
        xb = tensor(randn(Float32, 16, 4); device = dev)
        yb = tensor(randn(Float32, 16, 1); device = dev)
        yhat = m(xb)
        loss = mse_loss(yhat, yb)
        backward(loss)
        @test device(loss) isa AcceleratorDevice
        @test device_backend(device(loss)) == device_backend(dev)
        step!(SGD(params(m), lr = 1f-2))
        @test numel(loss) == 1
        m2 = train!(Sequential(Linear(3, 4; device = dev), Dropout(0.5)))
        z = dev_ones(Float32, (8, 3), dev)
        _ = m2(z)
        @test true
    else
        @test true
    end
end
