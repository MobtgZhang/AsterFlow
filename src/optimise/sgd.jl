mutable struct SGD
    params::Vector{Tensor}
    lr::Float32
    weight_decay::Float32
    momentum::Float32
    dampening::Float32
    nesterov::Bool
    vel::Dict{UInt64,Any}
end

function SGD(
    params;
    lr = 1f-2,
    weight_decay = 0f0,
    momentum = 0f0,
    dampening = 0f0,
    nesterov = false,
)
    (momentum > 0 || !nesterov) ||
        error("SGD: nesterov 需要 momentum > 0")
    return SGD(
        collect(Tensor, params),
        Float32(lr),
        Float32(weight_decay),
        Float32(momentum),
        Float32(dampening),
        nesterov,
        Dict{UInt64,Any}(),
    )
end

function _sgd_cpu!(
    p::Tensor,
    g::Tensor,
    lr::Float32,
    wd::Float32,
    vbuf::Union{Nothing,Vector{Float32}},
    momentum::Float32,
    dampening::Float32,
    nesterov::Bool,
)
    n = numel(p)
    @inbounds for i in 1:n
        pi = p.offset + i
        gi = g.offset + i
        gv = Float32(g.storage[gi])
        pv = Float32(p.storage[pi])
        if momentum > 0 && vbuf !== nothing
            vbuf[i] = momentum * vbuf[i] + (1f0 - dampening) * gv
            d = nesterov ? (gv + momentum * vbuf[i]) : vbuf[i]
            pv = pv - lr * (d + wd * pv)
        else
            pv = pv - lr * (gv + wd * pv)
        end
        p.storage[pi] = pv
    end
    return nothing
end

function _inplace_sgd!(
    p::Tensor,
    g::Tensor,
    opt::SGD,
)
    is_contiguous(p) && is_contiguous(g) || error("SGD: 参数与梯度需 contiguous")
    lr = opt.lr
    wd = opt.weight_decay
    if accelerator_storage(p.storage)
        vbuf = nothing
        if opt.momentum > 0
            id = objectid(p)
            n = numel(p)
            if !haskey(opt.vel, id)
                # 由扩展在 GPU 上分配；此处仅占位
                opt.vel[id] = dev_zeros(Float32, (n,), p.device)
            end
            vbuf = opt.vel[id]
        end
        _sgd_accelerator!(p, g, lr, wd, vbuf, opt.momentum, opt.dampening, opt.nesterov)
        return nothing
    end
    vbuf_cpu = nothing
    if opt.momentum > 0
        id = objectid(p)
        n = numel(p)
        if !haskey(opt.vel, id)
            opt.vel[id] = zeros(Float32, n)
        end
        vbuf_cpu = opt.vel[id]::Vector{Float32}
    end
    _sgd_cpu!(p, g, lr, wd, vbuf_cpu, opt.momentum, opt.dampening, opt.nesterov)
    return nothing
end

function step!(opt::SGD)
    for p in opt.params
        g = p.grad
        g === nothing && continue
        _inplace_sgd!(p, g, opt)
        zero_grad!(p)
    end
    return opt
end
