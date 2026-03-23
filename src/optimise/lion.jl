"""Lion（Chen et al.）：符号更新 + 动量；`weight_decay` 为解耦式（与 AdamW 相同，先乘 `1 - lr*weight_decay`）。"""
mutable struct Lion
    params::Vector{Tensor}
    lr::Float32
    beta1::Float32
    beta2::Float32
    weight_decay::Float32
    m::Dict{UInt64,Any}
end

function Lion(params; lr = 1f-4, beta1 = 9f-1, beta2 = 99f-2, weight_decay = 0f0)
    Lion(
        collect(Tensor, params),
        Float32(lr),
        Float32(beta1),
        Float32(beta2),
        Float32(weight_decay),
        Dict{UInt64,Any}(),
    )
end

@inline function _signf(x::Float32)
    return x > 0f0 ? 1f0 : (x < 0f0 ? -1f0 : 0f0)
end

function step!(opt::Lion)
    for p in opt.params
        g = p.grad
        g === nothing && continue
        is_contiguous(p) && is_contiguous(g) || error("Lion: 参数与梯度需 contiguous")
        accelerator_storage(p.storage) &&
            error("Lion: 当前实现仅支持 CPU 张量。")
        id = objectid(p)
        n = numel(p)
        if !haskey(opt.m, id)
            opt.m[id] = zeros(Float32, n)
        end
        mvec = opt.m[id]::Vector{Float32}
        lr = opt.lr
        b1 = opt.beta1
        b2 = opt.beta2
        wd = opt.weight_decay
        @inbounds for i in 1:n
            pi = p.offset + i
            gi = g.offset + i
            gv = Float32(g.storage[gi])
            pv = Float32(p.storage[pi])
            pv = pv * (1f0 - lr * wd)
            upd = b1 * mvec[i] + (1f0 - b1) * gv
            pv = pv - lr * _signf(upd)
            mvec[i] = b2 * mvec[i] + (1f0 - b2) * gv
            p.storage[pi] = pv
        end
        zero_grad!(p)
    end
    return opt
end
