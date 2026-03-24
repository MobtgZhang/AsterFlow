# 公式对齐 PyTorch AdamW：m = b1*m + (1-b1)*g；v = b2*v + (1-b2)*g²；
# θ = θ*(1 - lr*wd) - lr * (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps)

struct AdamWGroup
    params::Vector{Tensor}
    lr::Float32
    weight_decay::Float32
end

function AdamWGroup(params; lr = 1f-3, weight_decay = 1f-2)
    return AdamWGroup(collect(Tensor, params), Float32(lr), Float32(weight_decay))
end

"""
`AdamW`：与 PyTorch `AdamW` 同公式（解耦 weight decay）。

- 单组：`AdamW(params(m); lr=..., weight_decay=...)`
- 多组：`AdamW([AdamWGroup(ps1; lr=1f-3), AdamWGroup(ps2; lr=1f-4)]; beta1=...)`
"""
mutable struct AdamW
    groups::Vector{AdamWGroup}
    beta1::Float32
    beta2::Float32
    eps::Float32
    t::Int
    m::Dict{UInt64,Any}
    v::Dict{UInt64,Any}
end

function AdamW(params; lr = 1f-3, beta1 = 9f-1, beta2 = 999f-3, eps = 1f-8, weight_decay = 1f-2)
    return AdamW(
        [AdamWGroup(params; lr = lr, weight_decay = weight_decay)],
        Float32(beta1),
        Float32(beta2),
        Float32(eps),
        0,
        Dict{UInt64,Any}(),
        Dict{UInt64,Any}(),
    )
end

function AdamW(groups::Vector{AdamWGroup}; beta1 = 9f-1, beta2 = 999f-3, eps = 1f-8)
    return AdamW(groups, Float32(beta1), Float32(beta2), Float32(eps), 0, Dict(), Dict())
end

## 与旧代码兼容：`opt.params` 展开为所有组内参数
function Base.propertynames(::AdamW, private = false)
    return (:groups, :beta1, :beta2, :eps, :t, :m, :v, :params)
end

function Base.getproperty(opt::AdamW, name::Symbol)
    name === :params && return vcat([g.params for g in opt.groups]...)
    return getfield(opt, name)
end

function step!(opt::AdamW)
    opt.t += 1
    t = Float32(opt.t)
    for grp in opt.groups
        lr = grp.lr
        wd = grp.weight_decay
        for p in grp.params
            g = p.grad
            g === nothing && continue
            is_contiguous(p) && is_contiguous(g) || error("AdamW: 参数与梯度需 contiguous")
            if accelerator_storage(p.storage)
                _adamw_accelerator!(p, g, opt, t, lr, opt.beta1, opt.beta2, opt.eps, wd)
                zero_grad!(p)
                continue
            end
            id = objectid(p)
            n = numel(p)
            if !haskey(opt.m, id)
                opt.m[id] = zeros(Float32, n)
                opt.v[id] = zeros(Float32, n)
            end
            mvec = opt.m[id]::Vector{Float32}
            vvec = opt.v[id]::Vector{Float32}
            b1 = opt.beta1
            b2 = opt.beta2
            epsv = opt.eps
            @inbounds for i in 1:n
                pi = p.offset + i
                gi = g.offset + i
                gv = Float32(g.storage[gi])
                pv = Float32(p.storage[pi])
                mvec[i] = b1 * mvec[i] + (1f0 - b1) * gv
                vvec[i] = b2 * vvec[i] + (1f0 - b2) * (gv * gv)
                mhat = mvec[i] / (1f0 - b1^t)
                vhat = vvec[i] / (1f0 - b2^t)
                pv = pv * (1f0 - lr * wd)
                pv = pv - lr * mhat / (sqrt(vhat) + epsv)
                p.storage[pi] = pv
            end
            zero_grad!(p)
        end
    end
    return opt
end
