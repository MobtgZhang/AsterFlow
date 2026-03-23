"""Sophia-G 风格（对角 Hessian 用 `g^2` 估计并截断）：`theta = theta * (1 - lr*weight_decay) - lr * m / max(h, rho)`，`h` 为 `g^2` 的 EMA。"""
mutable struct Sophia
    params::Vector{Tensor}
    lr::Float32
    beta1::Float32
    beta2::Float32
    eps::Float32
    rho::Float32
    weight_decay::Float32
    m::Dict{UInt64,Any}
    h::Dict{UInt64,Any}
end

function Sophia(
    params;
    lr = 1f-3,
    beta1 = 9f-1,
    beta2 = 99f-2,
    eps = 1f-10,
    rho = 1f1,
    weight_decay = 0f0,
)
    Sophia(
        collect(Tensor, params),
        Float32(lr),
        Float32(beta1),
        Float32(beta2),
        Float32(eps),
        Float32(rho),
        Float32(weight_decay),
        Dict{UInt64,Any}(),
        Dict{UInt64,Any}(),
    )
end

function step!(opt::Sophia)
    b1 = opt.beta1
    b2 = opt.beta2
    for p in opt.params
        g = p.grad
        g === nothing && continue
        is_contiguous(p) && is_contiguous(g) || error("Sophia: 参数与梯度需 contiguous")
        accelerator_storage(p.storage) &&
            error("Sophia: 当前实现仅支持 CPU 张量。")
        id = objectid(p)
        n = numel(p)
        if !haskey(opt.m, id)
            opt.m[id] = zeros(Float32, n)
            opt.h[id] = zeros(Float32, n)
        end
        mvec = opt.m[id]::Vector{Float32}
        hvec = opt.h[id]::Vector{Float32}
        lr = opt.lr
        epsv = opt.eps
        rhov = opt.rho
        wd = opt.weight_decay
        @inbounds for i in 1:n
            pi = p.offset + i
            gi = g.offset + i
            gv = Float32(g.storage[gi])
            pv = Float32(p.storage[pi])
            pv = pv * (1f0 - lr * wd)
            mvec[i] = b1 * mvec[i] + (1f0 - b1) * gv
            g2 = gv * gv
            g2 = g2 > rhov ? rhov : g2
            hvec[i] = b2 * hvec[i] + (1f0 - b2) * g2
            denom = hvec[i] + epsv
            pv = pv - lr * mvec[i] / denom
            p.storage[pi] = pv
        end
        zero_grad!(p)
    end
    return opt
end
