"""Rectified Adam（Liu et al.）：在自适应学习率方差不可靠的早期减小步长，与 PyTorch `RAdam` 的 rectification 行为一致；`weight_decay` 为 Adam 式 L2（加在梯度上）。"""
mutable struct RAdam
    params::Vector{Tensor}
    lr::Float32
    beta1::Float32
    beta2::Float32
    eps::Float32
    weight_decay::Float32
    t::Int
    m::Dict{UInt64,Any}
    v::Dict{UInt64,Any}
end

function RAdam(
    params;
    lr = 1f-3,
    beta1 = 9f-1,
    beta2 = 999f-3,
    eps = 1f-8,
    weight_decay = 0f0,
)
    RAdam(
        collect(Tensor, params),
        Float32(lr),
        Float32(beta1),
        Float32(beta2),
        Float32(eps),
        Float32(weight_decay),
        0,
        Dict{UInt64,Any}(),
        Dict{UInt64,Any}(),
    )
end

function step!(opt::RAdam)
    opt.t += 1
    t = Float32(opt.t)
    b1 = opt.beta1
    b2 = opt.beta2
    rho_inf = 2f0 / (1f0 - b2) - 1f0
    b1t = b1^t
    b2t = b2^t
    rho_t = rho_inf - 2f0 * t * b2t / (1f0 - b2t)
    for p in opt.params
        g = p.grad
        g === nothing && continue
        is_contiguous(p) && is_contiguous(g) || error("RAdam: 参数与梯度需 contiguous")
        accelerator_storage(p.storage) &&
            error("RAdam: 当前实现仅支持 CPU 张量。")
        id = objectid(p)
        n = numel(p)
        if !haskey(opt.m, id)
            opt.m[id] = zeros(Float32, n)
            opt.v[id] = zeros(Float32, n)
        end
        mvec = opt.m[id]::Vector{Float32}
        vvec = opt.v[id]::Vector{Float32}
        lr = opt.lr
        epsv = opt.eps
        wd = opt.weight_decay
        @inbounds for i in 1:n
            pi = p.offset + i
            gi = g.offset + i
            gv = Float32(g.storage[gi])
            pv = Float32(p.storage[pi])
            gv = gv + wd * pv
            mvec[i] = b1 * mvec[i] + (1f0 - b1) * gv
            vvec[i] = b2 * vvec[i] + (1f0 - b2) * (gv * gv)
            mhat = mvec[i] / (1f0 - b1t)
            if rho_t > 4f0
                rt = sqrt(vvec[i] / (1f0 - b2t))
                l_t = sqrt(
                    (rho_t - 4f0) *
                    (rho_t - 2f0) *
                    rho_inf /
                    ((rho_inf - 4f0) * (rho_inf - 2f0) * rho_t),
                )
                pv = pv - lr * l_t * mhat / (rt + epsv)
            else
                pv = pv - lr * mhat
            end
            p.storage[pi] = pv
        end
        zero_grad!(p)
    end
    return opt
end
