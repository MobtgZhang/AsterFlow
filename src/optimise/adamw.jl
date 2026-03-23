mutable struct AdamW
    params::Vector{Tensor}
    lr::Float32
    beta1::Float32
    beta2::Float32
    eps::Float32
    wd::Float32
    t::Int
    m::Dict{UInt64,Any}
    v::Dict{UInt64,Any}
end

function AdamW(params; lr = 1f-3, beta1 = 9f-1, beta2 = 999f-3, eps = 1f-8, weight_decay = 1f-2)
    AdamW(
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

function step!(opt::AdamW)
    opt.t += 1
    t = Float32(opt.t)
    for p in opt.params
        g = p.grad
        g === nothing && continue
        is_contiguous(p) && is_contiguous(g) || error("AdamW: 参数与梯度需 contiguous")
        if accelerator_storage(p.storage)
            _adamw_accelerator!(
                p,
                g,
                opt,
                t,
                opt.lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                opt.wd,
            )
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
        lr = opt.lr
        b1 = opt.beta1
        b2 = opt.beta2
        eps = opt.eps
        wd = opt.wd
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
            pv = pv - lr * mhat / (sqrt(vhat) + eps)
            p.storage[pi] = pv
        end
        zero_grad!(p)
    end
    return opt
end
