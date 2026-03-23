mutable struct Adamax
    params::Vector{Tensor}
    lr::Float32
    beta1::Float32
    beta2::Float32
    eps::Float32
    weight_decay::Float32
    t::Int
    m::Dict{UInt64,Any}
    u::Dict{UInt64,Any}
end

function Adamax(
    params;
    lr = 2f-3,
    beta1 = 9f-1,
    beta2 = 999f-3,
    eps = 1f-8,
    weight_decay = 0f0,
)
    Adamax(
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

function step!(opt::Adamax)
    opt.t += 1
    t = Float32(opt.t)
    for p in opt.params
        g = p.grad
        g === nothing && continue
        is_contiguous(p) && is_contiguous(g) || error("Adamax: 参数与梯度需 contiguous")
        if accelerator_storage(p.storage)
            _adamax_accelerator!(
                p,
                g,
                opt,
                t,
                opt.lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                opt.weight_decay,
            )
            zero_grad!(p)
            continue
        end
        id = objectid(p)
        n = numel(p)
        if !haskey(opt.m, id)
            opt.m[id] = zeros(Float32, n)
            opt.u[id] = zeros(Float32, n)
        end
        mvec = opt.m[id]::Vector{Float32}
        uvec = opt.u[id]::Vector{Float32}
        lr = opt.lr
        b1 = opt.beta1
        b2 = opt.beta2
        epsv = opt.eps
        wd = opt.weight_decay
        @inbounds for i in 1:n
            pi = p.offset + i
            gi = g.offset + i
            gv = Float32(g.storage[gi])
            pv = Float32(p.storage[pi])
            gv = gv + wd * pv
            mvec[i] = b1 * mvec[i] + (1f0 - b1) * gv
            uvec[i] = max(b2 * uvec[i], abs(gv))
            mhat = mvec[i] / (1f0 - b1^t)
            pv = pv - lr * mhat / (uvec[i] + epsv)
            p.storage[pi] = pv
        end
        zero_grad!(p)
    end
    return opt
end
