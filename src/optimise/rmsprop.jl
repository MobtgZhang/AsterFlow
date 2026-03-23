mutable struct RMSprop
    params::Vector{Tensor}
    lr::Float32
    α::Float32
    eps::Float32
    weight_decay::Float32
    centered::Bool
    state::Dict{UInt64,Any}
    grad_avg::Dict{UInt64,Any}
end

function RMSprop(
    params;
    lr = 1f-3,
    α = 99f-2,
    eps = 1f-8,
    weight_decay = 0f0,
    centered = false,
)
    RMSprop(
        collect(Tensor, params),
        Float32(lr),
        Float32(α),
        Float32(eps),
        Float32(weight_decay),
        centered,
        Dict{UInt64,Any}(),
        Dict{UInt64,Any}(),
    )
end

function step!(opt::RMSprop)
    for p in opt.params
        g = p.grad
        g === nothing && continue
        is_contiguous(p) && is_contiguous(g) || error("RMSprop: 参数与梯度需 contiguous")
        if accelerator_storage(p.storage)
            _rmsprop_accelerator!(
                p,
                g,
                opt,
                opt.lr,
                opt.α,
                opt.eps,
                opt.weight_decay,
                opt.centered,
            )
            zero_grad!(p)
            continue
        end
        id = objectid(p)
        n = numel(p)
        if !haskey(opt.state, id)
            opt.state[id] = zeros(Float32, n)
            if opt.centered
                opt.grad_avg[id] = zeros(Float32, n)
            end
        end
        svec = opt.state[id]::Vector{Float32}
        α = opt.α
        lr = opt.lr
        epsv = opt.eps
        wd = opt.weight_decay
        gavg = opt.centered ? opt.grad_avg[id]::Vector{Float32} : nothing
        @inbounds for i in 1:n
            pi = p.offset + i
            gi = g.offset + i
            gv = Float32(g.storage[gi])
            pv = Float32(p.storage[pi])
            gv = gv + wd * pv
            svec[i] = α * svec[i] + (1f0 - α) * (gv * gv)
            if opt.centered
                gavg[i] = α * gavg[i] + (1f0 - α) * gv
                denom = sqrt(max(svec[i] - gavg[i] * gavg[i], 0f0)) + epsv
                pv = pv - lr * gv / denom
            else
                pv = pv - lr * gv / (sqrt(svec[i]) + epsv)
            end
            p.storage[pi] = pv
        end
        zero_grad!(p)
    end
    return opt
end
