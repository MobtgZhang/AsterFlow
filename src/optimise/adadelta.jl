mutable struct Adadelta
    params::Vector{Tensor}
    ρ::Float32
    eps::Float32
    weight_decay::Float32
    acc_grad::Dict{UInt64,Any}
    acc_delta::Dict{UInt64,Any}
end

function Adadelta(params; ρ = 9f-1, eps = 1f-6, weight_decay = 0f0)
    Adadelta(
        collect(Tensor, params),
        Float32(ρ),
        Float32(eps),
        Float32(weight_decay),
        Dict{UInt64,Any}(),
        Dict{UInt64,Any}(),
    )
end

function step!(opt::Adadelta)
    ρ = opt.ρ
    epsv = opt.eps
    wd = opt.weight_decay
    for p in opt.params
        g = p.grad
        g === nothing && continue
        is_contiguous(p) && is_contiguous(g) || error("Adadelta: 参数与梯度需 contiguous")
        if accelerator_storage(p.storage)
            _adadelta_accelerator!(p, g, opt, ρ, epsv, wd)
            zero_grad!(p)
            continue
        end
        id = objectid(p)
        n = numel(p)
        if !haskey(opt.acc_grad, id)
            opt.acc_grad[id] = zeros(Float32, n)
            opt.acc_delta[id] = zeros(Float32, n)
        end
        ag = opt.acc_grad[id]::Vector{Float32}
        ad = opt.acc_delta[id]::Vector{Float32}
        @inbounds for i in 1:n
            pi = p.offset + i
            gi = g.offset + i
            gv = Float32(g.storage[gi])
            pv = Float32(p.storage[pi])
            gv = gv + wd * pv
            ag[i] = ρ * ag[i] + (1f0 - ρ) * (gv * gv)
            upd = sqrt(ad[i] + epsv) / sqrt(ag[i] + epsv) * gv
            ad[i] = ρ * ad[i] + (1f0 - ρ) * (upd * upd)
            pv = pv - upd
            p.storage[pi] = pv
        end
        zero_grad!(p)
    end
    return opt
end
