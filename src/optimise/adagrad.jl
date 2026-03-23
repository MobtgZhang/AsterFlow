mutable struct Adagrad
    params::Vector{Tensor}
    lr::Float32
    eps::Float32
    weight_decay::Float32
    state::Dict{UInt64,Any}
end

function Adagrad(params; lr = 1f-2, eps = 1f-10, weight_decay = 0f0)
    Adagrad(
        collect(Tensor, params),
        Float32(lr),
        Float32(eps),
        Float32(weight_decay),
        Dict{UInt64,Any}(),
    )
end

function step!(opt::Adagrad)
    for p in opt.params
        g = p.grad
        g === nothing && continue
        is_contiguous(p) && is_contiguous(g) || error("Adagrad: 参数与梯度需 contiguous")
        if accelerator_storage(p.storage)
            _adagrad_accelerator!(p, g, opt, opt.lr, opt.eps, opt.weight_decay)
            zero_grad!(p)
            continue
        end
        id = objectid(p)
        n = numel(p)
        if !haskey(opt.state, id)
            opt.state[id] = zeros(Float32, n)
        end
        svec = opt.state[id]::Vector{Float32}
        lr = opt.lr
        epsv = opt.eps
        wd = opt.weight_decay
        @inbounds for i in 1:n
            pi = p.offset + i
            gi = g.offset + i
            gv = Float32(g.storage[gi])
            pv = Float32(p.storage[pi])
            gv = gv + wd * pv
            svec[i] = svec[i] + gv * gv
            pv = pv - lr * gv / (sqrt(svec[i]) + epsv)
            p.storage[pi] = pv
        end
        zero_grad!(p)
    end
    return opt
end
