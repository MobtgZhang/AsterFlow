"""AdaFactor：二维参数用行/列因子近似二阶矩；一维参数退化为对角 RMSprop 式更新。默认 `beta1=0`（无动量）。仅 CPU。"""
mutable struct AdaFactor
    params::Vector{Tensor}
    lr::Float32
    beta2::Float32
    eps::Float32
    eps_scale::Float32
    weight_decay::Float32
    row_state::Dict{UInt64,Any}
    col_state::Dict{UInt64,Any}
    diag_state::Dict{UInt64,Any}
end

function AdaFactor(
    params;
    lr = 1f-3,
    beta1 = 0f0,
    beta2 = 999f-3,
    eps = 1f-30,
    eps_scale = 1f-3,
    weight_decay = 0f0,
)
    beta1 != 0f0 && error("AdaFactor: 本实现仅支持 beta1=0；请使用 beta1=0。")
    AdaFactor(
        collect(Tensor, params),
        Float32(lr),
        Float32(beta2),
        Float32(eps),
        Float32(eps_scale),
        Float32(weight_decay),
        Dict{UInt64,Any}(),
        Dict{UInt64,Any}(),
        Dict{UInt64,Any}(),
    )
end

function _adafactor_step_2d!(
    p::Tensor,
    g::Tensor,
    lr::Float32,
    b2::Float32,
    epsv::Float32,
    eps_scale::Float32,
    wd::Float32,
    row_state::Vector{Float32},
    col_state::Vector{Float32},
)
    R, C = Int(p.size[1]), Int(p.size[2])
    (length(row_state) == R && length(col_state) == C) ||
        error("AdaFactor: 行/列状态长度与参数形状不一致。")
    row_inst = Vector{Float32}(undef, R)
    col_inst = Vector{Float32}(undef, C)
    fill!(row_inst, 0f0)
    fill!(col_inst, 0f0)
    o = p.offset
    go = g.offset
    @inbounds for j in 1:C
        base = o + (j - 1) * R
        gbase = go + (j - 1) * R
        for i in 1:R
            gv = Float32(g.storage[gbase+i]) + wd * Float32(p.storage[base+i])
            s = gv * gv
            row_inst[i] += s
            col_inst[j] += s
        end
    end
    invC = 1f0 / Float32(C)
    invR = 1f0 / Float32(R)
    @inbounds for i in 1:R
        row_inst[i] *= invC
        row_state[i] = b2 * row_state[i] + (1f0 - b2) * row_inst[i]
    end
    @inbounds for j in 1:C
        col_inst[j] *= invR
        col_state[j] = b2 * col_state[j] + (1f0 - b2) * col_inst[j]
    end
    rmax = 0f0
    cmax = 0f0
    @inbounds for i in 1:R
        rmax = max(rmax, row_state[i])
    end
    @inbounds for j in 1:C
        cmax = max(cmax, col_state[j])
    end
    corr = eps_scale * sqrt(max(rmax * cmax, epsv))
    @inbounds for j in 1:C
        base = o + (j - 1) * R
        gbase = go + (j - 1) * R
        for i in 1:R
            gv = Float32(g.storage[gbase+i]) + wd * Float32(p.storage[base+i])
            denom = sqrt(row_state[i] * col_state[j]) + corr
            pv = Float32(p.storage[base+i]) - lr * gv / denom
            p.storage[base+i] = pv
        end
    end
    return nothing
end

function step!(opt::AdaFactor)
    b2 = opt.beta2
    for p in opt.params
        g = p.grad
        g === nothing && continue
        is_contiguous(p) && is_contiguous(g) || error("AdaFactor: 参数与梯度需 contiguous")
        accelerator_storage(p.storage) &&
            error("AdaFactor: 当前实现仅支持 CPU 张量。")
        id = objectid(p)
        n = numel(p)
        lr = opt.lr
        epsv = opt.eps
        eps_scale = opt.eps_scale
        wd = opt.weight_decay
        if ndims(p) == 2
            R, C = Int(p.size[1]), Int(p.size[2])
            if !haskey(opt.row_state, id)
                opt.row_state[id] = zeros(Float32, R)
                opt.col_state[id] = zeros(Float32, C)
            end
            rs = opt.row_state[id]::Vector{Float32}
            cs = opt.col_state[id]::Vector{Float32}
            _adafactor_step_2d!(p, g, lr, b2, epsv, eps_scale, wd, rs, cs)
        else
            if !haskey(opt.diag_state, id)
                opt.diag_state[id] = zeros(Float32, n)
            end
            dvec = opt.diag_state[id]::Vector{Float32}
            @inbounds for i in 1:n
                pi = p.offset + i
                gi = g.offset + i
                gv = Float32(g.storage[gi])
                pv = Float32(p.storage[pi])
                gv = gv + wd * pv
                dvec[i] = b2 * dvec[i] + (1f0 - b2) * (gv * gv)
                denom = sqrt(dvec[i]) + eps_scale * sqrt(dvec[i] + epsv)
                pv = pv - lr * gv / denom
                p.storage[pi] = pv
            end
        end
        zero_grad!(p)
    end
    return opt
end
