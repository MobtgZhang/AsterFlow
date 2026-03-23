"""Lookahead（Zhang et al.）：每 `k` 次对内层优化步后，将慢权重向快权重插值并写回参数。`base` 须为带 `params` 与可调用 `step!` 的优化器。仅 CPU 参数。"""
mutable struct Lookahead
    base::Any
    k::Int
    alpha::Float32
    slow::Dict{UInt64,Vector{Float32}}
    step_count::Int
end

function Lookahead(base; k::Integer = 5, alpha = 5f-1)
    k >= 1 || error("Lookahead: k 须 >= 1")
    slow = Dict{UInt64,Vector{Float32}}()
    for p in base.params
        accelerator_storage(p.storage) &&
            error("Lookahead: 当前实现仅支持 CPU 张量。")
        id = objectid(p)
        n = numel(p)
        buf = Vector{Float32}(undef, n)
        @inbounds for i in 1:n
            buf[i] = Float32(p.storage[p.offset+i])
        end
        slow[id] = buf
    end
    Lookahead(base, Int(k), Float32(alpha), slow, 0)
end

function step!(opt::Lookahead)
    step!(opt.base)
    opt.step_count += 1
    if opt.step_count % opt.k != 0
        return opt
    end
    α = opt.alpha
    for p in opt.base.params
        accelerator_storage(p.storage) &&
            error("Lookahead: 当前实现仅支持 CPU 张量。")
        id = objectid(p)
        n = numel(p)
        if !haskey(opt.slow, id)
            buf = Vector{Float32}(undef, n)
            @inbounds for i in 1:n
                buf[i] = Float32(p.storage[p.offset+i])
            end
            opt.slow[id] = buf
        end
        s = opt.slow[id]
        @inbounds for i in 1:n
            pi = p.offset + i
            pv = Float32(p.storage[pi])
            si = s[i]
            si = si + α * (pv - si)
            s[i] = si
            p.storage[pi] = si
        end
    end
    return opt
end
