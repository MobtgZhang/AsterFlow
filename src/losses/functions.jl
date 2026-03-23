"""
均方误差，返回标量张量（1×1），用于 `backward`。
"""
function mse_loss(yhat::Tensor, y::Tensor)
    d = sub(yhat, y)
    s = mul(d, d)
    total = sum_tensor(s)
    invn = tensor(fill(eltype(yhat)(1 / numel(s)), 1, 1); device = yhat.device, requires_grad = false)
    return mul(total, invn)
end

function l1_loss(yhat::Tensor, y::Tensor)
    d = sub(yhat, y)
    ad = sqrt_tensor(mul_tensor(d, d))
    return mean_tensor(ad; dims = nothing)
end

"""
`logits`: `(B, C)`；`targets`: 长度 `B`，类别下标为 **1..C**（Julia 下标约定）。
"""
function cross_entropy_loss(logits::Tensor{T,2}, targets::AbstractVector{Int}) where {T}
    P = softmax_rows(logits)
    B, C = size(P)
    length(targets) == B || error("cross_entropy_loss: batch 维与 targets 长度不一致")
    pa = to_array(P)
    acc = zero(Float32)
    epsv = Float32(1.0e-8)
    for i in 1:B
        ti = targets[i]
        (1 <= ti <= C) || error("cross_entropy_loss: target 需在 1..C")
        acc -= log(max(pa[i, ti], epsv))
    end
    acc /= B
    out = tensor(fill(T(acc), 1, 1); device = logits.device, requires_grad = false)
    if grad_enabled() && logits.requires_grad
        out.requires_grad = true
        out.grad_fn = CrossEntropyBackward(logits, P, Vector{Int}(targets))
    end
    return out
end

function nll_loss(log_probs::Tensor{T,2}, targets::AbstractVector{Int}) where {T}
    B, C = size(log_probs)
    length(targets) == B || error("nll_loss: batch 不一致")
    la = to_array(log_probs)
    acc = zero(Float32)
    for i in 1:B
        ti = targets[i]
        (1 <= ti <= C) || error("nll_loss: target 需在 1..C")
        acc -= la[i, ti]
    end
    acc /= B
    out = tensor(fill(T(acc), 1, 1); device = log_probs.device, requires_grad = false)
    if grad_enabled() && log_probs.requires_grad
        out.requires_grad = true
        out.grad_fn = NLLBackward(log_probs, Vector{Int}(targets))
    end
    return out
end

mutable struct MSE <: Module
    training::Bool
end

MSE() = MSE(true)

function (m::MSE)(yhat::Tensor, y::Tensor)
    mse_loss(yhat, y)
end

mutable struct L1Loss <: Module
    training::Bool
end

L1Loss() = L1Loss(true)

(m::L1Loss)(yhat::Tensor, y::Tensor) = l1_loss(yhat, y)

mutable struct CrossEntropyLoss <: Module
    training::Bool
end

CrossEntropyLoss() = CrossEntropyLoss(true)

(m::CrossEntropyLoss)(logits::Tensor, targets::AbstractVector{Int}) = cross_entropy_loss(logits, targets)
