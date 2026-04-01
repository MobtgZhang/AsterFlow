"""
均方误差。`reduction=:mean`（默认）返回 1×1 标量张量；`:sum` 为总和；`:none` 为逐元素 `(yhat-y)²`。
"""
function mse_loss(yhat::Tensor, y::Tensor; reduction::Symbol = :mean)
    d = sub(yhat, y)
    s = mul(d, d)
    if reduction === :none
        return s
    end
    total = sum_tensor(s)
    if reduction === :sum
        return total
    end
    reduction === :mean || error("mse_loss: reduction 须为 :mean / :sum / :none")
    invn = tensor(fill(eltype(yhat)(1 / numel(s)), 1, 1); device = yhat.device, requires_grad = false)
    return mul(total, invn)
end

function l1_loss(yhat::Tensor, y::Tensor; reduction::Symbol = :mean)
    d = sub(yhat, y)
    ad = sqrt_tensor(mul_tensor(d, d))
    if reduction === :none
        return ad
    end
    if reduction === :sum
        return sum_tensor(ad)
    end
    reduction === :mean || error("l1_loss: reduction 须为 :mean / :sum / :none")
    return mean_tensor(ad; dims = nothing)
end

"""
`logits`: `(B, C)`；`targets`: 长度 `B`，类别下标为 **1..C**（Julia 下标约定）。
数值稳定：先 `softmax` 再 log；与 `logsoftmax`+`nll` 路线等价时可后续融合。
"""
function cross_entropy_loss(
    logits::Tensor{T,2},
    targets::AbstractVector{Int};
    reduction::Symbol = :mean,
) where {T}
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
    if reduction === :mean
        acc /= B
    elseif reduction === :sum
    elseif reduction === :none
        error("cross_entropy_loss: :none 尚未实现（需逐样本向量损失）")
    else
        error("cross_entropy_loss: reduction 须为 :mean 或 :sum")
    end
    out = tensor(fill(T(acc), 1, 1); device = logits.device, requires_grad = false)
    if grad_enabled() && logits.requires_grad
        out.requires_grad = true
        out.grad_fn = CrossEntropyBackward(logits, P, Vector{Int}(targets), tensor_version(logits))
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
        out.grad_fn = NLLBackward(log_probs, Vector{Int}(targets), tensor_version(log_probs))
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

(m::CrossEntropyLoss)(logits::Tensor, targets::AbstractVector{Int}; kwargs...) =
    cross_entropy_loss(logits, targets; kwargs...)
