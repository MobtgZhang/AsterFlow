mutable struct Dropout <: Module
    p::Float32
    training::Bool
end

Dropout(p::Real = 0.1f0) = Dropout(Float32(p), true)

function _bernoulli_keep_mask(::Type{T}, sz::Tuple{Vararg{Int}}, p::Float32, dev::Device) where {T}
    keep_h = T.(rand(T, sz...) .> p)
    return tensor_on_device(T, keep_h, dev; requires_grad = false)
end

function (m::Dropout)(x::Tensor{T,N}) where {T,N}
    (m.training && m.p > 0) || return x
    keep = _bernoulli_keep_mask(T, size(x), m.p, x.device)
    scale = T(1 / (1 - m.p))
    return mul_scalar_tensor(mul(x, keep), scale)
end
