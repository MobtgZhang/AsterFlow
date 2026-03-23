mutable struct Flatten <: Module
    training::Bool
end

Flatten() = Flatten(true)

function (m::Flatten)(x::Tensor{T,N}) where {T,N}
    b = size(x, 1)
    r = numel(x) ÷ b
    return reshape_op(x, (b, r))
end
