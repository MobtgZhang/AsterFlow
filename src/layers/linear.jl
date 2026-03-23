mutable struct Linear <: Module
    weight::Tensor{Float32,2}
    bias::Tensor{Float32,1}
    training::Bool
end

function Linear(
    in_features::Int,
    out_features::Int;
    dtype::Type{<:AbstractFloat} = Float32,
    device::Device = CPUDevice(),
)
    w = tensor(randn(dtype, in_features, out_features) .* dtype(0.02); device = device)
    b = tensor(zeros(dtype, out_features); device = device)
    requires_grad!(w, true)
    requires_grad!(b, true)
    return Linear(w, b, true)
end

function (m::Linear)(x::Tensor{T,2}) where {T}
    h = matmul(x, m.weight)
    ## bias broadcast: (B,Out) + (Out,) — expand bias to row
    B, O = size(h, 1), size(h, 2)
    bias2 = reshape_op(m.bias, (1, O))
    ones_b = dev_ones(Float32, (B, 1), h.device)
    bias_bc = matmul(ones_b, bias2)
    return add(h, bias_bc)
end
