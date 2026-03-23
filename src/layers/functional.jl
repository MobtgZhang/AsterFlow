# PyTorch 风格函数式 API（与 nn 层共享同一套算子）

relu(x::Tensor) = relu_tensor(x)
leaky_relu(x::Tensor, negative_slope::Real = 0.01f0) = leaky_relu_tensor(x, negative_slope)
tanh_act(x::Tensor) = tanh_tensor(x)
sigmoid_act(x::Tensor) = sigmoid_tensor(x)
gelu(x::Tensor) = GELU()(x)

function dropout(x::Tensor, p::Real; training::Bool = true)
    m = Dropout(Float32(p))
    m.training = training
    return m(x)
end

function functional_linear(x::Tensor, weight::Tensor, bias::Union{Nothing,Tensor} = nothing)
    h = matmul(x, weight)
    bias === nothing && return h
    B, O = size(h, 1), size(h, 2)
    bias2 = reshape_op(bias, (1, O))
    bias_bc = matmul(dev_ones(eltype(h), (B, 1), h.device), bias2)
    return add(h, bias_bc)
end
