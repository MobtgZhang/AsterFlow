mutable struct Identity <: Module
    training::Bool
end

Identity() = Identity(true)

(m::Identity)(x) = x

mutable struct ReLU <: Module
    training::Bool
end

ReLU() = ReLU(true)

function (m::ReLU)(x::Tensor)
    relu_tensor(x)
end

mutable struct LeakyReLU <: Module
    negative_slope::Float32
    training::Bool
end

LeakyReLU(negative_slope::Real = 0.01f0) = LeakyReLU(Float32(negative_slope), true)

function (m::LeakyReLU)(x::Tensor)
    leaky_relu_tensor(x, m.negative_slope)
end

mutable struct Tanh <: Module
    training::Bool
end

Tanh() = Tanh(true)

(m::Tanh)(x::Tensor) = tanh_tensor(x)

mutable struct Sigmoid <: Module
    training::Bool
end

Sigmoid() = Sigmoid(true)

(m::Sigmoid)(x::Tensor) = sigmoid_tensor(x)

mutable struct GELU <: Module
    training::Bool
end

GELU() = GELU(true)

function (m::GELU)(x::Tensor)
    T = eltype(x)
    k1 = T(sqrt(2 / pi))
    k2 = T(0.044715)
    x3 = mul_tensor(mul_tensor(x, x), x)
    inner = add(x, scale_tensor(x3, k2))
    t = tanh_tensor(scale_tensor(inner, k1))
    one_t = dev_fill(T, size(t), one(T), x.device)
    return mul_tensor(scale_tensor(x, T(0.5)), add(one_t, t))
end

mutable struct Softmax <: Module
    dim::Int
    training::Bool
end

Softmax(dim::Int = 2) = Softmax(dim, true)

function (m::Softmax)(x::Tensor{T,2}) where {T}
    m.dim == 2 || error("Softmax: 仅支持对最后一维（dim=2）的 2D 输入")
    softmax_rows(x)
end

mutable struct LogSoftmax <: Module
    dim::Int
    training::Bool
end

LogSoftmax(dim::Int = 2) = LogSoftmax(dim, true)

function (m::LogSoftmax)(x::Tensor{T,2}) where {T}
    m.dim == 2 || error("LogSoftmax: 仅支持 dim=2")
    log_tensor(softmax_rows(x))
end
