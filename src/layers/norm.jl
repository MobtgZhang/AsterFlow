mutable struct LayerNorm <: Module
    normalized_shape::Int
    weight::Tensor{Float32,1}
    bias::Tensor{Float32,1}
    eps::Float32
    training::Bool
end

function LayerNorm(
    normalized_shape::Int;
    eps::Real = 1.0f-5,
    device::Device = CPUDevice(),
)
    w = tensor(ones(Float32, normalized_shape); device = device)
    b = tensor(zeros(Float32, normalized_shape); device = device)
    requires_grad!(w, true)
    requires_grad!(b, true)
    return LayerNorm(normalized_shape, w, b, Float32(eps), true)
end

function (ln::LayerNorm)(x::Tensor{T,2}) where {T}
    size(x, 2) == ln.normalized_shape || error("LayerNorm: last dim must match normalized_shape")
    B, D = size(x)
    dev = x.device
    μ = mean_tensor(x; dims = 2)
    μb = matmul(μ, dev_ones(T, (1, D), dev))
    xc = sub(x, μb)
    v = mean_tensor(mul_tensor(xc, xc); dims = 2)
    ve = dev_fill(T, size(v), T(ln.eps), dev)
    stdv = sqrt_tensor(add(v, ve))
    inv_std = div_tensor(ones_like(stdv), stdv)
    invb = matmul(inv_std, dev_ones(T, (1, D), dev))
    y = mul_tensor(xc, invb)
    γ2 = reshape_op(ln.weight, (1, D))
    β2 = reshape_op(ln.bias, (1, D))
    y2 = mul_tensor(y, matmul(dev_ones(T, (B, 1), dev), γ2))
    return add(y2, matmul(dev_ones(T, (B, 1), dev), β2))
end

mutable struct BatchNorm1d <: Module
    num_features::Int
    weight::Tensor{Float32,1}
    bias::Tensor{Float32,1}
    eps::Float32
    momentum::Float32
    running_mean::Tensor{Float32,1}
    running_var::Tensor{Float32,1}
    training::Bool
end

function BatchNorm1d(
    num_features::Int;
    eps::Real = 1.0f-5,
    momentum::Real = 0.1f0,
    device::Device = CPUDevice(),
)
    w = tensor(ones(Float32, num_features); device = device)
    b = tensor(zeros(Float32, num_features); device = device)
    rm = tensor(zeros(Float32, num_features); device = device, requires_grad = false)
    rv = tensor(ones(Float32, num_features); device = device, requires_grad = false)
    requires_grad!(w, true)
    requires_grad!(b, true)
    return BatchNorm1d(num_features, w, b, Float32(eps), Float32(momentum), rm, rv, true)
end

function (bn::BatchNorm1d)(x::Tensor{T,2}) where {T}
    size(x, 2) == bn.num_features || error("BatchNorm1d: feature dim mismatch")
    B, C = size(x)
    dev = x.device
    if bn.training
        μ = mean_tensor(x; dims = 1)
        xc = sub(x, matmul(dev_ones(T, (B, 1), dev), μ))
        v = mean_tensor(mul_tensor(xc, xc); dims = 1)
        ve = dev_fill(T, size(v), T(bn.eps), dev)
        stdv = sqrt_tensor(add(v, ve))
        inv_std = div_tensor(ones_like(stdv), stdv)
        invb = matmul(dev_ones(T, (B, 1), dev), inv_std)
        y = mul_tensor(xc, invb)
        γ2 = reshape_op(bn.weight, (1, C))
        β2 = reshape_op(bn.bias, (1, C))
        y2 = add(mul_tensor(y, matmul(dev_ones(T, (B, 1), dev), γ2)), matmul(dev_ones(T, (B, 1), dev), β2))
        no_grad() do
            rm_arr = to_array(bn.running_mean)
            rv_arr = to_array(bn.running_var)
            μa = vec(to_array(μ))
            va = vec(to_array(v))
            rm_arr .= (1 - bn.momentum) .* rm_arr .+ bn.momentum .* μa
            rv_arr .= (1 - bn.momentum) .* rv_arr .+ bn.momentum .* va
            bn.running_mean = tensor(rm_arr; device = dev, requires_grad = false)
            bn.running_var = tensor(rv_arr; device = dev, requires_grad = false)
        end
        return y2
    else
        μ = reshape_op(bn.running_mean, (1, C))
        σ2 = reshape_op(bn.running_var, (1, C))
        ve = dev_fill(T, size(σ2), T(bn.eps), dev)
        stdv = sqrt_tensor(add(σ2, ve))
        inv_std = div_tensor(ones_like(stdv), stdv)
        xc = sub(x, matmul(dev_ones(T, (B, 1), dev), μ))
        invb = matmul(dev_ones(T, (B, 1), dev), inv_std)
        y = mul_tensor(xc, invb)
        γ2 = reshape_op(bn.weight, (1, C))
        β2 = reshape_op(bn.bias, (1, C))
        return add(mul_tensor(y, matmul(dev_ones(T, (B, 1), dev), γ2)), matmul(dev_ones(T, (B, 1), dev), β2))
    end
end
