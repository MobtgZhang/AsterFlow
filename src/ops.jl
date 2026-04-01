## 公开算子：经 Dispatcher，并在 GradMode 下挂 Node

function add(a::Tensor, b::Tensor)
    out = dispatch_op(:add, a.device, a, b)
    trace_try_record!(:add, (a, b), out)
    if grad_enabled() && (a.requires_grad || b.requires_grad)
        out.requires_grad = true
        out.grad_fn = AddBackward(a, b, tensor_version(a), tensor_version(b))
    end
    return out
end

function sub(a::Tensor, b::Tensor)
    out = dispatch_op(:sub, a.device, a, b)
    if grad_enabled() && (a.requires_grad || b.requires_grad)
        out.requires_grad = true
        out.grad_fn = SubBackward(a, b, tensor_version(a), tensor_version(b))
    end
    return out
end

function mul(a::Tensor, b::Tensor)
    out = dispatch_op(:mul, a.device, a, b)
    if grad_enabled() && (a.requires_grad || b.requires_grad)
        out.requires_grad = true
        out.grad_fn = MulBackward(a, b, a, b, tensor_version(a), tensor_version(b))
    end
    return out
end

function div_op(a::Tensor, b::Tensor)
    dispatch_op(:div, a.device, a, b)
end

function div_tensor(a::Tensor, b::Tensor)
    out = dispatch_op(:div, a.device, a, b)
    if grad_enabled() && (a.requires_grad || b.requires_grad)
        out.requires_grad = true
        out.grad_fn = DivBackward(a, b, tensor_version(a), tensor_version(b))
    end
    return out
end

function matmul(a::Tensor{T,2}, b::Tensor{T,2}) where {T}
    out = dispatch_op(:matmul, a.device, a, b)
    trace_try_record!(:matmul, (a, b), out)
    if grad_enabled() && (a.requires_grad || b.requires_grad)
        out.requires_grad = true
        out.grad_fn = MatMulBackward(a, b, tensor_version(a), tensor_version(b))
    end
    return out
end

function sum_tensor(a::Tensor; dims = nothing)
    out = dispatch_op(:sum, a.device, a; dims = dims)
    if grad_enabled() && a.requires_grad
        out.requires_grad = true
        out.grad_fn = SumBackward(a, dims, tensor_version(a))
    end
    return out
end

function mean_tensor(a::Tensor; dims = nothing)
    out = dispatch_op(:mean, a.device, a; dims = dims)
    if grad_enabled() && a.requires_grad
        if dims === nothing
            invn = eltype(a)(1 / numel(a))
            out.requires_grad = true
            out.grad_fn = MeanBackward(a, invn, tensor_version(a))
        elseif ndims(a) == 2
            d = dims isa Integer ? Int(dims) : (dims isa Tuple && length(dims) == 1 ? Int(dims[1]) : nothing)
            if d !== nothing && (d == 1 || d == 2)
                invn = eltype(a)(1 / size(a, d))
                out.requires_grad = true
                out.grad_fn = MeanDimsBackward(a, d, invn, tensor_version(a))
            end
        end
    end
    return out
end

function scale_tensor(a::Tensor, s::Real)
    T = eltype(a)
    out = dispatch_op(:scale, a.device, a, T(s))
    if grad_enabled() && a.requires_grad
        out.requires_grad = true
        out.grad_fn = ScaleBackward(a, T(s), tensor_version(a))
    end
    return out
end

function tanh_tensor(a::Tensor)
    out = dispatch_op(:tanh, a.device, a)
    if grad_enabled() && a.requires_grad
        out.requires_grad = true
        out.grad_fn = TanhBackward(a, out, tensor_version(a))
    end
    return out
end

function sigmoid_tensor(a::Tensor)
    out = dispatch_op(:sigmoid, a.device, a)
    if grad_enabled() && a.requires_grad
        out.requires_grad = true
        out.grad_fn = SigmoidBackward(a, out, tensor_version(a))
    end
    return out
end

function leaky_relu_tensor(a::Tensor, negative_slope::Real = 0.01f0)
    T = eltype(a)
    α = T(negative_slope)
    out = dispatch_op(:leaky_relu, a.device, a, α)
    if grad_enabled() && a.requires_grad
        out.requires_grad = true
        out.grad_fn = LeakyReLUBackward(a, Float32(α), tensor_version(a))
    end
    return out
end

function exp_tensor(a::Tensor)
    out = dispatch_op(:exp, a.device, a)
    if grad_enabled() && a.requires_grad
        out.requires_grad = true
        out.grad_fn = ExpBackward(a, out, tensor_version(a))
    end
    return out
end

function log_tensor(a::Tensor)
    out = dispatch_op(:log, a.device, a)
    if grad_enabled() && a.requires_grad
        out.requires_grad = true
        out.grad_fn = LogBackward(a, tensor_version(a))
    end
    return out
end

function sqrt_tensor(a::Tensor)
    out = dispatch_op(:sqrt, a.device, a)
    if grad_enabled() && a.requires_grad
        out.requires_grad = true
        out.grad_fn = SqrtBackward(a, out, tensor_version(a))
    end
    return out
end

function relu_tensor(a::Tensor)
    out = dispatch_op(:relu, a.device, a)
    if grad_enabled() && a.requires_grad
        out.requires_grad = true
        out.grad_fn = ReLUBackward(a, tensor_version(a))
    end
    return out
end

function softmax_rows(a::Tensor{T,2}) where {T}
    out = dispatch_op(:softmax_rows, a.device, a)
    if grad_enabled() && a.requires_grad
        out.requires_grad = true
        out.grad_fn = SoftmaxRowsBackward(a, out, tensor_version(a))
    end
    return out
end

reshape_op(t::Tensor, sz::Tuple{Vararg{Int}}) = reshape_tensor(t, sz)
