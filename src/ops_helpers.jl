ones_like(t::Tensor{T}) where {T} = dev_fill(T, size(t), one(T), device(t))

function accumulate_grad!(t::Tensor, g::Tensor)
    t.requires_grad || return
    t.grad = no_grad() do
        if t.grad === nothing
            g
        else
            dispatch_op(:add, t.device, t.grad, g)
        end
    end
    return nothing
end

function mul_tensor(a::Tensor, b::Tensor)
    no_grad() do
        dispatch_op(:mul, a.device, a, b)
    end
end

function div_op_tensor(a::Tensor, b::Tensor)
    no_grad() do
        dispatch_op(:div, a.device, a, b)
    end
end

function matmul_ng(a::Tensor{T,2}, b::Tensor{T,2}) where {T}
    no_grad() do
        dispatch_op(:matmul, a.device, a, b)
    end
end

function mul_scalar_tensor(t::Tensor{T}, s::Real) where {T}
    no_grad() do
        dispatch_op(:scale, t.device, t, T(s))
    end
end

function relu_backward_mul(grad::Tensor, inp::Tensor)
    no_grad() do
        dispatch_op(:relu_bwd, inp.device, grad, inp)
    end
end

function leaky_relu_backward_mul(grad::Tensor, inp::Tensor, negative_slope::Real)
    no_grad() do
        dispatch_op(:leaky_relu_bwd, inp.device, grad, inp, Float32(negative_slope))
    end
end

function softmax_rows_backward_nograd(prob::Tensor, grad::Tensor)
    no_grad() do
        dispatch_op(:softmax_rows_bwd, prob.device, prob, grad)
    end
end
