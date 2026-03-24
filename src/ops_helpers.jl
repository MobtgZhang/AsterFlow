ones_like(t::Tensor{T}) where {T} = dev_fill(T, size(t), one(T), device(t))

"""将 `grad_out` 的 shape 缩回 `target_sz`（broadcast 反传）。"""
function sum_grad_to_shape(grad_out::Tensor, target_sz::Tuple{Vararg{Int}})
    if size(grad_out) == target_sz
        return grad_out
    end
    arr = _sum_grad_array_to_shape(to_array(grad_out), target_sz)
    return tensor_on_device(eltype(grad_out), arr, grad_out.device; requires_grad = false)
end

function _sum_grad_array_to_shape(g::AbstractArray, target_sz::Tuple{Vararg{Int}})
    tg = length(target_sz)
    g2 = g
    while ndims(g2) < tg
        g2 = reshape(g2, (1, size(g2)...))
    end
    ndims(g2) > tg && error("sum_grad: 输出秩 $(ndims(g2)) 大于目标秩 $tg")
    for i in 1:tg
        if target_sz[i] == 1 && size(g2, i) != 1
            g2 = sum(g2, dims = i)
        end
    end
    return reshape(g2, target_sz)
end

function sum_grad_to_expand2d(g::Tensor, input_sz::Tuple{Int,Int}, expanded_sz::Tuple{Int,Int})
    ir, ic = input_sz
    er, ec = expanded_sz
    arr = to_array(g)
    if ir == er && ic == ec
        return tensor(arr; device = g.device, requires_grad = false)
    end
    if ir == 1 && ic == ec
        s = sum(arr, dims = 1)
        return tensor(reshape(s, (1, ec)); device = g.device, requires_grad = false)
    end
    if ic == 1 && ir == er
        s = sum(arr, dims = 2)
        return tensor(reshape(s, (er, 1)); device = g.device, requires_grad = false)
    end
    if ir == 1 && ic == 1
        return tensor(fill(sum(arr), (1, 1)); device = g.device, requires_grad = false)
    end
    error("sum_grad_to_expand2d: unsupported $input_sz -> $expanded_sz")
end

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
