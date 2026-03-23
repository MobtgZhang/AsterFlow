## CPU 原生算子：仅张量数学，不挂 autograd

function native_cpu_add(a::Tensor{T,N}, b::Tensor{T,N}) where {T,N}
    size(a) == size(b) || error("add: shape mismatch")
    aa = to_array(a)
    ba = to_array(b)
    tensor(aa .+ ba; device = a.device, requires_grad = false)
end

function native_cpu_sub(a::Tensor{T,N}, b::Tensor{T,N}) where {T,N}
    size(a) == size(b) || error("sub: shape mismatch")
    tensor(to_array(a) .- to_array(b); device = a.device, requires_grad = false)
end

function native_cpu_mul(a::Tensor{T,N}, b::Tensor{T,N}) where {T,N}
    size(a) == size(b) || error("mul: shape mismatch")
    tensor(to_array(a) .* to_array(b); device = a.device, requires_grad = false)
end

function native_cpu_div(a::Tensor{T,N}, b::Tensor{T,N}) where {T,N}
    size(a) == size(b) || error("div: shape mismatch")
    tensor(to_array(a) ./ to_array(b); device = a.device, requires_grad = false)
end

function native_cpu_matmul_julia(a::Tensor{T,2}, b::Tensor{T,2}) where {T}
    size(a, 2) == size(b, 1) || error("matmul: inner dim mismatch")
    return tensor(to_array(a) * to_array(b); device = a.device, requires_grad = false)
end

function native_cpu_matmul(a::Tensor{T,2}, b::Tensor{T,2}) where {T}
    if T === Float32 && libasterflow_version() >= 0
        return af_matmul_nograd(a, b)
    end
    return native_cpu_matmul_julia(a, b)
end

function native_cpu_sum(a::Tensor; dims = nothing)
    arr = to_array(a)
    if dims === nothing
        s = sum(arr)
        return tensor(fill(s, 1, 1); device = a.device, requires_grad = false)
    end
    tensor(sum(arr; dims = dims); device = a.device, requires_grad = false)
end

function native_cpu_mean(a::Tensor; dims = nothing)
    arr = to_array(a)
    if dims === nothing
        m = sum(arr) / length(arr)
        return tensor(fill(m, 1, 1); device = a.device, requires_grad = false)
    end
    d = dims isa Integer ? Int(dims) : Int(dims[1])
    n = size(arr, d)
    tensor(sum(arr; dims = dims) ./ n; device = a.device, requires_grad = false)
end

function native_cpu_exp(a::Tensor)
    tensor(exp.(to_array(a)); device = a.device, requires_grad = false)
end

function native_cpu_log(a::Tensor)
    tensor(log.(to_array(a)); device = a.device, requires_grad = false)
end

function native_cpu_sqrt(a::Tensor)
    tensor(sqrt.(to_array(a)); device = a.device, requires_grad = false)
end

function native_cpu_relu_julia(a::Tensor{T,N}) where {T,N}
    x = to_array(a)
    return tensor(max.(x, zero(T)); device = a.device, requires_grad = false)
end

function native_cpu_relu(a::Tensor{Float32,N}) where {N}
    if libasterflow_version() >= 0
        return af_relu_nograd(a)
    end
    return native_cpu_relu_julia(a)
end

function native_cpu_relu(a::Tensor{T,N}) where {T,N}
    return native_cpu_relu_julia(a)
end

function native_cpu_softmax_rows(a::Tensor{T,2}) where {T}
    x = to_array(a)
    m = maximum(x, dims = 2)
    e = exp.(x .- m)
    y = e ./ sum(e, dims = 2)
    tensor(y; device = a.device, requires_grad = false)
end

function native_cpu_scale(t::Tensor{T,N}, s::AbstractFloat) where {T,N}
    S = T(s)
    return tensor(to_array(t) .* S; device = t.device, requires_grad = false)
end

function native_cpu_relu_bwd(grad::Tensor{T,N}, inp::Tensor{T,N}) where {T,N}
    size(grad) == size(inp) || error("relu_bwd: shape mismatch")
    return tensor(to_array(grad) .* (to_array(inp) .> 0); device = inp.device, requires_grad = false)
end

function native_cpu_softmax_rows_bwd(p::Tensor{T,2}, g::Tensor{T,2}) where {T}
    size(p) == size(g) || error("softmax_rows_bwd: shape mismatch")
    s = to_array(p)
    gv = to_array(g)
    gin = s .* (gv .- sum(gv .* s, dims = 2))
    return tensor(gin; device = p.device, requires_grad = false)
end

function native_cpu_tanh(a::Tensor{T,N}) where {T,N}
    tensor(tanh.(to_array(a)); device = a.device, requires_grad = false)
end

function native_cpu_sigmoid(a::Tensor{T,N}) where {T,N}
    x = to_array(a)
    y = 1 ./ (1 .+ exp.(-x))
    tensor(y; device = a.device, requires_grad = false)
end

function native_cpu_leaky_relu(a::Tensor{T,N}, negative_slope::AbstractFloat) where {T,N}
    α = T(negative_slope)
    x = to_array(a)
    tensor(ifelse.(x .> 0, x, α .* x); device = a.device, requires_grad = false)
end

function native_cpu_leaky_relu_bwd(
    grad::Tensor{T,N},
    inp::Tensor{T,N},
    negative_slope::Float32,
) where {T,N}
    size(grad) == size(inp) || error("leaky_relu_bwd: shape mismatch")
    α = T(negative_slope)
    xi = to_array(inp)
    gi = to_array(grad)
    mask = ifelse.(xi .> 0, one(T), α)
    tensor(gi .* mask; device = inp.device, requires_grad = false)
end

function register_native_cpu!()
    b = BACKEND_CPU
    register_op!(:add, b, native_cpu_add)
    register_op!(:sub, b, native_cpu_sub)
    register_op!(:mul, b, native_cpu_mul)
    register_op!(:div, b, native_cpu_div)
    register_op!(:matmul, b, native_cpu_matmul)
    register_op!(:sum, b, native_cpu_sum)
    register_op!(:mean, b, native_cpu_mean)
    register_op!(:exp, b, native_cpu_exp)
    register_op!(:log, b, native_cpu_log)
    register_op!(:sqrt, b, native_cpu_sqrt)
    register_op!(:relu, b, native_cpu_relu)
    register_op!(:softmax_rows, b, native_cpu_softmax_rows)
    register_op!(:scale, b, native_cpu_scale)
    register_op!(:relu_bwd, b, native_cpu_relu_bwd)
    register_op!(:softmax_rows_bwd, b, native_cpu_softmax_rows_bwd)
    register_op!(:tanh, b, native_cpu_tanh)
    register_op!(:sigmoid, b, native_cpu_sigmoid)
    register_op!(:leaky_relu, b, native_cpu_leaky_relu)
    register_op!(:leaky_relu_bwd, b, native_cpu_leaky_relu_bwd)
    return nothing
end
