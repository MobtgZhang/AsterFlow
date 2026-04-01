abstract type Node end

inputs_of(n::Node) = ()

struct AddBackward <: Node
    a::Tensor
    b::Tensor
    ver_a::UInt64
    ver_b::UInt64
end
inputs_of(n::AddBackward) = (n.a, n.b)
function apply_backward!(n::AddBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.a, n.ver_a, "add:input_a")
    _verify_saved_tensor_version!(n.b, n.ver_b, "add:input_b")
    no_grad() do
        accumulate_grad!(n.a, sum_grad_to_shape(grad_out, size(n.a)))
        accumulate_grad!(n.b, sum_grad_to_shape(grad_out, size(n.b)))
    end
end

struct SubBackward <: Node
    a::Tensor
    b::Tensor
    ver_a::UInt64
    ver_b::UInt64
end
inputs_of(n::SubBackward) = (n.a, n.b)
function apply_backward!(n::SubBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.a, n.ver_a, "sub:input_a")
    _verify_saved_tensor_version!(n.b, n.ver_b, "sub:input_b")
    no_grad() do
        accumulate_grad!(n.a, sum_grad_to_shape(grad_out, size(n.a)))
        accumulate_grad!(n.b, sum_grad_to_shape(mul_scalar_tensor(grad_out, -1), size(n.b)))
    end
end

struct DivBackward <: Node
    a::Tensor
    b::Tensor
    ver_a::UInt64
    ver_b::UInt64
end
inputs_of(n::DivBackward) = (n.a, n.b)
function apply_backward!(n::DivBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.a, n.ver_a, "div:input_a")
    _verify_saved_tensor_version!(n.b, n.ver_b, "div:input_b")
    no_grad() do
        ga = div_op_tensor(grad_out, n.b)
        b2 = mul_tensor(n.b, n.b)
        gb = mul_scalar_tensor(div_op_tensor(mul_tensor(grad_out, n.a), b2), -1)
        accumulate_grad!(n.a, sum_grad_to_shape(ga, size(n.a)))
        accumulate_grad!(n.b, sum_grad_to_shape(gb, size(n.b)))
    end
end

struct MulBackward <: Node
    a::Tensor
    b::Tensor
    saved_a::Tensor
    saved_b::Tensor
    ver_a::UInt64
    ver_b::UInt64
end
inputs_of(n::MulBackward) = (n.a, n.b)
function apply_backward!(n::MulBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.a, n.ver_a, "mul:input_a")
    _verify_saved_tensor_version!(n.b, n.ver_b, "mul:input_b")
    no_grad() do
        ga = mul_tensor(grad_out, n.saved_b)
        gb = mul_tensor(grad_out, n.saved_a)
        accumulate_grad!(n.a, sum_grad_to_shape(ga, size(n.a)))
        accumulate_grad!(n.b, sum_grad_to_shape(gb, size(n.b)))
    end
end

struct MatMulBackward <: Node
    a::Tensor
    b::Tensor
    ver_a::UInt64
    ver_b::UInt64
end
inputs_of(n::MatMulBackward) = (n.a, n.b)
function apply_backward!(n::MatMulBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.a, n.ver_a, "matmul:input_a")
    _verify_saved_tensor_version!(n.b, n.ver_b, "matmul:input_b")
    a, b = n.a, n.b
    if ndims(a) == 2 && ndims(b) == 2
        no_grad() do
            accumulate_grad!(a, matmul_ng(grad_out, _permute_storage(b, (2, 1))))
            accumulate_grad!(b, matmul_ng(_permute_storage(a, (2, 1)), grad_out))
        end
    else
        error("MatMulBackward: only 2D in MVP")
    end
end

struct PermuteBackward <: Node
    input::Tensor
    forward_perm::Tuple{Vararg{Int}}
    ver_input::UInt64
end
inputs_of(n::PermuteBackward) = (n.input,)
function apply_backward!(n::PermuteBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "permute:input")
    invp = invperm(n.forward_perm)
    no_grad() do
        accumulate_grad!(n.input, _permute_storage(grad_out, invp))
    end
end

struct ReshapeBackward <: Node
    input::Tensor
    input_size::Tuple{Vararg{Int}}
    ver_input::UInt64
end
inputs_of(n::ReshapeBackward) = (n.input,)
function apply_backward!(n::ReshapeBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "reshape:input")
    no_grad() do
        accumulate_grad!(n.input, _reshape_storage(grad_out, n.input_size))
    end
end

struct ExpandBackward <: Node
    input::Tensor
    input_size::Tuple{Int,Int}
    expanded_size::Tuple{Int,Int}
    ver_input::UInt64
end
inputs_of(n::ExpandBackward) = (n.input,)
function apply_backward!(n::ExpandBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "expand:input")
    no_grad() do
        g = sum_grad_to_expand2d(grad_out, n.input_size, n.expanded_size)
        accumulate_grad!(n.input, g)
    end
end

struct SumBackward <: Node
    input::Tensor
    dims::Any
    ver_input::UInt64
end
inputs_of(n::SumBackward) = (n.input,)
function apply_backward!(n::SumBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "sum:input")
    inp = n.input
    g = expand_as_sumgrad(grad_out, inp.size, n.dims)
    accumulate_grad!(inp, g)
end

function expand_as_sumgrad(g::Tensor, target_sz::NTuple{N,Int}, dims) where {N}
    if dims === nothing
        return broadcast_fill_like(g, target_sz)
    end
    if N == 2 && ndims(g) == 2
        return expand_as_sumgrad_mat2d(g, target_sz, dims)
    end
    error("SumBackward: dims reduction for this shape not implemented")
end

function _sum_dim_to_int(dims)
    dims isa Integer && return Int(dims)
    (dims isa Tuple || dims isa AbstractVector) && length(dims) == 1 && return Int(dims[1])
    error("SumBackward: unsupported dims=$dims")
end

function expand_as_sumgrad_mat2d(g::Tensor{T,2}, target_sz::Tuple{Int,Int}, dims) where {T}
    B, D = target_sz
    di = _sum_dim_to_int(dims)
    if di == 1
        return matmul_ng(dev_ones(T, (B, 1), g.device), g)
    elseif di == 2
        return matmul_ng(g, dev_ones(T, (1, D), g.device))
    end
    error("SumBackward: only dims 1 or 2 for 2D tensors")
end

struct ScaleBackward <: Node
    input::Tensor
    scale::Any
    ver_input::UInt64
end
inputs_of(n::ScaleBackward) = (n.input,)
function apply_backward!(n::ScaleBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "scale:input")
    accumulate_grad!(n.input, mul_scalar_tensor(grad_out, n.scale))
end

struct MeanBackward <: Node
    input::Tensor
    inv_n::Any
    ver_input::UInt64
end
inputs_of(n::MeanBackward) = (n.input,)
function apply_backward!(n::MeanBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "mean:input")
    g = broadcast_fill_like(grad_out, n.input.size)
    accumulate_grad!(n.input, mul_scalar_tensor(g, n.inv_n))
end

struct MeanDimsBackward <: Node
    input::Tensor
    dim::Int
    inv_n::Any
    ver_input::UInt64
end
inputs_of(n::MeanDimsBackward) = (n.input,)
function apply_backward!(n::MeanDimsBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "mean_dims:input")
    g = expand_as_sumgrad_mat2d(grad_out, n.input.size, n.dim)
    accumulate_grad!(n.input, mul_scalar_tensor(g, n.inv_n))
end

struct TanhBackward <: Node
    input::Tensor
    out::Tensor
    ver_input::UInt64
end
inputs_of(n::TanhBackward) = (n.input,)
function apply_backward!(n::TanhBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "tanh:input")
    T = eltype(n.out)
    one_t = dev_fill(T, n.out.size, one(T), n.out.device)
    tsq = mul_tensor(n.out, n.out)
    factor = sub(one_t, tsq)
    accumulate_grad!(n.input, mul_tensor(grad_out, factor))
end

struct SigmoidBackward <: Node
    input::Tensor
    sig::Tensor
    ver_input::UInt64
end
inputs_of(n::SigmoidBackward) = (n.input,)
function apply_backward!(n::SigmoidBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "sigmoid:input")
    T = eltype(n.sig)
    one_t = dev_fill(T, n.sig.size, one(T), n.sig.device)
    one_m = sub(one_t, n.sig)
    accumulate_grad!(n.input, mul_tensor(grad_out, mul_tensor(n.sig, one_m)))
end

struct LeakyReLUBackward <: Node
    input::Tensor
    negative_slope::Float32
    ver_input::UInt64
end
inputs_of(n::LeakyReLUBackward) = (n.input,)
function apply_backward!(n::LeakyReLUBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "leaky_relu:input")
    accumulate_grad!(n.input, leaky_relu_backward_mul(grad_out, n.input, n.negative_slope))
end

struct CrossEntropyBackward <: Node
    logits::Tensor
    prob::Tensor
    targets::Vector{Int}
    ver_logits::UInt64
end
inputs_of(n::CrossEntropyBackward) = (n.logits,)
function apply_backward!(n::CrossEntropyBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.logits, n.ver_logits, "cross_entropy:logits")
    g0 = to_array(grad_out)[1]
    B, C = size(n.prob)
    pa = to_array(n.prob)
    diff = similar(pa)
    invB = Float32(g0 / B)
    for i in 1:B
        ti = n.targets[i]
        for j in 1:C
            diff[i, j] = (pa[i, j] - (j == ti ? 1f0 : 0f0)) * invB
        end
    end
    gt = tensor_on_device(Float32, diff, n.logits.device; requires_grad = false)
    accumulate_grad!(n.logits, gt)
end

struct NLLBackward <: Node
    log_probs::Tensor
    targets::Vector{Int}
    ver_log_probs::UInt64
end
inputs_of(n::NLLBackward) = (n.log_probs,)
function apply_backward!(n::NLLBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.log_probs, n.ver_log_probs, "nll:log_probs")
    g0 = to_array(grad_out)[1]
    B, C = size(n.log_probs)
    g = zeros(Float32, B, C)
    invB = Float32(g0 / B)
    for i in 1:B
        g[i, n.targets[i]] = -invB
    end
    gt = tensor_on_device(Float32, g, n.log_probs.device; requires_grad = false)
    accumulate_grad!(n.log_probs, gt)
end

function broadcast_fill_like(g::Tensor, target_sz::NTuple{N,Int}) where {N}
    if size(g) == target_sz
        return g
    end
    if numel(g) == 1
        T = eltype(g)
        v = to_array(g)[1]
        return dev_fill(T, target_sz, v, g.device)
    end
    error("broadcast_fill_like: unsupported $(size(g)) -> $target_sz")
end

struct ExpBackward <: Node
    input::Tensor
    saved_out::Tensor
    ver_input::UInt64
end
inputs_of(n::ExpBackward) = (n.input,)
function apply_backward!(n::ExpBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "exp:input")
    accumulate_grad!(n.input, mul_tensor(grad_out, n.saved_out))
end

struct LogBackward <: Node
    input::Tensor
    ver_input::UInt64
end
inputs_of(n::LogBackward) = (n.input,)
function apply_backward!(n::LogBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "log:input")
    accumulate_grad!(n.input, div_op_tensor(grad_out, n.input))
end

struct SqrtBackward <: Node
    input::Tensor
    saved_sqrt::Tensor
    ver_input::UInt64
end
inputs_of(n::SqrtBackward) = (n.input,)
function apply_backward!(n::SqrtBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "sqrt:input")
    half = eltype(grad_out)(0.5)
    accumulate_grad!(n.input, mul_scalar_tensor(div_op_tensor(grad_out, n.saved_sqrt), half))
end

struct ReLUBackward <: Node
    input::Tensor
    ver_input::UInt64
end
inputs_of(n::ReLUBackward) = (n.input,)
function apply_backward!(n::ReLUBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.input, n.ver_input, "relu:input")
    accumulate_grad!(n.input, relu_backward_mul(grad_out, n.input))
end

struct SoftmaxRowsBackward <: Node
    logits::Tensor
    prob::Tensor
    ver_logits::UInt64
end
inputs_of(n::SoftmaxRowsBackward) = (n.logits,)
function apply_backward!(n::SoftmaxRowsBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.logits, n.ver_logits, "softmax_rows:logits")
    gin = softmax_rows_backward_nograd(n.prob, grad_out)
    accumulate_grad!(n.logits, gin)
end

struct EmbeddingBackward <: Node
    weight::Tensor
    idx::Matrix{Int}
    ver_weight::UInt64
end
inputs_of(n::EmbeddingBackward) = (n.weight,)
function apply_backward!(n::EmbeddingBackward, grad_out::Tensor)
    _verify_saved_tensor_version!(n.weight, n.ver_weight, "embedding:weight")
    g = to_array(grad_out)
    gw = zeros(Float32, size(n.weight))
    B, L = size(n.idx)
    for i in 1:B, j in 1:L
        r = n.idx[i, j]
        @views gw[r, :] .+= g[(i-1)*L+j, :]
    end
    gw_t = tensor_on_device(Float32, gw, n.weight.device; requires_grad = false)
    accumulate_grad!(n.weight, gw_t)
end
