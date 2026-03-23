abstract type Node end

inputs_of(n::Node) = ()

struct AddBackward <: Node
    a::Tensor
    b::Tensor
end
inputs_of(n::AddBackward) = (n.a, n.b)
function apply_backward!(n::AddBackward, grad_out::Tensor)
    accumulate_grad!(n.a, grad_out)
    accumulate_grad!(n.b, grad_out)
end

struct SubBackward <: Node
    a::Tensor
    b::Tensor
end
inputs_of(n::SubBackward) = (n.a, n.b)
function apply_backward!(n::SubBackward, grad_out::Tensor)
    accumulate_grad!(n.a, grad_out)
    accumulate_grad!(n.b, mul_scalar_tensor(grad_out, -1))
end

struct DivBackward <: Node
    a::Tensor
    b::Tensor
end
inputs_of(n::DivBackward) = (n.a, n.b)
function apply_backward!(n::DivBackward, grad_out::Tensor)
    accumulate_grad!(n.a, div_op_tensor(grad_out, n.b))
    b2 = mul_tensor(n.b, n.b)
    accumulate_grad!(n.b, mul_scalar_tensor(div_op_tensor(mul_tensor(grad_out, n.a), b2), -1))
end

struct MulBackward <: Node
    a::Tensor
    b::Tensor
    saved_a::Tensor
    saved_b::Tensor
end
inputs_of(n::MulBackward) = (n.a, n.b)
function apply_backward!(n::MulBackward, grad_out::Tensor)
    accumulate_grad!(n.a, mul_tensor(grad_out, n.saved_b))
    accumulate_grad!(n.b, mul_tensor(grad_out, n.saved_a))
end

struct MatMulBackward <: Node
    a::Tensor
    b::Tensor
end
inputs_of(n::MatMulBackward) = (n.a, n.b)
function apply_backward!(n::MatMulBackward, grad_out::Tensor)
    a, b = n.a, n.b
    if ndims(a) == 2 && ndims(b) == 2
        accumulate_grad!(a, matmul_ng(grad_out, permute_tensor(b, (2, 1))))
        accumulate_grad!(b, matmul_ng(permute_tensor(a, (2, 1)), grad_out))
    else
        error("MatMulBackward: only 2D in MVP")
    end
end

struct SumBackward <: Node
    input::Tensor
    dims::Any
end
inputs_of(n::SumBackward) = (n.input,)
function apply_backward!(n::SumBackward, grad_out::Tensor)
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
end
inputs_of(n::ScaleBackward) = (n.input,)
function apply_backward!(n::ScaleBackward, grad_out::Tensor)
    accumulate_grad!(n.input, mul_scalar_tensor(grad_out, n.scale))
end

struct MeanBackward <: Node
    input::Tensor
    inv_n::Any
end
inputs_of(n::MeanBackward) = (n.input,)
function apply_backward!(n::MeanBackward, grad_out::Tensor)
    g = broadcast_fill_like(grad_out, n.input.size)
    accumulate_grad!(n.input, mul_scalar_tensor(g, n.inv_n))
end

struct MeanDimsBackward <: Node
    input::Tensor
    dim::Int
    inv_n::Any
end
inputs_of(n::MeanDimsBackward) = (n.input,)
function apply_backward!(n::MeanDimsBackward, grad_out::Tensor)
    g = expand_as_sumgrad_mat2d(grad_out, n.input.size, n.dim)
    accumulate_grad!(n.input, mul_scalar_tensor(g, n.inv_n))
end

struct TanhBackward <: Node
    input::Tensor
    out::Tensor
end
inputs_of(n::TanhBackward) = (n.input,)
function apply_backward!(n::TanhBackward, grad_out::Tensor)
    T = eltype(n.out)
    one_t = dev_fill(T, n.out.size, one(T), n.out.device)
    tsq = mul_tensor(n.out, n.out)
    factor = sub(one_t, tsq)
    accumulate_grad!(n.input, mul_tensor(grad_out, factor))
end

struct SigmoidBackward <: Node
    input::Tensor
    sig::Tensor
end
inputs_of(n::SigmoidBackward) = (n.input,)
function apply_backward!(n::SigmoidBackward, grad_out::Tensor)
    T = eltype(n.sig)
    one_t = dev_fill(T, n.sig.size, one(T), n.sig.device)
    one_m = sub(one_t, n.sig)
    accumulate_grad!(n.input, mul_tensor(grad_out, mul_tensor(n.sig, one_m)))
end

struct LeakyReLUBackward <: Node
    input::Tensor
    negative_slope::Float32
end
inputs_of(n::LeakyReLUBackward) = (n.input,)
function apply_backward!(n::LeakyReLUBackward, grad_out::Tensor)
    accumulate_grad!(n.input, leaky_relu_backward_mul(grad_out, n.input, n.negative_slope))
end

struct CrossEntropyBackward <: Node
    logits::Tensor
    prob::Tensor
    targets::Vector{Int}
end
inputs_of(n::CrossEntropyBackward) = (n.logits,)
function apply_backward!(n::CrossEntropyBackward, grad_out::Tensor)
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
end
inputs_of(n::NLLBackward) = (n.log_probs,)
function apply_backward!(n::NLLBackward, grad_out::Tensor)
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
end
inputs_of(n::ExpBackward) = (n.input,)
function apply_backward!(n::ExpBackward, grad_out::Tensor)
    accumulate_grad!(n.input, mul_tensor(grad_out, n.saved_out))
end

struct LogBackward <: Node
    input::Tensor
end
inputs_of(n::LogBackward) = (n.input,)
function apply_backward!(n::LogBackward, grad_out::Tensor)
    accumulate_grad!(n.input, div_op_tensor(grad_out, n.input))
end

struct SqrtBackward <: Node
    input::Tensor
    saved_sqrt::Tensor
end
inputs_of(n::SqrtBackward) = (n.input,)
function apply_backward!(n::SqrtBackward, grad_out::Tensor)
    half = eltype(grad_out)(0.5)
    accumulate_grad!(n.input, mul_scalar_tensor(div_op_tensor(grad_out, n.saved_sqrt), half))
end

struct ReLUBackward <: Node
    input::Tensor
end
inputs_of(n::ReLUBackward) = (n.input,)
function apply_backward!(n::ReLUBackward, grad_out::Tensor)
    accumulate_grad!(n.input, relu_backward_mul(grad_out, n.input))
end

struct SoftmaxRowsBackward <: Node
    logits::Tensor
    prob::Tensor
end
inputs_of(n::SoftmaxRowsBackward) = (n.logits,)
function apply_backward!(n::SoftmaxRowsBackward, grad_out::Tensor)
    gin = softmax_rows_backward_nograd(n.prob, grad_out)
    accumulate_grad!(n.logits, gin)
end

struct EmbeddingBackward <: Node
    weight::Tensor
    idx::Matrix{Int}
end
inputs_of(n::EmbeddingBackward) = (n.weight,)
function apply_backward!(n::EmbeddingBackward, grad_out::Tensor)
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
