mutable struct Embedding <: Module
    weight::Tensor{Float32,2}
    training::Bool
end

function Embedding(vocab::Int, dim::Int; device::Device = CPUDevice())
    w = tensor(randn(Float32, vocab, dim) .* Float32(0.02); device = device)
    requires_grad!(w, true)
    return Embedding(w, true)
end

function (e::Embedding)(idx::Matrix{Int})
    B, L = size(idx)
    D = size(e.weight, 2)
    out = Array{Float32}(undef, B, L, D)
    W = to_array(e.weight)
    for i in 1:B, j in 1:L
        out[i, j, :] = W[idx[i, j], :]
    end
    t = tensor(reshape(out, B * L, D); device = e.weight.device, requires_grad = false)
    if grad_enabled() && e.weight.requires_grad
        t.requires_grad = true
        t.grad_fn = EmbeddingBackward(e.weight, copy(idx))
    end
    return t
end

mutable struct CausalSelfAttention <: Module
    wq::Linear
    wk::Linear
    wv::Linear
    wo::Linear
    head_dim::Int
    training::Bool
end

function CausalSelfAttention(d_model::Int, n_heads::Int; device::Device = CPUDevice())
    (d_model % n_heads == 0) || error("d_model must divide n_heads")
    n_heads == 1 || error("MVP: 仅支持 n_heads==1（B=1 因果注意力）")
    dh = div(d_model, n_heads)
    return CausalSelfAttention(
        Linear(d_model, d_model; device = device),
        Linear(d_model, d_model; device = device),
        Linear(d_model, d_model; device = device),
        Linear(d_model, d_model; device = device),
        dh,
        true,
    )
end

function (m::CausalSelfAttention)(x::Tensor{Float32,2})
    L, D = size(x)
    q = m.wq(x)
    k = m.wk(x)
    v = m.wv(x)
    scale = Float32(1 / sqrt(Float32(m.head_dim)))
    s0 = matmul(q, permute_tensor(k, (2, 1)))
    s = mul_scalar_tensor(s0, scale)
    Ma = zeros(Float32, L, L)
    for i in 1:L, j in 1:L
        if j > i
            Ma[i, j] = -1f4
        end
    end
    mask_t = tensor_on_device(Float32, Ma, x.device; requires_grad = false)
    s2 = add(s, mask_t)
    p = softmax_rows(s2)
    ctx = matmul(p, v)
    return m.wo(ctx)
end

mutable struct TransformerBlock <: Module
    attn::CausalSelfAttention
    ffn1::Linear
    ffn2::Linear
    training::Bool
end

function TransformerBlock(d_model::Int, ff_dim::Int; device::Device = CPUDevice())
    return TransformerBlock(
        CausalSelfAttention(d_model, 1; device = device),
        Linear(d_model, ff_dim; device = device),
        Linear(ff_dim, d_model; device = device),
        true,
    )
end

function (m::TransformerBlock)(x::Tensor{Float32,2})
    y = m.attn(x)
    x2 = add(x, y)
    h = relu_tensor(m.ffn1(x2))
    z = m.ffn2(h)
    return add(x2, z)
end

mutable struct TinyGPT <: Module
    embed::Embedding
    block::TransformerBlock
    head::Linear
    training::Bool
end

function TinyGPT(vocab::Int, d_model::Int, ff_dim::Int; device::Device = CPUDevice())
    return TinyGPT(
        Embedding(vocab, d_model; device = device),
        TransformerBlock(d_model, ff_dim; device = device),
        Linear(d_model, vocab; device = device),
        true,
    )
end

function (g::TinyGPT)(idx::Matrix{Int})
    size(idx, 1) == 1 || error("TinyGPT MVP: batch 维度需为 1")
    L = size(idx, 2)
    h = g.embed(idx)
    d = size(g.embed.weight, 2)
    h2 = reshape_op(h, (L, d))
    y = g.block(h2)
    return g.head(y)
end
