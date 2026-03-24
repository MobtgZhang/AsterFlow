"""
NVIDIA CUDA 后端：注册 `device_backend == :cuda` 的算子与运行时探测。
ROCm/NPU 等应使用独立扩展注册 `:rocm`、`:npu` 等符号。
"""
module AsterFlowCUDAExt

using AsterFlow
using CUDA

function _require_cuda_device(dev::AsterFlow.AcceleratorDevice)
    dev.backend == :cuda || error("CuArray 仅与 AcceleratorDevice(:cuda, id) 联用，当前为 :$(dev.backend)")
    return nothing
end

AsterFlow.accelerator_storage(::CUDA.CuArray) = true

function AsterFlow.tensor(
    data::CUDA.CuArray{T,N};
    device::AsterFlow.AcceleratorDevice = AsterFlow.cuda_device(0),
    requires_grad::Bool = false,
) where {T,N}
    _require_cuda_device(device)
    sz = size(data)
    v = vec(copy(data))
    st = AsterFlow.column_major_strides(sz)
    return AsterFlow.Tensor{T,N}(v, sz, st, 0, device, requires_grad, nothing, nothing)
end

function cuda_tensor_upload!(
    a::Array{T,N},
    dev::AsterFlow.AcceleratorDevice;
    requires_grad::Bool = false,
) where {T,N}
    _require_cuda_device(dev)
    return AsterFlow.tensor(CuArray(a); device = dev, requires_grad = requires_grad)
end

function cuda_dev_ones!(::Type{T}, sz::Tuple{Vararg{Int}}, dev::AsterFlow.AcceleratorDevice) where {T}
    _require_cuda_device(dev)
    return AsterFlow.tensor(CUDA.ones(T, sz...); device = dev, requires_grad = false)
end

function cuda_dev_zeros!(::Type{T}, sz::Tuple{Vararg{Int}}, dev::AsterFlow.AcceleratorDevice) where {T}
    _require_cuda_device(dev)
    return AsterFlow.tensor(CUDA.zeros(T, sz...); device = dev, requires_grad = false)
end

function cuda_dev_fill!(::Type{T}, sz::Tuple{Vararg{Int}}, v, dev::AsterFlow.AcceleratorDevice) where {T}
    _require_cuda_device(dev)
    return AsterFlow.tensor(CUDA.fill(T(v), sz...); device = dev, requires_grad = false)
end

function cuda_materialize_strided!(t::AsterFlow.Tensor{T,N}) where {T,N}
    CUDA.@allowscalar begin
        out = Array{T}(undef, t.size)
        for I in CartesianIndices(t.size)
            out[I] = AsterFlow.getindex_tensor(t, I)
        end
        return out
    end
end

function cuda_contiguous_accel!(t::AsterFlow.Tensor{T,N}) where {T,N}
    h = cuda_materialize_strided!(t)
    v = vec(CuArray(h))
    return AsterFlow.Tensor{T,N}(
        v,
        t.size,
        AsterFlow.column_major_strides(t.size),
        0,
        t.device,
        t.requires_grad,
        nothing,
        t.grad_fn,
    )
end

function _as_cuarray_shaped(t::AsterFlow.Tensor{T,N}) where {T,N}
    if AsterFlow.is_contiguous(t)
        r = @view t.storage[(t.offset+1):(t.offset+AsterFlow.numel(t))]
        return reshape(r, t.size)
    end
    return CuArray(AsterFlow.to_array(t))
end

function _tensor_from_cuarray(ca::CuArray{T,N}, dev::AsterFlow.AcceleratorDevice) where {T,N}
    _require_cuda_device(dev)
    sz = size(ca)
    v = vec(copy(ca))
    st = AsterFlow.column_major_strides(sz)
    return AsterFlow.Tensor{T,N}(v, sz, st, 0, dev, false, nothing, nothing)
end

function cuda_add(a::AsterFlow.Tensor, b::AsterFlow.Tensor)
    A = _as_cuarray_shaped(a)
    B = _as_cuarray_shaped(b)
    return _tensor_from_cuarray(A .+ B, a.device)
end

function cuda_sub(a::AsterFlow.Tensor, b::AsterFlow.Tensor)
    A = _as_cuarray_shaped(a)
    B = _as_cuarray_shaped(b)
    return _tensor_from_cuarray(A .- B, a.device)
end

function cuda_mul(a::AsterFlow.Tensor, b::AsterFlow.Tensor)
    A = _as_cuarray_shaped(a)
    B = _as_cuarray_shaped(b)
    return _tensor_from_cuarray(A .* B, a.device)
end

function cuda_div(a::AsterFlow.Tensor, b::AsterFlow.Tensor)
    A = _as_cuarray_shaped(a)
    B = _as_cuarray_shaped(b)
    return _tensor_from_cuarray(A ./ B, a.device)
end

function cuda_matmul(a::AsterFlow.Tensor{T,2}, b::AsterFlow.Tensor{T,2}) where {T}
    A = _as_cuarray_shaped(a)
    B = _as_cuarray_shaped(b)
    return _tensor_from_cuarray(A * B, a.device)
end

function cuda_scale(t::AsterFlow.Tensor{T,N}, s::AbstractFloat) where {T,N}
    A = _as_cuarray_shaped(t)
    S = T(s)
    return _tensor_from_cuarray(A .* S, t.device)
end

function cuda_relu(a::AsterFlow.Tensor{T,N}) where {T,N}
    A = _as_cuarray_shaped(a)
    return _tensor_from_cuarray(max.(A, zero(T)), a.device)
end

function cuda_relu_bwd(grad::AsterFlow.Tensor{T,N}, inp::AsterFlow.Tensor{T,N}) where {T,N}
    G = _as_cuarray_shaped(grad)
    X = _as_cuarray_shaped(inp)
    return _tensor_from_cuarray(G .* (X .> 0), inp.device)
end

function cuda_exp(a::AsterFlow.Tensor)
    A = _as_cuarray_shaped(a)
    return _tensor_from_cuarray(exp.(A), a.device)
end

function cuda_log(a::AsterFlow.Tensor)
    A = _as_cuarray_shaped(a)
    return _tensor_from_cuarray(log.(A), a.device)
end

function cuda_sqrt(a::AsterFlow.Tensor)
    A = _as_cuarray_shaped(a)
    return _tensor_from_cuarray(sqrt.(A), a.device)
end

function cuda_softmax_rows(a::AsterFlow.Tensor{T,2}) where {T}
    X = _as_cuarray_shaped(a)
    m = maximum(X; dims = 2)
    e = exp.(X .- m)
    y = e ./ sum(e; dims = 2)
    return _tensor_from_cuarray(y, a.device)
end

function cuda_softmax_rows_bwd(p::AsterFlow.Tensor{T,2}, g::AsterFlow.Tensor{T,2}) where {T}
    P = _as_cuarray_shaped(p)
    G = _as_cuarray_shaped(g)
    gin = P .* (G .- sum(G .* P; dims = 2))
    return _tensor_from_cuarray(gin, p.device)
end

function cuda_sum(a::AsterFlow.Tensor; dims = nothing)
    A = _as_cuarray_shaped(a)
    if dims === nothing
        s = sum(A)
        return _tensor_from_cuarray(CUDA.fill(s, 1, 1), a.device)
    end
    return _tensor_from_cuarray(sum(A; dims = dims), a.device)
end

function cuda_mean(a::AsterFlow.Tensor; dims = nothing)
    A = _as_cuarray_shaped(a)
    if dims === nothing
        m = mean(A)
        return _tensor_from_cuarray(CUDA.fill(m, 1, 1), a.device)
    end
    return _tensor_from_cuarray(mean(A; dims = dims), a.device)
end

function cuda_tanh(a::AsterFlow.Tensor)
    A = _as_cuarray_shaped(a)
    return _tensor_from_cuarray(tanh.(A), a.device)
end

function cuda_sigmoid(a::AsterFlow.Tensor)
    A = _as_cuarray_shaped(a)
    y = 1 ./ (1 .+ exp.(-A))
    return _tensor_from_cuarray(y, a.device)
end

function cuda_leaky_relu(a::AsterFlow.Tensor{T,N}, α::AbstractFloat) where {T,N}
    A = _as_cuarray_shaped(a)
    αt = T(α)
    return _tensor_from_cuarray(ifelse.(A .> 0, A, αt .* A), a.device)
end

function cuda_leaky_relu_bwd(
    grad::AsterFlow.Tensor{T,N},
    inp::AsterFlow.Tensor{T,N},
    α::Float32,
) where {T,N}
    G = _as_cuarray_shaped(grad)
    X = _as_cuarray_shaped(inp)
    αt = T(α)
    return _tensor_from_cuarray(G .* ifelse.(X .> 0, one(T), αt), inp.device)
end

function cuda_sgd!(
    p::AsterFlow.Tensor,
    g::AsterFlow.Tensor,
    lr::Float32,
    wd::Float32,
    vbuf,
    momentum::Float32,
    dampening::Float32,
    nesterov::Bool,
)
    n = AsterFlow.numel(p)
    ps = @view p.storage[(p.offset+1):(p.offset+n)]
    gs = @view g.storage[(g.offset+1):(g.offset+n)]
    if momentum == 0f0
        CUDA.@. ps = ps - lr * (gs + wd * ps)
        return nothing
    end
    vbuf === nothing && error("CUDA SGD: momentum>0 时需要速度缓冲")
    vs = @view vbuf.storage[(vbuf.offset+1):(vbuf.offset+n)]
    CUDA.@. vs = momentum * vs + (1f0 - dampening) * gs
    if nesterov
        CUDA.@. ps = ps - lr * (gs + momentum * vs + wd * ps)
    else
        CUDA.@. ps = ps - lr * (vs + wd * ps)
    end
    return nothing
end

function cuda_adamw!(
    p::AsterFlow.Tensor,
    g::AsterFlow.Tensor,
    opt::AsterFlow.AdamW,
    t::Float32,
    lr::Float32,
    b1::Float32,
    b2::Float32,
    eps::Float32,
    wd::Float32,
)
    id = objectid(p)
    n = AsterFlow.numel(p)
    if !haskey(opt.m, id)
        opt.m[id] = CUDA.zeros(Float32, n)
        opt.v[id] = CUDA.zeros(Float32, n)
    end
    mvec = opt.m[id]::CuVector{Float32}
    vvec = opt.v[id]::CuVector{Float32}
    ps = @view p.storage[(p.offset+1):(p.offset+n)]
    gs = @view g.storage[(g.offset+1):(g.offset+n)]
    @. mvec = b1 * mvec + (1f0 - b1) * gs
    @. vvec = b2 * vvec + (1f0 - b2) * (gs * gs)
    fac_m = 1f0 / (1f0 - b1^t)
    fac_v = 1f0 / (1f0 - b2^t)
    @. ps = ps * (1f0 - lr * wd) - lr * ((mvec * fac_m) / (sqrt(vvec * fac_v) + eps))
    return nothing
end

function cuda_adam!(
    p::AsterFlow.Tensor,
    g::AsterFlow.Tensor,
    opt::AsterFlow.Adam,
    t::Float32,
    lr::Float32,
    b1::Float32,
    b2::Float32,
    eps::Float32,
    wd::Float32,
)
    id = objectid(p)
    n = AsterFlow.numel(p)
    if !haskey(opt.m, id)
        opt.m[id] = CUDA.zeros(Float32, n)
        opt.v[id] = CUDA.zeros(Float32, n)
    end
    mvec = opt.m[id]::CuVector{Float32}
    vvec = opt.v[id]::CuVector{Float32}
    ps = @view p.storage[(p.offset+1):(p.offset+n)]
    gs = @view g.storage[(g.offset+1):(g.offset+n)]
    CUDA.@. mvec = b1 * mvec + (1f0 - b1) * (gs + wd * ps)
    CUDA.@. vvec = b2 * vvec + (1f0 - b2) * ((gs + wd * ps) * (gs + wd * ps))
    fac_m = 1f0 / (1f0 - b1^t)
    fac_v = 1f0 / (1f0 - b2^t)
    CUDA.@. ps = ps - lr * ((mvec * fac_m) / (sqrt(vvec * fac_v) + eps))
    return nothing
end

const _CUDA_BACKEND = :cuda

function __init__()
    AsterFlow.register_backend_runtime!(_CUDA_BACKEND, _ -> CUDA.functional())
    AsterFlow.register_tensor_upload!(_CUDA_BACKEND, cuda_tensor_upload!)
    AsterFlow.register_dev_ones_gpu!(_CUDA_BACKEND, cuda_dev_ones!)
    AsterFlow.register_dev_zeros_gpu!(_CUDA_BACKEND, cuda_dev_zeros!)
    AsterFlow.register_dev_fill_gpu!(_CUDA_BACKEND, cuda_dev_fill!)
    AsterFlow.register_materialize_strided_gpu!(_CUDA_BACKEND, cuda_materialize_strided!)
    AsterFlow.register_contiguous_accelerator!(_CUDA_BACKEND, cuda_contiguous_accel!)
    AsterFlow.register_sgd_accelerator!(_CUDA_BACKEND, cuda_sgd!)
    AsterFlow.register_adamw_accelerator!(_CUDA_BACKEND, cuda_adamw!)
    AsterFlow.register_adam_accelerator!(_CUDA_BACKEND, cuda_adam!)
    AsterFlow.register_op!(:add, _CUDA_BACKEND, cuda_add)
    AsterFlow.register_op!(:sub, _CUDA_BACKEND, cuda_sub)
    AsterFlow.register_op!(:mul, _CUDA_BACKEND, cuda_mul)
    AsterFlow.register_op!(:div, _CUDA_BACKEND, cuda_div)
    AsterFlow.register_op!(:matmul, _CUDA_BACKEND, cuda_matmul)
    AsterFlow.register_op!(:scale, _CUDA_BACKEND, cuda_scale)
    AsterFlow.register_op!(:relu, _CUDA_BACKEND, cuda_relu)
    AsterFlow.register_op!(:relu_bwd, _CUDA_BACKEND, cuda_relu_bwd)
    AsterFlow.register_op!(:softmax_rows, _CUDA_BACKEND, cuda_softmax_rows)
    AsterFlow.register_op!(:softmax_rows_bwd, _CUDA_BACKEND, cuda_softmax_rows_bwd)
    AsterFlow.register_op!(:exp, _CUDA_BACKEND, cuda_exp)
    AsterFlow.register_op!(:log, _CUDA_BACKEND, cuda_log)
    AsterFlow.register_op!(:sqrt, _CUDA_BACKEND, cuda_sqrt)
    AsterFlow.register_op!(:sum, _CUDA_BACKEND, cuda_sum)
    AsterFlow.register_op!(:mean, _CUDA_BACKEND, cuda_mean)
    AsterFlow.register_op!(:tanh, _CUDA_BACKEND, cuda_tanh)
    AsterFlow.register_op!(:sigmoid, _CUDA_BACKEND, cuda_sigmoid)
    AsterFlow.register_op!(:leaky_relu, _CUDA_BACKEND, cuda_leaky_relu)
    AsterFlow.register_op!(:leaky_relu_bwd, _CUDA_BACKEND, cuda_leaky_relu_bwd)
    return nothing
end

end
