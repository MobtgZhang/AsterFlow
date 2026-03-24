"""
AMD ROCm / HIP 后端：注册 `device_backend == :rocm`，依赖 `AMDGPU.jl` 与系统 ROCm。
与 `AsterFlowCUDAExt` 对称；算子与优化器内核基于 `ROCArray` 广播实现。
"""
module AsterFlowROCMExt

using AsterFlow
using AMDGPU

function _require_rocm_device(dev::AsterFlow.AcceleratorDevice)
    dev.backend == :rocm || error("ROCArray 仅与 AcceleratorDevice(:rocm, id) 联用，当前为 :$(dev.backend)")
    return nothing
end

AsterFlow.accelerator_storage(::ROCArray) = true

function AsterFlow.tensor(
    data::ROCArray{T,N};
    device::AsterFlow.AcceleratorDevice = AsterFlow.rocm_device(0),
    requires_grad::Bool = false,
) where {T,N}
    _require_rocm_device(device)
    sz = size(data)
    v = vec(copy(data))
    st = AsterFlow.column_major_strides(sz)
    return AsterFlow.Tensor{T,N}(v, sz, st, 0, device, requires_grad, nothing, nothing)
end

function rocm_tensor_upload!(
    a::Array{T,N},
    dev::AsterFlow.AcceleratorDevice;
    requires_grad::Bool = false,
) where {T,N}
    _require_rocm_device(dev)
    return AsterFlow.tensor(ROCArray(a); device = dev, requires_grad = requires_grad)
end

function rocm_dev_ones!(::Type{T}, sz::Tuple{Vararg{Int}}, dev::AsterFlow.AcceleratorDevice) where {T}
    _require_rocm_device(dev)
    return AsterFlow.tensor(AMDGPU.ones(T, sz...); device = dev, requires_grad = false)
end

function rocm_dev_zeros!(::Type{T}, sz::Tuple{Vararg{Int}}, dev::AsterFlow.AcceleratorDevice) where {T}
    _require_rocm_device(dev)
    return AsterFlow.tensor(AMDGPU.zeros(T, sz...); device = dev, requires_grad = false)
end

function rocm_dev_fill!(::Type{T}, sz::Tuple{Vararg{Int}}, v, dev::AsterFlow.AcceleratorDevice) where {T}
    _require_rocm_device(dev)
    return AsterFlow.tensor(ROCArray(fill(T(v), sz...)); device = dev, requires_grad = false)
end

function rocm_materialize_strided!(t::AsterFlow.Tensor{T,N}) where {T,N}
    AMDGPU.@allowscalar begin
        out = Array{T}(undef, t.size)
        for I in CartesianIndices(t.size)
            out[I] = AsterFlow.getindex_tensor(t, I)
        end
        return out
    end
end

function rocm_contiguous_accel!(t::AsterFlow.Tensor{T,N}) where {T,N}
    h = rocm_materialize_strided!(t)
    v = vec(ROCArray(h))
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

function _as_roc_shaped(t::AsterFlow.Tensor{T,N}) where {T,N}
    if AsterFlow.is_contiguous(t)
        r = @view t.storage[(t.offset+1):(t.offset+AsterFlow.numel(t))]
        return reshape(r, t.size)
    end
    return ROCArray(AsterFlow.to_array(t))
end

function _tensor_from_roc(ca::ROCArray{T,N}, dev::AsterFlow.AcceleratorDevice) where {T,N}
    _require_rocm_device(dev)
    sz = size(ca)
    v = vec(copy(ca))
    st = AsterFlow.column_major_strides(sz)
    return AsterFlow.Tensor{T,N}(v, sz, st, 0, dev, false, nothing, nothing)
end

function rocm_add(a::AsterFlow.Tensor, b::AsterFlow.Tensor)
    A = _as_roc_shaped(a)
    B = _as_roc_shaped(b)
    return _tensor_from_roc(A .+ B, a.device)
end

function rocm_sub(a::AsterFlow.Tensor, b::AsterFlow.Tensor)
    A = _as_roc_shaped(a)
    B = _as_roc_shaped(b)
    return _tensor_from_roc(A .- B, a.device)
end

function rocm_mul(a::AsterFlow.Tensor, b::AsterFlow.Tensor)
    A = _as_roc_shaped(a)
    B = _as_roc_shaped(b)
    return _tensor_from_roc(A .* B, a.device)
end

function rocm_div(a::AsterFlow.Tensor, b::AsterFlow.Tensor)
    A = _as_roc_shaped(a)
    B = _as_roc_shaped(b)
    return _tensor_from_roc(A ./ B, a.device)
end

function rocm_matmul(a::AsterFlow.Tensor{T,2}, b::AsterFlow.Tensor{T,2}) where {T}
    A = _as_roc_shaped(a)
    B = _as_roc_shaped(b)
    return _tensor_from_roc(A * B, a.device)
end

function rocm_scale(t::AsterFlow.Tensor{T,N}, s::AbstractFloat) where {T,N}
    A = _as_roc_shaped(t)
    S = T(s)
    return _tensor_from_roc(A .* S, t.device)
end

function rocm_relu(a::AsterFlow.Tensor{T,N}) where {T,N}
    A = _as_roc_shaped(a)
    return _tensor_from_roc(max.(A, zero(T)), a.device)
end

function rocm_relu_bwd(grad::AsterFlow.Tensor{T,N}, inp::AsterFlow.Tensor{T,N}) where {T,N}
    G = _as_roc_shaped(grad)
    X = _as_roc_shaped(inp)
    return _tensor_from_roc(G .* (X .> 0), inp.device)
end

function rocm_exp(a::AsterFlow.Tensor)
    A = _as_roc_shaped(a)
    return _tensor_from_roc(exp.(A), a.device)
end

function rocm_log(a::AsterFlow.Tensor)
    A = _as_roc_shaped(a)
    return _tensor_from_roc(log.(A), a.device)
end

function rocm_sqrt(a::AsterFlow.Tensor)
    A = _as_roc_shaped(a)
    return _tensor_from_roc(sqrt.(A), a.device)
end

function rocm_softmax_rows(a::AsterFlow.Tensor{T,2}) where {T}
    X = _as_roc_shaped(a)
    m = maximum(X; dims = 2)
    e = exp.(X .- m)
    y = e ./ sum(e; dims = 2)
    return _tensor_from_roc(y, a.device)
end

function rocm_softmax_rows_bwd(p::AsterFlow.Tensor{T,2}, g::AsterFlow.Tensor{T,2}) where {T}
    P = _as_roc_shaped(p)
    G = _as_roc_shaped(g)
    gin = P .* (G .- sum(G .* P; dims = 2))
    return _tensor_from_roc(gin, p.device)
end

function rocm_sum(a::AsterFlow.Tensor; dims = nothing)
    A = _as_roc_shaped(a)
    if dims === nothing
        s = sum(A)
        return _tensor_from_roc(ROCArray(fill(s, 1, 1)), a.device)
    end
    return _tensor_from_roc(sum(A; dims = dims), a.device)
end

function rocm_mean(a::AsterFlow.Tensor; dims = nothing)
    A = _as_roc_shaped(a)
    if dims === nothing
        m = sum(A) / length(A)
        return _tensor_from_roc(ROCArray(fill(m, 1, 1)), a.device)
    end
    d = dims isa Integer ? Int(dims) : Int(dims[1])
    n = size(A, d)
    return _tensor_from_roc(sum(A; dims = dims) ./ n, a.device)
end

function rocm_tanh(a::AsterFlow.Tensor)
    A = _as_roc_shaped(a)
    return _tensor_from_roc(tanh.(A), a.device)
end

function rocm_sigmoid(a::AsterFlow.Tensor)
    A = _as_roc_shaped(a)
    y = 1 ./ (1 .+ exp.(-A))
    return _tensor_from_roc(y, a.device)
end

function rocm_leaky_relu(a::AsterFlow.Tensor{T,N}, α::AbstractFloat) where {T,N}
    A = _as_roc_shaped(a)
    αt = T(α)
    return _tensor_from_roc(ifelse.(A .> 0, A, αt .* A), a.device)
end

function rocm_leaky_relu_bwd(
    grad::AsterFlow.Tensor{T,N},
    inp::AsterFlow.Tensor{T,N},
    α::Float32,
) where {T,N}
    G = _as_roc_shaped(grad)
    X = _as_roc_shaped(inp)
    αt = T(α)
    return _tensor_from_roc(G .* ifelse.(X .> 0, one(T), αt), inp.device)
end

function rocm_sgd!(
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
        @. ps = ps - lr * (gs + wd * ps)
        return nothing
    end
    vbuf === nothing && error("ROCm SGD: momentum>0 时需要速度缓冲")
    vs = @view vbuf.storage[(vbuf.offset+1):(vbuf.offset+n)]
    @. vs = momentum * vs + (1f0 - dampening) * gs
    if nesterov
        @. ps = ps - lr * (gs + momentum * vs + wd * ps)
    else
        @. ps = ps - lr * (vs + wd * ps)
    end
    return nothing
end

function rocm_adamw!(
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
        opt.m[id] = AMDGPU.zeros(Float32, n)
        opt.v[id] = AMDGPU.zeros(Float32, n)
    end
    mvec = opt.m[id]
    vvec = opt.v[id]
    ps = @view p.storage[(p.offset+1):(p.offset+n)]
    gs = @view g.storage[(g.offset+1):(g.offset+n)]
    @. mvec = b1 * mvec + (1f0 - b1) * gs
    @. vvec = b2 * vvec + (1f0 - b2) * (gs * gs)
    fac_m = 1f0 / (1f0 - b1^t)
    fac_v = 1f0 / (1f0 - b2^t)
    @. ps = ps * (1f0 - lr * wd) - lr * ((mvec * fac_m) / (sqrt(vvec * fac_v) + eps))
    return nothing
end

function rocm_adam!(
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
        opt.m[id] = AMDGPU.zeros(Float32, n)
        opt.v[id] = AMDGPU.zeros(Float32, n)
    end
    mvec = opt.m[id]
    vvec = opt.v[id]
    ps = @view p.storage[(p.offset+1):(p.offset+n)]
    gs = @view g.storage[(g.offset+1):(g.offset+n)]
    @. mvec = b1 * mvec + (1f0 - b1) * (gs + wd * ps)
    @. vvec = b2 * vvec + (1f0 - b2) * ((gs + wd * ps) * (gs + wd * ps))
    fac_m = 1f0 / (1f0 - b1^t)
    fac_v = 1f0 / (1f0 - b2^t)
    @. ps = ps - lr * ((mvec * fac_m) / (sqrt(vvec * fac_v) + eps))
    return nothing
end

function _rocm_runtime_probe(device_id::Int)::Bool
    try
        AMDGPU.functional() || return false
        devs = AMDGPU.devices()
        return length(devs) > device_id
    catch
        return false
    end
end

const _ROCM_BACKEND = :rocm

function __init__()
    AsterFlow.register_backend_runtime!(_ROCM_BACKEND, _rocm_runtime_probe)
    AsterFlow.register_tensor_upload!(_ROCM_BACKEND, rocm_tensor_upload!)
    AsterFlow.register_dev_ones_gpu!(_ROCM_BACKEND, rocm_dev_ones!)
    AsterFlow.register_dev_zeros_gpu!(_ROCM_BACKEND, rocm_dev_zeros!)
    AsterFlow.register_dev_fill_gpu!(_ROCM_BACKEND, rocm_dev_fill!)
    AsterFlow.register_materialize_strided_gpu!(_ROCM_BACKEND, rocm_materialize_strided!)
    AsterFlow.register_contiguous_accelerator!(_ROCM_BACKEND, rocm_contiguous_accel!)
    AsterFlow.register_sgd_accelerator!(_ROCM_BACKEND, rocm_sgd!)
    AsterFlow.register_adamw_accelerator!(_ROCM_BACKEND, rocm_adamw!)
    AsterFlow.register_adam_accelerator!(_ROCM_BACKEND, rocm_adam!)
    AsterFlow.register_op!(:add, _ROCM_BACKEND, rocm_add)
    AsterFlow.register_op!(:sub, _ROCM_BACKEND, rocm_sub)
    AsterFlow.register_op!(:mul, _ROCM_BACKEND, rocm_mul)
    AsterFlow.register_op!(:div, _ROCM_BACKEND, rocm_div)
    AsterFlow.register_op!(:matmul, _ROCM_BACKEND, rocm_matmul)
    AsterFlow.register_op!(:scale, _ROCM_BACKEND, rocm_scale)
    AsterFlow.register_op!(:relu, _ROCM_BACKEND, rocm_relu)
    AsterFlow.register_op!(:relu_bwd, _ROCM_BACKEND, rocm_relu_bwd)
    AsterFlow.register_op!(:softmax_rows, _ROCM_BACKEND, rocm_softmax_rows)
    AsterFlow.register_op!(:softmax_rows_bwd, _ROCM_BACKEND, rocm_softmax_rows_bwd)
    AsterFlow.register_op!(:exp, _ROCM_BACKEND, rocm_exp)
    AsterFlow.register_op!(:log, _ROCM_BACKEND, rocm_log)
    AsterFlow.register_op!(:sqrt, _ROCM_BACKEND, rocm_sqrt)
    AsterFlow.register_op!(:sum, _ROCM_BACKEND, rocm_sum)
    AsterFlow.register_op!(:mean, _ROCM_BACKEND, rocm_mean)
    AsterFlow.register_op!(:tanh, _ROCM_BACKEND, rocm_tanh)
    AsterFlow.register_op!(:sigmoid, _ROCM_BACKEND, rocm_sigmoid)
    AsterFlow.register_op!(:leaky_relu, _ROCM_BACKEND, rocm_leaky_relu)
    AsterFlow.register_op!(:leaky_relu_bwd, _ROCM_BACKEND, rocm_leaky_relu_bwd)
    return nothing
end

end # module
