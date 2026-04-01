const ScalarLike = Union{AbstractFloat,Integer,Bool}

## Column-major strides (Julia / BLAS layout): dim 1 varies fastest.
@inline function column_major_strides(sz::NTuple{N,Int}) where {N}
    if N == 0
        return ()
    end
    s = Vector{Int}(undef, N)
    s[1] = 1
    for k in 2:N
        s[k] = s[k-1] * sz[k-1]
    end
    return NTuple{N,Int}(s)
end

"""
`Tensor` 使用 1D `storage` + `size` + `strides` + `offset`（元素偏移），与 PyTorch 式元数据一致。

- **`storage`**：底层缓冲（CPU 为 `Vector`，CUDA 等为 `CuArray` 等）。
- **`size` / `strides` / `offset`**：列主（dim 1 最快），与 Julia `Array` / BLAS 一致。
- **`device`**：逻辑设备，供 `dispatch_op` 选择内核。
- **`requires_grad`**：是否参与 autograd；叶子可写 `requires_grad!`。
- **`grad`**：反向累加缓冲区；`zero_grad!` 清空。
- **`grad_fn`**：产生本张量的 `Node`；`detach_tensor` 会置空。
- **`version_ref`**：与共享 `storage` 的视图共用同一 `Ref{UInt64}`；inplace 写入递增，供 autograd 检测陈旧计算图。

共享 storage 的视图（`view_tensor` / `permute` / 部分 `reshape`）在反传时通过各自 `grad_fn` 把梯度写回同一 storage 上的逻辑位置。
"""
mutable struct Tensor{T,N}
    storage::AbstractVector{T}
    size::NTuple{N,Int}
    strides::NTuple{N,Int}
    offset::Int
    device::Device
    requires_grad::Bool
    grad::Union{Nothing,Tensor{T,N}}
    grad_fn::Any
    version_ref::Ref{UInt64}
end

@inline function _new_tensor_version_ref()
    return Ref{UInt64}(UInt64(0))
end

@inline function _bump_tensor_version!(t::Tensor)
    r = t.version_ref
    r[] = r[] + UInt64(1)
    return nothing
end

"""当前 autograd 版本号（与同 storage 视图共享）。"""
tensor_version(t::Tensor) = t.version_ref[]

@inline function _verify_saved_tensor_version!(t::Tensor, saved::UInt64, name::String = "tensor")
    t.version_ref[] == saved && return nothing
    error("backward: tensor was modified in-place after forward ($name)")
end

Base.eltype(::Tensor{T,N}) where {T,N} = T
Base.ndims(::Tensor{T,N}) where {T,N} = N

device(t::Tensor) = t.device
Base.size(t::Tensor) = t.size
Base.size(t::Tensor, d::Integer) = t.size[Int(d)]
strides_tensor(t::Tensor) = t.strides
storage_offset(t::Tensor) = t.offset
numel(t::Tensor) = prod(t.size)

function is_contiguous(t::Tensor)
    st = column_major_strides(t.size)
    return t.strides == st && t.offset == 0
end

function _linear_index(t::Tensor, I::CartesianIndex{N}) where {N}
    o = t.offset
    @inbounds for k in 1:N
        o += (I[k] - 1) * t.strides[k]
    end
    o + 1 # 1-based Vector index
end

@inline function getindex_tensor(t::Tensor{T,N}, I::CartesianIndex{N}) where {T,N}
    @inbounds t.storage[_linear_index(t, I)]::T
end

function setindex_tensor!(t::Tensor{T,N}, v, I::CartesianIndex{N}) where {T,N}
    if t.requires_grad && grad_enabled()
        error("setindex_tensor!: 在追踪梯度时禁止 inplace 写入 requires_grad 张量")
    end
    @inbounds t.storage[_linear_index(t, I)] = v
    _bump_tensor_version!(t)
    return t
end

function tensor(data::Array{T,N}; device::Device = CPUDevice(), requires_grad::Bool = false) where {T,N}
    if is_accelerator(device)
        return tensor_on_device(T, data, device; requires_grad = requires_grad)
    end
    sz = size(data)
    v = Base.vec(copy(data)) # own column-major copy
    st = column_major_strides(sz)
    Tensor{T,N}(v, sz, st, 0, device, requires_grad, nothing, nothing, _new_tensor_version_ref())
end

function _storage_to_cpu_array(s::AbstractVector{T}) where {T}
    s isa Vector{T} && return s
    return Vector{T}(s)
end

function to_array(t::Tensor{T,N}) where {T,N}
    if !is_contiguous(t)
        return _materialize_strided(t)
    end
    buf = _storage_to_cpu_array(t.storage)
    reshape(buf[(t.offset+1):(t.offset+numel(t))], t.size)
end

function _materialize_strided(t::Tensor{T,N}) where {T,N}
    if accelerator_storage(t.storage)
        return _materialize_strided_gpu(t)
    end
    out = Array{T,N}(undef, t.size)
    for I in CartesianIndices(t.size)
        out[I] = getindex_tensor(t, I)
    end
    return out
end

function contiguous(t::Tensor{T,N}) where {T,N}
    is_contiguous(t) && return t
    if accelerator_storage(t.storage)
        return contiguous_accelerator_impl(t)
    end
    return Tensor{T,N}(
        Base.vec(_materialize_strided(t)),
        t.size,
        column_major_strides(t.size),
        0,
        t.device,
        t.requires_grad,
        nothing,
        t.grad_fn,
        _new_tensor_version_ref(),
    )
end

## View: 共享 storage，新 shape/strides/offset
function view_tensor(
    t::Tensor{T,N},
    newsize::NTuple{M,Int},
    newstrides::NTuple{M,Int},
    newoffset::Int,
) where {T,N,M}
    Tensor{T,M}(
        t.storage,
        newsize,
        newstrides,
        t.offset + newoffset,
        t.device,
        t.requires_grad,
        nothing,
        t.grad_fn,
        t.version_ref,
    )
end

"""无 autograd 记录的 permute（内部反向与 matmul 等使用）。"""
function _permute_storage(t::Tensor{T,N}, perm::NTuple{N,Int}) where {T,N}
    isperm(perm) || error("invalid permutation")
    newsz = ntuple(k -> t.size[perm[k]], N)
    newst = ntuple(k -> t.strides[perm[k]], N)
    return Tensor{T,N}(t.storage, newsz, newst, t.offset, t.device, false, nothing, nothing, t.version_ref)
end

"""无 autograd 记录的 reshape（假定调用方已保证 contiguous 或语义正确）。"""
function _reshape_storage(t::Tensor, newsize::Tuple{Vararg{Int}})
    if prod(newsize) != numel(t)
        error("reshape: numel mismatch")
    end
    if !is_contiguous(t)
        return _reshape_storage(contiguous(t), newsize)
    end
    N = length(newsize)
    return Tensor{eltype(t),N}(
        t.storage,
        newsize,
        column_major_strides(newsize),
        t.offset,
        t.device,
        false,
        nothing,
        nothing,
        t.version_ref,
    )
end

"""截断计算图，保留数据与设备（PyTorch `tensor.detach()`；Julia 中避免与 `Base.detach(::Cmd)` 同名冲突）。"""
function detach_tensor(t::Tensor{T,N}) where {T,N}
    Tensor{T,N}(
        t.storage,
        t.size,
        t.strides,
        t.offset,
        t.device,
        false,
        nothing,
        nothing,
        t.version_ref,
    )
end

function Base.copy(t::Tensor{T,N}) where {T,N}
    Tensor{T,N}(
        copy(t.storage),
        t.size,
        t.strides,
        t.offset,
        t.device,
        false,
        nothing,
        nothing,
        _new_tensor_version_ref(),
    )
end

function _materialize_strided_gpu(t::Tensor)
    return _dispatch_materialize_strided_gpu(t)
end
