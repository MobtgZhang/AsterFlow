"""
`dev_*` / `tensor_on_device`：CPU 直接构造；`AcceleratorDevice` 由扩展通过 `register_*_gpu!` 注册。
"""

function contiguous_accelerator_impl(t::Tensor)
    return _dispatch_contiguous_accelerator(t)
end

function dev_ones_gpu(::Type{T}, sz::Tuple{Vararg{Int}}, dev::AcceleratorDevice) where {T}
    return _dispatch_dev_ones_gpu(T, sz, dev)
end

function dev_zeros_gpu(::Type{T}, sz::Tuple{Vararg{Int}}, dev::AcceleratorDevice) where {T}
    return _dispatch_dev_zeros_gpu(T, sz, dev)
end

function dev_fill_gpu(::Type{T}, sz::Tuple{Vararg{Int}}, v, dev::AcceleratorDevice) where {T}
    return _dispatch_dev_fill_gpu(T, sz, v, dev)
end

function dev_ones(::Type{T}, sz::Tuple{Vararg{Int}}, dev::CPUDevice) where {T}
    return tensor(ones(T, sz); device = dev, requires_grad = false)
end

function dev_ones(::Type{T}, sz::Tuple{Vararg{Int}}, dev::AcceleratorDevice) where {T}
    return dev_ones_gpu(T, sz, dev)
end

function dev_zeros(::Type{T}, sz::Tuple{Vararg{Int}}, dev::CPUDevice) where {T}
    return tensor(zeros(T, sz); device = dev, requires_grad = false)
end

function dev_zeros(::Type{T}, sz::Tuple{Vararg{Int}}, dev::AcceleratorDevice) where {T}
    return dev_zeros_gpu(T, sz, dev)
end

function dev_fill(::Type{T}, sz::Tuple{Vararg{Int}}, v, dev::CPUDevice) where {T}
    return tensor(fill(T(v), sz); device = dev, requires_grad = false)
end

function dev_fill(::Type{T}, sz::Tuple{Vararg{Int}}, v, dev::AcceleratorDevice) where {T}
    return dev_fill_gpu(T, sz, v, dev)
end

function tensor_on_device(::Type{T}, a::Array{T,N}, dev::CPUDevice; requires_grad::Bool = false) where {T,N}
    sz = size(a)
    v = Base.vec(copy(a))
    st = column_major_strides(sz)
    return Tensor{T,N}(v, sz, st, 0, dev, requires_grad, nothing, nothing)
end

function tensor_on_device(::Type{T}, a::Array{T,N}, dev::AcceleratorDevice; requires_grad::Bool = false) where {T,N}
    return tensor_on_device_gpu(a, dev; requires_grad = requires_grad)
end

function tensor_on_device_gpu(a::Array{T,N}, dev::AcceleratorDevice; requires_grad::Bool = false) where {T,N}
    return _dispatch_tensor_upload(a, dev; requires_grad = requires_grad)
end
