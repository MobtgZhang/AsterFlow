## 多后端扩展共享入口：按 `device.backend` 查表，避免 CUDA/ROCm 等方法覆盖冲突。

const _REG_TENSOR_UPLOAD = Dict{Symbol,Base.Callable}()
const _REG_DEV_ONES = Dict{Symbol,Base.Callable}()
const _REG_DEV_ZEROS = Dict{Symbol,Base.Callable}()
const _REG_DEV_FILL = Dict{Symbol,Base.Callable}()
const _REG_MATERIALIZE_STRIDED_GPU = Dict{Symbol,Base.Callable}()
const _REG_CONTIGUOUS_ACCEL = Dict{Symbol,Base.Callable}()

function register_tensor_upload!(backend::Symbol, fn!::Base.Callable)
    _REG_TENSOR_UPLOAD[backend] = fn!
    return nothing
end

function register_dev_ones_gpu!(backend::Symbol, fn!::Base.Callable)
    _REG_DEV_ONES[backend] = fn!
    return nothing
end

function register_dev_zeros_gpu!(backend::Symbol, fn!::Base.Callable)
    _REG_DEV_ZEROS[backend] = fn!
    return nothing
end

function register_dev_fill_gpu!(backend::Symbol, fn!::Base.Callable)
    _REG_DEV_FILL[backend] = fn!
    return nothing
end

function register_materialize_strided_gpu!(backend::Symbol, fn!::Base.Callable)
    _REG_MATERIALIZE_STRIDED_GPU[backend] = fn!
    return nothing
end

function register_contiguous_accelerator!(backend::Symbol, fn!::Base.Callable)
    _REG_CONTIGUOUS_ACCEL[backend] = fn!
    return nothing
end

function _dispatch_tensor_upload(a::Array{T,N}, dev::AcceleratorDevice; requires_grad::Bool = false) where {T,N}
    fn = get(_REG_TENSOR_UPLOAD, dev.backend, nothing)
    fn === nothing && error(
        "tensor_on_device(AcceleratorDevice(:$(dev.backend), ...)) 需要加载对应扩展（如 CUDA.jl / AMDGPU.jl）。",
    )
    return fn(a, dev; requires_grad = requires_grad)
end

function _dispatch_dev_ones_gpu(::Type{T}, sz::Tuple{Vararg{Int}}, dev::AcceleratorDevice) where {T}
    fn = get(_REG_DEV_ONES, dev.backend, nothing)
    fn === nothing &&
        error("dev_ones(AcceleratorDevice(:$(dev.backend), ...)) 需要加载对应扩展。")
    return fn(T, sz, dev)
end

function _dispatch_dev_zeros_gpu(::Type{T}, sz::Tuple{Vararg{Int}}, dev::AcceleratorDevice) where {T}
    fn = get(_REG_DEV_ZEROS, dev.backend, nothing)
    fn === nothing &&
        error("dev_zeros(AcceleratorDevice(:$(dev.backend), ...)) 需要加载对应扩展。")
    return fn(T, sz, dev)
end

function _dispatch_dev_fill_gpu(::Type{T}, sz::Tuple{Vararg{Int}}, v, dev::AcceleratorDevice) where {T}
    fn = get(_REG_DEV_FILL, dev.backend, nothing)
    fn === nothing &&
        error("dev_fill(AcceleratorDevice(:$(dev.backend), ...)) 需要加载对应扩展。")
    return fn(T, sz, v, dev)
end

function _dispatch_materialize_strided_gpu(t)
    b = device_backend(t.device)
    fn = get(_REG_MATERIALIZE_STRIDED_GPU, b, nothing)
    fn === nothing && error("加速器非连续张量物化需要加载对应后端扩展（:$(b)）。")
    return fn(t)
end

function _dispatch_contiguous_accelerator(t)
    b = device_backend(t.device)
    fn = get(_REG_CONTIGUOUS_ACCEL, b, nothing)
    fn === nothing && error("加速器 contiguous 需要加载对应后端扩展（:$(b)）。")
    return fn(t)
end
