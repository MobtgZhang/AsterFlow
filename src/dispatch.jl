## Dispatcher: (op, backend, dtype?) -> kernel；支持 fallback 链与执行模式键

const OpRegistryKey = Tuple{Symbol,Symbol,Union{Nothing,Type}}
const OpRegistry = Dict{OpRegistryKey,Any}()

"""与 PyTorch DispatchKey 概念对齐的简化版：`:eager`、`:debug`、预留 `:compiled`。"""
const ASTERFLOW_EXECUTION_MODE = Ref{Symbol}(:eager)

"""
后端符号的 fallback 顺序，例如 `[:cuda, :cpu]`。
仅当主后端未注册对应 `op`（含 dtype 解析）时依次尝试；**不会**自动搬运张量到 CPU，
但 CPU 内核可通过 `to_array` 读回设备数据再写回（慢路径）。
"""
const DISPATCH_FALLBACK_CHAIN = Dict{Symbol,Vector{Symbol}}()

function register_dispatch_fallback!(backend::Symbol, chain::Vector{Symbol})
    DISPATCH_FALLBACK_CHAIN[backend] = chain
    return nothing
end

"""
注册算子实现。`dtype=nothing` 表示通配（任意 `eltype`）；否则仅当首个 `Tensor` 参数的类型为 `dtype` 时优先匹配。
"""
function register_op!(op::Symbol, backend::Symbol, fn; dtype::Union{Nothing,Type} = nothing)
    OpRegistry[(op, backend, dtype)] = fn
    return nothing
end

function _dispatch_chain(dev::Device)
    b = device_backend(dev)
    ch = get(DISPATCH_FALLBACK_CHAIN, b, Symbol[])
    return vcat([b], ch)
end

function _first_tensor_eltype_from_args(args::Tuple)
    for a in args
        a isa Tensor && return eltype(a)
    end
    return nothing
end

function _lookup_kernel(op::Symbol, backend::Symbol, T::Union{Nothing,Type})
    if T !== nothing
        fn = get(OpRegistry, (op, backend, T), nothing)
        fn !== nothing && return fn
    end
    return get(OpRegistry, (op, backend, nothing), nothing)
end

function dispatch_op(op::Symbol, dev::Device, args...; kwargs...)
    mode = ASTERFLOW_EXECUTION_MODE[]
    if mode === :debug
        # 占位：可在此接入日志 / 断言
    end
    T = _first_tensor_eltype_from_args(args)
    for b in _dispatch_chain(dev)
        fn = _lookup_kernel(op, b, T)
        if fn !== nothing
            return fn(args...; kwargs...)
        end
    end
    error("No kernel registered for op=$(repr(op)) backend=$(repr(device_backend(dev))) dtype=$(repr(T))")
end

"""返回 `Dict{Symbol, Vector{Symbol}}`：backend -> 已注册的 op 列表（去重排序）。"""
function registered_ops_report()
    d = Dict{Symbol,Vector{Symbol}}()
    for (op, b, _) in keys(OpRegistry)
        push!(get!(Vector{Symbol}, d, b), op)
    end
    for v in values(d)
        sort!(unique!(v))
    end
    return d
end

"""注册自定义算子 + 可选反向节点工厂（第三方扩展用）。"""
function register_custom_op!(
    op::Symbol,
    backend::Symbol,
    forward_fn;
    backward_factory = nothing,
    dtype::Union{Nothing,Type} = nothing,
)
    register_op!(op, backend, forward_fn; dtype = dtype)
    if backward_factory !== nothing
        _CUSTOM_OP_BACKWARD[(op, backend, dtype)] = backward_factory
    end
    return nothing
end

const _CUSTOM_OP_BACKWARD = Dict{Tuple{Symbol,Symbol,Union{Nothing,Type}},Any}()

function custom_backward_factory(op::Symbol, backend::Symbol, T::Union{Nothing,Type} = nothing)
    if T !== nothing
        fac = get(_CUSTOM_OP_BACKWARD, (op, backend, T), nothing)
        fac !== nothing && return fac
    end
    return get(_CUSTOM_OP_BACKWARD, (op, backend, nothing), nothing)
end
