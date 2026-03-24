## Dispatcher: (op, backend) -> kernel；支持 fallback 链与执行模式键

const OpRegistry = Dict{Tuple{Symbol,Symbol},Any}()

"""与 PyTorch DispatchKey 概念对齐的简化版：`:eager`、`:debug`、预留 `:compiled`。"""
const ASTERFLOW_EXECUTION_MODE = Ref{Symbol}(:eager)

"""
后端符号的 fallback 顺序，例如 `[:cuda, :cpu]`。
仅当主后端未注册对应 `op` 时依次尝试；**不会**自动搬运张量到 CPU。
"""
const DISPATCH_FALLBACK_CHAIN = Dict{Symbol,Vector{Symbol}}()

function register_dispatch_fallback!(backend::Symbol, chain::Vector{Symbol})
    DISPATCH_FALLBACK_CHAIN[backend] = chain
    return nothing
end

function register_op!(op::Symbol, backend::Symbol, fn)
    OpRegistry[(op, backend)] = fn
    return nothing
end

function _dispatch_chain(dev::Device)
    b = device_backend(dev)
    ch = get(DISPATCH_FALLBACK_CHAIN, b, Symbol[])
    return vcat([b], ch)
end

function _lookup_kernel(op::Symbol, backend::Symbol)
    return get(OpRegistry, (op, backend), nothing)
end

function dispatch_op(op::Symbol, dev::Device, args...; kwargs...)
    mode = ASTERFLOW_EXECUTION_MODE[]
    if mode === :debug
        # 占位：可在此接入日志 / 断言
    end
    for b in _dispatch_chain(dev)
        fn = _lookup_kernel(op, b)
        if fn !== nothing
            return fn(args...; kwargs...)
        end
    end
    error("No kernel registered for op=$(repr(op)) backend=$(repr(device_backend(dev)))")
end

"""返回 `Dict{Symbol, Vector{Symbol}}`：backend -> 已注册的 op 列表（排序）。"""
function registered_ops_report()
    d = Dict{Symbol,Vector{Symbol}}()
    for (op, b) in keys(OpRegistry)
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
)
    register_op!(op, backend, forward_fn)
    if backward_factory !== nothing
        _CUSTOM_OP_BACKWARD[(op, backend)] = backward_factory
    end
    return nothing
end

const _CUSTOM_OP_BACKWARD = Dict{Tuple{Symbol,Symbol},Any}()

function custom_backward_factory(op::Symbol, backend::Symbol)
    return get(_CUSTOM_OP_BACKWARD, (op, backend), nothing)
end
