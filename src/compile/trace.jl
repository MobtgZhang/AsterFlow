mutable struct TraceState
    active::Bool
    graph::Union{Nothing,IRGraph}
    tensor_to_id::IdDict{Tensor,Int}
end

const TRACE = TraceState(false, nothing, IdDict{Tensor,Int}())

function trace_begin!()
    g = IRGraph()
    TRACE.active = true
    TRACE.graph = g
    empty!(TRACE.tensor_to_id)
    return g
end

function trace_end!()
    g = TRACE.graph
    TRACE.active = false
    TRACE.graph = nothing
    empty!(TRACE.tensor_to_id)
    return g
end

tracing() = TRACE.active

"""
在 trace 激活时记录算子（占位）。动态 shape / 控制流分支可能导致单次 trace 无法覆盖全图；
生产路径应在静态子图边界调用 `trace_graph`。
"""
function trace_try_record!(::Symbol, ::Any, ::Any)
    return nothing
end

function trace_register_tensor!(t::Tensor, id::Int)
    TRACE.tensor_to_id[t] = id
    return id
end

function trace_tensor_id(t::Tensor)
    get(TRACE.tensor_to_id, t, nothing)
end

macro trace(expr)
    quote
        trace_begin!()
        local g
        try
            $(esc(expr))
        finally
            g = trace_end!()
        end
        g
    end
end

function trace_graph(f)
    trace_begin!()
    local g
    try
        f()
        g = TRACE.graph
    finally
        TRACE.active = false
        TRACE.graph = nothing
        empty!(TRACE.tensor_to_id)
    end
    return g
end
