## 子图 IR（Julia 侧），可序列化后交给 C++ 后端（当前为占位）。

struct IRValue
    id::Int
    dtype::String
    shape::Vector{Int}
end

@enum IROpKind begin
    IR_Add
    IR_Mul
    IR_MatMul
    IR_ReLU
    IR_Sum
end

struct IRNode
    kind::IROpKind
    inputs::Vector{Int}
    outputs::Vector{Int}
    attrs::Dict{String,Any}
end

mutable struct IRGraph
    next_id::Int
    inputs::Vector{IRValue}
    nodes::Vector{IRNode}
    outputs::Vector{Int}
end

IRGraph() = IRGraph(1, IRValue[], IRNode[], Int[])

function ir_new_input!(g::IRGraph, dtype::String, shape::Vector{Int})
    id = g.next_id
    g.next_id += 1
    v = IRValue(id, dtype, shape)
    push!(g.inputs, v)
    return id
end

function ir_append_node!(g::IRGraph, kind::IROpKind, inputs::Vector{Int}, out_shapes::Vector{Vector{Int}}; attrs = Dict{String,Any}())
    out_ids = Int[]
    for sh in out_shapes
        id = g.next_id
        g.next_id += 1
        push!(out_ids, id)
    end
    push!(g.nodes, IRNode(kind, copy(inputs), copy(out_ids), attrs))
    return out_ids
end

function ir_set_outputs!(g::IRGraph, ids::Vector{Int})
    g.outputs = copy(ids)
    return g
end

"""占位：由输入 shape 推导单输出 shape（静态规则）；动态 rank 需运行时填充 attrs。"""
function ir_infer_binary_output_shape(sh1::Vector{Int}, sh2::Vector{Int}, op::IROpKind)
    if op == IR_MatMul && length(sh1) >= 2 && length(sh2) >= 2
        return [sh1[1], sh2[2]]
    end
    if op == IR_Add || op == IR_Mul
        return length(sh1) >= length(sh2) ? copy(sh1) : copy(sh2)
    end
    return copy(sh1)
end
