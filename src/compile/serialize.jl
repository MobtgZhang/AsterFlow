using TOML

function _graph_to_dict(g::IRGraph)
    Dict{String,Any}(
        "inputs" => [
            Dict{String,Any}("id" => v.id, "dtype" => v.dtype, "shape" => collect(v.shape)) for v in g.inputs
        ],
        "nodes" => [
            Dict{String,Any}(
                "kind" => Int(node.kind),
                "inputs" => collect(node.inputs),
                "outputs" => collect(node.outputs),
                "attrs" => node.attrs,
            ) for node in g.nodes
        ],
        "outputs" => collect(g.outputs),
    )
end

function graph_to_json(g::IRGraph)
    io = IOBuffer()
    TOML.print(io, _graph_to_dict(g))
    return String(take!(io))
end

function graph_from_json(s::String)
    d = TOML.parse(s)
    g = IRGraph()
    g.next_id = 1
    for inp in d["inputs"]
        id = Int(inp["id"])
        g.next_id = max(g.next_id, id + 1)
        sh = inp["shape"]
        shape = sh isa AbstractVector ? Int.(sh) : Int[sh]
        push!(g.inputs, IRValue(id, string(inp["dtype"]), shape))
    end
    for nd in d["nodes"]
        kind = IROpKind(Int(nd["kind"]))
        attrs = Dict{String,Any}()
        if haskey(nd, "attrs") && nd["attrs"] isa AbstractDict
            for (k, v) in nd["attrs"]
                attrs[string(k)] = v
            end
        end
        ins = Int.(nd["inputs"])
        outs = Int.(nd["outputs"])
        push!(g.nodes, IRNode(kind, ins, outs, attrs))
    end
    g.outputs = Int.(d["outputs"])
    return g
end
