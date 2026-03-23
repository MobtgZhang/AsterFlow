abstract type Module end

mutable struct Sequential <: Module
    layers::Tuple
    training::Bool
end

Sequential(layers...) = Sequential(layers, true)

function (m::Sequential)(x)
    y = x
    for layer in m.layers
        y = layer(y)
    end
    return y
end

function train!(m::Module)
    set_training!(m, true)
    return m
end

function evalmode!(m::Module)
    set_training!(m, false)
    return m
end

function set_training!(m::Module, v::Bool)
    if hasfield(typeof(m), :training)
        setfield!(m, :training, v)
    end
    for nm in fieldnames(typeof(m))
        x = getfield(m, nm)
        x isa Module && set_training!(x, v)
        if x isa Tuple || x isa AbstractVector
            for y in x
                y isa Module && set_training!(y, v)
            end
        end
        if x isa AbstractDict
            for y in values(x)
                y isa Module && set_training!(y, v)
            end
        end
    end
    return m
end

function params(m::Module)
    ps = Tensor[]
    _collect_params!(m, ps)
    return ps
end

function _collect_params!(m::Sequential, ps::Vector{Tensor})
    for layer in m.layers
        layer isa Module && _collect_params!(layer, ps)
    end
    return ps
end

mutable struct ModuleList <: Module
    layers::Vector
    training::Bool
end

ModuleList(xs...) = ModuleList(collect(Any, xs), true)

function Base.push!(m::ModuleList, layer)
    push!(m.layers, layer)
    return m
end

function Base.length(m::ModuleList)
    return length(m.layers)
end

function Base.getindex(m::ModuleList, i::Integer)
    return m.layers[i]
end

function (m::ModuleList)(x)
    y = x
    for layer in m.layers
        y = layer(y)
    end
    return y
end

function _collect_params!(m::ModuleList, ps::Vector{Tensor})
    for layer in m.layers
        layer isa Module && _collect_params!(layer, ps)
    end
    return ps
end

mutable struct ModuleDict <: Module
    layers::Dict{Symbol,Any}
    training::Bool
end

function ModuleDict(pairs::Pair{Symbol}...)
    d = Dict{Symbol,Any}()
    for (k, v) in pairs
        d[k] = v
    end
    return ModuleDict(d, true)
end

function Base.getindex(m::ModuleDict, k::Symbol)
    return m.layers[k]
end

function Base.setindex!(m::ModuleDict, v, k::Symbol)
    m.layers[k] = v
    return m
end

function _collect_params!(m::ModuleDict, ps::Vector{Tensor})
    for (_, layer) in m.layers
        layer isa Module && _collect_params!(layer, ps)
    end
    return ps
end

function _collect_params!(m::Module, ps::Vector{Tensor})
    m isa Sequential && return _collect_params!(m, ps)
    m isa ModuleList && return _collect_params!(m, ps)
    m isa ModuleDict && return _collect_params!(m, ps)
    for nm in fieldnames(typeof(m))
        x = getfield(m, nm)
        x isa Tensor && x.requires_grad && push!(ps, x)
        x isa Module && _collect_params!(x, ps)
        if x isa Tuple || x isa AbstractVector
            for y in x
                y isa Tensor && y.requires_grad && push!(ps, y)
                y isa Module && _collect_params!(y, ps)
            end
        end
    end
    return ps
end
