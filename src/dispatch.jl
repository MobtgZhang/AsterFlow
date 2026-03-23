## Dispatcher: (op, backend::Symbol) -> kernel，与具体厂商解耦

const OpRegistry = Dict{Tuple{Symbol,Symbol},Any}()

function register_op!(op::Symbol, backend::Symbol, fn)
    OpRegistry[(op, backend)] = fn
    return nothing
end

function dispatch_op(op::Symbol, dev::Device, args...; kwargs...)
    b = device_backend(dev)
    key = (op, b)
    if !haskey(OpRegistry, key)
        error("No kernel registered for op=$(repr(op)) backend=$(repr(b))")
    end
    return OpRegistry[key](args...; kwargs...)
end
