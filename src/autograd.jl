function backward(root::Tensor; retain_graph::Bool = false)
    if numel(root) != 1
        error("backward: MVP 仅支持标量 loss（numel==1）")
    end
    if root.grad === nothing
        T = eltype(root)
        root.grad = dev_ones(T, (1, 1), root.device)
    end
    backward_from(root, retain_graph)
    return nothing
end

function backward_from(root::Tensor, retain_graph::Bool)
    visited = Set{Tensor}()
    stack = Tensor[]
    function visit(t::Tensor)
        t in visited && return
        push!(visited, t)
        fn = t.grad_fn
        if fn !== nothing
            for p in inputs_of(fn)
                p isa Tensor && visit(p)
            end
        end
        push!(stack, t)
    end
    visit(root)
    for t in Iterators.reverse(stack)
        g = t.grad
        g === nothing && continue
        fn = t.grad_fn
        fn === nothing && continue
        apply_backward!(fn, g)
    end
    if !retain_graph
        for t in stack
            t.grad_fn = nothing
        end
    end
    return nothing
end

function zero_grad!(t::Tensor)
    t.grad = nothing
    return t
end

function requires_grad!(t::Tensor, v::Bool = true)
    t.requires_grad = v
    return t
end

grad(t::Tensor) = t.grad
set_grad!(t::Tensor, g) = (t.grad = g; t)
