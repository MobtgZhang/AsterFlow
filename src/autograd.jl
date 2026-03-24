function backward(root::Tensor; retain_graph::Bool = false, gradient::Union{Nothing,Tensor} = nothing)
    if gradient === nothing
        numel(root) != 1 &&
            error("backward: 非标量张量须提供 gradient=（与 root 同形状的梯度张量）")
        root.grad === nothing || error("backward: 请先 zero_grad! 再对同一 loss 反传")
        T = eltype(root)
        root.grad = dev_ones(T, size(root), root.device)
    else
        size(gradient) == size(root) ||
            error("backward: gradient 形状须与 root 一致 $(size(gradient)) vs $(size(root))")
        root.grad === nothing || error("backward: root.grad 已存在，请先 zero_grad!")
        root.grad = gradient
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
        no_grad() do
            apply_backward!(fn, g)
        end
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
