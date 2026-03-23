mutable struct GradMode
    enabled::Bool
end

const _grad_mode = GradMode(true)

function no_grad(f)
    prev = _grad_mode.enabled
    _grad_mode.enabled = false
    try
        return f()
    finally
        _grad_mode.enabled = prev
    end
end

grad_enabled() = _grad_mode.enabled
