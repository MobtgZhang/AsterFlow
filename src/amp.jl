## 混合精度：autocast 上下文 + GradScaler（与 PyTorch AMP 方向一致）

const _AUTOCAST_ENABLED = Ref(false)
const _AUTOCAST_ELTYPE = Ref{Type{<:AbstractFloat}}(Float32)

autocast_enabled() = _AUTOCAST_ENABLED[]

"""当前 autocast 目标浮点类型（白名单算子可在此类型上执行）。"""
autocast_eltype() = _AUTOCAST_ELTYPE[]

"""
    @autocast T expr

在 `expr` 求值期间开启 `autocast`，目标元素类型 `T`（如 `Float16`）。
算子是否在低精度执行取决于各层实现；核心算子可逐步接入白名单。
"""
macro autocast(T, expr)
    quote
        prev = _AUTOCAST_ENABLED[]
        prevT = _AUTOCAST_ELTYPE[]
        _AUTOCAST_ENABLED[] = true
        _AUTOCAST_ELTYPE[] = $(esc(T))
        try
            $(esc(expr))
        finally
            _AUTOCAST_ENABLED[] = prev
            _AUTOCAST_ELTYPE[] = prevT
        end
    end
end

"""
动态 loss scaling：与 `backward` 配合时，在 `scale(loss)` 上反传，再在 `step!` 前 `unscale!` / `update!`。
当前为 CPU 浮点状态机占位；与 GPU 内核融合前可先用于算法验证。
"""
mutable struct GradScaler
    scale::Float32
    growth_factor::Float32
    backoff_factor::Float32
    growth_interval::Int
    step_count::Int
    found_inf::Bool
end

function GradScaler(;
    init_scale = 2.0f0,
    growth_factor = 2.0f0,
    backoff_factor = 0.5f0,
    growth_interval = 2000,
)
    return GradScaler(init_scale, growth_factor, backoff_factor, growth_interval, 0, false)
end

"""对 loss 张量乘以 scaler（再 `backward`）。"""
function scale_loss(gs::GradScaler, t::Tensor)
    return scale_tensor(t, gs.scale)
end

"""将参数梯度除以当前 scale（调用方应保证已 `backward`）。"""
function unscale_grads!(gs::GradScaler, params::AbstractVector{Tensor})
    invs = 1.0f0 / gs.scale
    for p in params
        g = p.grad
        g === nothing && continue
        p.grad = mul_scalar_tensor(g, invs)
    end
    return gs
end

function update!(gs::GradScaler)
    gs.step_count += 1
    if gs.found_inf
        gs.scale = max(gs.scale * gs.backoff_factor, 1.0f-4)
        gs.found_inf = false
    elseif gs.step_count % gs.growth_interval == 0
        gs.scale = min(gs.scale * gs.growth_factor, 65536.0f0)
    end
    return gs
end
