## 加速器上的优化器步进：各后端扩展在 __init__ 中 `register_*_accelerator!` 注册实现。
## 同步：默认在默认 CUDA 流上异步执行；`to_array` / 主机读回会隐式同步。多流策略见 docs/accelerator-streams.md。

const _OPT_SGD = Dict{Symbol,Base.Callable}()
const _OPT_ADAMW = Dict{Symbol,Base.Callable}()
const _OPT_ADAM = Dict{Symbol,Base.Callable}()
const _OPT_RMSPROP = Dict{Symbol,Base.Callable}()
const _OPT_ADAGRAD = Dict{Symbol,Base.Callable}()
const _OPT_ADADELTA = Dict{Symbol,Base.Callable}()
const _OPT_ADAMAX = Dict{Symbol,Base.Callable}()

function register_sgd_accelerator!(backend::Symbol, fn!::Base.Callable)
    _OPT_SGD[backend] = fn!
    return nothing
end

function register_adamw_accelerator!(backend::Symbol, fn!::Base.Callable)
    _OPT_ADAMW[backend] = fn!
    return nothing
end

function register_adam_accelerator!(backend::Symbol, fn!::Base.Callable)
    _OPT_ADAM[backend] = fn!
    return nothing
end

function register_rmsprop_accelerator!(backend::Symbol, fn!::Base.Callable)
    _OPT_RMSPROP[backend] = fn!
    return nothing
end

function register_adagrad_accelerator!(backend::Symbol, fn!::Base.Callable)
    _OPT_ADAGRAD[backend] = fn!
    return nothing
end

function register_adadelta_accelerator!(backend::Symbol, fn!::Base.Callable)
    _OPT_ADADELTA[backend] = fn!
    return nothing
end

function register_adamax_accelerator!(backend::Symbol, fn!::Base.Callable)
    _OPT_ADAMAX[backend] = fn!
    return nothing
end

"""占位：未来对接 buffer pool / 显存峰值统计。"""
function gpu_memory_stats_placeholder!()
    return nothing
end

"""
`vbuf`：与参数同长度的速度缓冲（CPU 为 `Vector{Float32}`，CUDA/ROCm 为 `CuVector`/`ROCVector`）。
`momentum>0` 时使用；否则可传 `nothing`。
"""
function _sgd_accelerator!(
    p::Tensor,
    g::Tensor,
    lr::Float32,
    wd::Float32,
    vbuf,
    momentum::Float32,
    dampening::Float32,
    nesterov::Bool,
)
    p.device isa AcceleratorDevice || error("SGD 加速器步进: 需要 AcceleratorDevice")
    b = p.device.backend
    fn = get(_OPT_SGD, b, nothing)
    fn === nothing && error("SGD: 后端 :$(b) 未注册优化器内核，请加载对应扩展。")
    return fn(p, g, lr, wd, vbuf, momentum, dampening, nesterov)
end

function _adamw_accelerator!(
    p::Tensor,
    g::Tensor,
    opt::Any,
    t::Float32,
    lr::Float32,
    b1::Float32,
    b2::Float32,
    eps::Float32,
    wd::Float32,
)
    p.device isa AcceleratorDevice || error("AdamW: 需要 AcceleratorDevice")
    b = p.device.backend
    fn = get(_OPT_ADAMW, b, nothing)
    fn === nothing && error("AdamW: 后端 :$(b) 未注册优化器内核。")
    return fn(p, g, opt, t, lr, b1, b2, eps, wd)
end

function _adam_accelerator!(
    p::Tensor,
    g::Tensor,
    opt::Any,
    t::Float32,
    lr::Float32,
    b1::Float32,
    b2::Float32,
    eps::Float32,
    wd::Float32,
)
    p.device isa AcceleratorDevice || error("Adam: 需要 AcceleratorDevice")
    b = p.device.backend
    fn = get(_OPT_ADAM, b, nothing)
    fn === nothing && error("Adam: 后端 :$(b) 未注册优化器内核。")
    return fn(p, g, opt, t, lr, b1, b2, eps, wd)
end

function _rmsprop_accelerator!(
    p::Tensor,
    g::Tensor,
    opt::Any,
    lr::Float32,
    α::Float32,
    eps::Float32,
    wd::Float32,
    centered::Bool,
)
    p.device isa AcceleratorDevice || error("RMSprop: 后端 :$(p.device.backend) 未实现。")
    b = p.device.backend
    fn = get(_OPT_RMSPROP, b, nothing)
    fn === nothing && error("RMSprop: 后端 :$(b) 未注册优化器内核。")
    return fn(p, g, opt, lr, α, eps, wd, centered)
end

function _adagrad_accelerator!(p::Tensor, g::Tensor, opt::Any, lr::Float32, eps::Float32, wd::Float32)
    p.device isa AcceleratorDevice || error("Adagrad: 后端 :$(p.device.backend) 未实现。")
    b = p.device.backend
    fn = get(_OPT_ADAGRAD, b, nothing)
    fn === nothing && error("Adagrad: 后端 :$(b) 未注册优化器内核。")
    return fn(p, g, opt, lr, eps, wd)
end

function _adadelta_accelerator!(p::Tensor, g::Tensor, opt::Any, ρ::Float32, eps::Float32, wd::Float32)
    p.device isa AcceleratorDevice || error("Adadelta: 后端 :$(p.device.backend) 未实现。")
    b = p.device.backend
    fn = get(_OPT_ADADELTA, b, nothing)
    fn === nothing && error("Adadelta: 后端 :$(b) 未注册优化器内核。")
    return fn(p, g, opt, ρ, eps, wd)
end

function _adamax_accelerator!(
    p::Tensor,
    g::Tensor,
    opt::Any,
    t::Float32,
    lr::Float32,
    b1::Float32,
    b2::Float32,
    eps::Float32,
    wd::Float32,
)
    p.device isa AcceleratorDevice || error("Adamax: 后端 :$(p.device.backend) 未实现。")
    b = p.device.backend
    fn = get(_OPT_ADAMAX, b, nothing)
    fn === nothing && error("Adamax: 后端 :$(b) 未注册优化器内核。")
    return fn(p, g, opt, t, lr, b1, b2, eps, wd)
end
