"""
    checkpoint(f, args...; kwargs...)

梯度检查点占位：当前等价于直接 `f(args...; kwargs...)`。
后续可改为：前向不保留中间激活、反向内重算 `f` 以降低显存。
"""
function checkpoint(f, args...; kwargs...)
    return f(args...; kwargs...)
end
