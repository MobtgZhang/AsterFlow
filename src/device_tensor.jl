## 设备迁移（与 PyTorch `.to(device)` 对齐）

"""
    to_device(t::Tensor, dev::Device)

拷贝数据到 `dev` 上的新张量，**不保留** `grad_fn`（与跨设备后图截断的常见语义一致）。
"""
function to_device(t::Tensor{T,N}, dev::Device) where {T,N}
    device(t) == dev && return t
    arr = to_array(t)
    return tensor(arr; device = dev, requires_grad = t.requires_grad)
end

function module_to_device!(m::Module, dev::Device)
    _module_to_device_impl!(m, dev)
    return m
end

function _module_to_device_impl!(m::Sequential, dev::Device)
    for layer in m.layers
        layer isa Module && _module_to_device_impl!(layer, dev)
    end
end

function _module_to_device_impl!(m::ModuleList, dev::Device)
    for layer in m.layers
        layer isa Module && _module_to_device_impl!(layer, dev)
    end
end

function _module_to_device_impl!(m::ModuleDict, dev::Device)
    for (_, layer) in m.layers
        layer isa Module && _module_to_device_impl!(layer, dev)
    end
end

function _module_to_device_impl!(m::Module, dev::Device)
    m isa Sequential && return _module_to_device_impl!(m, dev)
    m isa ModuleList && return _module_to_device_impl!(m, dev)
    m isa ModuleDict && return _module_to_device_impl!(m, dev)
    for nm in fieldnames(typeof(m))
        x = getfield(m, nm)
        if x isa Tensor
            setfield!(m, nm, to_device(x, dev))
        elseif x isa Module
            _module_to_device_impl!(x, dev)
        elseif x isa Tuple
            ys = map(x) do y
                y isa Tensor ? to_device(y, dev) : y
            end
            setfield!(m, nm, ys)
        elseif x isa AbstractVector
            ys = map(x) do y
                y isa Tensor ? to_device(y, dev) : y
            end
            setfield!(m, nm, ys)
        end
    end
end
