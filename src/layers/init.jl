function _fan_in_out(w::Tensor{<:AbstractFloat,2})
    fan_in = size(w, 1)
    fan_out = size(w, 2)
    return fan_in, fan_out
end

function xavier_uniform!(w::Tensor{Float32,2}; gain::Real = 1.0f0)
    fin, fout = _fan_in_out(w)
    lim = Float32(gain * sqrt(6 / (fin + fout)))
    arr = to_array(w)
    arr .= rand(Float32, size(arr)...) .* (2lim) .- lim
    w2 = tensor(arr; device = device(w), requires_grad = w.requires_grad)
    w.storage = w2.storage
    w.size = w2.size
    w.strides = w2.strides
    w.offset = w2.offset
    return w
end

function xavier_normal!(w::Tensor{Float32,2}; gain::Real = 1.0f0)
    fin, fout = _fan_in_out(w)
    std = Float32(gain * sqrt(2 / (fin + fout)))
    arr = to_array(w)
    arr .= std .* randn(Float32, size(arr)...)
    w2 = tensor(arr; device = device(w), requires_grad = w.requires_grad)
    w.storage = w2.storage
    w.size = w2.size
    w.strides = w2.strides
    w.offset = w2.offset
    return w
end

function kaiming_uniform!(w::Tensor{Float32,2}; mode::Symbol = :fan_in, a::Real = 0.0f0)
    fin, fout = _fan_in_out(w)
    fan = mode == :fan_out ? fout : fin
    gain = Float32(sqrt(2.0f0 / (1 + Float32(a)^2)))
    bound = Float32(gain * sqrt(6 / fan))
    arr = to_array(w)
    arr .= rand(Float32, size(arr)...) .* (2bound) .- bound
    w2 = tensor(arr; device = device(w), requires_grad = w.requires_grad)
    w.storage = w2.storage
    w.size = w2.size
    w.strides = w2.strides
    w.offset = w2.offset
    return w
end

function kaiming_normal!(w::Tensor{Float32,2}; mode::Symbol = :fan_in, a::Real = 0.0f0)
    fin, fout = _fan_in_out(w)
    fan = mode == :fan_out ? fout : fin
    gain = Float32(sqrt(2.0f0 / (1 + Float32(a)^2)))
    std = Float32(gain / sqrt(fan))
    arr = to_array(w)
    arr .= std .* randn(Float32, size(arr)...)
    w2 = tensor(arr; device = device(w), requires_grad = w.requires_grad)
    w.storage = w2.storage
    w.size = w2.size
    w.strides = w2.strides
    w.offset = w2.offset
    return w
end

function init_linear!(m::Linear; weight_init = xavier_uniform!, bias_zero::Bool = true)
    weight_init(m.weight)
    if bias_zero
        arr = to_array(m.bias)
        fill!(arr, 0)
        b2 = tensor(arr; device = device(m.bias), requires_grad = m.bias.requires_grad)
        m.bias.storage = b2.storage
        m.bias.size = b2.size
        m.bias.strides = b2.strides
        m.bias.offset = b2.offset
    end
    return m
end
