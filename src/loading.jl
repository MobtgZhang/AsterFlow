using BSON
using JSON

## --- safetensors：C 连续（行主 / NumPy 序）与 Julia 列主互转 ---

function _st_c_order_to_array(::Type{T}, span::AbstractVector{UInt8}, shp::Vector{Int}) where {T}
    n = prod(shp)
    need = n * sizeof(T)
    length(span) ≥ need || error("safetensors: 缓冲区长度不足 ($(length(span)) < $need)")
    v = Vector{T}(undef, n)
    copyto!(v, reinterpret(T, view(span, 1:need)))
    N = length(shp)
    N == 0 && return reshape(v, ())
    N == 1 && return vec(v)
    sh = Tuple(shp)
    tmp = reshape(v, reverse(sh))
    return permutedims(tmp, ntuple(i -> N - i + 1, Val(N)))
end

function _st_array_to_c_order_bytes(a::AbstractArray{T,N}) where {T,N}
    N == 0 && return copy(reinterpret(UInt8, vec([a])))
    if N == 1
        return copy(reinterpret(UInt8, vec(collect(a))))
    end
    tmp = permutedims(collect(a), ntuple(i -> N - i + 1, Val(N)))
    return copy(reinterpret(UInt8, vec(tmp)))
end

const _ST_JL_TO_DTYPE = Dict{Type,String}(
    Float32 => "F32",
    Float64 => "F64",
    Float16 => "F16",
    Int64 => "I64",
    Int32 => "I32",
    Int16 => "I16",
    Int8 => "I8",
    UInt8 => "U8",
    Bool => "BOOL",
)

const _ST_DTYPE_TO_JL = Dict{String,Type}(
    "F32" => Float32,
    "F64" => Float64,
    "F16" => Float16,
    "I64" => Int64,
    "I32" => Int32,
    "I16" => Int16,
    "I8" => Int8,
    "U8" => UInt8,
    "BOOL" => Bool,
)

function _st_dtype_and_payload(a::AbstractArray)
    T = eltype(a)
    haskey(_ST_JL_TO_DTYPE, T) || error("safetensors: 不支持的元素类型 $T")
    _ST_JL_TO_DTYPE[T], _st_array_to_c_order_bytes(a)
end

"""
    load_safetensors(path) -> Dict{String, Array}

读取 Hugging Face `safetensors` 文件（C 连续 / 与 PyTorch 导出一致的字节序）。
不支持 `BF16`（需在 Python 侧先转为 F32）。
"""
function load_safetensors(path::AbstractString)
    data = read(path)
    length(data) ≥ 8 || error("safetensors: 文件过短")
    hlen = Int(reinterpret(UInt64, data[1:8])[1])
    8 + hlen ≤ length(data) || error("safetensors: 头长度无效")
    header = String(@view data[9:(8+hlen)])
    meta = JSON.parse(header)
    blob = @view data[(9+hlen):end]
    out = Dict{String,Array}()
    for (name, desc) in meta
        name == "__metadata__" && continue
        desc isa AbstractDict || continue
        dtype = string(desc["dtype"])
        shape = Int.(desc["shape"])
        offs = desc["data_offsets"]
        o0, o1 = Int(offs[1]), Int(offs[2])
        o0 < o1 || error("safetensors: 无效偏移 $name")
        span = @view blob[(o0+1):o1]
        if dtype == "BF16"
            error("safetensors: BF16 请先在外部转为 F32 再导入")
        end
        haskey(_ST_DTYPE_TO_JL, dtype) || error("safetensors: 未知 dtype $dtype")
        T = _ST_DTYPE_TO_JL[dtype]
        if dtype == "BOOL"
            n = prod(shape)
            length(span) ≥ n || error("safetensors: BOOL 缓冲区过短")
            raw = Bool[@inbounds(span[i] != 0x00) for i in 1:n]
            N = length(shape)
            if N == 0
                out[name] = reshape(raw, ())
            elseif N == 1
                out[name] = raw
            else
                sh = Tuple(shape)
                tmp = reshape(raw, reverse(sh))
                out[name] = permutedims(tmp, ntuple(i -> N - i + 1, Val(N)))
            end
        else
            out[name] = _st_c_order_to_array(T, span, shape)
        end
    end
    return out
end

"""
    save_safetensors(path, tensors::AbstractDict{<:AbstractString,<:AbstractArray}; metadata=nothing)

将 `name => array` 写入 `safetensors`。张量名按字典序排列，与常见工具链一致。
`metadata` 为 `Dict{String,String}` 时写入 `__metadata__`。
"""
function save_safetensors(
    path::AbstractString,
    tensors::AbstractDict{<:AbstractString,<:AbstractArray};
    metadata::Union{Nothing,AbstractDict{<:AbstractString,<:AbstractString}} = nothing,
)
    names = sort!(collect(String.(keys(tensors))))
    isempty(names) && error("save_safetensors: 空字典")
    meta = Dict{String,Any}()
    if metadata !== nothing
        meta["__metadata__"] = Dict{String,Any}(string(k) => string(v) for (k, v) in metadata)
    end
    blob = UInt8[]
    offset = 0
    for name in names
        arr = tensors[name]
        dtype, bytes = _st_dtype_and_payload(arr)
        n = length(bytes)
        meta[name] = Dict{String,Any}(
            "dtype" => dtype,
            "shape" => collect(Int.(size(arr))),
            "data_offsets" => [offset, offset + n],
        )
        append!(blob, bytes)
        offset += n
    end
    header = JSON.json(meta)
    hbytes = codeunits(header)
    open(path, "w") do io
        write(io, UInt64(length(hbytes)))
        write(io, hbytes)
        write(io, blob)
    end
    return path
end

## --- BSON：Julia 生态常用二进制格式（你提到的 BOSN 一般指 BSON）---

"""
    save_weights_bson(path, tensors::AbstractDict{<:AbstractString,<:AbstractArray})

将权重字典存为 BSON。键为 `String`，值为普通 `Array`（CPU）。
"""
function save_weights_bson(path::AbstractString, tensors::AbstractDict{<:AbstractString,<:AbstractArray})
    plain = Dict{String,Any}(string(k) => collect(v) for (k, v) in tensors)
    BSON.bson(path, Dict("asterflow_state_dict" => plain, "format" => "asterflow_bson_v1"))
    return path
end

"""
    load_weights_bson(path) -> Dict{String, Array}

从 BSON 读回 `Dict{String, Array}`。
"""
function load_weights_bson(path::AbstractString)
    raw = BSON.load(path)
    raw isa AbstractDict || error("BSON: 根对象不是字典")
    d = raw isa Dict{String,Any} ? raw : Dict{String,Any}(string(k) => v for (k, v) in raw)
    if haskey(d, "asterflow_state_dict")
        inner = d["asterflow_state_dict"]
    else
        inner = d
    end
    inner isa AbstractDict || error("BSON: 缺少 state_dict")
    out = Dict{String,Array}()
    for (k, v) in inner
        v isa AbstractArray || continue
        out[string(k)] = Array(v)
    end
    return out
end

## --- PyTorch .pt：通过 Python（torch + safetensors）转存为临时 safetensors 再读 ---

const _PYTORCH_EXPORT_TO_ST = raw"""
import sys
path_in, path_out = sys.argv[1], sys.argv[2]
try:
    import torch
except ImportError:
    sys.stderr.write("ASTERROR: 需要安装 PyTorch: pip install torch\n")
    sys.exit(2)
try:
    from safetensors.torch import save_file
except ImportError:
    sys.stderr.write("ASTERROR: 需要安装 safetensors: pip install safetensors\n")
    sys.exit(3)
try:
    obj = torch.load(path_in, map_location="cpu", weights_only=True)
except TypeError:
    obj = torch.load(path_in, map_location="cpu")
if isinstance(obj, dict):
    sd = obj
elif hasattr(obj, "state_dict"):
    sd = obj.state_dict()
else:
    sys.stderr.write("ASTERROR: 不支持的 checkpoint 类型\n")
    sys.exit(4)
clean = {k: v.detach().cpu() for k, v in sd.items() if hasattr(v, "numpy")}
save_file(clean, path_out)
"""

function _run_python_script(py::AbstractString, script::AbstractString, args::AbstractString...)
    scriptpath, io = mktemp()
    try
        write(io, script)
        close(io)
        argv = String[string(py), scriptpath, map(String, args)...]
        run(pipeline(Cmd(argv); stderr = stderr))
    finally
        isfile(scriptpath) && rm(scriptpath; force = true)
    end
    return nothing
end

"""
    load_pytorch_state_dict(path; python=nothing) -> Dict{String, Array}

从 PyTorch `.pt` / `.pth`（zip + pickle 或旧格式，`torch.save`）加载 `state_dict`。
依赖系统上的 `python`，且需安装 `torch` 与 `safetensors`（`pip install torch safetensors`）。

`python` 默认依次为环境变量 `JULIA_PYTHONCALL_EXE`、`PYTHON`、`python3`。
"""
function load_pytorch_state_dict(path::AbstractString; python::Union{Nothing,AbstractString} = nothing)
    isfile(path) || error("文件不存在: $path")
    py = something(
        python,
        get(ENV, "JULIA_PYTHONCALL_EXE", nothing),
        get(ENV, "PYTHON", nothing),
        "python3",
    )
    tmp = string(tempname(), ".safetensors")
    try
        _run_python_script(py, _PYTORCH_EXPORT_TO_ST, path, tmp)
        return load_safetensors(tmp)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

const _PYTORCH_SAVE_FROM_ST = raw"""
import sys
path_in, path_out = sys.argv[1], sys.argv[2]
try:
    import torch
    from safetensors.torch import load_file
except ImportError:
    sys.stderr.write("ASTERROR: 需要 torch 与 safetensors\n")
    sys.exit(2)
sd = load_file(path_in)
torch.save(sd, path_out)
"""

"""
    save_pytorch_state_dict(path, tensors::AbstractDict{<:AbstractString,<:AbstractArray}; python=nothing)

先将权重写成临时 `safetensors`，再调用 Python `torch.save`，得到与 PyTorch 兼容的 `.pt`。
同样需要本机 `torch` + `safetensors`。
"""
function save_pytorch_state_dict(
    path::AbstractString,
    tensors::AbstractDict{<:AbstractString,<:AbstractArray};
    python::Union{Nothing,AbstractString} = nothing,
)
    py = something(
        python,
        get(ENV, "JULIA_PYTHONCALL_EXE", nothing),
        get(ENV, "PYTHON", nothing),
        "python3",
    )
    tmp = string(tempname(), ".safetensors")
    try
        save_safetensors(tmp, tensors)
        _run_python_script(py, _PYTORCH_SAVE_FROM_ST, tmp, path)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
    return path
end

## --- state_dict / load_state_dict!（与 PyTorch 命名对齐）---

function state_dict(m::Module)
    d = Dict{String,Array}()
    _state_dict!(m, d, "")
    return d
end

function _state_dict!(m::Sequential, d::Dict{String,Array}, prefix::String)
    for (i, layer) in enumerate(m.layers)
        layer isa Module && _state_dict!(layer, d, prefix * string(i - 1) * ".")
    end
    return d
end

function _state_dict!(m::ModuleList, d::Dict{String,Array}, prefix::String)
    for (i, layer) in enumerate(m.layers)
        layer isa Module && _state_dict!(layer, d, prefix * string(i - 1) * ".")
    end
    return d
end

function _state_dict!(m::ModuleDict, d::Dict{String,Array}, prefix::String)
    for (k, layer) in m.layers
        layer isa Module || continue
        kp = prefix * string(k) * "."
        _state_dict!(layer, d, kp)
    end
    return d
end

function _state_dict!(m::Module, d::Dict{String,Array}, prefix::String)
    m isa Sequential && return _state_dict!(m, d, prefix)
    m isa ModuleList && return _state_dict!(m, d, prefix)
    m isa ModuleDict && return _state_dict!(m, d, prefix)
    for nm in fieldnames(typeof(m))
        nm == :training && continue
        x = getfield(m, nm)
        if x isa Tensor
            d[prefix * string(nm)] = to_array(x)
        elseif x isa Module
            _state_dict!(x, d, prefix * string(nm) * ".")
        elseif x isa Tuple
            for (i, y) in enumerate(x)
                y isa Module && _state_dict!(y, d, prefix * string(i - 1) * ".")
            end
        elseif x isa AbstractVector
            for (i, y) in enumerate(x)
                y isa Module && _state_dict!(y, d, prefix * string(i - 1) * ".")
            end
        elseif x isa AbstractDict
            for (k, y) in x
                y isa Module || continue
                _state_dict!(y, d, prefix * string(k) * ".")
            end
        end
    end
    return d
end

function _expected_keys!(m::Sequential, keys::Vector{String}, prefix::String)
    for (i, layer) in enumerate(m.layers)
        layer isa Module && _expected_keys!(layer, keys, prefix * string(i - 1) * ".")
    end
end

function _expected_keys!(m::ModuleList, keys::Vector{String}, prefix::String)
    for (i, layer) in enumerate(m.layers)
        layer isa Module && _expected_keys!(layer, keys, prefix * string(i - 1) * ".")
    end
end

function _expected_keys!(m::ModuleDict, keys::Vector{String}, prefix::String)
    for (k, layer) in m.layers
        layer isa Module || continue
        _expected_keys!(layer, keys, prefix * string(k) * ".")
    end
end

function _expected_keys!(m::Module, keys::Vector{String}, prefix::String)
    m isa Sequential && return _expected_keys!(m, keys, prefix)
    m isa ModuleList && return _expected_keys!(m, keys, prefix)
    m isa ModuleDict && return _expected_keys!(m, keys, prefix)
    for nm in fieldnames(typeof(m))
        nm == :training && continue
        x = getfield(m, nm)
        if x isa Tensor
            push!(keys, prefix * string(nm))
        elseif x isa Module
            _expected_keys!(x, keys, prefix * string(nm) * ".")
        elseif x isa Tuple
            for (i, y) in enumerate(x)
                y isa Module && _expected_keys!(y, keys, prefix * string(i - 1) * ".")
            end
        elseif x isa AbstractVector
            for (i, y) in enumerate(x)
                y isa Module && _expected_keys!(y, keys, prefix * string(i - 1) * ".")
            end
        elseif x isa AbstractDict
            for (k, y) in x
                y isa Module || continue
                _expected_keys!(y, keys, prefix * string(k) * ".")
            end
        end
    end
end

function expected_state_keys(m::Module)
    ks = String[]
    _expected_keys!(m, ks, "")
    return ks
end

function _float_array_for_tensor(t::Tensor, arr::AbstractArray)
    T = eltype(t)
    if eltype(arr) == T
        return Array(arr)
    end
    return Array{T}(arr)
end

function _match_linear_weight!(t_dest::Tensor, arr::AbstractArray, pytorch_compat::Bool)
    a = _float_array_for_tensor(t_dest, arr)
    if size(a) == size(t_dest)
        return a
    end
    if pytorch_compat && ndims(t_dest) == 2 && size(a) == reverse(size(t_dest))
        return collect(permutedims(a, (2, 1)))
    end
    error("形状不匹配: 目标 $(size(t_dest))，checkpoint $(size(a))")
end

function load_state_dict!(
    m::Module,
    d::AbstractDict{<:AbstractString,<:AbstractArray};
    pytorch_compat::Bool = false,
    strict::Bool = true,
)
    used = Set{String}()
    _load_state_dict!(m, d, "", used; pytorch_compat = pytorch_compat)
    if strict
        exp = Set(expected_state_keys(m))
        missing_k = setdiff(exp, used)
        !isempty(missing_k) && error("load_state_dict!: 缺少键: $(collect(missing_k))")
        extra = setdiff(Set(String.(keys(d))), used)
        !isempty(extra) && error("load_state_dict!: 多余键: $(collect(extra))")
    end
    return m
end

function _load_state_dict!(m::Sequential, d, prefix, used; pytorch_compat)
    for (i, layer) in enumerate(m.layers)
        layer isa Module && _load_state_dict!(layer, d, prefix * string(i - 1) * ".", used; pytorch_compat = pytorch_compat)
    end
end

function _load_state_dict!(m::ModuleList, d, prefix, used; pytorch_compat)
    for (i, layer) in enumerate(m.layers)
        layer isa Module && _load_state_dict!(layer, d, prefix * string(i - 1) * ".", used; pytorch_compat = pytorch_compat)
    end
end

function _load_state_dict!(m::ModuleDict, d, prefix, used; pytorch_compat)
    for (k, layer) in m.layers
        layer isa Module || continue
        _load_state_dict!(layer, d, prefix * string(k) * ".", used; pytorch_compat = pytorch_compat)
    end
end

function _load_state_dict!(m::Linear, d, prefix, used; pytorch_compat)
    devw = device(m.weight)
    devb = device(m.bias)
    wk = prefix * "weight"
    if haskey(d, wk)
        arr = _match_linear_weight!(m.weight, d[wk], pytorch_compat)
        m.weight = tensor(arr; device = devw, requires_grad = m.weight.requires_grad)
        push!(used, wk)
    end
    bk = prefix * "bias"
    if haskey(d, bk)
        m.bias = tensor(_float_array_for_tensor(m.bias, d[bk]); device = devb, requires_grad = m.bias.requires_grad)
        push!(used, bk)
    end
end

function _load_state_dict!(m::LayerNorm, d, prefix, used; pytorch_compat)
    devw = device(m.weight)
    for (nm, fld) in ((:weight, m.weight), (:bias, m.bias))
        k = prefix * string(nm)
        haskey(d, k) || continue
        t = getfield(m, nm)
        setfield!(m, nm, tensor(_float_array_for_tensor(t, d[k]); device = devw, requires_grad = t.requires_grad))
        push!(used, k)
    end
end

function _load_state_dict!(m::BatchNorm1d, d, prefix, used; pytorch_compat)
    for (nm, fld) in ((:weight, m.weight), (:bias, m.bias), (:running_mean, m.running_mean), (:running_var, m.running_var))
        k = prefix * string(nm)
        haskey(d, k) || continue
        t = getfield(m, nm)
        dev = device(t)
        setfield!(
            m,
            nm,
            tensor(_float_array_for_tensor(t, d[k]); device = dev, requires_grad = t.requires_grad),
        )
        push!(used, k)
    end
end

function _load_state_dict!(m::Embedding, d, prefix, used; pytorch_compat)
    wk = prefix * "weight"
    if haskey(d, wk)
        t = m.weight
        m.weight = tensor(
            _float_array_for_tensor(t, d[wk]);
            device = device(t),
            requires_grad = t.requires_grad,
        )
        push!(used, wk)
    end
end

function _load_state_dict!(m::Module, d, prefix, used; pytorch_compat)
    m isa Sequential && return _load_state_dict!(m, d, prefix, used; pytorch_compat = pytorch_compat)
    m isa ModuleList && return _load_state_dict!(m, d, prefix, used; pytorch_compat = pytorch_compat)
    m isa ModuleDict && return _load_state_dict!(m, d, prefix, used; pytorch_compat = pytorch_compat)
    m isa Linear && return _load_state_dict!(m, d, prefix, used; pytorch_compat = pytorch_compat)
    m isa LayerNorm && return _load_state_dict!(m, d, prefix, used; pytorch_compat = pytorch_compat)
    m isa BatchNorm1d && return _load_state_dict!(m, d, prefix, used; pytorch_compat = pytorch_compat)
    m isa Embedding && return _load_state_dict!(m, d, prefix, used; pytorch_compat = pytorch_compat)
    ## CausalSelfAttention、TransformerBlock、TinyGPT 等由子模块字段递归覆盖
    for nm in fieldnames(typeof(m))
        nm == :training && continue
        x = getfield(m, nm)
        if x isa Tensor
            k = prefix * string(nm)
            if haskey(d, k)
                t = x
                setfield!(
                    m,
                    nm,
                    tensor(_float_array_for_tensor(t, d[k]); device = device(t), requires_grad = t.requires_grad),
                )
                push!(used, k)
            end
        elseif x isa Module
            _load_state_dict!(x, d, prefix * string(nm) * ".", used; pytorch_compat = pytorch_compat)
        elseif x isa Tuple
            for (i, y) in enumerate(x)
                y isa Module && _load_state_dict!(y, d, prefix * string(i - 1) * ".", used; pytorch_compat = pytorch_compat)
            end
        elseif x isa AbstractVector
            for (i, y) in enumerate(x)
                y isa Module && _load_state_dict!(y, d, prefix * string(i - 1) * ".", used; pytorch_compat = pytorch_compat)
            end
        elseif x isa AbstractDict
            for (k, y) in x
                y isa Module || continue
                _load_state_dict!(y, d, prefix * string(k) * ".", used; pytorch_compat = pytorch_compat)
            end
        end
    end
end
