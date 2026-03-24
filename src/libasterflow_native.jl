## 可选 `libasterflow_native`（C++ aster_native 共享库）。未找到或未开启开关时不使用。

const _AN_LIB = Ref{String}("")

function _find_libasterflow_native()
    env = get(ENV, "ASTERFLOW_NATIVE_LIB", "")
    isempty(env) || return env
    pkg_root = dirname(dirname(@__FILE__))
    rel = joinpath("aster_native", "build", "AsterNative")
    cands = if Sys.iswindows()
        (joinpath(pkg_root, rel, "asterflow_native.dll"),)
    elseif Sys.isapple()
        (joinpath(pkg_root, rel, "libasterflow_native.dylib"),)
    else
        (joinpath(pkg_root, rel, "libasterflow_native.so"),)
    end
    for p in cands
        ispath(p) && return abspath(p)
    end
    return ""
end

function _init_libasterflow_native!()
    _AN_LIB[] = _find_libasterflow_native()
    return nothing
end

asterflow_native_path() = _AN_LIB[]

function asterflow_native_version()::Int32
    isempty(_AN_LIB[]) && return Int32(-1)
    try
        sym = dlsym(dlopen(_AN_LIB[]; throw_error = false), :af_native_version)
        sym === nothing && return Int32(-1)
        return ccall(sym, Int32, ())
    catch
        return Int32(-1)
    end
end

function an_native_matmul_nograd(a::Tensor{Float32,2}, b::Tensor{Float32,2})
    m, k = size(a)
    k2, n = size(b)
    k == k2 || error("matmul: inner dim mismatch")
    if isempty(_AN_LIB[])
        return native_cpu_matmul_julia(a, b)
    end
    h = dlopen(_AN_LIB[]; throw_error = false)
    h === nothing && return native_cpu_matmul_julia(a, b)
    f = dlsym(h, :af_native_matmul_f32_colmajor)
    f === nothing && return native_cpu_matmul_julia(a, b)
    A = to_array(a)
    B = to_array(b)
    C = Matrix{Float32}(undef, m, n)
    ccall(
        f,
        Cvoid,
        (Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Int64, Int64, Int64),
        A,
        B,
        C,
        m,
        k,
        n,
    )
    return tensor(C; device = a.device, requires_grad = false)
end

function an_native_relu_nograd(a::Tensor{Float32,N}) where {N}
    if isempty(_AN_LIB[])
        return native_cpu_relu_julia(a)
    end
    h = dlopen(_AN_LIB[]; throw_error = false)
    h === nothing && return native_cpu_relu_julia(a)
    f = dlsym(h, :af_native_relu_f32)
    f === nothing && return native_cpu_relu_julia(a)
    x = to_array(a)
    y = similar(x)
    n = length(x)
    ccall(f, Cvoid, (Ptr{Float32}, Ptr{Float32}, Int64), x, y, n)
    return tensor(y; device = a.device, requires_grad = false)
end

function an_native_add_nograd(a::Tensor{Float32,N}, b::Tensor{Float32,N}) where {N}
    size(a) == size(b) || error("an_native_add: shape mismatch")
    if isempty(_AN_LIB[])
        return tensor(to_array(a) .+ to_array(b); device = a.device, requires_grad = false)
    end
    h = dlopen(_AN_LIB[]; throw_error = false)
    h === nothing && return tensor(to_array(a) .+ to_array(b); device = a.device, requires_grad = false)
    f = dlsym(h, :af_native_add_f32)
    f === nothing && return tensor(to_array(a) .+ to_array(b); device = a.device, requires_grad = false)
    aa = to_array(a)
    ba = to_array(b)
    y = similar(aa)
    n = length(aa)
    ccall(f, Cvoid, (Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Int64), aa, ba, y, n)
    return tensor(y; device = a.device, requires_grad = false)
end
