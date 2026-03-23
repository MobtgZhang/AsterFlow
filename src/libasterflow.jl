## 可选动态库 `libasterflow`（C ABI）。未找到库时 matmul 等回退到纯 Julia。

const _AF_LIB = Ref{String}("")

function _find_libasterflow()
    env = get(ENV, "ASTERFLOW_LIB", "")
    isempty(env) || return env
    ## `libasterflow.jl` 位于 `src/`，父目录即 Julia 包根（与 Flux 等仓库布局一致）
    pkg_root = dirname(dirname(@__FILE__))
    for p in (joinpath(pkg_root, "libasterflow", "build", "libasterflow.so"), joinpath(pkg_root, "deps", "libasterflow.so"))
        ispath(p) && return abspath(p)
    end
    return ""
end

function _init_libasterflow!()
    _AF_LIB[] = _find_libasterflow()
    return nothing
end

libasterflow_path() = _AF_LIB[]

function libasterflow_version()::Int32
    isempty(_AF_LIB[]) && return Int32(-1)
    try
        sym = dlsym(dlopen(_AF_LIB[]; throw_error=false), :af_version)
        sym === nothing && return Int32(-1)
        return ccall(sym, Int32, ())
    catch
        return Int32(-1)
    end
end

function af_alloc(n::Integer)::Ptr{Cvoid}
    isempty(_AF_LIB[]) && return Ptr{Cvoid}(0)
    f = dlsym(dlopen(_AF_LIB[]; throw_error=false), :af_alloc_bytes)
    f === nothing && return Ptr{Cvoid}(0)
    return ccall(f, Ptr{Cvoid}, (Csize_t,), Csize_t(n))
end

function af_free(p::Ptr{Cvoid})
    (p == C_NULL || isempty(_AF_LIB[])) && return
    f = dlsym(dlopen(_AF_LIB[]; throw_error=false), :af_free)
    f === nothing && return
    ccall(f, Cvoid, (Ptr{Cvoid},), p)
    return nothing
end

"""
`af_matmul_nograd(a,b)`：若加载 `libasterflow` 则走 C 列主 matmul，否则 `native_cpu_matmul`。
"""
function af_matmul_nograd(a::Tensor{Float32,2}, b::Tensor{Float32,2})
    m, k = size(a)
    k2, n = size(b)
    k == k2 || error("matmul: inner dim mismatch")
    if isempty(_AF_LIB[])
        return native_cpu_matmul_julia(a, b)
    end
    h = dlopen(_AF_LIB[]; throw_error=false)
    h === nothing && return native_cpu_matmul_julia(a, b)
    f = dlsym(h, :af_matmul_f32_colmajor)
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

function af_relu_nograd(a::Tensor{Float32,N}) where {N}
    if isempty(_AF_LIB[])
        return native_cpu_relu_julia(a)
    end
    h = dlopen(_AF_LIB[]; throw_error=false)
    h === nothing && return native_cpu_relu_julia(a)
    f = dlsym(h, :af_relu_f32)
    f === nothing && return native_cpu_relu_julia(a)
    x = to_array(a)
    y = similar(x)
    n = length(x)
    ccall(f, Cvoid, (Ptr{Float32}, Ptr{Float32}, Int64), x, y, n)
    return tensor(y; device = a.device, requires_grad = false)
end
