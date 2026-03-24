# 微基准（可选依赖 BenchmarkTools）：
#   julia --project=. benchmark/micro.jl
using AsterFlow
using Random

function bench_matmul(n::Int = 256)
    rng = MersenneTwister(0)
    a = tensor(randn(rng, Float32, n, n))
    b = tensor(randn(rng, Float32, n, n))
    @time matmul(a, b)
    return nothing
end

if isdefined(Main, :BenchmarkTools)
    using BenchmarkTools
    @info "BenchmarkTools 已加载，可使用 @btime matmul(...)"
else
    @info "安装 BenchmarkTools 后可启用 @btime：using Pkg; Pkg.add(\"BenchmarkTools\")"
    bench_matmul(128)
end
