## 占位：将 IR JSON 写入缓存目录并生成可读的 “CUDA stub” 文本（非真实 NVRTC 流水线）。

mutable struct CompiledStub
    cache_key::String
    cuda_stub_path::String
    spec_path::String
end

function _cache_dir()
    d = joinpath(first(DEPOT_PATH), "asterflow_compile_cache")
    mkpath(d)
    return d
end

function compile_stub_launch(stub::CompiledStub, args::Vector)
    # 占位：仅校验参数个数与 key 存在
    isempty(stub.cache_key) && error("CompiledStub: empty key")
    return args
end

"""
将 `IRGraph` 导出为 JSON + `.cu.stub` 文本，供后续 C++ 后端替换为真实 codegen。
"""
function codegen_stub_cache(g::IRGraph)
    js = graph_to_json(g)
    ## 缓存键：`hash(JSON)`；IR schema（`IROpKind` 等）变更时旧缓存自然失效（不同 hash）。
    key = string(hash(js))
    dir = _cache_dir()
    spec_path = joinpath(dir, "$key.toml")
    stub_path = joinpath(dir, "$key.cu.stub")
    write(spec_path, js)
    write(
        stub_path,
        """
        // AsterFlow codegen stub (key=$key)
        // 后续由 aster_native 编译管线替换为 NVRTC / 静态 cubin
        extern "C" void asterflow_stub_launch() {}
        """,
    )
    return CompiledStub(key, stub_path, spec_path)
end
