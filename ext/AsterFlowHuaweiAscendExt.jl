# 华为昇腾 NPU：注册 `device_backend == :ascend` 的运行时探测。
# 本文件由主包 `include` 入 `AsterFlow` 模块（非 Pkg 扩展），不引入 CANN 依赖。
# 在昇腾主机上可通过环境变量或设备节点判定；算子需另行 `register_op!` 对接 CANN/自定义内核。

function _ascend_npu_runtime_probe(device_id::Int)::Bool
    if haskey(ENV, "ASCEND_VISIBLE_DEVICES")
        s = strip(ENV["ASCEND_VISIBLE_DEVICES"])
        !isempty(s) && s != "-1" && return true
    end
    if haskey(ENV, "ASCEND_RT_VISIBLE_DEVICES")
        s = strip(ENV["ASCEND_RT_VISIBLE_DEVICES"])
        !isempty(s) && s != "-1" && return true
    end
    # 常见驱动设备节点（随 CANN 版本可能变化）
    if ispath("/dev/davinci_manager")
        return true
    end
    if device_id >= 0 && ispath("/dev/davinci$(device_id)")
        return true
    end
    return false
end

function _register_ascend_npu_backend!()
    register_backend_runtime!(:ascend, _ascend_npu_runtime_probe)
    return nothing
end
