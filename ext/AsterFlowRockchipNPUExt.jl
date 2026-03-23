# 瑞芯微 RK NPU：注册 `device_backend == :rknpu` 的运行时探测。
# 由主包 `include`（非 Pkg 扩展）。RKNN Runtime 的 Julia 数组后端需自行对接后再 `register_op!`。

function _rockchip_npu_runtime_probe(device_id::Int)::Bool
    if haskey(ENV, "RKNN_TARGET_SOC") || haskey(ENV, "RKDEVICE_NUM")
        return true
    end
    if ispath("/sys/class/rknpu")
        return true
    end
    # 部分板卡暴露 rknn 相关节点
    if ispath("/dev/rknpu")
        return true
    end
    return false
end

function _register_rockchip_npu_backend!()
    register_backend_runtime!(:rknpu, _rockchip_npu_runtime_probe)
    return nothing
end
