"""
设备抽象：与具体厂商/runtime 解耦。算子按 `device_backend(dev)` 分发到 C/C++/CUDA/HIP 等实现。
"""
abstract type Device end

struct CPUDevice <: Device end

"""
    AcceleratorDevice(:cuda, 0)   # NVIDIA，由 CUDA 扩展注册
    AcceleratorDevice(:rocm, 0)  # AMD，由未来 ROCm 扩展注册
    AcceleratorDevice(:npu, 0)   # NPU，由厂商扩展 + libasterflow 注册

底层内核仍在 C/C++/CUDA/HIP 中实现；Julia 只持有 `backend` 符号与设备号。
"""
struct AcceleratorDevice <: Device
    backend::Symbol
    device_id::Int
end

Base.:(==)(a::AcceleratorDevice, b::AcceleratorDevice) =
    a.backend == b.backend && a.device_id == b.device_id

Base.show(io::IO, d::AcceleratorDevice) =
    print(io, "AcceleratorDevice(:$(d.backend), $(d.device_id))")

## --- 构造便捷函数（不绑定具体 Julia 包名）---

cuda_device(device_id::Int = 0) = AcceleratorDevice(:cuda, device_id)
rocm_device(device_id::Int = 0) = AcceleratorDevice(:rocm, device_id)
npu_device(device_id::Int = 0) = AcceleratorDevice(:npu, device_id)
"""华为昇腾（逻辑后端 `:ascend`）；算子需对接 CANN 或自定义扩展。"""
ascend_device(device_id::Int = 0) = AcceleratorDevice(:ascend, device_id)
"""瑞芯微 RK NPU（逻辑后端 `:rknpu`）；需 RKNN Runtime 与自定义 `register_op!`。"""
rknpu_device(device_id::Int = 0) = AcceleratorDevice(:rknpu, device_id)

"""兼容旧代码：`CUDADevice(0)` 等价于 `cuda_device(0)`。"""
CUDADevice(device_id::Int = 0) = cuda_device(device_id)

## --- PyTorch 风格：`device("cuda")` + `isavailable(dev)` ---

"""
    device("cpu")           -> CPUDevice()
    device("cuda")          -> AcceleratorDevice(:cuda, 0)
    device("cuda:1")        -> AcceleratorDevice(:cuda, 1)
    device("rocm:0") / device("npu:0")  同理

对应 PyTorch 的 `torch.device("cuda:0")`；可用性请用 `isavailable(device(...))`。
"""
function device(desc::AbstractString)
    s = lowercase(strip(desc))
    isempty(s) && error("device: 空字符串无效")
    if s == "cpu"
        return CPUDevice()
    end
    parts = split(s, ':'; limit = 2)
    b = Symbol(strip(parts[1]))
    id = length(parts) >= 2 ? parse(Int, strip(parts[2])) : 0
    return AcceleratorDevice(b, id)
end

"""
当前 `Device` 是否可用（已加载对应扩展且底层 runtime 就绪）。
- `CPUDevice`：恒为 `true`
- `AcceleratorDevice`：查询各扩展注册的 probe（等价于原 `runtime_available`）
"""
function isavailable(d::CPUDevice)
    return true
end

function isavailable(d::AcceleratorDevice)
    f = get(_BACKEND_RUNTIME_PROBES, d.backend, _accel_probe_unregistered)
    try
        return f(d.device_id)::Bool
    catch
        return false
    end
end

## --- 统一“逻辑后端”符号，供 Dispatcher 使用 ---

const BACKEND_CPU = :cpu

device_backend(::CPUDevice) = BACKEND_CPU
device_backend(d::AcceleratorDevice) = d.backend

is_accelerator(::Device) = false
is_accelerator(::AcceleratorDevice) = true

"""
判断 `AbstractVector` 是否位于加速器上（非主机 `Vector`）。
各后端扩展为 `CuArray`、`ROCArray` 等添加方法，默认 `false`。
"""
accelerator_storage(::AbstractVector) = false

## --- 运行时可用性（扩展在 __init__ 中注册）---

const _BACKEND_RUNTIME_PROBES = Dict{Symbol,Base.Callable}()

function register_backend_runtime!(backend::Symbol, probe!::Base.Callable)
    _BACKEND_RUNTIME_PROBES[backend] = probe!
    return nothing
end

function _accel_probe_unregistered(::Int)
    return false
end

"""兼容旧名：请优先使用 `isavailable(dev)`。"""
runtime_available(d::Device) = isavailable(d)

"""
从环境变量解析设备：`ASTERFLOW_DEVICE=cuda:0`（推荐）或 `ASTERFLOW_ACCELERATOR=cuda:0`（兼容）。
格式与 `device("cuda:0")` 一致；未设置时返回 `nothing`。
"""
function accelerator_from_env()
    s = strip(get(ENV, "ASTERFLOW_DEVICE", get(ENV, "ASTERFLOW_ACCELERATOR", "")))
    isempty(s) && return nothing
    try
        return device(s)
    catch
        return nothing
    end
end

"""
在 `backends` 顺序中返回第一个 `isavailable` 为真的 `AcceleratorDevice`；都不可用则 `nothing`。
`specs` 可为符号向量 `[:cuda, :rocm]`，或字符串向量 `["cuda", "rocm:0"]`（与 `device` 语法一致）。
"""
function first_available_accelerator(
    backends::Vector{Symbol} = [:cuda, :rocm, :ascend, :rknpu, :npu],
)
    for b in backends
        dev = AcceleratorDevice(b, 0)
        isavailable(dev) && return dev
    end
    return nothing
end

function first_available_accelerator(specs::Vector{<:AbstractString})
    for s in specs
        dev = device(s)
        dev isa AcceleratorDevice || continue
        isavailable(dev) && return dev
    end
    return nothing
end
