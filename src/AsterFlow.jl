"""
AsterFlow: Julia 前端 + 可插拔加速器后端 +（可选）C++ `libasterflow` 与编译路径。

目录布局参考 [Flux.jl](https://github.com/FluxML/Flux.jl)：`layers/`、`losses/`、`optimise/`、`compile/`、`data/`，
顶层保留张量、设备、自动微分等核心文件；权重序列化见 `loading.jl`（类比 Flux 的 `loading.jl`）。
"""
module AsterFlow

using LinearAlgebra
using Libdl

export Tensor, Device, CPUDevice, AcceleratorDevice, ScalarLike
export cuda_device, rocm_device, npu_device, ascend_device, rknpu_device, CUDADevice
export device, isavailable
export device_backend, BACKEND_CPU, is_accelerator
export accelerator_storage, register_backend_runtime!, runtime_available
export register_tensor_upload!, register_dev_ones_gpu!, register_dev_zeros_gpu!, register_dev_fill_gpu!
export register_materialize_strided_gpu!, register_contiguous_accelerator!
export accelerator_from_env, first_available_accelerator
export tensor, to_array, numel, is_contiguous, contiguous, reshape_tensor, view_tensor, permute_tensor
export column_major_strides, softmax_rows
export register_op!, dispatch_op, register_dispatch_fallback!, register_custom_op!, registered_ops_report
export ASTERFLOW_EXECUTION_MODE
export detach_tensor, expand_tensor, broadcast_to_tensor, setindex_tensor!
export to_device, module_to_device!
export ddp_barrier!, ddp_allreduce_mean_grads!
export autocast_enabled, @autocast, GradScaler, scale_loss, unscale_grads!, update!
export checkpoint
export fuse_linear_relu_chain!
export list_native_cpu_ops
export add, sub, mul, div_op, div_tensor, matmul, sum_tensor, mean_tensor, scale_tensor
export exp_tensor, log_tensor, sqrt_tensor, relu_tensor, tanh_tensor, sigmoid_tensor, leaky_relu_tensor
export reshape_op
export backward, zero_grad!, requires_grad!, no_grad, grad, set_grad!
export @trace, TraceState, trace_graph, graph_to_json, graph_from_json, CompiledStub, compile_stub_launch
export IRGraph, IRNode, IRValue, IROpKind, IR_Add, IR_Mul, IR_MatMul, IR_ReLU, IR_Sum
export ir_new_input!, ir_append_node!, ir_set_outputs!, codegen_stub_cache, ir_infer_binary_output_shape
export Linear, ReLU, LeakyReLU, Tanh, Sigmoid, GELU, Identity
export Softmax, LogSoftmax, MSE, L1Loss, CrossEntropyLoss
export Sequential, ModuleList, ModuleDict, Dropout, Flatten, LayerNorm, BatchNorm1d, buffers
export Conv2d, params, train!, evalmode!
export mse_loss, l1_loss, cross_entropy_loss, nll_loss
export xavier_uniform!, xavier_normal!, kaiming_uniform!, kaiming_normal!, init_linear!
export relu, tanh_act, sigmoid_act, gelu, dropout, functional_linear
export dev_ones, dev_zeros, dev_fill, tensor_on_device
export SGD, AdamW, AdamWGroup, Adam, RMSprop, Adagrad, Adadelta, Adamax
export RAdam, Lookahead, AdaFactor, Lion, Sophia, step!
export TransformerBlock, Embedding, TinyGPT, CausalSelfAttention, CAUSAL_ATTN_MASK_NEG
export state_dict, load_state_dict!, expected_state_keys
export load_safetensors, save_safetensors, save_weights_bson, load_weights_bson
export load_pytorch_state_dict, save_pytorch_state_dict
export AbstractDataset, TensorDataset, Subset, DataLoader, random_split
export libasterflow_version, af_alloc, af_free, af_matmul_nograd, af_add_nograd
export asterflow_native_version, asterflow_native_path
export gpu_memory_stats_placeholder!

## --- 核心：梯度模式、设备、张量、算子、自动微分 ---

include("grad_mode.jl")
include("devices.jl")
include("tensor.jl")
include("accelerator_dispatch.jl")
include("storage.jl")
include("dispatch.jl")
include("libasterflow_native.jl")
include("native.jl")
include("ops_helpers.jl")
include("graph.jl")
include("layout.jl")
include("compile/ir.jl")
include("compile/trace.jl")
include("ops.jl")
include("autograd.jl")
include("checkpoint.jl")
include("libasterflow.jl")
include("amp.jl")

## --- 编译：序列化 / 代码生成桩 / 融合 ---

include("compile/serialize.jl")
include("compile/codegen_stub.jl")
include("compile/fusion.jl")

## --- 网络层（Flux 风格 `layers/`）---

include("layers/module.jl")
include("layers/linear.jl")
include("layers/activation.jl")
include("layers/dropout.jl")
include("layers/flatten.jl")
include("layers/norm.jl")
include("layers/init.jl")
include("layers/functional.jl")
include("layers/conv.jl")
include("device_tensor.jl")

## --- 损失函数（Flux 风格 `losses/`）---

include("losses/functions.jl")

## --- 数据加载 ---

include(joinpath(@__DIR__, "data", "dataset.jl"))

## --- 优化器（Flux 风格 `optimise/` 拼写）---

include("optimise/gpu_optim.jl")
include("optimise/fused.jl")
include("optimise/sgd.jl")
include("optimise/adamw.jl")
include("optimise/adam.jl")
include("optimise/rmsprop.jl")
include("optimise/adagrad.jl")
include("optimise/adadelta.jl")
include("optimise/adamax.jl")
include("optimise/radam.jl")
include("optimise/lookahead.jl")
include("optimise/adafactor.jl")
include("optimise/lion.jl")
include("optimise/sophia.jl")

## --- Transformer 块 ---

include("transformer/blocks.jl")

## --- 权重与 checkpoint（类比 Flux `loading.jl`）---

include("loading.jl")

include("distributed/ddp_stub.jl")

## --- 厂商占位扩展（与 `Project.toml` 中 `[extensions]` 对应）---

include(joinpath(dirname(@__DIR__), "ext", "AsterFlowHuaweiAscendExt.jl"))
include(joinpath(dirname(@__DIR__), "ext", "AsterFlowRockchipNPUExt.jl"))

function __init__()
    _init_libasterflow!()
    _init_libasterflow_native!()
    register_native_cpu!()
    _register_ascend_npu_backend!()
    _register_rockchip_npu_backend!()
    return nothing
end

end # module
