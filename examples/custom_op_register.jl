# 第三方包中注册自定义 CPU 内核示例（需在 `using AsterFlow` 之后执行）：
#
#   my_add(a, b) = AsterFlow.tensor(AsterFlow.to_array(a) .+ AsterFlow.to_array(b); device = a.device)
#   AsterFlow.register_op!(:my_add, AsterFlow.BACKEND_CPU, my_add)
#
# 若需 autograd，请在前端用 `no_grad` 包装内核，并在 `ops.jl` 模式外单独封装 `grad_fn`。

using AsterFlow

function example_register_my_add()
    fn(a::Tensor, b::Tensor) = tensor(to_array(a) .+ to_array(b); device = a.device, requires_grad = false)
    register_op!(:my_add, BACKEND_CPU, fn)
    return nothing
end
