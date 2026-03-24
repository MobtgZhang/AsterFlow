## 单机空实现的 DDP 钩子（设计见 docs/ddp-design.md）

function ddp_barrier!()
    return nothing
end

"""占位：全规约梯度并平均；单机无通信。"""
function ddp_allreduce_mean_grads!(params::AbstractVector{Tensor}; nprocs::Int = 1)
    nprocs == 1 && return nothing
    error("ddp_allreduce_mean_grads!: 分布式尚未接入，请使用 nprocs=1")
end
