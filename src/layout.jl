## 视图类算子（依赖 graph.jl 中的 Node）

function reshape_tensor(t::Tensor, newsize::Tuple{Vararg{Int}})
    if prod(newsize) != numel(t)
        error("reshape: numel mismatch")
    end
    if !is_contiguous(t)
        return reshape_tensor(contiguous(t), newsize)
    end
    N = length(newsize)
    in_sz = size(t)
    out = Tensor{eltype(t),N}(
        t.storage,
        newsize,
        column_major_strides(newsize),
        t.offset,
        t.device,
        false,
        nothing,
        nothing,
        t.version_ref,
    )
    if grad_enabled() && t.requires_grad
        out.requires_grad = true
        out.grad_fn = ReshapeBackward(t, in_sz, tensor_version(t))
    end
    return out
end

function permute_tensor(t::Tensor{T,N}, perm::NTuple{N,Int}) where {T,N}
    out = _permute_storage(t, perm)
    if grad_enabled() && t.requires_grad
        out.requires_grad = true
        out.grad_fn = PermuteBackward(t, perm, tensor_version(t))
    end
    return out
end

"""
`expand_tensor`：二维广播视图（stride=0 模拟 PyTorch `expand`）。
"""
function expand_tensor(t::Tensor{T,2}, newsize::Tuple{Int,Int}) where {T}
    r, c = size(t)
    nr, nc = newsize
    ((nr == r || r == 1) && (nc == c || c == 1)) || error("expand_tensor: 形状不可广播到 $newsize")
    st1 = r == 1 ? 0 : t.strides[1]
    st2 = c == 1 ? 0 : t.strides[2]
    out = Tensor{T,2}(
        t.storage,
        newsize,
        (st1, st2),
        t.offset,
        t.device,
        false,
        nothing,
        nothing,
        t.version_ref,
    )
    if grad_enabled() && t.requires_grad
        out.requires_grad = true
        out.grad_fn = ExpandBackward(t, (r, c), newsize, tensor_version(t))
    end
    return out
end

broadcast_to_tensor(t::Tensor{T,2}, target::Tuple{Int,Int}) where {T} = expand_tensor(t, target)
