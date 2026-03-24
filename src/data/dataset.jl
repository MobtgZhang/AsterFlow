# 与 PyTorch Dataset / DataLoader 类似：按第 1 维切片并打包为 Tensor。

abstract type AbstractDataset end

Base.length(d::AbstractDataset) = error("AbstractDataset: 请实现 `length`")
Base.getindex(d::AbstractDataset, i::Int) = error("AbstractDataset: 请实现 `getindex`")

function _tensor_rows(t::Tensor, idxs::AbstractVector{Int})
    arr = to_array(t)
    nd = ndims(arr)
    if nd == 1
        sel = arr[idxs]
        return tensor(sel; device = device(t), requires_grad = false)
    end
    sel = arr[idxs, ntuple(_ -> Colon(), nd - 1)...]
    return tensor(sel; device = device(t), requires_grad = false)
end

struct TensorDataset <: AbstractDataset
    tensors::Tuple
    n::Int
end

function TensorDataset(ts::Tensor...)
    isempty(ts) && error("TensorDataset: 至少一个张量")
    n0 = size(ts[1], 1)
    for k in 2:length(ts)
        size(ts[k], 1) == n0 || error("TensorDataset: 第 1 维（batch）长度须一致")
    end
    return TensorDataset(ts, n0)
end

Base.length(d::TensorDataset) = d.n

function Base.getindex(d::TensorDataset, i::Int)
    (1 <= i <= d.n) || throw(BoundsError(d, i))
    return ntuple(k -> _tensor_rows(d.tensors[k], [i]), length(d.tensors))
end

struct Subset{D<:AbstractDataset} <: AbstractDataset
    dataset::D
    indices::Vector{Int}
end

Base.length(s::Subset) = length(s.indices)

function Base.getindex(s::Subset, i::Int)
    (1 <= i <= length(s.indices)) || throw(BoundsError(s, i))
    return s.dataset[s.indices[i]]
end

function _shuffle_perm!(xs::Vector{Int})
    n = length(xs)
    @inbounds for i in n:-1:2
        j = rand(1:i)
        xs[i], xs[j] = xs[j], xs[i]
    end
    return xs
end

function random_split(dataset::AbstractDataset, lengths::Vector{Int})
    n = length(dataset)
    sum(lengths) == n || error("random_split: lengths 之和须等于数据集长度 ($n)")
    ord = sortperm(rand(n))
    out = Subset[]
    start = 1
    for len in lengths
        push!(out, Subset(dataset, ord[start:(start+len-1)]))
        start += len
    end
    return Tuple(out)
end

function _take_batch_tensor(ds::TensorDataset, idxs::Vector{Int})
    return ntuple(k -> _tensor_rows(ds.tensors[k], idxs), length(ds.tensors))
end

function _take_batch(dataset::AbstractDataset, idxs::Vector{Int})
    if dataset isa TensorDataset
        return _take_batch_tensor(dataset, idxs)
    elseif dataset isa Subset && dataset.dataset isa TensorDataset
        subidxs = [dataset.indices[i] for i in idxs]
        return _take_batch_tensor(dataset.dataset, subidxs)
    else
        error("DataLoader: 仅支持 `TensorDataset` 与 `Subset{TensorDataset}` 的批拼接")
    end
end

mutable struct DataLoader
    dataset::AbstractDataset
    batchsize::Int
    shuffle::Bool
    collate_fn::Any
    num_workers::Int
end

function DataLoader(
    dataset::AbstractDataset;
    batchsize::Int = 1,
    shuffle::Bool = false,
    collate_fn = nothing,
    num_workers::Int = 0,
)
    batchsize >= 1 || error("DataLoader: batchsize >= 1")
    num_workers > 0 &&
        @warn "DataLoader: num_workers>0 尚未实现多进程预取，当前等价于 0" maxlog = 1
    return DataLoader(dataset, batchsize, shuffle, collate_fn, num_workers)
end

function Base.iterate(dl::DataLoader, state = nothing)
    n = length(dl.dataset)
    if state === nothing
        ord = collect(1:n)
        dl.shuffle && _shuffle_perm!(ord)
        state = (ord, 1)
    end
    ord, pos = state
    pos > length(ord) && return nothing
    last = min(pos + dl.batchsize - 1, length(ord))
    idxs = ord[pos:last]
    batch = _take_batch(dl.dataset, idxs)
    if dl.collate_fn !== nothing
        batch = dl.collate_fn(batch)
    end
    return (batch, (ord, last + 1))
end

Base.IteratorSize(::Type{DataLoader}) = Base.SizeUnknown()
