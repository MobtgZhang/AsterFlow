"""
与 `torch.nn.Conv2d` 默认参数对齐的占位结构（前向尚未实现）。

默认：`stride=1`、`padding=0`、`dilation=1`、`groups=1`；`kernel_size` 可为整数或 `(H,W)`。
"""
mutable struct Conv2d <: Module
    in_channels::Int
    out_channels::Int
    kernel_size::Tuple{Int,Int}
    stride::Tuple{Int,Int}
    padding::Tuple{Int,Int}
    dilation::Tuple{Int,Int}
    groups::Int
    training::Bool
end

function Conv2d(
    in_ch::Int,
    out_ch::Int,
    kernel_size::Union{Int,Tuple{Int,Int}};
    stride::Union{Int,Tuple{Int,Int}} = 1,
    padding::Union{Int,Tuple{Int,Int}} = 0,
    dilation::Union{Int,Tuple{Int,Int}} = 1,
    groups::Int = 1,
)
    k = kernel_size isa Int ? (kernel_size, kernel_size) : kernel_size
    st = stride isa Int ? (stride, stride) : stride
    pd = padding isa Int ? (padding, padding) : padding
    di = dilation isa Int ? (dilation, dilation) : dilation
    return Conv2d(in_ch, out_ch, k, st, pd, di, groups, true)
end

function (::Conv2d)(::Tensor)
    error("Conv2d: 尚未实现；当前 AsterFlow 无通用卷积算子，请使用 Linear/MLP 或后续版本。")
end
