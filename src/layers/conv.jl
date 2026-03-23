mutable struct Conv2d <: Module
    in_channels::Int
    out_channels::Int
    kernel_size::Tuple{Int,Int}
    training::Bool
end

function Conv2d(
    in_ch::Int,
    out_ch::Int,
    kernel_size::Union{Int,Tuple{Int,Int}};
)
    k = kernel_size isa Int ? (kernel_size, kernel_size) : kernel_size
    return Conv2d(in_ch, out_ch, k, true)
end

function (::Conv2d)(::Tensor)
    error("Conv2d: 尚未实现；当前 AsterFlow 无通用卷积算子，请使用 Linear/MLP 或后续版本。")
end
