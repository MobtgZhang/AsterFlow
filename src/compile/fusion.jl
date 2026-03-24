## 子图融合（占位）：与 eager 数值对照测试在测试中扩展。

"""
    fuse_linear_relu_chain!(g::IRGraph) -> g

占位：未来对 `IR_MatMul` + `IR_ReLU` 链做模式匹配并合并节点。
当前原样返回 `g`。
"""
function fuse_linear_relu_chain!(g::IRGraph)
    return g
end
