## Fused 优化器占位（与 CUDA 单 kernel 或多 tensor fusion 路线对齐）

"""占位类型：未来可挂接 `foreach` / fused AdamW kernel。"""
struct FusedOptimizerStub end
