#!/usr/bin/env julia
# 用法：在仓库根目录执行
#   julia --project=. examples/tiny_mlp_train.jl

using AsterFlow

function main()
    m = train!(Sequential(Linear(8, 16), ReLU(), Linear(16, 1)))
    x = tensor(randn(Float32, 32, 8))
    y = tensor(randn(Float32, 32, 1))
    opt = AdamW(params(m), lr = 5f-3)
    for it in 1:20
        yhat = m(x)
        loss = mse_loss(yhat, y)
        zero_grad!.(params(m))
        backward(loss)
        step!(opt)
        if it % 5 == 0
            println("iter $it loss = ", to_array(loss)[1])
        end
    end
end

main()
