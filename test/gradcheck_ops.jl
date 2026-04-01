# 有限差分梯度检验（核心算子）

function _central_diff_scalar(f, x::Float32, eps::Float32)
    return (f(x + eps) - f(x - eps)) / (2 * eps)
end

@testset "gradcheck add matmul relu" begin
    epsv = 1.0f-2

    ## add 对左输入的偏导
    xa = 0.3f0
    function fb(w::Float32)
        ta = tensor(fill(w, 1, 1); requires_grad = true)
        tb = tensor(fill(0.2f0, 1, 1))
        return to_array(sum_tensor(add(ta, tb)))[1]
    end
    fd = _central_diff_scalar(fb, xa, epsv)
    ta = tensor(fill(xa, 1, 1); requires_grad = true)
    tb = tensor(fill(0.2f0, 1, 1))
    backward(sum_tensor(add(ta, tb)))
    @test isapprox(to_array(ta.grad)[1], fd; rtol = 0.15f0)

    ## 1×1 matmul：∂(u*v)/∂u = v
    u0 = 1.5f0
    v0 = 2.0f0
    function fm(u::Float32)
        tu = tensor(fill(u, 1, 1); requires_grad = true)
        tv = tensor(fill(v0, 1, 1))
        return to_array(sum_tensor(matmul(tu, tv)))[1]
    end
    fd_m = _central_diff_scalar(fm, u0, epsv)
    u = tensor(fill(u0, 1, 1); requires_grad = true)
    v = tensor(fill(v0, 1, 1))
    backward(sum_tensor(matmul(u, v)))
    @test isapprox(to_array(u.grad)[1], fd_m; rtol = 0.15f0)

    ## relu 在正部
    xr = 0.7f0
    function fr(z::Float32)
        t = tensor(fill(z, 1, 1); requires_grad = true)
        return to_array(sum_tensor(relu_tensor(t)))[1]
    end
    fd_r = _central_diff_scalar(fr, xr, epsv)
    tr = tensor(fill(xr, 1, 1); requires_grad = true)
    backward(sum_tensor(relu_tensor(tr)))
    @test isapprox(to_array(tr.grad)[1], fd_r; rtol = 0.15f0)
end
