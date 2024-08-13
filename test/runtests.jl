using Test
using SupportPoints
using Random
using FiniteDifferences
using StableRNGs
using LinearAlgebra

include("Aqua.jl")

# Simulate data for testing
function gendat(n, f, sigma, rng)
    Y = zeros(2, n)
    for i = 1:n
        x = 2 * pi * rand(rng)
        y = f(x) + sigma * randn(rng)
        Y[:, i] = [x, y]
    end
    return Y
end

function same_solution(x0, x1; tol=1e-3)

    for c0 in eachcol(x0)
        d = [norm(c0 - c1) for c1 in eachcol(x1)]
        if minimum(d) > tol
            return false
        end
    end

    return true
end

@testset "test loss and grad with weights" begin

    rng = StableRNG(123)
    n = 1000
    p = 2
    f = x -> x
    sigma = 1
    npt = 10

    Y = gendat(n, f, sigma, rng)
    m = 100
    YY = hcat(Y, Y[:, 1:m], Y[:, 1:m])
    wgts = vcat(fill(3, m), fill(1, n-m))

    X0 = supportpoints(Y, npt; maxit_grad=10)
    par0 = vec(X0)
    agrad1 = zeros(p * npt)
    agrad2 = zeros(p * npt)

    # The gradient of the loss function with weights
    grad1! = (Xv, Gv) -> SupportPoints.grad!(YY, reshape(Xv, p, npt), reshape(Gv, p, npt))

    # The gradient of the loss function without weights
    grad2! = (Xv, Gv) -> SupportPoints.grad!(Y, reshape(Xv, p, npt), reshape(Gv, p, npt); wgts=wgts)

    for it in 1:3
        par = par0 + 0.05*randn(p*npt)
        grad1!(par, agrad1)
        grad2!(par, agrad2)
        @test isapprox(agrad1, agrad2, atol=1e-4, rtol=1e-4)
    end
end

@testset "test fitting with weights" begin

    rng = StableRNG(123)
    n = 1000
    p = 2
    sigma = 1
    rng = StableRNG(123)

    for f in [x->x, x->sin(x)]
        for npt in [1, 2, 3]

            Y = gendat(n, f, sigma, rng)
            m = 100
            YY = hcat(Y, Y[:, 1:m], Y[:, 1:m])
            wgts = vcat(fill(3, m), fill(1, n-m))
            wgts = n * wgts / sum(wgts)

            X0 = supportpoints(Y, npt; wgts=wgts)
            X1 = supportpoints(YY, npt)

            @test same_solution(X0, X1; tol=1e-2)
        end
    end
end

@testset "test versus loss" begin

    rng = StableRNG(123)

    # Simulate data
    n = 1000

    for f in [x->x, x->sin(x), x->x^2]
        for npt in [5, 10]

            XX = zeros(2, npt)

            for sigma in [0.1, 1, 10]
                Y = gendat(n, f, sigma, rng)

                # Get the support points
                X = supportpoints(Y, npt)

                # The loss function at the support points
                v0 = SupportPoints.loss(Y, X)

                # Check that a random subset of the data has
                # worse fit than the estimated support points.
                for i in 1:100
                    ii = randperm(rng, n)[1:npt]
                    for (k,j) in enumerate(ii)
                        XX[:, k] = Y[:, j] + 0.01*randn(rng, 2)
                    end
                    v = SupportPoints.loss(Y, XX)
                    @test v > v0
                end
            end
        end
    end
end

@testset "test loss gradient" begin

    rng = StableRNG(123)

    # Simulate data
    n = 1000

    # Number of variables
    p = 2

    for f in [x->x, x->sin(x), x->x^2]
        for npt in [5, 10]

            XX = zeros(2, npt)

            for sigma in [0.1, 1, 10]
                Y = gendat(n, f, sigma, rng)

                X0 = supportpoints(Y, npt; maxit_grad=10)
                par0 = vec(X0)
                agrad = zeros(p * npt)

                # The loss function, passing the support points as a vector
                loss = Xv -> SupportPoints.loss(Y, reshape(Xv, p, npt))

                # The gradient of the loss function, passing the support points and gradient as vectors
                grad! = (Xv, Gv) -> SupportPoints.grad!(Y, reshape(Xv, p, npt), reshape(Gv, p, npt))

                for it in 1:3
                    par = par0 + 0.05*randn(p*npt)
                    grad!(par, agrad)
                    ngrad = grad(central_fdm(5, 1), loss, par)[1]
                    @test isapprox(agrad, ngrad, atol=1e-4, rtol=1e-4)
                end
            end
        end
    end
end

