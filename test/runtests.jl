using Test
using SupportPoints
using Random
using FiniteDifferences
using StableRNGs

rng = StableRNG(123)

include("Aqua.jl")

# Simulate data for testing
function gendat(n, f, sigma)
    Y = zeros(2, n)
    for i = 1:n
        x = 2 * pi * rand(rng)
        y = f(x) + sigma * randn(rng)
        Y[:, i] = [x, y]
    end
    return Y
end

@testset "test versus loss" begin

    # Simulate data
    n = 1000

    for f in [x->x, x->sin(x), x->x^2]
        for npt in [5, 10]

            XX = zeros(2, npt)

            for sigma in [0.1, 1, 10]
                Y = gendat(n, f, sigma)

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

    # Simulate data
    n = 1000

    # Number of variables
    p = 2

    for f in [x->x, x->sin(x), x->x^2]
        for npt in [5, 10]

            XX = zeros(2, npt)

            for sigma in [0.1, 1, 10]
                Y = gendat(n, f, sigma)

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

