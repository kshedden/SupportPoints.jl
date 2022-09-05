using Test, SupportPoints, Random

# Simulate data for testing
function gendat(n, f, sigma)
    Y = zeros(2, n)
    for i = 1:n
        x = 2 * pi * rand()
        y = f(x) + sigma * randn()
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
                v0 = support_loss(Y, X)

                # Check that a random subset of the data has
                # worse fit than the estimated support points.
                for i in 1:100
                    ii = randperm(n)[1:npt]
                    for (k,j) in enumerate(ii)
                        XX[:, k] = Y[:, j] + 0.01*randn(2)
                    end
                    v = support_loss(Y, XX)
                    @test v > v0
                end
            end
        end
    end
end
