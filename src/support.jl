#=
Find support points that represent a distribution.

Given a collection Y of vectors, find a set of support 
points of a specified size that captures the distribution
represented by Y.

The algorithm implemented here is based on equation 22 of
the following reference:

https://arxiv.org/abs/1609.01811
=#

using Random, LinearAlgebra

# The target function to be minimized, not explicitly used in the
# optimization algorithm.  'Y' contains the data and 'X' contains
# the candidate set of support points.
function support_loss(Y::Matrix{T}, X::Matrix{T}) where {T<:AbstractFloat}

    # Number of support points
    n = size(X, 2)

    # Number of data points
    N = size(Y, 2)

    f = 0.0
    for y in eachcol(Y)
        for x in eachcol(X)
            f += norm(x - y)
        end
    end
    f *= 2 / (n * N)

    for (j1, x1) in enumerate(eachcol(X))
        for j2 = 1:j1-1
            f -= 2 * norm(x1 - X[:, j2]) / n^2
        end
    end

    return f
end

# One iteration of fitting.  Returns an updated set of support points
# based on the current support points in X and the data in Y.
function update_support(Y::Matrix{T}, X::Matrix{T}, X1::Matrix{T}) where {T<:AbstractFloat}

    # Size of the sample data.
    N = size(Y, 2)

    # Dimension of the vectors
    d = size(Y, 1)

    # Number of support points
    n = size(X, 2)

    # Update each support point in turn
    u = zeros(d)
    for (i, xi) in enumerate(eachcol(X))

        X1[:, i] .= 0
        for (j, xj) in enumerate(eachcol(X))
            if j != i
                u .= xi - xj
                X1[:, i] += u / norm(u)
            end
        end
        X1[:, i] *= N / n

        q = 0.0
        for y in eachcol(Y)
            nm = norm(y - xi)
            X1[:, i] += y / nm
            q += 1 / nm
        end

        X1[:, i] /= q
    end
end

#=
Find a set of 'n' support points that represent the distribution of
the values in 'Y'.
=#
function supportpoints(
    Y::Matrix{T},
    n::Int;
    maxit = 1000,
    tol = 1e-4,
)::Matrix{T} where {T<:AbstractFloat}

    d, N = size(Y)

    # Starting values are a random sample from the data.
    # Need to perturb since the algorithm has a singularity
    # when a support point is exactly equal to a data point.
    X = zeros(d, n)
    ii = randperm(N)[1:n]
    for (j, i) in enumerate(ii)
        X[:, j] = Y[:, i] + 0.1 * randn(d)
    end

    # Storage for the next iterate
    X1 = zeros(d, n)

    success = false
    for itr = 1:maxit
        update_support(Y, X, X1)

        # Assess convergence based on the L2 distance from the
        # previous support points to the current ones.
        di = 0.0
        for j = 1:size(X, 2)
            di += norm(X1[:, j] - X[:, j])^2
        end
        di = sqrt(di)
        if di < tol
            success = true
            break
        end

        for j = 1:n
            X[:, j] .= X1[:, j]
        end
    end

    if !success
        @warn "Support point estimation did not converge"
    end

    return X
end
