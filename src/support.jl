#=
Find support points that represent a distribution.

Given a collection Y of vectors, find a set of support
points of a specified size that captures the distribution
represented by Y.

The algorithm implemented here is based on equation 22 of
the following reference:

https://arxiv.org/abs/1609.01811
=#

"""
    loss(Y, X)

The target function to be minimized.  The columns of 'Y' contain the data and
the columns of 'X' contain the candidate set of support points.
"""
function loss(Y::Matrix{T}, X::Matrix{T}; wgts=[]) where {T<:AbstractFloat}

    # Number of support points
    n = size(X, 2)
    p = size(Y, 1)

    # Normalize weights to Union{Nothing,Vector{T}} for type stability
    W = isempty(wgts) ? nothing : convert(Vector{T}, wgts)
    N = W === nothing ? T(size(Y, 2)) : sum(W)

    f = zero(T)
    for (i, y) in enumerate(eachcol(Y))
        w = W === nothing ? one(T) : W[i]
        for x in eachcol(X)
            nm = zero(T)
            @inbounds for k in 1:p
                nm += (x[k] - y[k])^2
            end
            f += w * sqrt(nm)
        end
    end
    f *= 2 / (n * N)

    for (j1, x1) in enumerate(eachcol(X))
        for j2 in 1:j1-1
            x2 = @view X[:, j2]
            nm = zero(T)
            @inbounds for k in 1:p
                nm += (x1[k] - x2[k])^2
            end
            f -= 2 * sqrt(nm) / n^2
        end
    end

    return f
end

"""
    grad(Y, X, G)

The gradient of the target function to be minimized.  The columns of 'Y' contain the data and
the columns of 'X' contain the candidate set of support points.
"""
function grad!(Y::Matrix{T}, X::Matrix{T}, G::Matrix{T}; wgts=[]) where {T<:AbstractFloat}

    # Number of support points
    n = size(X, 2)

    # Normalize weights to Union{Nothing,Vector{T}} for type stability
    W = isempty(wgts) ? nothing : convert(Vector{T}, wgts)
    N = W === nothing ? T(size(Y, 2)) : sum(W)

    # Number of variables
    p = size(X, 1)

    u = zeros(T, p)
    G .= 0

    # norm(y_i - x_j) terms
    for (i, y) in enumerate(eachcol(Y))
        w = W === nothing ? one(T) : W[i]
        for (j, x) in enumerate(eachcol(X))
            nm = zero(T)
            @inbounds for k in 1:p
                u[k] = x[k] - y[k]
                nm += u[k]^2
            end
            c = w / sqrt(nm)
            @inbounds for k in 1:p
                G[k, j] += c * u[k]
            end
        end
    end
    G .*= 2 / (n * N)

    # norm(x_j - x_k) terms
    for (j1, x1) in enumerate(eachcol(X))
        for j2 in 1:j1-1
            x2 = @view X[:, j2]
            nm = zero(T)
            @inbounds for k in 1:p
                u[k] = x1[k] - x2[k]
                nm += u[k]^2
            end
            c = 2 / (n^2 * sqrt(nm))
            @inbounds for k in 1:p
                G[k, j1] -= c * u[k]
                G[k, j2] += c * u[k]
            end
        end
    end
end

# One iteration of fitting.  Returns an updated set of support points
# based on the current support points in X and the data in Y.
function update_support(Y::Matrix{T}, X::Matrix{T}, X1::Matrix{T}; wgts=[]) where {T<:AbstractFloat}

    # Normalize weights to Union{Nothing,Vector{T}} for type stability
    W = isempty(wgts) ? nothing : convert(Vector{T}, wgts)
    N = W === nothing ? T(size(Y, 2)) : sum(W)

    # Dimension of the vectors
    d = size(Y, 1)

    # Number of support points
    n = size(X, 2)

    # Update each support point in turn
    u = zeros(T, d)
    for (i, xi) in enumerate(eachcol(X))

        xi1 = @view X1[:, i]
        xi1 .= 0
        for (j, xj) in enumerate(eachcol(X))
            if j != i
                nm = zero(T)
                @inbounds for k in 1:d
                    u[k] = xi[k] - xj[k]
                    nm += u[k]^2
                end
                nm = sqrt(nm)
                @inbounds for k in 1:d
                    xi1[k] += u[k] / nm
                end
            end
        end
        xi1 .*= N / n

        q = zero(T)
        for (j, y) in enumerate(eachcol(Y))
            w = W === nothing ? one(T) : W[j]
            nm = zero(T)
            @inbounds for k in 1:d
                u[k] = y[k] - xi[k]
                nm += u[k]^2
            end
            nm = sqrt(nm)
            inv_nm = w / nm
            @inbounds for k in 1:d
                xi1[k] += inv_nm * y[k]
            end
            q += inv_nm
        end

        xi1 ./= q
    end
end

# Optimize the support points (X) for the data (Y) using MM iterations, starting
# from the provided value of X.
function fit_mm!(Y, X; maxit, tol, verbosity, wgts=[], rng=Random.default_rng())

    # Number of support points
    npt = size(X, 2)

    d, N = size(Y)

    # Storage for the next iterate
    X1 = zeros(d, npt)

    if verbosity > 1
        println("Support points majorization iterations:")
    end

    success = false
    for itr = 1:maxit
        update_support(Y, X, X1; wgts=wgts)

        # Assess convergence based on the L2 distance from the
        # previous support points to the current ones.
        di = norm(X1 - X)
        if verbosity > 1
            println(@sprintf("%5d %12.5f %12.5f", itr, di, loss(Y, X; wgts=wgts)))
        end
        if di < tol
            success = true
            break
        end

        X .= X1
    end

    return success
end

# Starting values are a random sample from the data.
# Need to perturb since the algorithm has a singularity
# when a support point is exactly equal to a data point.
function get_start(Y, npt; rng=Random.default_rng())

    d, N = size(Y)

    X = zeros(d, npt)
    ii = randperm(rng, N)[1:npt]
    for (j, i) in enumerate(ii)
        X[:, j] = Y[:, i] + 0.1 * randn(rng, d)
    end

    return X
end

function fit_grad!(Y, X; meth=LBFGS(), opts=Optim.options(), wgts=[], verbosity=0)

    # Data dimension (d) and number of observations (N)
    d, N = size(Y)

    # Number of support points
    npt = size(X, 2)

    # The loss function, passing the support points as a vector
    loss = Xv -> SupportPoints.loss(Y, reshape(Xv, d, npt); wgts=wgts)

    # The gradient of the loss function, passing the support points and gradient as vectors
    grad! = (Gv, Xv) -> SupportPoints.grad!(Y, reshape(Xv, d, npt), reshape(Gv, d, npt); wgts=wgts)

    if verbosity > 0
        print("Starting support points gradient optimization...")
    end

    rr = optimize(loss, grad!, copy(vec(X)), meth, opts)

    if !Optim.converged(rr)
        @warn "Gradient optimization did not converge"
    end

    if verbosity > 0
        println("Done")
    end

    X .= reshape(Optim.minimizer(rr), d, npt)

    return rr
end

"""
    supportpoints(Y, npt; maxit_grad=1000, tol_mm=1e-4, verbosity=0, rng=Random.default_rng())

Find a set of 'npt' support points that represent the distribution of
the values in 'Y', which is a d x n matrix.  The features are in the
columns of 'Y'.  The support points are returned as a d x npt matrix.

The default algorithm is up to 'maxit_mm' majorization/maximization iterations,
followed by up to 'maxit_grad' gradient descent iterations.
"""
function supportpoints(Y, npt; maxit_grad=1000, maxit_mm=5, wgts=[], tol_mm=1e-4, verbosity=0, rng=Random.default_rng())

    Y = copy(Y)

    # d = feature dimension, N = number of observations
    d, N = size(Y)

    X = get_start(Y, npt; rng=rng)

    # Start with some MM iterations
    _ = fit_mm!(Y, X; maxit=maxit_mm, wgts=wgts, tol=tol_mm, verbosity=verbosity, rng=rng)

    # Gradient iterations
    if maxit_grad > 0
        opts = Optim.Options(g_tol=1e-4, iterations=maxit_grad)
        rr = fit_grad!(Y, X; opts, wgts=wgts, verbosity=verbosity)
        success = Optim.converged(rr)
        if !success && verbosity > 0
            @warn "Support point estimation did not converge"
        end
    end

    return X
end
