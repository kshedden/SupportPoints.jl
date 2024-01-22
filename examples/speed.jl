using SupportPoints

function gendat(d, n, sigma)
    Y = zeros(d, n)
    u = rand(d)
    f = rand(d)
    for i = 1:n
        x = 2 * pi * rand(d)
        Y[:, i] = sin.(f.*x + u) + sigma * randn(d)
    end
    return Y
end

npt = 5
d = 5
n = 1000
sigma = 1.0
Y = gendat(d, n, sigma)

numit = 10
rr = zeros(numit, 4)

for it in 1:numit

    # Default approach (MM followed by gradient descent)
    t1 = time()
    X1 = supportpoints(Y, npt)
    t1 = time() - t1

    # MM-only approach
    t2 = time()
    X2 = supportpoints(Y, npt; maxit_mm=1000, maxit=0)
    t2 = time() - t2

    rr[it, :] = [t1, t2, SupportPoints.loss(Y, X1), SupportPoints.loss(Y, X2)]
end
