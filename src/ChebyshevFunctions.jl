
struct ChebyshevApproximator

    N::Int # degree of polynomial
    M::Int # number of nodes in each dimension
    D::Int
    a::Vector{Float64} # lower bounds
    b::Vector{Float64} # upper bounds
    X::Vector{Vector{Float64}} # points of evaluation
    T::Vector{Vector{Float64}} # polynomial basis

    function initializeChebyshevApproximator(N, M, a, b)
        D = size(a)[1]
        ν = [-cos((2*m - 1)*π/(2*M)) for m in 1:M]
        ξ = [(ν .+ 1) .* (b[d] - a[d])/2 .+ a[d] for d in 1:D]
        X = collect.(vec(collect(Base.product(ξ...))))
        T = calculateChebyshevPolynomials(X, N, a, b)
        return X, T
    end

    function ChebyshevApproximator(N, M, a, b)
        X, T = initializeChebyshevApproximator(N, M, a, b)
        D = size(a)[1]
        return new(N, M, D, a, b, X, T)
    end
end

function calculateChebyshevPolynomials(X, N, a, b)
    Xnorm = [2 .* (x - a)./(b - a) .- 1 for x in X]
    B = [ [ [cos(n * acos(xx)) for n in 0:N] for xx in x] for x in Xnorm]
    T = [kron(x...) for x in B]
    return T
end

function calculateChebyshevCoefficients(func, C::ChebyshevApproximator)
    y = func.(C.X)
    Tmat = hcat(C.T...)'
    TprimeT = Diagonal([Tmat[:,i]'*Tmat[:,i] for i in 1:size(Tmat)[2]])
    θ = TprimeT \ Tmat'*y
    return θ
end

function evaluateChebyshev(func, X, C::ChebyshevApproximator)
    θ = calculateChebyshevCoefficients(func, C)
    T = calculateChebyshevPolynomials(X, C.N, C.a, C.b)
    y = hcat(T...)' * θ
    return y
end
