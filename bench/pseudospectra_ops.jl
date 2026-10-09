# Operators for `bench/pseudospectra_run.jl`.
#
# The matrix definitions are taken unchanged from `bench/fpsweep/operators.jl` of the main tree
# (the previous experiment), which records their sources: EigTool demos
# (github.com/eigtool/eigtool, commit 70c8a3c) for Grcar, Kahan, Hatano–Nelson, Airy, Davies,
# Landau, Orr–Sommerfeld and the two convection–diffusion operators, and Matrix Market / SuiteSparse
# for the Brusselator (NEP/brussel), Olmstead (NEP/olmstead), Québec (NEP/quebec) and
# Harwell–Boeing (lanpro) matrices.
#
# Two deliberate departures from that file:
#
# * complex EigTool matrices are used as they are, not through the real embedding
#   [Re −Im; Im Re]. The package handles ComplexF64 operators (`gammau` uses γ_{w+2} for complex
#   data), and the embedding duplicates every eigenvalue modulus, which would make "exactly 10
#   eigenvalues outside R" unreachable for half the counts;
# * the matrix-free convection–diffusion stencil is written here as a pair of closures with the
#   package's `LinOp(Amul, Aadj, n; epsA)` interface, instead of FPSweep's `Stencil2D` type.

using LinearAlgebra, SparseArrays, Random
using BallArithmetic: mul_up, sqrt_up, gamma_bound

const PS_MATDIR = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "fpsweep",
    "matrices")
const PS_MMCACHE = joinpath(homedir(), "Code", "RigorousPseudospectra.jl", "bench", "mm_cache")

# --- Matrix Market reader (verbatim from bench/fpsweep/operators.jl) ------------------------------
function ps_read_mtx(path)
    open(path) do io
        hdr = lowercase(readline(io))
        occursin("coordinate", hdr) || error("only coordinate format supported: $hdr")
        occursin("complex", hdr) && error("complex matrix: $hdr")
        sym = occursin("symmetric", hdr)
        skew = occursin("skew-symmetric", hdr)
        pattern = occursin("pattern", hdr)
        line = readline(io)
        while startswith(line, "%")
            line = readline(io)
        end
        m, n, nz = parse.(Int, split(line))
        I = Int[]
        J = Int[]
        V = Float64[]
        sizehint!(I, 2nz)
        sizehint!(J, 2nz)
        sizehint!(V, 2nz)
        for _ in 1:nz
            p = split(readline(io))
            i = parse(Int, p[1])
            j = parse(Int, p[2])
            v = pattern ? 1.0 : parse(Float64, p[3])
            push!(I, i)
            push!(J, j)
            push!(V, v)
            if (sym || skew) && i != j
                push!(I, j)
                push!(J, i)
                push!(V, skew ? -v : v)
            end
        end
        sparse(I, J, V, m, n)
    end
end

"Read a cached Matrix Market file by bare name (no download; both caches are searched)."
function ps_mm(name::AbstractString)
    for d in (PS_MMCACHE, PS_MATDIR)
        p = joinpath(d, "$name.mtx")
        isfile(p) && return ps_read_mtx(p)
    end
    error("$name.mtx not cached in $PS_MMCACHE or $PS_MATDIR")
end

# --- EigTool definitions (verbatim from bench/fpsweep/operators.jl) -------------------------------
function ps_cheb(N)
    x = cos.(pi * (0:N) / N)
    c = [2; ones(N - 1); 2] .* (-1) .^ (0:N)
    X = repeat(x, 1, N + 1)
    dX = X - X'
    D = (c * (1 ./ c)') ./ (dX + I)
    D = D - Diagonal(vec(sum(D; dims = 2)))
    return D, x
end

ps_grcar(N) = spdiagm(0 => ones(N), -1 => -ones(N - 1), 1 => ones(N - 1), 2 => ones(N - 2),
    3 => ones(N - 3))

function ps_kahan(N)
    s = (1 / 10)^(1 / (N - 1))
    c = sqrt(1 - s^2)
    col = -(s .^ (0:(N - 1))) * c
    K = triu(repeat(col, 1, N), 1) + Diagonal(s .^ (0:(N - 1)))
    return Matrix(K)
end

function ps_hatano(N; seed = 1)
    g = 0.4
    rng = MersenneTwister(seed)
    return spdiagm(-1 => exp(-g) * ones(N - 1), 0 => 3 * rand(rng, N) .- 1.5,
        1 => exp(g) * ones(N - 1))
end

function ps_airy(N)
    D, x = ps_cheb(N)
    D2 = D^2
    D2 = D2[2:N, 2:N]
    x = x[2:N]
    return 3e-4 * D2 + im * Diagonal(x)
end

function ps_davies(N)
    D, x = ps_cheb(N)
    x = x[2:N]
    L = N > 130 ? 16 : 9
    x = L * x
    D = D / L
    A = -D^2
    return A[2:N, 2:N] + im * Diagonal(x .^ 2)
end

function ps_landau(N)
    F = N > 200 ? 32 : 12
    beta = 0.5 * (1 .- (2 * (1:(N - 1))) .^ (-2.0)) .^ (-0.5)
    T = SymTridiagonal(zeros(N), beta)
    E = eigen(T)
    idx = sortperm(E.values)
    nodes = E.values[idx]
    weights = 2 * E.vectors[1, idx] .^ 2
    B = zeros(ComplexF64, N, N)
    for k in 1:N
        B[k, :] = weights' .* (sqrt(im * F) .* exp.(-im * pi * F * (nodes[k] .- nodes') .^ 2))
    end
    w = sqrt.(weights)
    for j in 1:N
        B[:, j] = w .* B[:, j] / w[j]
    end
    return B
end

function ps_orrsommerfeld(N)
    R = 5772
    D, x = ps_cheb(N)
    D2 = D^2
    D2i = D2[2:N, 2:N]
    S = Diagonal([0; 1 ./ (1 .- x[2:N] .^ 2); 0])
    D4 = (Diagonal(1 .- x .^ 2) * D^4 - 8 * Diagonal(x) * D^3 - 12 * D^2) * S
    D4 = D4[2:N, 2:N]
    Id = Matrix{Float64}(I, N - 1, N - 1)
    A = (D4 - 2 * D2i + Id) / R - 2im * Id - im * Diagonal(1 .- x[2:N] .^ 2) * (D2i - Id)
    B = D2i - Id
    return B \ A
end

"EigTool convdiff_demo: e D² + D on the Chebyshev grid, interior."
function ps_convdiff_cheb(N)
    e = 1 / 30
    D, _ = ps_cheb(N)
    C = e * D^2 + D
    return C[2:(end - 1), 2:(end - 1)]
end

# --- the matrix-free 2D convection–diffusion stencil ----------------------------------------------
# EigTool convdiff_fd_demo: C = kron(I, A1) + kron(A2, I) on an n1×n2 grid, with
# A1 = tridiag(-α1-β1, 2β1, α1-β1), A2 = tridiag(-α2-β2, 2β2, α2-β2),
# e = 1e-2, w1 = 0, w2 = 1, h_i = 1/(n_i+1), α_i = w_i/h_i, β_i = 2e/h_i².
# Five nonzeros per row and per column, so eq:roundA holds with ε_A = √5 γ_5: every entry of the
# computed image is a sum of at most five products, giving |fl(Ay) − Ay| ≤ γ_5 |A||y|
# componentwise, and ‖|A|‖ ≤ √(‖A‖₁‖A‖_∞) ≤ √5 ‖A‖.
struct ConvDiffFD
    n1::Int
    n2::Int
    d::Float64
    s1::Float64   # sub-diagonal in direction 1
    p1::Float64   # super-diagonal in direction 1
    s2::Float64
    p2::Float64
end

function ps_convdiff_fd(n1::Integer, n2::Integer)
    e = 1e-2
    w1 = 0.0
    w2 = 1.0
    h1 = 1 / (n1 + 1)
    h2 = 1 / (n2 + 1)
    α1 = w1 / h1
    β1 = 2e / h1^2
    α2 = w2 / h2
    β2 = 2e / h2^2
    return ConvDiffFD(n1, n2, 2β1 + 2β2, -α1 - β1, α1 - β1, -α2 - β2, α2 - β2)
end

Base.size(S::ConvDiffFD) = (S.n1 * S.n2, S.n1 * S.n2)

function _cdapply(S::ConvDiffFD, x::AbstractVector, adj::Bool)
    n1, n2 = S.n1, S.n2
    X = reshape(x, n1, n2)
    Y = similar(X)
    s1, p1, s2, p2 = adj ? (S.p1, S.s1, S.p2, S.s2) : (S.s1, S.p1, S.s2, S.p2)
    @inbounds for j in 1:n2, i in 1:n1
        v = S.d * X[i, j]
        i > 1 && (v += s1 * X[i - 1, j])
        i < n1 && (v += p1 * X[i + 1, j])
        j > 1 && (v += s2 * X[i, j - 1])
        j < n2 && (v += p2 * X[i, j + 1])
        Y[i, j] = v
    end
    return vec(Y)
end

ps_cd_mul(S) = x -> _cdapply(S, x, false)
ps_cd_adj(S) = x -> _cdapply(S, x, true)

"The same operator as a sparse matrix (for floating-point baselines only)."
function ps_convdiff_fd_sparse(S::ConvDiffFD)
    n1, n2 = S.n1, S.n2
    A1 = spdiagm(-1 => fill(S.s1, n1 - 1), 0 => fill(S.d / 2, n1), 1 => fill(S.p1, n1 - 1))
    A2 = spdiagm(-1 => fill(S.s2, n2 - 1), 0 => fill(S.d / 2, n2), 1 => fill(S.p2, n2 - 1))
    return kron(sparse(1.0I, n2, n2), A1) + kron(A2, sparse(1.0I, n1, n1))
end

# --- the catalogue --------------------------------------------------------------------------------
# Each entry returns a NamedTuple:
#   name, A (matrix or `nothing` for matrix-free), n, matvec, adjoint, epsA (matrix-free only),
#   hermitian, source, kmax, dense (whether a dense eigendecomposition is affordable).

const PS_EIGT = "EigTool demo (github.com/eigtool/eigtool, commit 70c8a3c)"

_stored(name, A, source; hermitian = false, kmax = 200, dense = true) =
    (; name, A, n = size(A, 1), matvec = x -> A * x, adjoint = x -> A' * x,
        epsA = nothing, hermitian, source, kmax, dense)

const PS_CATALOGUE = Dict{String, Function}(
    # Brusselator, Matrix Market NEP brussel
    "rdb200" => () -> _stored("rdb200", ps_mm("rdb200"),
        "Matrix Market NEP/brussel rdb200"; kmax = 200),
    "rdb800l" => () -> _stored("rdb800l", ps_mm("rdb800l"),
        "Matrix Market NEP/brussel rdb800l"; kmax = 250),
    "rdb2048" => () -> _stored("rdb2048", ps_mm("rdb2048"),
        "Matrix Market NEP/brussel rdb2048"; kmax = 300),
    "rdb5000" => () -> _stored("rdb5000", ps_mm("rdb5000"),
        "SuiteSparse Bai/rdb5000 (Brusselator)"; kmax = 300),
    # pseudospectra literature
    "grcar500" => () -> _stored("grcar500", ps_grcar(500), "$PS_EIGT grcar_demo(500)";
        kmax = 300),
    "convdiff_cheb200" => () -> _stored("convdiff_cheb200", ps_convdiff_cheb(200),
        "$PS_EIGT convdiff_demo(200), Chebyshev"; kmax = 190),
    "davies200" => () -> _stored("davies200", ps_davies(200),
        "$PS_EIGT davies_demo(200), complex, no real embedding"; kmax = 190),
    "kahan500" => () -> _stored("kahan500", ps_kahan(500), "$PS_EIGT kahan_demo(500)";
        kmax = 300),
    "airy200" => () -> _stored("airy200", ps_airy(200),
        "$PS_EIGT airy_demo(200), complex, no real embedding"; kmax = 190),
    "orrsom200" => () -> _stored("orrsom200", ps_orrsommerfeld(200),
        "$PS_EIGT orrsommerfeld_demo(200), complex, no real embedding"; kmax = 190),
    "landau400" => () -> _stored("landau400", ps_landau(400),
        "$PS_EIGT landau_demo(400), complex, no real embedding"; kmax = 300),
    "hatano2000" => () -> _stored("hatano2000", ps_hatano(2000),
        "$PS_EIGT hatano_demo(2000), diagonal from MersenneTwister(1)"; kmax = 300),
    # SuiteSparse / Matrix Market, two kinds
    "qh1484" => () -> _stored("qh1484", ps_mm("qh1484"),
        "Matrix Market NEP/quebec qh1484 (power system, non-normal)"; kmax = 300),
    "nos3" => () -> _stored("nos3", ps_mm("nos3"),
        "Matrix Market Harwell-Boeing/lanpro nos3 (symmetric positive definite)";
        hermitian = true, kmax = 250),
    # the case where the method declines
    "olm1000" => () -> _stored("olm1000", ps_mm("olm1000"),
        "Matrix Market NEP/olmstead olm1000"; kmax = 400),
)

"""
    ps_operator(name) -> NamedTuple

The matrix-free stencil is built by `ps_free_operator`; everything else comes from
`PS_CATALOGUE`.
"""
ps_operator(name::AbstractString) = PS_CATALOGUE[name]()

"""
    ps_free_operator(N1, N2) -> NamedTuple

The matrix-free convection–diffusion stencil on an `N1×N2` grid, `n = N1·N2`, with `ε_A = √5 γ_5`.
"""
function ps_free_operator(N1::Integer, N2::Integer; kmax = 150)
    S = ps_convdiff_fd(N1, N2)
    n = N1 * N2
    epsA = mul_up(sqrt_up(5.0), gamma_bound(5, Float64))
    return (; name = "convdiff_fd_$(N1)x$(N2)", A = nothing, n, matvec = ps_cd_mul(S),
        adjoint = ps_cd_adj(S), epsA, hermitian = false,
        source = "$PS_EIGT convdiff_fd_demo, matrix-free 5-point stencil, grid $(N1)×$(N2)",
        kmax, dense = false, stencil = S)
end
