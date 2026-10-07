# Test operators for the floating-point sweep experiments, with their sources.
#
# Every operator is a real Float64 operator in one of the forms of FPSweep (SparseOp, DenseOp,
# Stencil2D, Stencil3D). Complex matrices from EigTool are replaced by their real embedding
# [Re −Im; Im Re] (2n×2n), whose singular values are those of the complex matrix, each twice;
# this keeps every computation in real arithmetic, where the constants of eq:roundA and eq:epsP
# (Higham's real inner-product bounds) apply.
#
# Downloaded matrices go to ~/ResearchData/KrylovPseudospectra_Data/fpsweep/matrices.

using LinearAlgebra, SparseArrays, Random, Downloads

const DATA = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "fpsweep")
const MATDIR = joinpath(DATA, "matrices")
const NIST = "https://math.nist.gov/pub/MatrixMarket2"
const SSMC = "https://sparse.tamu.edu/MM"

# ---------------------------------------------------------------------------------------------
# Matrix Market reader (real/integer/pattern, general/symmetric, coordinate)
# ---------------------------------------------------------------------------------------------
function read_mtx(path)
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
        I = Int[]; J = Int[]; V = Float64[]
        sizehint!(I, 2nz); sizehint!(J, 2nz); sizehint!(V, 2nz)
        for _ in 1:nz
            p = split(readline(io))
            i = parse(Int, p[1]); j = parse(Int, p[2])
            v = pattern ? 1.0 : parse(Float64, p[3])
            push!(I, i); push!(J, j); push!(V, v)
            if (sym || skew) && i != j
                push!(I, j); push!(J, i); push!(V, skew ? -v : v)
            end
        end
        sparse(I, J, V, m, n)
    end
end

"Download (once) and read a Matrix Market file. Returns (A, url)."
function fetch_nist(path::String)          # e.g. "NEP/olmstead/olm2000"
    name = basename(path)
    mtx = joinpath(MATDIR, "$name.mtx")
    url = "$NIST/$path.mtx.gz"
    if !isfile(mtx)
        gz = mtx * ".gz"
        Downloads.download(url, gz)
        run(pipeline(`gunzip -c $gz`, stdout = mtx))
    end
    return read_mtx(mtx), url
end
function fetch_ssmc(group::String, name::String)
    mtx = joinpath(MATDIR, "$name.mtx")
    url = "$SSMC/$group/$name.tar.gz"
    if !isfile(mtx)
        tgz = joinpath(MATDIR, "$name.tar.gz")
        Downloads.download(url, tgz)
        run(`tar -xzf $tgz -C $MATDIR`)
        cp(joinpath(MATDIR, name, "$name.mtx"), mtx)
    end
    return read_mtx(mtx), url
end

# ---------------------------------------------------------------------------------------------
# EigTool demo matrices (definitions read from github.com/eigtool/eigtool, *_demo.m and
# private/cheb.m, commit 70c8a3c; copies in DATA/eigtool_demos)
# ---------------------------------------------------------------------------------------------

"cheb.m of Trefethen, Spectral Methods in MATLAB (as in EigTool private/cheb.m)."
function cheb(N)
    x = cos.(pi * (0:N) / N)
    c = [2; ones(N - 1); 2] .* (-1) .^ (0:N)
    X = repeat(x, 1, N + 1)
    dX = X - X'
    D = (c * (1 ./ c)') ./ (dX + I)
    D = D - Diagonal(vec(sum(D; dims = 2)))
    return D, x
end

realembed(A::AbstractMatrix{<:Complex}) = [real(A) -imag(A); imag(A) real(A)]

grcar_eig(N) = spdiagm(0 => ones(N), -1 => -ones(N - 1), 1 => ones(N - 1), 2 => ones(N - 2),
    3 => ones(N - 3))
function kahan_eig(N)
    s = (1 / 10)^(1 / (N - 1)); c = sqrt(1 - s^2)
    col = -(s .^ (0:(N - 1))) * c
    K = triu(repeat(col, 1, N), 1) + Diagonal(s .^ (0:(N - 1)))
    return Matrix(K)
end
function hatano_eig(N; seed = 1)
    g = 0.4
    rng = MersenneTwister(seed)
    return spdiagm(-1 => exp(-g) * ones(N - 1), 0 => 3 * rand(rng, N) .- 1.5,
        1 => exp(g) * ones(N - 1))
end
function airy_eig(N)
    D, x = cheb(N)
    D2 = D^2
    D2 = D2[2:N, 2:N]; x = x[2:N]
    return 3e-4 * D2 + im * Diagonal(x)
end
function davies_eig(N)
    D, x = cheb(N)
    x = x[2:N]
    L = N > 130 ? 16 : 9
    x = L * x; D = D / L
    A = -D^2
    return A[2:N, 2:N] + im * Diagonal(x .^ 2)
end
function landau_eig(N)
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
function orrsommerfeld_eig(N)
    R = 5772
    D, x = cheb(N)
    D2 = D^2; D2i = D2[2:N, 2:N]
    S = Diagonal([0; 1 ./ (1 .- x[2:N] .^ 2); 0])
    D4 = (Diagonal(1 .- x .^ 2) * D^4 - 8 * Diagonal(x) * D^3 - 12 * D^2) * S
    D4 = D4[2:N, 2:N]
    Id = Matrix{Float64}(I, N - 1, N - 1)
    A = (D4 - 2 * D2i + Id) / R - 2im * Id - im * Diagonal(1 .- x[2:N] .^ 2) * (D2i - Id)
    B = D2i - Id
    return B \ A
end
function basor_eig(N)
    r = -im ./ (1:N)
    c = [-im; pi; im ./ (1:(N - 2))]
    T = zeros(ComplexF64, N, N)
    for i in 1:N, j in 1:N
        T[i, j] = i >= j ? c[i - j + 1] : r[j - i + 1]
    end
    return T
end
"Chebyshev differentiation D_N of cheb.m with the first row and column removed (u(1) = 0)."
chebdiff(N) = cheb(N)[1][2:(N + 1), 2:(N + 1)]
"EigTool convdiff_demo: e D² + D on the Chebyshev grid, interior."
function convdiff_cheb(N)
    e = 1 / 30
    D, _ = cheb(N)
    C = e * D^2 + D
    return C[2:(end - 1), 2:(end - 1)]
end

"EigTool convdiff_fd_demo as a matrix-free 5-point stencil on the n1×n2 grid (n1 = N, n2 = 4N)."
function convdiff_fd_stencil(N; n1 = N, n2 = 4N)
    e = 1e-2; w1 = 0.0; w2 = 1.0
    h1 = 1 / (n1 + 1); h2 = 1 / (n2 + 1)
    α1 = w1 / h1; β1 = 2e / h1^2
    α2 = w2 / h2; β2 = 2e / h2^2
    # A1 = spdiags([-α1-β1, 2β1, α1-β1], -1:1): sub, diag, super; C = kron(I,A1) + kron(A2,I)
    return FPSweep.Stencil2D(n1, n2, 2β1 + 2β2, -α1 - β1, α1 - β1, -α2 - β2, α2 - β2,
        "convdiff_fd_$(n1)x$(n2)")
end
"EigTool skewlap3d_demo as a matrix-free 7-point stencil, n = (N-1)³."
function skewlap3d_stencil(N)
    M = N - 1
    d = fill(-6.0 * N^2, M^3)
    cm = 1.5 * N^2; cp = 0.5 * N^2       # D = N² toeplitz([-2 1.5 ...],[-2 .5 ...])
    return FPSweep.Stencil3D(M^3, M, M, M, d, cm, cp, cm, cp, cm, cp, "skewlap3d_N$N")
end

# ---------------------------------------------------------------------------------------------
# Hermitian constructed operators
# ---------------------------------------------------------------------------------------------
"2D Laplacian, 5-point stencil (4, −1), N×N grid, Dirichlet."
laplace2d_stencil(N) = FPSweep.Stencil2D(N, N, 4.0, -1.0, -1.0, -1.0, -1.0, "laplace2d_$(N)x$(N)")
"""3D Anderson model: H = −(nearest-neighbour hopping) + diag(ε_i), ε_i uniform in [−W/2, W/2],
L×L×L cube, Dirichlet boundary (the literature often uses periodic boundary conditions)."""
function anderson3d_stencil(L; W = 16.5, seed = 3)
    rng = MersenneTwister(seed)
    d = W .* (rand(rng, L^3) .- 0.5)
    return FPSweep.Stencil3D(L^3, L, L, L, d, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        "anderson3d_L$(L)_W$(W)")
end
"Graph Laplacian D − A of an undirected graph given by its (symmetric) adjacency pattern."
function graph_laplacian(Adj::SparseMatrixCSC)
    n = size(Adj, 1)
    I, J, _ = findnz(Adj + Adj')
    keep = I .!= J                                  # drop self-loops
    A = sparse(I[keep], J[keep], 1.0, n, n)          # unit weights, symmetric pattern
    return spdiagm(0 => vec(sum(A; dims = 2))) - A
end

# ---------------------------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------------------------
gaussian(n; seed = 5) = randn(MersenneTwister(seed), n, n) ./ sqrt(n)
"""Graded bidiagonal (constructed for the κ scan, not from the literature):
A[i,i] = ρ^(i-1), A[i,i+1] = ρ^(i-1)/2."""
graded_bidiag(n, ρ) = spdiagm(0 => ρ .^ (0:(n - 1)), 1 => 0.5 .* ρ .^ (0:(n - 2)))

# ---------------------------------------------------------------------------------------------
# The catalogue: name => () -> (op, source::String, symmetric::Bool)
# ---------------------------------------------------------------------------------------------
_sp(A, name) = FPSweep.SparseOp(sparse(Float64.(A)), name)
_dn(A, name) = FPSweep.DenseOp(Matrix{Float64}(A), name)

function nist_op(path; sym = false)
    A, url = fetch_nist(path)
    return (_sp(A, basename(path)), "Matrix Market (NIST), $(dirname(path)), $(basename(path)), $url", sym)
end
function ssmc_op(group, name; laplacian = false, sym = false)
    A, url = fetch_ssmc(group, name)
    laplacian && (A = graph_laplacian(A))
    lab = laplacian ? "graph Laplacian D−A of " : ""
    return (_sp(A, name * (laplacian ? "_lap" : "")),
        "$(lab)SuiteSparse Matrix Collection, group $group, $name, $url", sym)
end
const EIGT = "EigTool demo (github.com/eigtool/eigtool, commit 70c8a3c)"

const CATALOGUE = Dict{String, Function}(
    # --- Matrix Market NEP, non-Hermitian -----------------------------------------------------
    "olm2000" => () -> nist_op("NEP/olmstead/olm2000"),
    "olm5000" => () -> nist_op("NEP/olmstead/olm5000"),
    "rdb200" => () -> nist_op("NEP/brussel/rdb200"),
    "rdb800l" => () -> nist_op("NEP/brussel/rdb800l"),
    "rdb2048" => () -> nist_op("NEP/brussel/rdb2048"),
    "rdb5000" => () -> ssmc_op("Bai", "rdb5000"),
    "dw2048" => () -> nist_op("NEP/dwave/dw2048"),
    "dw8192" => () -> nist_op("NEP/dwave/dw8192"),
    "tols2000" => () -> nist_op("NEP/mvmtls/tols2000"),
    "tols4000" => () -> nist_op("NEP/mvmtls/tols4000"),
    "pde2961" => () -> nist_op("NEP/matpde/pde2961"),
    "bwm2000" => () -> nist_op("NEP/mvmbwm/bwm2000"),
    "ck656" => () -> nist_op("NEP/chuck/ck656"),
    "qh1484" => () -> nist_op("NEP/quebec/qh1484"),
    "rw5151" => () -> nist_op("NEP/mvmrwk/rw5151"),
    "af23560" => () -> nist_op("NEP/airfoil/af23560"),
    # --- pseudospectra literature (EigTool demos) ---------------------------------------------
    "grcar2000" => () -> (_sp(grcar_eig(2000), "grcar2000"), "$EIGT grcar_demo(2000)", false),
    "kahan500" => () -> (_sp(sparse(kahan_eig(500)), "kahan500"), "$EIGT kahan_demo(500)", false),
    "hatano2000" => () -> (_sp(hatano_eig(2000), "hatano2000"),
        "$EIGT hatano_demo(2000), diagonal from MersenneTwister(1)", false),
    "airy200" => () -> (_dn(realembed(airy_eig(200)), "airy200"),
        "$EIGT airy_demo(200), real embedding 2(N-1)", false),
    "davies200" => () -> (_dn(realembed(davies_eig(200)), "davies200"),
        "$EIGT davies_demo(200), real embedding 2(N-1)", false),
    "landau400" => () -> (_dn(realembed(landau_eig(400)), "landau400"),
        "$EIGT landau_demo(400), real embedding 2N", false),
    "orrsom200" => () -> (_dn(realembed(orrsommerfeld_eig(200)), "orrsom200"),
        "$EIGT orrsommerfeld_demo(200), real embedding 2(N-1)", false),
    "basor500" => () -> (_dn(realembed(basor_eig(500)), "basor500"),
        "$EIGT basor_demo(500) (Toeplitz), real embedding 2N", false),
    "convdiff_cheb200" => () -> (_dn(convdiff_cheb(200), "convdiff_cheb200"),
        "$EIGT convdiff_demo(200) (Chebyshev)", false),
    "chebdiff300" => () -> (_dn(chebdiff(300), "chebdiff300"),
        "Chebyshev differentiation from cheb.m (EigTool private/cheb.m), N=300, first row/column removed", false),
    # --- matrix-free stencils -----------------------------------------------------------------
    "convdiff_fd_N30" => () -> (convdiff_fd_stencil(30), "$EIGT convdiff_fd_demo(30), matrix-free stencil", false),
    "convdiff_fd_N60" => () -> (convdiff_fd_stencil(60), "$EIGT convdiff_fd_demo(60), matrix-free stencil", false),
    "convdiff_fd_N125" => () -> (convdiff_fd_stencil(125), "$EIGT convdiff_fd_demo(125), matrix-free stencil", false),
    "convdiff_fd_N250" => () -> (convdiff_fd_stencil(250), "$EIGT convdiff_fd_demo(250), matrix-free stencil", false),
    "convdiff_fd_N500" => () -> (convdiff_fd_stencil(500), "$EIGT convdiff_fd_demo(500), matrix-free stencil", false),
    "skewlap3d_N13" => () -> (skewlap3d_stencil(13), "$EIGT skewlap3d_demo(13), matrix-free stencil", false),
    "skewlap3d_N41" => () -> (skewlap3d_stencil(41), "$EIGT skewlap3d_demo(41), matrix-free stencil", false),
    "graded1000" => () -> (_sp(graded_bidiag(1000, 0.9), "graded1000"),
        "constructed: graded bidiagonal, ρ=0.9, n=1000", false),
    "graded400" => () -> (_sp(graded_bidiag(400, 0.9), "graded400"),
        "constructed: graded bidiagonal, ρ=0.9, n=400", false),
    "grcar500" => () -> (_sp(grcar_eig(500), "grcar500"), "$EIGT grcar_demo(500)", false),
    "olm1000" => () -> nist_op("NEP/olmstead/olm1000"),
    "landau150" => () -> (_dn(realembed(landau_eig(150)), "landau150"),
        "$EIGT landau_demo(150), real embedding 2N", false),
    # --- controls -----------------------------------------------------------------------------
    "gauss1000" => () -> (_dn(gaussian(1000), "gauss1000"), "Gaussian N(0,1/n), MersenneTwister(5)", false),
    "gauss2000" => () -> (_dn(gaussian(2000), "gauss2000"), "Gaussian N(0,1/n), MersenneTwister(5)", false),
    "jordan2000" => () -> (_sp(spdiagm(1 => ones(1999)), "jordan2000"),
        "bench/matrices.jl jordan_block(2000, 0) (sparse)", false),
    "chain2000" => () -> (_sp(spdiagm(0 => Float64.(1:2000), 1 => 0.7 * ones(1999)), "chain2000"),
        "bench/matrices.jl graded_chain(2000; s=0.7) (sparse)", false),
    # --- Hermitian: Harwell-Boeing (Matrix Market) ----------------------------------------------
    "nos3" => () -> nist_op("Harwell-Boeing/lanpro/nos3"; sym = true),
    "nos5" => () -> nist_op("Harwell-Boeing/lanpro/nos5"; sym = true),
    "nos6" => () -> nist_op("Harwell-Boeing/lanpro/nos6"; sym = true),
    "nos7" => () -> nist_op("Harwell-Boeing/lanpro/nos7"; sym = true),
    "bcsstk08" => () -> nist_op("Harwell-Boeing/bcsstruc1/bcsstk08"; sym = true),
    "bcsstk13" => () -> nist_op("Harwell-Boeing/bcsstruc1/bcsstk13"; sym = true),
    "bcsstk14" => () -> nist_op("Harwell-Boeing/bcsstruc2/bcsstk14"; sym = true),
    "bcsstk15" => () -> nist_op("Harwell-Boeing/bcsstruc2/bcsstk15"; sym = true),
    "bcsstk16" => () -> nist_op("Harwell-Boeing/bcsstruc2/bcsstk16"; sym = true),
    # --- Hermitian: SuiteSparse -----------------------------------------------------------------
    "Si2" => () -> ssmc_op("PARSEC", "Si2"; sym = true),
    "SiH4" => () -> ssmc_op("PARSEC", "SiH4"; sym = true),
    "benzene" => () -> ssmc_op("PARSEC", "benzene"; sym = true),
    "USpowerGrid_lap" => () -> ssmc_op("Pajek", "USpowerGrid"; laplacian = true, sym = true),
    "fv1" => () -> ssmc_op("Norris", "fv1"; sym = true),
    "Andrews" => () -> ssmc_op("Andrews", "Andrews"; sym = true),
    # --- Hermitian: matrix-free ---------------------------------------------------------------
    "laplace2d_N60" => () -> (laplace2d_stencil(60), "2D Laplacian (4,−1) stencil, 60×60, Dirichlet", true),
    "laplace2d_N300" => () -> (laplace2d_stencil(300), "2D Laplacian (4,−1) stencil, 300×300, Dirichlet", true),
    "anderson3d_L16" => () -> (anderson3d_stencil(16), "3D Anderson model, W=16.5, 16³, Dirichlet, MersenneTwister(3)", true),
    "anderson3d_L50" => () -> (anderson3d_stencil(50), "3D Anderson model, W=16.5, 50³, Dirichlet, MersenneTwister(3)", true),
)
