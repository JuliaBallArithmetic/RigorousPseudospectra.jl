# Dense diagnostic of what the frame of def:Q can reach, for every operator of pseudospectra_ops.jl.
# Nothing here is certified. For each gap radius R = sqrt(|λ_N||λ_{N+1}|) between distinct
# eigenvalue moduli, N eigenvalues lie outside R; prop:frame gives ḡ ≥ ‖G22‖ ≥ σ_{2d+1}(A) for any
# frame of dimension d, so d_min(R) = min{d : σ_{2d+1}(A) < R} is the least frame dimension at which
# ḡ < R is possible at all. `sep_ok` records σ_{2N+1}(A) < R: whether a frame carrying only those N
# eigenvalues could have ‖G22‖ < R.
#
#   JULIA_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 julia --project=bench bench/reach_diagnostic.jl
using LinearAlgebra, Printf
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "16")))

const HOST = readchomp(`hostname -s`)
const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra", "reach")
mkpath(OUT)
names = isempty(ARGS) ? ["rdb200", "rdb800l", "rdb2048", "rdb5000", "nos3", "olm1000", "qh1484",
    "grcar500", "kahan500", "davies200", "airy200", "orrsom200", "landau400", "hatano2000",
    "convdiff_cheb200"] : split(ARGS[1], ",")
NMAX = typemax(Int)

io = open(joinpath(OUT, "reach_diagnostic.$(HOST).csv"), "w")
println(io, "operator,n,N,R,lamN,lamN1,sigma_2N1,dmin,sep_ok")
for nm in names
    name = String(nm)
    t = @elapsed begin
        A = Matrix(ps_operator(name).A)
        n = size(A, 1)
        lam = sort(abs.(eigvals(A)); rev = true)
        s = svdvals(A)
    end
    for N in 1:min(NMAX, n - 1)
        lam[N] > lam[N + 1] * (1 + 1e-9) || continue
        R = sqrt(lam[N] * lam[N + 1])
        j = findfirst(<(R), s)
        dmin = j === nothing ? -1 : max(cld(j - 1, 2), 0)
        sep_ok = 2N + 1 <= n && s[2N + 1] < R
        @printf(io, "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%s\n", name, n, N, R, lam[N], lam[N + 1],
            2N + 1 <= n ? s[2N + 1] : NaN, dmin, sep_ok)
    end
    flush(io)
    @printf("%s n=%d done in %.1f s\n", name, size(A, 1), t)
end
close(io)
println("wrote ", joinpath(OUT, "reach_diagnostic.$(HOST).csv"))
