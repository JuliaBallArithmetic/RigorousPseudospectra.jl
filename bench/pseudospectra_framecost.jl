# Matvec count and wall clock of the frame construction alone (sec:frame, sec:depth), for the same
# operators and the same R as `bench/pseudospectra_run.jl`. The certificate itself is not run: the
# sweep's matvec count is exact in `frame.csv` (`nmatvec_far`), the frame's was not instrumented
# there, and this script supplies it.
#
#   julia --project=. -t 8 bench/pseudospectra_framecost.jl <op> [<op> ...]
#   julia --project=. -t 8 bench/pseudospectra_framecost.jl free 50 200
#
# Writes ~/ResearchData/KrylovPseudospectra_Data/pseudospectra/framecost.csv.

using RigorousPseudospectra
using LinearAlgebra, SparseArrays, Random, Printf

include(joinpath(@__DIR__, "pseudospectra_ops.jl"))

BLAS.set_num_threads(parse(Int, get(ENV, "PS_BLAS", "8")))

const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
const HDR = ["op", "n", "target", "count_used", "R", "dimQ", "depth_rule", "nmatvec_frame",
    "niter_depth", "time_frame"]

function emit(vals)
    p = joinpath(OUT, "framecost.csv")
    new = !isfile(p)
    open(p, "a") do io
        new && println(io, join(HDR, ","))
        println(io, join(vals, ","))
    end
end

"Eigenvalues by decreasing modulus, as in the driver."
function eigs_desc(o)
    if o.dense && o.A !== nothing && o.n <= 6000
        λ = eigvals(Matrix(o.A))
        return λ[sortperm(abs.(λ); rev = true)]
    end
    V, H = arnoldi(o.matvec, o.n, min(o.n - 1, 300); T = ComplexF64)
    Lh = size(H, 2)
    λ = eigvals(H[1:Lh, 1:Lh])
    return λ[sortperm(abs.(λ); rev = true)]
end

function adm(a, c)
    c = min(c, length(a) - 1)
    while c < length(a) - 1 && abs(a[c] - a[c + 1]) <= 1e-9 * a[c]
        c += 1
    end
    return c
end

function run_one(o)
    λ = eigs_desc(o)
    a = abs.(λ)
    for target in (10, 5)
        c = adm(a, target)
        R = sqrt(a[c] * a[c + 1])
        cnt = Ref(0)
        mv(x) = (cnt[] += 1; o.matvec(x))
        ad(x) = (cnt[] += 1; o.adjoint(x))
        op = o.A === nothing ? LinOp(mv, ad, o.n; epsA = o.epsA) : LinOp(o.A)
        # for a stored matrix LinOp does its own products, so count them through the closures the
        # frame uses; `choose_frame` calls `apply_cols(op, Q, ·)` too, which is `dim Q` matvecs each
        cnt[] = 0
        t = @elapsed ((Q, fi) = choose_frame(op, mv, ad, R; kmax = o.kmax))
        # For a stored matrix `LinOp(A)` multiplies with its own copy, so the calls the certificate
        # side makes are not seen by the closures: add the `apply_cols(op, Q, ·)` of
        # `float_split_estimates` (2 per column per depth iteration) and the stochastic order
        # estimate (`nsamples = 6` vectors, `mmax = 8` applications each). For a matrix-free
        # operator every application already goes through the closures.
        extra = o.A === nothing ? 0 :
                2 * sum(h -> h.dimQ, fi.history; init = 0) + 48 * length(fi.history)
        emit([o.name, o.n, target, c, @sprintf("%.10g", R), size(Q, 2), fi.rule,
            cnt[] + extra, length(fi.history), @sprintf("%.4g", t)])
        println(o.name, " target=", target, " R=", R, " dimQ=", size(Q, 2), " matvecs=",
            cnt[] + extra, " iterations=", length(fi.history), " t=", round(t; digits = 2), " s")
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    while !isempty(args)
        nm = popfirst!(args)
        if nm == "free"
            N1 = parse(Int, popfirst!(args))
            N2 = parse(Int, popfirst!(args))
            km = (!isempty(args) && all(isdigit, args[1])) ? parse(Int, popfirst!(args)) : 150
            run_one(ps_free_operator(N1, N2; kmax = km))
        else
            run_one(ps_operator(nm))
        end
    end
end
