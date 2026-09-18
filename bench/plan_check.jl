# The planning phase against the certified run, on the rows that declined in
# `20260914_compression_deepsplit.md`.
#
# For each operator: `plan_split` over a grid of (R, m, dim Q) in floating point, the feasible
# region of the gate and of the assembly test, and then one certified run at the point the planner
# picks (or, when there is none, what the planner predicts the frame would have to be). The
# planner's cost is reported beside the certified one.
#
# Usage: julia --project=. -t 16 bench/plan_check.jl [operator ...]

using RigorousPseudospectra
using LinearAlgebra, SparseArrays, Random, Printf, Serialization
import BallArithmetic
const RP = RigorousPseudospectra
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))

BLAS.set_num_threads(parse(Int, get(ENV, "PS_BLAS", "8")))
const OUTDIR = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
mkpath(OUTDIR)
const NTHETA = parse(Int, get(ENV, "PS_NTHETA", "64"))
const MAXDEPTH = parse(Int, get(ENV, "PS_MAXDEPTH", "6"))

# the rows of the deep-split report that declined: two on the gate, three on the assembly test
const DEFAULT_OPS = ["grcar500", "rdb800l", "kahan500", "nos3", "qh1484"]

function run_plan(name::AbstractString; kss = nothing, mmax::Integer = 12, nR::Integer = 5,
        amargin::Real = 1.5)
    o = ps_operator(name)
    op = o.A === nothing ? RP.LinOp(o.matvec, o.adjoint, o.n; epsA = o.epsA) : RP.LinOp(o.A)
    ks = kss === nothing ? unique(clamp.([16, 32, 64, 128], 1, max(o.kmax ÷ 2, 8))) :
         collect(kss)
    t_plan = @elapsed (pl = RP.plan_split(op, o.matvec, o.adjoint; kss = ks, mmax, nR, amargin,
        Leig = min(o.n - 1, 200)))
    println("\n================ $name ================")
    @printf("n = %d, planner: %d grid points, %d feasible, %d matvecs, %.2f s (arnoldi %.2f, gkl %.2f, pool %.2f, grid %.2f)\n",
        o.n, length(pl.rows), length(pl.feasible), pl.nmatvec, pl.time.total, pl.time.arnoldi,
        pl.time.gkl, pl.time.pool, pl.time.grid)
    RP.plan_table(pl)

    row = (; name, n = o.n, nrows = length(pl.rows), nfeas = length(pl.feasible),
        nmatvec = pl.nmatvec, t_plan = pl.time.total,
        Rs = pl.Rs, feasible = pl.feasible, predicted = pl.predicted)
    pl.best === nothing && return merge(row, (; certified = false, best = nothing,
        t_cert = NaN, closed = false, count = -1, note = "no feasible point on the grid"))

    b = pl.best
    @printf("\ncertified run at the planner's point: R = %.6g, m = %d, ks = %d (dim Q ≈ %d)\n",
        b.R, b.m, b.ks, b.dimQ)
    t_cert = @elapsed begin
        # the same frame the planner scored: same Leig, same Lsv, same droptol
        Q, finfo = RP.two_sided_frame(o.matvec, o.adjoint, o.n, b.R; ks = b.ks,
            Leig = pl.params.Leig, Lsv = pl.params.Lsv, droptol = pl.params.droptol,
            T = o.A === nothing ? Float64 : (eltype(o.A) <: Complex ? ComplexF64 : Float64))
        sc = RP.certify_split(op, Q; target = b.R, order = b.m, compression = :split,
            blocksize = 32)
        # the planner's σ11 is the floating-point σ_min of the pencil, so the matching certified
        # route is the exact-SVD escape; the split route is reported beside it
        cnt, okc, chk = RP.count_exterior(sc, b.R; nθ = NTHETA, maxdepth = MAXDEPTH,
            method = :svd)
    end
    @printf("  dim Q = %d, α ≤ %.2e, ḡ ≤ %.6g (planner %.6g), γ1 ≤ %.4e (planner %.4e), γ2 ≤ %.4e (planner %.4e)\n",
        sc.k, sc.α, sc.gbar, b.gbar_m, sc.γ1, b.γ1, sc.γ2, b.γ2)
    s11 = minimum(RP.sigma11_lo(sc, b.R * cis(2π * t / 24); method = :svd) for t in 0:23)
    s11s = minimum(RP.sigma11_lo(sc, b.R * cis(2π * t / 24); method = :split) for t in 0:23)
    @printf("  σ11(R) ≥ %.4e by svdbox, %.4e by the split (planner %.4e); R − ḡ = %.4e (planner %.4e), γ1γ2 = %.4e (planner %.4e)\n",
        s11, s11s, b.s11, b.R - sc.gbar, b.s22, sc.γ1 * sc.γ2, b.γ1 * b.γ2)
    @printf("  arc cover %s, exterior count %s; certified run %.1f s against %.2f s of planning\n",
        chk.ok, okc ? string(cnt) : "none", t_cert, pl.time.total)
    return merge(row, (; certified = true, best = b, t_cert, closed = chk.ok && okc,
        count = okc ? cnt : -1, dimQ = sc.k, gbar = sc.gbar, γ1 = sc.γ1, γ2 = sc.γ2, s11, s11s,
        note = ""))
end

if abspath(PROGRAM_FILE) == @__FILE__
    ops = isempty(ARGS) ? DEFAULT_OPS : ARGS
    out = NamedTuple[]
    for nm in ops
        try
            push!(out, run_plan(String(nm)))
        catch e
            @warn "operator failed" nm e
            for l in stacktrace(catch_backtrace())[1:min(8, end)]
                println("   ", l)
            end
        end
    end
    serialize(joinpath(OUTDIR, "plan_check.$(gethostname()).jls"), out)
    println("\nserialised to ", joinpath(OUTDIR, "plan_check.$(gethostname()).jls"))
    println("\nsummary")
    @printf("%-12s %6s %8s %8s %9s %8s %8s %8s %8s\n", "operator", "n", "grid", "feasible",
        "matvecs", "plan s", "cert s", "closed", "count")
    for r in out
        @printf("%-12s %6d %8d %8d %9d %8.2f %8s %8s %8s\n", r.name, r.n, r.nrows, r.nfeas,
            r.nmatvec, r.t_plan, isnan(r.t_cert) ? "-" : @sprintf("%.1f", r.t_cert),
            string(r.closed), r.count < 0 ? "none" : string(r.count))
    end
end
