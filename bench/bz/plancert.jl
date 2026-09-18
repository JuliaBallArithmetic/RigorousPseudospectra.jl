# The planning phase on the Belousov-Zhabotinsky configurations that declined: a floating-point
# grid of (R, m, dim Q), the feasible region of the gate and of the assembly test, and then one
# certified run at the point the planner picks.
#
# Usage as a script:  julia -t 24 --project=. bench/bz/plancert.jl KE XI NEIG

include(joinpath(@__DIR__, "bzcommon.jl"))
using Serialization, Printf

const NTHETA = 64
const MAXDEPTH = 4

function run_plan_bz(ke::Integer, ξ::Real, neig::Integer; kss = [20, 50, 100, 150],
        mmax::Integer = 12, nR::Integer = 5, amargin::Real = 1.5)
    k = 2^ke
    tag = "plan_k$(ke)_xi$(ξ)_n$(neig)"
    @info "run" tag
    L, binfo = build_L(k)
    l = noise_halfwidth(ξ, k)
    Amul, Aadj, η, oi = make_operator(L, l)
    op = RP.LinOp(Amul, Aadj, k; epsA = 0.0, etaA = η)

    t_plan = @elapsed (pl = RP.plan_split(op, Amul, Aadj; kss, mmax, nR, amargin, Leig = 200,
        T = Float64))
    println("\n================ $tag ================")
    @printf("n = %d, ξ = %.0e: planner %d grid points, %d feasible, %d matvecs, %.2f s (arnoldi %.2f, gkl %.2f, pool %.2f, grid %.2f)\n",
        k, ξ, length(pl.rows), length(pl.feasible), pl.nmatvec, pl.time.total, pl.time.arnoldi,
        pl.time.gkl, pl.time.pool, pl.time.grid)
    RP.plan_table(pl)

    res = (; tag, ke, k, ξ, neig, nrows = length(pl.rows), nfeas = length(pl.feasible),
        nmatvec = pl.nmatvec, t_plan = pl.time.total, Rs = pl.Rs, feasible = pl.feasible,
        predicted = pl.predicted)
    if pl.best === nothing
        res = merge(res, (; certified = false, best = nothing, t_cert = NaN, closed = false,
            count = -1))
        serialize(joinpath(OUT, "$(tag).$(gethostname()).jls"), res)
        return res
    end
    b = pl.best
    @printf("\ncertified run at the planner's point: R = %.6g, m = %d, ks = %d (dim Q ≈ %d)\n",
        b.R, b.m, b.ks, b.dimQ)
    t_cert = @elapsed begin
        # the same frame the planner scored: same Leig, same Lsv, same droptol
        Q, finfo = RP.two_sided_frame(Amul, Aadj, k, b.R; ks = b.ks, Leig = pl.params.Leig,
            Lsv = pl.params.Lsv, droptol = pl.params.droptol, T = Float64)
        sc = RP.certify_split(op, Q; target = b.R, order = b.m, compression = :split)
        # the planner's σ11 is the floating-point σ_min of the pencil, so the matching certified
        # route is the exact-SVD escape; the split route is reported beside it
        cnt, okc, chk = RP.count_exterior(sc, b.R; nθ = NTHETA, maxdepth = MAXDEPTH,
            method = :svd)
    end
    s11 = minimum(RP.sigma11_lo(sc, b.R * cis(2π * t / 24); method = :svd) for t in 0:23)
    s11s = minimum(RP.sigma11_lo(sc, b.R * cis(2π * t / 24); method = :split) for t in 0:23)
    @printf("  dim Q = %d, α ≤ %.2e, ḡ ≤ %.6g (planner %.6g), γ1γ2 = %.4e (planner %.4e)\n",
        sc.k, sc.α, sc.gbar, b.gbar_m, sc.γ1 * sc.γ2, b.γ1 * b.γ2)
    @printf("  σ11(R) ≥ %.4e by svdbox, %.4e by the split (planner %.4e); R − ḡ = %.4e (planner %.4e)\n",
        s11, s11s, b.s11, b.R - sc.gbar, b.s22)
    @printf("  arc cover %s, exterior count %s; certified run %.1f s against %.2f s of planning\n",
        chk.ok, okc ? string(cnt) : "none", t_cert, pl.time.total)
    res = merge(res, (; certified = true, best = b, t_cert, closed = chk.ok && okc,
        count = okc ? cnt : -1, dimQ = sc.k, gbar = sc.gbar, γ1 = sc.γ1, γ2 = sc.γ2, s11, s11s))
    serialize(joinpath(OUT, "$(tag).$(gethostname()).jls"), res)
    return res
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && !isempty(ARGS)
    run_plan_bz(parse(Int, ARGS[1]), parse(Float64, ARGS[2]), parse(Int, ARGS[3]))
end
