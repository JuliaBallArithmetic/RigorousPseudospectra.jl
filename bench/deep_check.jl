# The three-tier split (`deep_split`, `deep_certificate`) against the two-tier one and against the
# exact-SVD escape, on the operators of `bench/pseudospectra_ops.jl` and on the dense gallery.
#
# For each (operator, target):
#
#   * one frame of `def:Q`, one far-block sweep, one set of couplings (`compression = :none`);
#   * the two-tier split by modulus on that compression;
#   * the three-tier cut: the sizes of E, N and D, the cut radii, ḡ_D, the residual coupling c21,
#     the merged far-block bound `‖N^j‖^{1/j}` against R, the new couplings, and whether it closes;
#   * the reduced certificate on `Q' = Q·U`, its arc cover, its exterior count and its isolation
#     discs, against the same on the full frame;
#   * the per-z cost of σ11 by the three routes, and the soundness of every certified floor against
#     a dense `σ_min(A − zI)` where n allows.
#
# Usage: julia --project=. -t 16 bench/deep_check.jl [op:target ...]

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
const NDENSE = 2200

const DEFAULT_ROWS = ["rdb200:10", "rdb200:5", "airy200:10", "airy200:5",
    "convdiff_cheb200:10", "convdiff_cheb200:5", "davies200:10", "davies200:5",
    "orrsom200:10", "orrsom200:5", "grcar500:10", "kahan500:10", "landau400:10",
    "nos3:10", "rdb800l:10", "qh1484:10"]

fp_eigs(o) = begin
    if o.dense && o.A !== nothing && o.n <= 6000
        λ = eigvals(Matrix(o.A))
        λ[sortperm(abs.(λ); rev = true)]
    else
        V, H = RP.arnoldi(o.matvec, o.n, min(o.n - 1, 300); T = ComplexF64)
        Lh = size(H, 2)
        λ = eigvals(H[1:Lh, 1:Lh])
        λ[sortperm(abs.(λ); rev = true)]
    end
end

function admissible_count(a::AbstractVector, c::Integer)
    c = min(c, length(a) - 1)
    while c < length(a) - 1 && abs(a[c] - a[c + 1]) <= 1e-9 * a[c]
        c += 1
    end
    return c
end

with_comp(sc; comp = nothing, msplit = nothing) = RP.SplitCertificate(sc.n, sc.k, sc.Q, sc.α,
    sc.G, sc.M, sc.gbar, sc.abar, sc.γ1, sc.γ2, sc.snorm, sc.far, sc.fd, comp, msplit,
    sc.hermitian, sc.gbars_power, sc.time)

function run_row(name::AbstractString, target::Integer)
    o = ps_operator(name)
    λ = fp_eigs(o)
    a = abs.(λ)
    c = admissible_count(a, target)
    R = sqrt(a[c] * a[c + 1])
    op = o.A === nothing ? RP.LinOp(o.matvec, o.adjoint, o.n; epsA = o.epsA) : RP.LinOp(o.A)
    t_frame = @elapsed ((Q, finfo) = RP.choose_frame(op, o.matvec, o.adjoint, R; kmax = o.kmax))
    t_split = @elapsed (sc = RP.certify_split(op, Q; target = R, compression = :none,
        blocksize = 32))
    gate = R * (R - sc.gbar)
    @info "row" name target count = c R dimQ = size(Q, 2) ḡ = sc.gbar γ1γ2 = sc.γ1 * sc.γ2 gate

    # the two-tier split on the full frame, for the comparison
    t_ms = @elapsed (ms = RP.split_compression(sc.G, R; M = sc.M))
    scs = with_comp(sc; msplit = ms)
    # the full block diagonalisation on the full frame
    t_mr = @elapsed (mr = RP.miyajima_rump(sc.G; M = sc.M))
    scf = with_comp(sc; comp = mr)

    # the three-tier cut
    kscan = unique(clamp.(round.(Int, exp.(range(log(2), log(max(size(Q, 2), 3));
        length = 8))), 1, size(Q, 2)))
    t_ds = @elapsed (ds = RP.deep_split(sc, R; scan = kscan))
    sc2 = nothing
    t_dc = NaN
    # the couplings of the reduced frame computed directly from the operator (2·k_EN applications
    # of A), as a baseline for the derived ones; the derivation applies A not at all
    g1d = NaN
    g2d = NaN
    gd_far = NaN
    if ds.ok
        t_dc = @elapsed (sc2 = RP.deep_certificate(sc, ds; compression = :full))
        try
            fdn = RP.frame_data(op, sc.Q * ds.U)
            _, g1d, g2d = RP.isometry_bounds(fdn, sc.abar)
        catch e
            @warn "direct couplings failed" e
        end
        if o.A !== nothing && o.n <= NDENSE
            Zn = Matrix(qr(sc.Q * ds.U).Q)[:, 1:size(ds.U, 2)]
            Pn = I - Zn * Zn'
            gd_far = opnorm(Pn * Matrix{ComplexF64}(o.A) * Pn)
        end
    end

    # the triangular route of sec:triangular on the same compression: an ordered Schur cut in the
    # unitary basis, no Sylvester similarity and no kappa. Added 2026-09-26; deepsplit.jl landed on
    # 09-14 and triangular.jl on 09-19, so the two had never been run against each other.
    t_ts = @elapsed (ts = RP.triangular_split(sc.G, sc.M, R; α = sc.α))
    zs = ComplexF64[R * cis(2π * t / NTHETA) for t in 0:(NTHETA - 1)]
    t_z_tri = NaN
    s11_tri = NaN
    tri_c = -1
    tri_gbar = NaN
    tri_div = NaN
    if ts !== nothing
        t_z_tri = @elapsed s11_tri = minimum(RP.sigma_tri_lo(ts, z) for z in zs)
        tri_c = ts.c
        tri_gbar = ts.gbar
        tri_div = ts.divisor
    end
    t_z_svd = @elapsed s11_svd = minimum(RP.sigma11_lo(sc, z; method = :svd) for z in zs)
    t_z_two = @elapsed s11_two = minimum(RP.sigma11_lo(scs, z; method = :split) for z in zs)
    t_z_deep = NaN
    s11_deep = NaN
    t_z_deep2 = NaN
    s11_deep2 = NaN
    if sc2 !== nothing
        t_z_deep = @elapsed s11_deep = minimum(RP.sigma11_lo(sc2, z; method = :svd) for z in zs)
        # the same reduced compression through the two-block floor (E against N) instead of one SVD
        ms2 = RP.split_compression(sc2.G, R; M = sc2.M)
        sc2s = with_comp(sc2; msplit = ms2)
        if ms2.cert_ok
            t_z_deep2 = @elapsed s11_deep2 = minimum(RP.sigma11_lo(sc2s, z; method = :split)
            for z in zs)
        end
    end

    cover(s, meth) = begin
        t = @elapsed chk = RP.check_circle(s, 0.0, R; region = :exterior, nθ = NTHETA,
            maxdepth = MAXDEPTH, method = meth)
        (; chk.ok, chk.region_ok, chk.floor, chk.minσ11, chk.minσ22, chk.nevals, t)
    end
    cv_old = cover(scf, :svd)
    cv_deep = sc2 === nothing ? nothing : cover(sc2, :svd)
    cnt_old = RP.count_exterior(scf, R; nθ = NTHETA, maxdepth = MAXDEPTH, method = :svd)
    cnt_deep = sc2 === nothing ? (0, false, nothing) :
               RP.count_exterior(sc2, R; nθ = NTHETA, maxdepth = MAXDEPTH, method = :svd)
    ext_deep = (sc2 !== nothing && cnt_deep[2]) ?
               RP.certify_exterior(sc2, R; nθ = NTHETA, maxdepth = MAXDEPTH, method = :svd) :
               nothing
    ext_old = cnt_old[2] ? RP.certify_exterior(scf, R; nθ = NTHETA, maxdepth = MAXDEPTH,
        method = :svd) : nothing

    # soundness: every certified floor against a dense σ_min
    nbad = 0
    ncmp = 0
    minratio = NaN
    if o.A !== nothing && o.n <= NDENSE
        Ad = Matrix{ComplexF64}(o.A)
        rs = Float64[]
        for ρ in (1.0, 1.2, 2.0), t in 0:11
            z = ρ * R * cis(2π * t / 12)
            tr = minimum(svdvals(Ad - z * I))
            for s in (sc2 === nothing ? (scf,) : (scf, sc2))
                f = RP.resolvent_floor(s, z; method = :svd).floor
                ncmp += 1
                f > tr * (1 + 1e-9) && (nbad += 1)
                f > 0 && push!(rs, f / tr)
            end
        end
        minratio = isempty(rs) ? NaN : minimum(rs)
    end

    fp_out = count(z -> abs(z) > R, λ)
    row = (; name, target, count_used = c, n = o.n, R, dimQ = size(Q, 2), rule = finfo.rule,
        α = sc.α, gbar = sc.gbar, γ1 = sc.γ1, γ2 = sc.γ2, gate, snorm = sc.snorm,
        t_frame, t_split, t_ms, t_mr, t_ds, t_dc,
        ds_ok = ds.ok, ds_reason = ds.reason, reason_text = RP.deep_reason(ds),
        kE = ds.kE, kN = ds.kN, kD = ds.kD, rE = ds.rE, rD = ds.rD, kEN = ds.kE + ds.kN,
        gD = ds.gD, c12 = ds.c12, c21 = ds.c21, gamma1n = ds.gamma1n, gamma2n = ds.gamma2n,
        snormn = ds.snormn, gbars = ds.gbars, gbarn = ds.gbarn, ntrials = ds.ntrials,
        local_min = ds.inner.local_min, u = ds.inner.u, v = ds.inner.v, w = ds.inner.w,
        scan = ds.inner.scan, s11_fp = ds.inner.s11, marg_fp = ds.inner.marg,
        kwanted = ds.inner.kwanted, tz_wanted = ds.inner.tz, kcap = ds.inner.kcap,
        alpha_new = sc2 === nothing ? NaN : sc2.α,
        two_kplus = ms.kplus, two_ok = ms.ok, two_reason = ms.reason,
        full_ok = mr.ok, full_nclusters = mr.ok ? length(mr.clusters) : -1,
        s11_svd, s11_two, s11_deep, s11_deep2,
        t_z_svd = t_z_svd / NTHETA, t_z_two = t_z_two / NTHETA, t_z_deep = t_z_deep / NTHETA,
        t_z_deep2 = t_z_deep2 / NTHETA, t_z_tri = t_z_tri / NTHETA, t_ts, s11_tri,
        tri_c, tri_gbar, tri_div,
        cv_old, cv_deep,
        cnt_old = cnt_old[2] ? cnt_old[1] : -1, cnt_deep = cnt_deep[2] ? cnt_deep[1] : -1,
        discs_old = ext_old === nothing ? -1 : count(d -> isfinite(d.radius), ext_old.discs),
        agree_old = ext_old === nothing ? false : ext_old.ok,
        discs_deep = ext_deep === nothing ? -1 : count(d -> isfinite(d.radius), ext_deep.discs),
        agree_deep = ext_deep === nothing ? false : ext_deep.ok,
        fp_out, nbad, ncmp, minratio, g1d, g2d, gd_far)
    print_row(row)
    return row
end

function print_row(r)
    println("\n================ $(r.name), target $(r.target) ================")
    @printf("n = %d, count = %d, R = %.6g, dim Q = %d, ḡ ≤ %.6g, γ1γ2 = %.4e, R(R − ḡ) = %.4e\n",
        r.n, r.count_used, r.R, r.dimQ, r.gbar, r.γ1 * r.γ2, r.gate)
    @printf("deep cut: %s -- %s\n", r.ds_ok, r.reason_text)
    r.kwanted > 0 && @printf("  the conditions want k_EN = %d of %d at %.4e s per z (ceiling %d)\n",
        r.kwanted, r.dimQ, r.tz_wanted, r.kcap)
    if r.ds_ok
        @printf("  E|N|D = %d|%d|%d of %d, r_E = %.6g, r_D = %.6g, %d trials, local minimum %s\n",
            r.kE, r.kN, r.kD, r.dimQ, r.rE, r.rD, r.ntrials, r.local_min)
        @printf("  ḡ_D ≤ %.6g, c₁₂ ≤ %.4e, c₂₁ ≤ %.4e (‖U*G̃S̃‖ %.4e, ‖S̃*G̃U‖ %.4e, ‖S̃*G̃S̃‖ %.4e)\n",
            r.gD, r.c12, r.c21, r.u, r.v, r.w)
        @printf("  γ'₁ ≤ %.4e, γ'₂ ≤ %.4e, γ'₁γ'₂ = %.4e against γ1γ2 = %.4e (ratio %.4g)\n",
            r.gamma1n, r.gamma2n, r.gamma1n * r.gamma2n, r.γ1 * r.γ2,
            r.gamma1n * r.gamma2n / (r.γ1 * r.γ2))
        @printf("  ḡ'_j^(1/j): ")
        for (j, g) in enumerate(r.gbars)
            @printf("%d:%.5g ", j, g^(1 / j))
        end
        @printf(" -> ḡ' ≤ %.6g against R = %.6g; α' = %.2e; frame %d -> %d\n",
            r.gbarn, r.R, r.alpha_new, r.dimQ, r.kEN)
    end
    @printf("%-10s %12s %12s %8s %12s %8s %8s\n", "route", "min σ11", "s per z", "cover",
        "floor on Γ", "count", "discs")
    @printf("%-10s %12.4e %12.4e %8s %12.4e %8d %8d\n", "svd, full", r.s11_svd, r.t_z_svd,
        r.cv_old.ok, r.cv_old.floor, r.cnt_old, r.discs_old)
    isnan(r.s11_tri) ||
        @printf("%-10s %12.4e %12.4e %8s %12s %8d %8s   cut %d of %d, ḡ_22 %.4g, divisor %.6f\n",
            "triangular", r.s11_tri, r.t_z_tri, "-", "-", r.tri_c, "-", r.tri_c, r.dimQ,
            r.tri_gbar, r.tri_div)
    @printf("%-10s %12.4e %12.4e %8s %12s %8s %8s\n", "two-tier", r.s11_two, r.t_z_two, "-", "-",
        "-", "-")
    if r.cv_deep !== nothing
        @printf("%-10s %12.4e %12.4e %8s %12.4e %8d %8d\n", "deep, svd", r.s11_deep, r.t_z_deep,
            r.cv_deep.ok, r.cv_deep.floor, r.cnt_deep, r.discs_deep)
        isnan(r.s11_deep2) ||
            @printf("%-10s %12.4e %12.4e %8s %12s %8s %8s\n", "deep, E|N", r.s11_deep2,
                r.t_z_deep2, "-", "-", "-", "-")
    end
    @printf("cor:agree: full frame %s, reduced frame %s; fp count outside R: %d\n",
        r.agree_old, r.agree_deep, r.fp_out)
    isnan(r.g1d) ||
        @printf("couplings of the reduced frame: derived γ'₁ ≤ %.4e, γ'₂ ≤ %.4e; recomputed from the operator %.4e, %.4e (ratios %.3g, %.3g); ḡ' derived %.4e, dense %.4e (ratio %.3g)\n",
            r.gamma1n, r.gamma2n, r.g1d, r.g2d, r.gamma1n / r.g1d, r.gamma2n / r.g2d,
            r.gbars[1], r.gd_far, r.gbars[1] / r.gd_far)
    if !isempty(r.scan)
        println("scan of the cut (σ̂11 and the margin are floating-point estimates)")
        @printf("%8s %12s %12s %12s %12s %12s %12s %10s %6s %6s\n", "k_EN", "gD", "gbar'",
            "R - gbar'", "g1n", "g2n", "s11hat", "margin", "far", "pass")
        for x in sort(r.scan; by = t -> t.kEN)
            @printf("%8d %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %10.3g %6s %6s\n", x.kEN,
                x.gD, x.gbarn, r.R - x.gbarn, x.γ1n, x.γ2n, x.s11, x.margin, x.far, x.pass)
        end
    end
    r.ncmp > 0 && @printf("soundness: %d of %d certified floors above the dense σ_min; smallest positive ratio %.3e\n",
        r.nbad, r.ncmp, r.minratio)
end

if abspath(PROGRAM_FILE) == @__FILE__
    rows = isempty(ARGS) ? DEFAULT_ROWS : ARGS
    out = NamedTuple[]
    for spec in rows
        nm, tg = split(spec, ':')
        try
            push!(out, run_row(String(nm), parse(Int, tg)))
        catch e
            @warn "row failed" spec e
            for l in stacktrace(catch_backtrace())[1:min(8, end)]
                println("   ", l)
            end
        end
    end
    serialize(joinpath(OUTDIR, "deep_check.$(gethostname()).jls"), out)
    println("\nserialised to ", joinpath(OUTDIR, "deep_check.$(gethostname()).jls"))
    println("\nsummary")
    @printf("%-18s %6s %6s %6s %5s %5s %5s %10s %10s %10s %10s %8s %8s %14s\n", "operator",
        "target", "dimQ", "k_EN", "E", "N", "D", "ḡ_D", "ḡ'", "R", "γ'γ'/γγ", "cnt old",
        "cnt deep", "reason")
    for r in out
        @printf("%-18s %6d %6d %6s %5d %5d %5d %10.4g %10.4g %10.4g %10.4g %8d %8d %14s\n",
            r.name, r.target, r.dimQ, r.ds_ok ? string(r.kEN) : "-", r.kE, r.kN, r.kD, r.gD,
            r.gbarn, r.R, r.gamma1n * r.gamma2n / (r.γ1 * r.γ2), r.cnt_old, r.cnt_deep,
            string(r.ds_reason))
    end
end
