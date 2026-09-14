# A handful of the rows of `20260912_pseudospectra_runs.md` re-run with the compression split by
# modulus in place of the full block diagonalisation, everything else the same.
#
# For each (operator, target) the frame of `def:Q`, the far block and the couplings are computed
# once; the two compression layers are then built from the same ball `G` and compared on
#
#   * the arc cover of `|z| = R` with each of the three routes to σ11 (`:split`, `:svd`,
#     `:blocks`) and its cost;
#   * the exterior count of `prop:count` and the isolation discs of `cor:agree`;
#   * `k₊`, `‖X‖`, `‖R2‖`, `β_Λ`, `κ`, `ḡ_J` for the split and the corresponding numbers for the
#     full diagonalisation.
#
# Usage: julia --project=. -t 16 bench/pseudospectra_split_rerun.jl [op:target ...]

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

# the rows of the report that are re-run by default: the three where `prop:count` held on the
# circle and no count followed, and five where a count did follow (so that nothing is lost)
const DEFAULT_ROWS = ["convdiff_cheb200:5", "davies200:5", "orrsom200:5",
    "convdiff_cheb200:10", "davies200:10", "orrsom200:10", "airy200:5", "rdb200:10"]

function fp_eigs(o)
    if o.dense && o.A !== nothing && o.n <= 6000
        λ = eigvals(Matrix(o.A))
        return λ[sortperm(abs.(λ); rev = true)]
    end
    V, H = RP.arnoldi(o.matvec, o.n, min(o.n - 1, 300); T = ComplexF64)
    Lh = size(H, 2)
    λ = eigvals(H[1:Lh, 1:Lh])
    return λ[sortperm(abs.(λ); rev = true)]
end

function admissible_count(a::AbstractVector, c::Integer)
    c = min(c, length(a) - 1)
    while c < length(a) - 1 && abs(a[c] - a[c + 1]) <= 1e-9 * a[c]
        c += 1
    end
    return c
end

function with_compression(sc::RP.SplitCertificate; comp = nothing, msplit = nothing)
    return RP.SplitCertificate(sc.n, sc.k, sc.Q, sc.α, sc.G, sc.M, sc.gbar, sc.abar, sc.γ1,
        sc.γ2, sc.snorm, sc.far, sc.fd, comp, msplit, sc.hermitian, sc.gbars_power, sc.time)
end

function run_row(name::AbstractString, target::Integer)
    o = ps_operator(name)
    λ = fp_eigs(o)
    a = abs.(λ)
    c = admissible_count(a, target)
    R = sqrt(a[c] * a[c + 1])
    op = o.A === nothing ? RP.LinOp(o.matvec, o.adjoint, o.n; epsA = o.epsA) : RP.LinOp(o.A)
    t_frame = @elapsed ((Q, finfo) = RP.choose_frame(op, o.matvec, o.adjoint, R; kmax = o.kmax))
    t_split = @elapsed (sc0 = RP.certify_split(op, Q; target = R, compression = :none,
        blocksize = 32))
    @info "row" name target count = c R dimQ = size(Q, 2) ḡ = sc0.gbar γ1γ2 = sc0.γ1 * sc0.γ2

    t_ms = @elapsed (ms = RP.split_compression(sc0.G, R; M = sc0.M))
    scs = with_compression(sc0; msplit = ms)
    t_mr = @elapsed (mr = RP.miyajima_rump(sc0.G; M = sc0.M))
    scf = with_compression(sc0; comp = mr)

    zs = ComplexF64[R * cis(2π * t / NTHETA) for t in 0:(NTHETA - 1)]
    t_z_split = @elapsed s11_split = minimum(RP.sigma11_lo(scs, z; method = :split) for z in zs)
    t_z_svd = @elapsed s11_svd = minimum(RP.sigma11_lo(sc0, z; method = :svd) for z in zs)
    t_z_blk = NaN
    s11_blk = NaN
    if mr.ok
        t_z_blk = @elapsed s11_blk = minimum(RP.sigma11_lo(scf, z; method = :blocks) for z in zs)
    end

    function cover(sc, meth)
        t = @elapsed chk = RP.check_circle(sc, 0.0, R; region = :exterior, nθ = NTHETA,
            maxdepth = MAXDEPTH, method = meth)
        return (; chk.ok, chk.region_ok, chk.floor, chk.minσ11, chk.minσ22, chk.nevals, t)
    end
    cv_split = cover(scs, :split)
    cv_svd = cover(sc0, :svd)
    cv_blk = mr.ok ? cover(scf, :blocks) : nothing

    # the count, with the circle certified by each route and the components of the matching layer,
    # plus the "components only" question: can a count be read at all
    cnt_split = RP.count_exterior(scs, R; nθ = NTHETA, maxdepth = MAXDEPTH, method = :split)
    cnt_full = mr.ok ? RP.count_exterior(scf, R; nθ = NTHETA, maxdepth = MAXDEPTH,
        method = :blocks) : (0, false, nothing)
    # the circle by the exact-SVD escape with the components of each layer: this separates
    # "the circle failed" from "the components give no count"
    cnt_split_svd = RP.count_exterior(scs, R; nθ = NTHETA, maxdepth = MAXDEPTH, method = :svd)
    cnt_full_svd = mr.ok ? RP.count_exterior(scf, R; nθ = NTHETA, maxdepth = MAXDEPTH,
        method = :svd) : (0, false, nothing)
    comp_split = RP.eigencount_outside(RP.split_components(ms), R)
    comp_full = mr.ok ? RP.eigencount_outside(RP._comps(scf), R) : (0, false)

    ext_split = cnt_split[2] ? RP.certify_exterior(scs, R; nθ = NTHETA, maxdepth = MAXDEPTH,
        method = :split) : nothing
    ext_svd = cnt_split_svd[2] ? RP.certify_exterior(scs, R; nθ = NTHETA, maxdepth = MAXDEPTH,
        method = :svd) : nothing

    fp_out = count(z -> abs(z) > R, λ)
    ev11 = size(Q, 2) <= 2000 ?
           eigvals(Matrix(BallArithmetic.mid(sc0.M)) \ Matrix(BallArithmetic.mid(sc0.G))) :
           ComplexF64[]
    nout11 = isempty(ev11) ? -1 : count(z -> abs(z) > R, ev11)

    row = (; name, target, count_used = c, R, n = o.n, dimQ = size(Q, 2), rule = finfo.rule,
        α = sc0.α, gbar = sc0.gbar, γ1 = sc0.γ1, γ2 = sc0.γ2, t_frame, t_split, t_ms, t_mr,
        kplus = ms.kplus, ms.nout_fp, nout11, ms.tries, nrmX = ms.nrmX, nrmR2 = ms.nrmR2,
        betaΛ = ms.betaΛ, kappa = ms.kappa, gbarJ = ms.gbarJ, rho_fp = ms.rho_fp, rhoJ = ms.rhoJ,
        gbar_order = ms.gbar_order, ms_ok = ms.ok, cert_ok = ms.cert_ok, reason = ms.reason,
        kcap = ms.kcap, kwanted = ms.kwanted, tz_split = ms.tz,
        reason_text = RP.split_reason(ms),
        full_ok = mr.ok, full_nblocks = mr.ok ? length(mr.cert.blocks) : -1,
        full_nclusters = mr.ok ? length(mr.clusters) : -1,
        full_nmerge = mr.cert.candidate.info.nmerge, full_maxX = mr.cert.candidate.info.maxX,
        full_nrmR2 = mr.nrmR2, full_beta = mr.ok ? mr.betaΛ : NaN,
        full_kappa = mr.ok ? mr.kappa : NaN,
        full_maxrad = mr.ok ? maximum(C -> C.hull_radius, mr.clusters) : NaN,
        s11_split, s11_svd, s11_blk, t_z_split = t_z_split / NTHETA,
        t_z_svd = t_z_svd / NTHETA, t_z_blk = t_z_blk / NTHETA,
        cv_split, cv_svd, cv_blk,
        cnt_split = cnt_split[2] ? cnt_split[1] : -1,
        cnt_full = cnt_full[2] ? cnt_full[1] : -1,
        cnt_split_svd = cnt_split_svd[2] ? cnt_split_svd[1] : -1,
        cnt_full_svd = cnt_full_svd[2] ? cnt_full_svd[1] : -1,
        comp_split = comp_split[2] ? comp_split[1] : -1,
        comp_full = comp_full[2] ? comp_full[1] : -1,
        fp_out,
        discs_split = ext_split === nothing ? -1 : count(d -> isfinite(d.radius),
            ext_split.discs),
        agree_split = ext_split === nothing ? false : ext_split.ok,
        discs_svd = ext_svd === nothing ? -1 : count(d -> isfinite(d.radius), ext_svd.discs),
        agree_svd = ext_svd === nothing ? false : ext_svd.ok)
    print_row(row)
    return row
end

function print_row(r)
    println("\n================ $(r.name), target $(r.target) ================")
    @printf("n = %d, count = %d, R = %.6g, dim Q = %d (depth rule %s), ḡ ≤ %.6g, γ1γ2 = %.4e\n",
        r.n, r.count_used, r.R, r.dimQ, r.rule, r.gbar, r.γ1 * r.γ2)
    @printf("split:  k₊ = %d (ceiling %d; fp count outside R: compression %d, operator %d), tries %d, ‖X‖ = %.3e\n",
        r.kplus, r.kcap, r.nout11, r.fp_out, r.tries, r.nrmX)
    @printf("        both conditions: %s -- %s\n", r.ms_ok, r.reason_text)
    @printf("        ‖R2‖ ≤ %.3e, β_Λ ≤ %.3e, κ ≤ %.3e, ḡ_J ≤ %.6g (ρ(J) fp %.6g), ρ_J = %.6g, %.2f s\n",
        r.nrmR2, r.betaΛ, r.kappa, r.gbarJ, r.rho_fp, r.rhoJ, r.t_ms)
    @printf("        ḡ_J by order: ")
    for (j, g) in enumerate(r.gbar_order)
        @printf("%d:%.5g ", j, g)
    end
    println()
    @printf("full:   blocks %d, clusters %d, merges %d, max ‖X‖ %.3e, ‖R2‖ ≤ %.3e, β_Λ ≤ %.3e, κ ≤ %.3e, largest radius %.3e, %.2f s\n",
        r.full_nblocks, r.full_nclusters, r.full_nmerge, r.full_maxX, r.full_nrmR2, r.full_beta,
        r.full_kappa, r.full_maxrad, r.t_mr)
    @printf("%-8s %12s %12s %8s %12s %12s %8s %10s %8s\n", "route", "min σ11", "s per z",
        "cover", "floor on Γ", "min σ11 arc", "nevals", "cover (s)", "count")
    for (nm, s, tz, cv, ct) in (("split", r.s11_split, r.t_z_split, r.cv_split, r.cnt_split),
        ("svdbox", r.s11_svd, r.t_z_svd, r.cv_svd, r.cnt_split_svd),
        ("blocks", r.s11_blk, r.t_z_blk, r.cv_blk, r.cnt_full))
        cv === nothing && continue
        @printf("%-8s %12.4e %12.4e %8s %12.4e %12.4e %8d %10.2f %8d\n", nm, s, tz, cv.ok,
            cv.floor, cv.minσ11, cv.nevals, cv.t, ct)
    end
    @printf("components alone: split gives %s, full gives %s; fp count outside R: %d (compression %d)\n",
        r.comp_split < 0 ? "no count" : string(r.comp_split),
        r.comp_full < 0 ? "no count" : string(r.comp_full), r.fp_out, r.nout11)
    @printf("cor:agree with the split: %d discs, complete = %s (svd circle: %d discs, complete = %s)\n",
        r.discs_split, r.agree_split, r.discs_svd, r.agree_svd)
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
    serialize(joinpath(OUTDIR, "split_rerun.jls"), out)
    println("\nserialised to ", joinpath(OUTDIR, "split_rerun.jls"))
end
