# The split by modulus against the full block diagonalisation on the Belousov-Zhabotinsky
# compressions, one configuration at a time and on the same data: one frame, one far-block sweep,
# one set of couplings, and then the two compression layers built from the same ball `G`.
#
#   run_split(ke, ξ, neig, ks) -> NamedTuple
#
# Everything printed is a certified bound (outward rounded) or is labelled an estimate.
# Usage as a script:  julia -t 24 --project=. bench/bz/splitcert.jl KE XI NEIG KS [NTHETA]

include(joinpath(@__DIR__, "certify.jl"))          # bzcommon.jl, build_L, make_operator, classes

using Serialization

# A copy of `sc` carrying a different compression layer (nothing else is recomputed).
function with_compression(sc::RP.SplitCertificate; comp = nothing, msplit = nothing)
    return RP.SplitCertificate(sc.n, sc.k, sc.Q, sc.α, sc.G, sc.M, sc.gbar, sc.abar, sc.γ1,
        sc.γ2, sc.snorm, sc.far, sc.fd, comp, msplit, sc.hermitian, sc.gbars_power, sc.time)
end

function run_split(ke::Integer, ξ::Real, neig::Integer, ksarg::Integer;
        order = :auto, nθ::Integer = 64, maxdepth::Integer = 4, m::Integer = 8,
        do_svd::Bool = true, do_full::Bool = true, tagsuffix::AbstractString = "")
    k = 2^ke
    tag = "split_k$(ke)_xi$(ξ)_n$(neig)_ks$(ksarg)$(tagsuffix)"
    @info "run" tag nthreads = Threads.nthreads()
    L, binfo = build_L(k)
    l = noise_halfwidth(ξ, k)
    Amul, Aadj, η, oi = make_operator(L, l)
    op = RP.LinOp(Amul, Aadj, k; epsA = 0.0, etaA = η)

    λ = RP._ritz_leading(Amul, k, Float64; L = max(120, 6neig))
    cl = classes(λ)
    length(cl) > neig || error("only $(length(cl)) eigenvalue classes found")
    Rwin = (cl[neig + 1], cl[neig])
    R0 = sqrt(Rwin[1] * Rwin[2])

    t0 = time()
    Q, finfo = RP.two_sided_frame(Amul, Aadj, k, R0; ks = ksarg, Leig = max(120, 6neig),
        Lsv = min(k, 2ksarg + 60), T = Float64)
    t_frame = time() - t0

    # one split, no compression layer: the frame data, the sweep and the couplings
    t0 = time()
    sc0 = RP.certify_split(op, Q; order, target = R0, compression = :none)
    t_split = time() - t0
    @info "split" α = sc0.α ḡ = sc0.gbar ā = sc0.abar γ1 = sc0.γ1 γ2 = sc0.γ2 t_split

    # R inside the window, chosen in floating point exactly as bench/bz/certify.jl does
    Gm = Matrix(BallArithmetic.mid(sc0.G)); Mm = Matrix(BallArithmetic.mid(sc0.M))
    σ11_fp(Rc) = minimum(minimum(svdvals(Gm - Rc * cis(2π * t / 24) * Mm)) for t in 0:23) /
                 (1 + sc0.α)
    best = nothing
    for Rc in range(Rwin[1] * 1.002, Rwin[2] * 0.998; length = 25)
        s = σ11_fp(Rc)
        mg = s * max(Rc - sc0.gbar, 0.0) / (sc0.γ1 * sc0.γ2)
        (best === nothing || mg > best.margin) && (best = (; R = Rc, σ11 = s, margin = mg))
    end
    R = best.R

    # ---- the two compression layers on the same ball G
    t0 = time()
    ms = RP.split_compression(sc0.G, R; M = sc0.M, m)
    t_ms = time() - t0
    scs = with_compression(sc0; msplit = ms)
    mr = nothing
    t_mr = NaN
    scf = nothing
    if do_full
        t0 = time()
        mr = RP.miyajima_rump(sc0.G; M = sc0.M)
        t_mr = time() - t0
        scf = with_compression(sc0; comp = mr)
    end

    # the floating-point eigenvalues of the compression, for the comparison only
    ev11 = eigvals(Mm \ Gm)
    nout11 = count(z -> abs(z) > R, ev11)

    # ---- per-z cost and value of the three routes to σ11
    zs = ComplexF64[R * cis(2π * t / nθ) for t in 0:(nθ - 1)]
    t0 = time(); s11_split = minimum(RP.sigma11_lo(scs, z; method = :split) for z in zs)
    t_z_split = (time() - t0) / nθ
    s11_svd = NaN; t_z_svd = NaN
    if do_svd
        t0 = time(); s11_svd = minimum(RP.sigma11_lo(sc0, z; method = :svd) for z in zs)
        t_z_svd = (time() - t0) / nθ
    end
    s11_blocks = NaN; t_z_blocks = NaN
    if do_full && mr.ok
        t0 = time(); s11_blocks = minimum(RP.sigma11_lo(scf, z; method = :blocks) for z in zs)
        t_z_blocks = (time() - t0) / nθ
    end

    # ---- the arc cover of |z| = R with each route, and the exterior count
    function circle_and_count(sc, meth)
        t0 = time()
        chk = RP.check_circle(sc, 0.0, R; region = :exterior, nθ, maxdepth, method = meth)
        tc = time() - t0
        cnt, okc, _ = RP.count_exterior(sc, R; nθ, maxdepth, method = meth)
        return (; ok = chk.ok, region_ok = chk.region_ok, floor = chk.floor,
            minσ11 = chk.minσ11, minσ22 = chk.minσ22, nevals = chk.nevals, t = tc,
            count = okc ? cnt : -1)
    end
    c_split = circle_and_count(scs, :split)
    c_svd = do_svd ? circle_and_count(sc0, :svd) : nothing
    c_full = (do_full && mr.ok) ? circle_and_count(scf, :blocks) : nothing
    # the count when the circle is certified by the exact-SVD escape but the components come from
    # one layer or the other: this separates "the circle failed" from "no count"
    cnt_split_comps = RP.eigencount_outside(RP.split_components(ms), R)
    cnt_full_comps = (do_full && mr.ok) ? RP.eigencount_outside(RP._comps(scf), R) : (0, false)

    comps = RP.split_components(ms)
    lead = sort(comps; by = C -> -abs(C.center))[1:min(neig + 4, length(comps))]

    res = (; tag, ke, k, ξ, l, neig, ksarg, dimQ = size(Q, 2), R, Rwin, R0, α = sc0.α,
        gbar = sc0.gbar, abar = sc0.abar, γ1 = sc0.γ1, γ2 = sc0.γ2, m_far = sc0.far.m,
        t_frame, t_split, t_ms, t_mr,
        # the split
        kplus = ms.kplus, nout_fp = ms.nout_fp, nout11, tries = ms.tries, Rp = ms.Rp,
        split_ok = ms.ok, cert_ok = ms.cert_ok, reason = ms.reason, kcap = ms.kcap,
        kwanted = ms.kwanted, tz_split = ms.tz, reason_text = RP.split_reason(ms),
        nrmX = ms.nrmX, nrmR2 = ms.nrmR2, betaΛ = ms.betaΛ, beta1 = ms.beta1, beta2 = ms.beta2,
        kappa = ms.kappa, gbarJ = ms.gbarJ, gbar_order = ms.gbar_order, rhoJ = ms.rhoJ,
        rho_fp = ms.rho_fp, m_used = length(ms.gbars), ms_time = ms.time,
        nblocks_inner = ms.inner === nothing ? -1 : length(ms.inner.blocks),
        # the full diagonalisation
        full_ok = mr === nothing ? false : mr.ok,
        full_nblocks = (mr !== nothing && mr.ok) ? length(mr.cert.blocks) : -1,
        full_nclusters = (mr !== nothing && mr.ok) ? length(mr.clusters) : -1,
        full_nmerge = (mr !== nothing) ? mr.cert.candidate.info.nmerge : -1,
        full_maxX = (mr !== nothing) ? mr.cert.candidate.info.maxX : NaN,
        full_nrmR2 = mr === nothing ? NaN : mr.nrmR2,
        full_beta = (mr !== nothing && mr.ok) ? mr.betaΛ : NaN,
        full_kappa = (mr !== nothing && mr.ok) ? mr.kappa : NaN,
        full_maxrad = (mr !== nothing && mr.ok) ? maximum(C -> C.hull_radius, mr.clusters) : NaN,
        # σ11 and cost
        s11_split, s11_svd, s11_blocks, t_z_split, t_z_svd, t_z_blocks,
        c_split, c_svd, c_full, cnt_split_comps, cnt_full_comps,
        lead = [(; C.center, C.radius, C.count) for C in lead], nθ, maxdepth)
    serialize(joinpath(OUT, "$(tag).jls"), res)
    print_split(res)
    return res
end

function print_split(r)
    println("\n================ $(r.tag) ================")
    @printf("n = %d, ξ = %.0e, l = %d, neig = %d, dim Q = %d, R = %.6f (window %.4f..%.4f)\n",
        r.k, r.ξ, r.l, r.neig, r.dimQ, r.R, r.Rwin[1], r.Rwin[2])
    @printf("α ≤ %.2e, ḡ ≤ %.6f, ā ≤ %.4f, γ1 ≤ %.4e, γ2 ≤ %.4e, γ1γ2 = %.4e, R − ḡ = %.4e\n",
        r.α, r.gbar, r.abar, r.γ1, r.γ2, r.γ1 * r.γ2, r.R - r.gbar)
    println("\nthe split by modulus")
    @printf("  k₊ = %d (ceiling %d; eigenvalues of the compression outside R: %d), tries = %d, R' = %.6f\n",
        r.kplus, r.kcap, r.nout11, r.tries, r.Rp)
    @printf("  both conditions: %s -- %s\n", r.split_ok, r.reason_text)
    @printf("  ‖X‖ = %.4e, ‖R2‖ ≤ %.4e, β_Λ ≤ %.4e (β₁ %.4e, β₂ %.4e), κ ≤ %.4e\n",
        r.nrmX, r.nrmR2, r.betaΛ, r.beta1, r.beta2, r.kappa)
    @printf("  ḡ_J ≤ %.6f at m = %d (ρ(J) fp = %.6f, ratio %.4f); region radius ρ_J = %.6f %s R\n",
        r.gbarJ, r.m_used, r.rho_fp, r.gbarJ / max(r.rho_fp, eps()), r.rhoJ,
        r.rhoJ < r.R ? "<" : "≥")
    @printf("  ḡ_J by order: ")
    for (j, g) in enumerate(r.gbar_order)
        @printf("%d:%.5f ", j, g)
    end
    println()
    @printf("  blocks of the nested candidate: %d; built in %.2f s (candidate %.2f, certificate %.2f, powers %.2f, inner %.2f)\n",
        r.nblocks_inner, r.t_ms, r.ms_time.candidate, r.ms_time.certificate, r.ms_time.powers,
        r.ms_time.inner)
    if r.full_ok
        println("\nthe full block diagonalisation")
        @printf("  blocks %d, clusters %d, merges %d, max ‖X‖ %.3e\n", r.full_nblocks,
            r.full_nclusters, r.full_nmerge, r.full_maxX)
        @printf("  ‖R2‖ ≤ %.4e, β_Λ ≤ %.4e, κ ≤ %.4e, largest cluster radius %.4e; %.2f s\n",
            r.full_nrmR2, r.full_beta, r.full_kappa, r.full_maxrad, r.t_mr)
    else
        @printf("\nthe full block diagonalisation: ‖R2‖ ≤ %.4e, not certified (%.2f s)\n",
            r.full_nrmR2, r.t_mr)
    end
    println("\nσ11 on |z| = R and the arc cover")
    @printf("%-10s %14s %12s %10s %8s %12s %12s %8s %7s\n", "route", "min σ11 (pts)", "s per z",
        "cover", "floor", "min σ11 arc", "cover time", "nevals", "count")
    for (nm, s11, tz, c) in (("split", r.s11_split, r.t_z_split, r.c_split),
        ("svdbox", r.s11_svd, r.t_z_svd, r.c_svd),
        ("blocks", r.s11_blocks, r.t_z_blocks, r.c_full))
        c === nothing && continue
        @printf("%-10s %14.4e %12.4e %10s %8.2e %12.4e %12.2f %8d %7d\n", nm, s11, tz,
            c.ok, c.floor, c.minσ11, c.t, c.nevals, c.count)
    end
    @printf("count from the split components: %s; from the full diagonalisation: %s; dense (fp) count of the compression outside R: %d\n",
        r.cnt_split_comps[2] ? string(r.cnt_split_comps[1]) : "none",
        r.cnt_full_comps[2] ? string(r.cnt_full_comps[1]) : "none", r.nout11)
    println("\nleading certified components of the enclosure (split path)")
    for C in r.lead
        @printf("  %+.8f %+.8fim  radius %.3e  count %d  |c| = %.8f\n",
            real(C.center), imag(C.center), C.radius, C.count, abs(C.center))
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && !isempty(ARGS)
    run_split(parse(Int, ARGS[1]), parse(Float64, ARGS[2]), parse(Int, ARGS[3]),
        parse(Int, ARGS[4]);
        nθ = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 64)
end
