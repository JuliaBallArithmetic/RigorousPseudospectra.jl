# The three-tier split on the Belousov-Zhabotinsky compressions: the deep tier leaves the frame and
# joins the far block, and the reduced certificate is compared with the two-tier one and with the
# exact-SVD escape on the full frame, all from one sweep and one set of couplings.
#
# Usage as a script:  julia -t 24 --project=. bench/bz/deepcert.jl KE XI NEIG KS [NTHETA]

include(joinpath(@__DIR__, "splitcert.jl"))     # certify.jl, bzcommon.jl, with_compression

function run_deep(ke::Integer, ξ::Real, neig::Integer, ksarg::Integer;
        order = :auto, nθ::Integer = 64, maxdepth::Integer = 4, m::Integer = 8,
        tagsuffix::AbstractString = "")
    k = 2^ke
    tag = "deep_k$(ke)_xi$(ξ)_n$(neig)_ks$(ksarg)$(tagsuffix)"
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
    t_frame = @elapsed ((Q, finfo) = RP.two_sided_frame(Amul, Aadj, k, R0; ks = ksarg,
        Leig = max(120, 6neig), Lsv = min(k, 2ksarg + 60), T = Float64))
    t_split = @elapsed (sc0 = RP.certify_split(op, Q; order, target = R0, compression = :none))

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
    gate = R * (R - sc0.gbar)
    @info "split" α = sc0.α ḡ = sc0.gbar γ1 = sc0.γ1 γ2 = sc0.γ2 γ1γ2 = sc0.γ1 * sc0.γ2 gate R

    t_ms = @elapsed (ms = RP.split_compression(sc0.G, R; M = sc0.M, m))
    scs = with_compression(sc0; msplit = ms)
    kscan = unique(clamp.(round.(Int, exp.(range(log(2), log(max(size(Q, 2), 3));
        length = 8))), 1, size(Q, 2)))
    t_ds = @elapsed (ds = RP.deep_split(sc0, R; m, scan = kscan))
    sc2 = nothing
    t_dc = NaN
    ms2 = nothing
    sc2s = nothing
    if ds.ok
        t_dc = @elapsed (sc2 = RP.deep_certificate(sc0, ds; compression = :full))
        ms2 = RP.split_compression(sc2.G, R; M = sc2.M, m)
        ms2.cert_ok && (sc2s = with_compression(sc2; msplit = ms2))
    end

    ev11 = eigvals(Mm \ Gm)
    nout11 = count(z -> abs(z) > R, ev11)

    zs = ComplexF64[R * cis(2π * t / nθ) for t in 0:(nθ - 1)]
    t_z_svd = @elapsed s11_svd = minimum(RP.sigma11_lo(sc0, z; method = :svd) for z in zs)
    t_z_two = @elapsed s11_two = minimum(RP.sigma11_lo(scs, z; method = :split) for z in zs)
    t_z_deep = NaN; s11_deep = NaN; t_z_deep2 = NaN; s11_deep2 = NaN
    if sc2 !== nothing
        t_z_deep = @elapsed s11_deep = minimum(RP.sigma11_lo(sc2, z; method = :svd) for z in zs)
        if sc2s !== nothing
            t_z_deep2 = @elapsed s11_deep2 = minimum(RP.sigma11_lo(sc2s, z; method = :split)
            for z in zs)
        end
    end

    function circle_and_count(s, meth)
        t = @elapsed chk = RP.check_circle(s, 0.0, R; region = :exterior, nθ, maxdepth,
            method = meth)
        cnt, okc, _ = RP.count_exterior(s, R; nθ, maxdepth, method = meth)
        (; chk.ok, chk.region_ok, chk.floor, chk.minσ11, chk.minσ22, chk.nevals, t,
            count = okc ? cnt : -1)
    end
    c_two = circle_and_count(scs, :split)
    c_svd = circle_and_count(sc0, :svd)
    c_deep = sc2 === nothing ? nothing : circle_and_count(sc2, :svd)
    ext_deep = (c_deep !== nothing && c_deep.count >= 0) ?
               RP.certify_exterior(sc2, R; nθ, maxdepth, method = :svd) : nothing

    res = (; tag, ke, k, ξ, l, neig, ksarg, dimQ = size(Q, 2), R, Rwin, α = sc0.α,
        gbar = sc0.gbar, abar = sc0.abar, γ1 = sc0.γ1, γ2 = sc0.γ2, gate, snorm = sc0.snorm,
        t_frame, t_split, t_ms, t_ds, t_dc, nout11,
        ds_ok = ds.ok, ds_reason = ds.reason, reason_text = RP.deep_reason(ds),
        kE = ds.kE, kN = ds.kN, kD = ds.kD, kEN = ds.kE + ds.kN, rE = ds.rE, rD = ds.rD,
        gD = ds.gD, c12 = ds.c12, c21 = ds.c21, gamma1n = ds.gamma1n, gamma2n = ds.gamma2n,
        snormn = ds.snormn, gbars = ds.gbars, gbarn = ds.gbarn, ntrials = ds.ntrials,
        local_min = ds.inner.local_min, u = ds.inner.u, v = ds.inner.v, w = ds.inner.w,
        scan = ds.inner.scan, s11_fp = ds.inner.s11, marg_fp = ds.inner.marg,
        kwanted = ds.inner.kwanted, tz_wanted = ds.inner.tz, kcap = ds.inner.kcap,
        alpha_new = sc2 === nothing ? NaN : sc2.α,
        two_kplus = ms.kplus, two_ok = ms.ok, two_reason = ms.reason,
        s11_svd, s11_two, s11_deep, s11_deep2,
        t_z_svd = t_z_svd / nθ, t_z_two = t_z_two / nθ, t_z_deep = t_z_deep / nθ,
        t_z_deep2 = t_z_deep2 / nθ, c_two, c_svd, c_deep,
        discs_deep = ext_deep === nothing ? -1 : count(d -> isfinite(d.radius), ext_deep.discs),
        agree_deep = ext_deep === nothing ? false : ext_deep.ok, nθ, maxdepth)
    serialize(joinpath(OUT, "$(tag).jls"), res)
    print_deep(res)
    return res
end

function print_deep(r)
    println("\n================ $(r.tag) ================")
    @printf("n = %d, ξ = %.0e, neig = %d, dim Q = %d, R = %.6f, ḡ ≤ %.6f, γ1γ2 = %.4e, R(R − ḡ) = %.4e\n",
        r.k, r.ξ, r.neig, r.dimQ, r.R, r.gbar, r.γ1 * r.γ2, r.gate)
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
    @printf("%-10s %12s %12s %8s %12s %8s\n", "route", "min σ11", "s per z", "cover",
        "floor on Γ", "count")
    @printf("%-10s %12.4e %12.4e %8s %12.4e %8d\n", "svd, full", r.s11_svd, r.t_z_svd,
        r.c_svd.ok, r.c_svd.floor, r.c_svd.count)
    @printf("%-10s %12.4e %12.4e %8s %12.4e %8d\n", "two-tier", r.s11_two, r.t_z_two,
        r.c_two.ok, r.c_two.floor, r.c_two.count)
    if r.c_deep !== nothing
        @printf("%-10s %12.4e %12.4e %8s %12.4e %8d\n", "deep, svd", r.s11_deep, r.t_z_deep,
            r.c_deep.ok, r.c_deep.floor, r.c_deep.count)
        isnan(r.s11_deep2) ||
            @printf("%-10s %12.4e %12.4e %8s %12s %8s\n", "deep, E|N", r.s11_deep2, r.t_z_deep2,
                "-", "-", "-")
    end
    @printf("cor:agree on the reduced frame: %s (%d discs); eigenvalues of the compression outside R: %d\n",
        r.agree_deep, r.discs_deep, r.nout11)
    if !isempty(r.scan)
        println("scan of the cut (floating-point estimates except ḡ_D, ḡ', γ')")
        @printf("%8s %12s %12s %12s %12s %12s %12s %6s %6s\n", "k_EN", "ḡ_D", "ḡ'", "R − ḡ'",
            "γ'₁", "γ'₂", "σ̂11(R)", "far", "pass")
        for x in sort(r.scan; by = t -> t.kEN)
            @printf("%8d %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %6s %6s\n", x.kEN, x.gD,
                x.gbarn, r.R - x.gbarn, x.γ1n, x.γ2n, x.s11, x.far, x.pass)
        end
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && !isempty(ARGS)
    run_deep(parse(Int, ARGS[1]), parse(Float64, ARGS[2]), parse(Int, ARGS[3]),
        parse(Int, ARGS[4]); nθ = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 64)
end
