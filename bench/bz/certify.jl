# Step 2 of the brief: the certified splitting of A = N∘L, one configuration at a time.
#
#   run_one(ke, ξ, neig, ks; order, dense) -> NamedTuple
#
# `ke` gives the partition 2^ke, `ξ` the noise half-width as a fraction of the interval
# (l = round(ξ·2^ke)), `neig` how many leading eigenvalues must lie outside R (a conjugate pair
# counted once), `ks` the number of singular directions per side in the frame of def:Q (0 uses
# `choose_frame`). Everything printed is either a certified bound (outward rounded) or is
# labelled an estimate. Results are serialised to OUT.
#
# Usage as a script:
#   julia -t 24 --project=. certify.jl KE XI NEIG KS [ORDER] [DENSE]

include("bzcommon.jl")
using Serialization

"Moduli of the Ritz values, a conjugate pair counted once, in decreasing order."
function classes(λ; tol = 1e-8)
    a = sort(abs.(λ); rev = true)
    out = Float64[]
    for x in a
        (isempty(out) || abs(x - out[end]) > tol * max(1.0, x)) && push!(out, x)
    end
    return out
end

function run_one(ke::Integer, ξ::Real, neig::Integer, ksarg::Integer;
        order = :auto, dense::Bool = false, method::Symbol = :svd, nθ::Integer = 64,
        maxdepth::Integer = 4, rmin::Bool = true, full::Bool = true, xmax::Real = 1e8,
        vbd_method::Symbol = :sylvester, radii_factors = [1.0, 1.05, 1.2, 1.5, 2.0],
        tagsuffix::AbstractString = "", blocksize = nothing, sweep_threads = nothing)
    k = 2^ke
    tag = "k$(ke)_xi$(ξ)_n$(neig)_ks$(ksarg)$(tagsuffix)"
    @info "run" tag nthreads = Threads.nthreads()
    t0 = time()
    L, binfo = build_L(k)
    t_build = time() - t0
    l = noise_halfwidth(ξ, k)
    Amul, Aadj, η, oi = make_operator(L, l)
    op = RP.LinOp(Amul, Aadj, k; epsA = 0.0, etaA = η)
    @info "operator" k l η w = oi.w εL = oi.εL normL = oi.normL t_build

    λ = RP._ritz_leading(Amul, k, Float64; L = max(120, 6neig))
    cl = classes(λ)
    length(cl) > neig || error("only $(length(cl)) eigenvalue classes found")
    Rwin = (cl[neig + 1], cl[neig])          # any R here leaves exactly `neig` classes outside
    R0 = sqrt(Rwin[1] * Rwin[2])
    @info "radius window" Rwin R0 cl[1:min(neig + 2, length(cl))]

    # ------------------------------------------------------------------------------------------------
    # the frame
    t0 = time()
    if ksarg == 0
        Q, finfo = RP.choose_frame(op, Amul, Aadj, R0; kmax = min(k - 1, 400), T = Float64,
            verbose = true)
        how = "choose_frame (eq:depthrule, floating point)"
    else
        Q, finfo = RP.two_sided_frame(Amul, Aadj, k, R0; ks = ksarg, Leig = max(120, 6neig),
            Lsv = min(k, 2ksarg + 60), T = Float64)
        how = "two_sided_frame, ks = $ksarg per side (def:Q)"
    end
    t_frame = time() - t0
    @info "frame" how dimQ = size(Q, 2) t_frame

    # ------------------------------------------------------------------------------------------------
    # the certified split
    GC.gc()
    mem0 = Sys.maxrss()
    t0 = time()
    swkw = (; (blocksize === nothing ? () : (:blocksize => blocksize,))...,
              (sweep_threads === nothing ? () : (:nthreads => sweep_threads,))...)
    sc = RP.certify_split(op, Q; order, target = R0, xmax, vbd_method, swkw...)
    t_split = time() - t0
    mem_MB = (Sys.maxrss() - mem0) / 2^20
    far = sc.far
    @info "split" α = sc.α ḡ = sc.gbar ā = sc.abar γ1 = sc.γ1 γ2 = sc.γ2 m = far.m t_split mem_MB

    # ------------------------------------------------------------------------------------------------
    # R inside the window, chosen in floating point (the choice is not part of the certificate): the
    # point of the window at which the estimated margin σ11(R)(R − ḡ)/(γ1γ2) of eq:assembly is largest.
    Gm = Matrix(BallArithmetic.mid(sc.G)); Mm = Matrix(BallArithmetic.mid(sc.M))
    σ11_fp(R) = minimum(minimum(svdvals(Gm - R * cis(2π * t / 24) * Mm)) for t in 0:23) / (1 + sc.α)
    Rscan = NamedTuple[]
    for Rc in range(Rwin[1] * 1.002, Rwin[2] * 0.998; length = 25)
        s = σ11_fp(Rc)
        push!(Rscan, (; R = Rc, σ11 = s, margin = s * max(Rc - sc.gbar, 0.0) / (sc.γ1 * sc.γ2)))
    end
    best = Rscan[argmax(getfield.(Rscan, :margin))]
    R = best.R
    @info "R chosen in the window" Rwin R margin = best.margin σ11 = best.σ11

    # the circle at R and the outer circles. `full = false` reports the POINTWISE certified floor
    # at the `nθ` samples instead of the certified arc cover of prop:count: the cover needs a
    # sample spacing below σ11 (its slack is h = rΔθ/2), so it needs about πr/σ11 samples, each one
    # `svdbox` of the k×k ball pencil; at large k that is hours, and the estimate is reported.
    radii = Float64.(radii_factors) .* R
    circles = NamedTuple[]
    for r in radii
        t0 = time()
        if full
            chk = RP.check_circle(sc, 0.0, r; region = :exterior, nθ, maxdepth, method)
            push!(circles, (; r, ok = chk.ok, region_ok = chk.region_ok, floor = chk.floor,
                resolvent = chk.ok ? 1 / chk.floor : Inf, minσ11 = chk.minσ11,
                minσ22 = chk.minσ22, nevals = chk.nevals, cover = true, t = time() - t0))
        else
            fl = [RP.resolvent_floor(sc, r * cis(2π * t / nθ); method) for t in 0:(nθ - 1)]
            okp = all(f -> f.ok, fl)
            mn = minimum(f -> f.floor, fl)
            push!(circles, (; r, ok = okp, region_ok = r > sc.gbar, floor = mn,
                resolvent = okp ? 1 / mn : Inf, minσ11 = minimum(f -> f.σ11, fl),
                minσ22 = minimum(f -> f.σ22, fl), nevals = nθ, cover = false,
                t = time() - t0))
        end
    end

    # the two routes to σ11 at the samples of the circle at R (no cover, diagnostic only)
    zs = [R * cis(2π * t / nθ) for t in 0:(nθ - 1)]
    s11_blocks = minimum(RP.sigma11_lo(sc, z; method = :blocks) for z in zs)
    t0 = time()
    s11_svd = minimum(RP.sigma11_lo(sc, z; method = :svd) for z in zs)
    t_s11svd = time() - t0
    @info "σ11 at the samples" s11_blocks s11_svd γ1γ2 = sc.γ1 * sc.γ2 σ22 = R - sc.gbar

    # prop:complete
    incl = NamedTuple[]
    for e in 10.0 .^ (-1:-1:-8)
        δ, ok = RP.pseudospectrum_outside(sc, e, R)
        push!(incl, (; ε = e, δ, ratio = δ / e, ok))
    end

    # cor:agree and prop:count
    t0 = time()
    ext = full ? RP.certify_exterior(sc, R; nθ, maxdepth, method) :
          (; ok = false, R = Float64(R), count = -1, discs = NamedTuple[],
             exterior = RP.CircleCheck(0.0 + 0im, Float64(R), false, R > sc.gbar, :exterior,
                 0.0, 0, NaN, NaN))
    t_ext = time() - t0

    # prop:projector
    projs = NamedTuple[]
    if ext.exterior.ok
        pc = RP.certify_projector(sc, 0.0, R; region = :exterior, check = ext.exterior,
            nθ, maxdepth, method)
        push!(projs, (; region = :exterior, c = 0.0 + 0im, r = R, ok = pc.ok, bound = pc.bound,
            M = pc.M, normP = pc.normP, dim = pc.count, gap = pc.gap, sgap = pc.subspace_gap))
    end
    for d in ext.discs
        isfinite(d.radius) || continue
        pc = RP.certify_projector(sc, d.center, d.radius; region = :interior,
            check = d.check, nθ, maxdepth, method)
        push!(projs, (; region = :interior, c = d.center, r = d.radius, ok = pc.ok,
            bound = pc.bound, M = pc.M, normP = pc.normP, dim = pc.count, gap = pc.gap,
            sgap = pc.subspace_gap))
    end

    # the smallest R at which the exterior certificate still passes, frame fixed
    t0 = time()
    function passes(r)
        r > sc.gbar || return false
        c, ok, chk = RP.count_exterior(sc, r; nθ, maxdepth, method)
        return ok && chk.ok && chk.region_ok
    end
    lo = nextfloat(sc.gbar)
    # `passes` needs the count as well as the hypothesis: when the enclosure of G11 gives no
    # count (one merged cluster, say) no R passes and the search is skipped.
    ok_at_R = rmin && ext.exterior.ok && ext.exterior.region_ok && ext.count > 0
    Rmin = ok_at_R ? R : NaN
    if ok_at_R
        hi = R
        for _ in 1:8
            mid_ = (lo + hi) / 2
            hi - lo < 2e-3 * R && break
            if passes(mid_); hi = mid_; else; lo = mid_; end
        end
        Rmin = hi
    end
    t_rmin = time() - t0

    # the leading certified components of the enclosure of G11 (thm:enclosure)
    comps_all = RP._comps(sc)
    lead = sort(comps_all; by = C -> -abs(C.center))[1:min(neig + 4, length(comps_all))]
    # how many cover samples prop:count would need on |z| = R, and what one sample costs
    ncover = isfinite(s11_svd) && s11_svd > 0 ? ceil(Int, π * R / s11_svd) : -1
    tcover = ncover > 0 ? ncover * t_s11svd / nθ : NaN
    @info "cover estimate" ncover t_per_sample = t_s11svd / nθ tcover

    # ------------------------------------------------------------------------------------------------
    # dense truth
    dtruth = nothing
    if dense
        t0 = time()
        Ad = dense_A(L, l)
        Pq = I - Q * Q'
        G22d = Pq * Ad * Pq
        nG22 = opnorm(G22d)
        nA = opnorm(Ad)
        ev = eigvals(Ad)
        ev = ev[sortperm(abs.(ev); rev = true)]
        nout = count(z -> abs(z) > R, ev)
        # every certified component of the enclosure of G11 against the dense eigenvalues
        G11d = (Q' * Q) \ (Q' * Ad * Q)          # eigenvalues of Z*AZ (prop:almostortho)
        ev11 = eigvals(G11d)
        comps = RP._comps(sc)
        hits = [(; c = C.center, rad = C.radius, cnt = C.count,
                  inside = count(z -> abs(z - C.center) <= C.radius, ev11)) for C in comps]
        nout11 = count(z -> abs(z) > R, ev11)
        dtruth = (; nG22, nA, ev = ev[1:min(30, length(ev))],
            ev11 = ev11[sortperm(abs.(ev11); rev = true)][1:min(30, length(ev11))],
            nout, nout11, hits, t = time() - t0)
        @info "dense" nG22 gbar = sc.gbar nA abar = sc.abar nout count = ext.count
        if sc.gbar < nG22 * (1 - 1e-10)
            error("STOP: certified ḡ = $(sc.gbar) is below the dense ‖G22‖ = $nG22")
        end
        if sc.abar < nA * (1 - 1e-10)
            error("STOP: certified ā = $(sc.abar) is below the dense ‖A‖ = $nA")
        end
        # only a certified count may be compared: `count_exterior` returns 0 with ok = false
        # when the enclosure of G11 gives no count, and that 0 is not a claim
        if ext.ok && ext.count != nout
            error("STOP: certified exterior count $(ext.count) ≠ dense count $nout")
        end
        for h in hits
            h.cnt == h.inside || @warn "component count differs from dense" h
        end
    end

    # ------------------------------------------------------------------------------------------------
    res = (; ke, k, ξ, l, neig, ksarg, η, oi, binfo = (; binfo.nnz, binfo.t_assemble, binfo.maxrad),
        t_build, R, Rwin, R0, Rmargin = best.margin, Rscan, classes = cl[1:min(neig + 4, length(cl))], λ = λ[1:min(24, length(λ))],
        how, dimQ = size(Q, 2), α = sc.α, t_frame,
        gbar = sc.gbar, abar = sc.abar, γ1 = sc.γ1, γ2 = sc.γ2, snorm = sc.snorm,
        m = far.m, passed = far.passed, tpass = far.tpass, epsB = far.epsB, etaB = far.etaB,
        epsP = far.epsP, scale = far.scale, nmatvec = far.nmatvec,
        gbar_trace = far.gbar_trace, gbar_interp = far.gbar_interp, trace_fp = far.trace_fp,
        estimates = far.estimates,
        t_split, t_sweep = far.time_sweep, t_checks = far.time_checks,
        t_estimate = far.time_estimate, t_framedata = sc.time.frame, t_comp = sc.time.compression,
        mem_MB, circles, incl, s11_blocks, s11_svd, t_s11svd, method, nθ, maxdepth, full,
        lead = [(; C.center, C.radius, C.count) for C in lead], ncover, tcover, ext_ok = ext.ok, count = ext.count,
        circle_ok = ext.exterior.ok, circle_region_ok = ext.exterior.region_ok,
        circle_floor = ext.exterior.floor, circle_minσ11 = ext.exterior.minσ11,
        discs = [(; d.center, d.radius, d.count) for d in ext.discs], t_ext, projs,
        Rmin, t_rmin, dtruth,
        comps = [(; C.center, C.radius, C.count) for C in RP._comps(sc)])

    serialize(joinpath(OUT, "cert_$(tag).jls"), res)

    # ------------------------------------------------------------------------------------------------
    println("\n================ $tag ================")
    @printf("n = %d, ξ = %.0e, l = %d cells, neig = %d, R = %.6f\n", k, ξ, l, neig, R)
    @printf("mid(L): nnz = %d, assembled in %.1f s, max interval radius %.2e\n",
        binfo.nnz, binfo.t_assemble, binfo.maxrad)
    @printf("rounding: w = %d, ε_L = %.3e, ‖L‖ ≤ %.4f, γ_{2l+2} = %.3e, η = %.4e (ε_A = 0)\n",
        oi.w, oi.εL, oi.normL, oi.γN, η)
    @printf("frame: %s, dim Q = %d, α ≤ %.3e, %.1f s\n", how, size(Q, 2), sc.α, t_frame)
    @printf("far block: m = %d (%s, t = %d), ḡ ≤ %.6f, ā ≤ %.6f, ε_P = %.3e, ε_B = %.3e, η_B = %.3e\n",
        far.m, far.passed, far.tpass, sc.gbar, sc.abar, far.epsP, far.epsB, far.etaB)
    @printf("       scale c = %.3e, %d applications of B, sweep %.1f s, checks %.1f s, %.0f MB\n",
        far.scale, far.nmatvec, far.time_sweep, far.time_checks, mem_MB)
    @printf("couplings: γ1 ≤ %.6e, γ2 ≤ %.6e, ‖G11‖ ≤ %.4f; split total %.1f s\n",
        sc.γ1, sc.γ2, sc.snorm, t_split)
    @printf("σ11 over %d samples of |z| = R: eq:blockres %.4e, svdbox %.4e (%.1f s); γ1γ2 = %.4e, R − ḡ = %.4e\n",
        nθ, s11_blocks, s11_svd, t_s11svd, sc.γ1 * sc.γ2, R - sc.gbar)
    @printf("ḡ by order: "); for t in 1:far.m; @printf("t=%d:%.4f ", t, far.gbar_trace[t]); end
    @printf("| interp:%.4f\n", far.gbar_interp)

    println("\ncircles (region = exterior)")
    @printf("%10s %8s %10s %14s %14s %12s %12s %8s\n",
        "r", "ok", "region", "floor ≥", "resolvent ≤", "min σ11", "min σ22", "nevals")
    for c in circles
        @printf("%10.6f %8s %10s %14.6e %14.6e %12.4e %12.4e %8d\n",
            c.r, c.ok, c.region_ok, c.floor, c.resolvent, c.minσ11, c.minσ22, c.nevals)
    end

    println("\nprop:complete at R (Λ_δ(A) ∩ {|z|>R} ⊆ Λ_ε(G11) ∩ {|z|>R})")
    @printf("%10s %14s %12s %6s\n", "ε", "δ", "δ/ε", "ok")
    for i in incl
        @printf("%10.1e %14.6e %12.4e %6s\n", i.ε, i.δ, i.ratio, i.ok)
    end

    @printf("\nexterior at R = %.6f: hypothesis %s, region %s, count %d, isolation discs %s (%.1f s)\n",
        R, ext.exterior.ok, ext.exterior.region_ok, ext.count, ext.ok, t_ext)
    for d in ext.discs
        @printf("  disc c = %+.6f %+.6fim, r = %.4e, count %d\n",
            real(d.center), imag(d.center), d.radius, d.count)
    end

    println("\nprop:projector")
    @printf("%10s %22s %12s %12s %12s %6s %12s %12s\n",
        "region", "centre", "r", "M ≤", "‖P−P₀‖ ≤", "dim", "‖P‖ ≤", "subspace gap")
    for p in projs
        @printf("%10s %10.5f%+10.5fim %12.4e %12.4e %12.4e %6d %12.4e %12.4e\n",
            p.region, real(p.c), imag(p.c), p.r, p.M, p.bound, p.dim, p.normP, p.sgap)
    end

    println("\nleading components of the certified enclosure of G11 (thm:enclosure)")
    for C in lead
        @printf("  %+.8f %+.8fim  radius %.3e  count %d  |c| = %.8f\n",
            real(C.center), imag(C.center), C.radius, C.count, abs(C.center))
    end
    @printf("cover of prop:count on |z| = R would need about %d samples at %.2f s each: %.0f s\n",
        ncover, t_s11svd / nθ, tcover)

    @printf("\nsmallest R passing the exterior certificate with this frame: %.6f (ḡ = %.6f), %.1f s\n",
        Rmin, sc.gbar, t_rmin)

    if dtruth !== nothing
        println("\ndense truth")
        @printf("‖G22‖ = %.6f  vs certified ḡ = %.6f  (ratio %.4f)\n",
            dtruth.nG22, sc.gbar, sc.gbar / dtruth.nG22)
        @printf("‖A‖   = %.6f  vs certified ā = %.6f  (ratio %.4f)\n",
            dtruth.nA, sc.abar, sc.abar / dtruth.nA)
        @printf("eigenvalues of A with |λ| > R: %d; of G11: %d; certified count %d\n",
            dtruth.nout, dtruth.nout11, ext.count)
        for h in dtruth.hits
            @printf("  component c = %+.6f %+.6fim r = %.3e: certified %d, dense inside %d\n",
                real(h.c), imag(h.c), h.rad, h.cnt, h.inside)
        end
    end
    println("\nserialized to ", joinpath(OUT, "cert_$(tag).jls"))
    return res
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && !isempty(ARGS)
    run_one(parse(Int, ARGS[1]), parse(Float64, ARGS[2]), parse(Int, ARGS[3]),
        parse(Int, ARGS[4]);
        order = (length(ARGS) >= 5 && parse(Int, ARGS[5]) > 0) ? parse(Int, ARGS[5]) : :auto,
        dense = length(ARGS) >= 6 && ARGS[6] == "1")
end

