# Tests of RigorousPseudospectra against dense ground truth. Run with
#     julia --project=. test/runtests.jl            (add -t N to exercise the threaded sweep)
#
# "Truth" is computed densely in Float64 (LAPACK), so it carries its own rounding error, of
# relative size about 1e-14 at these dimensions; comparisons "certified bound vs truth" therefore
# allow a tolerance TOL·max(1, |truth|), and the tests print the smallest margin found and the
# number of trials.

using RigorousPseudospectra
using BallArithmetic
using LinearAlgebra
using Random
using SparseArrays
using Test
using Printf

include(joinpath(@__DIR__, "..", "bench", "matrices.jl"))

const TOL = 1e-10
le(a, b) = a <= b + TOL * max(1.0, abs(b))          # certified a ≤ true b, up to the truth's error

# Blocks of A in the frame of the isometry Z of the polar decomposition Q = ZS.
function truth_blocks(A, Q)
    S = sqrt(Hermitian(Q' * Q))
    Z = Q / S
    P = I - Z * Z'
    return (; Z, P, G11 = Z' * A * Z, G12 = Z' * A * P, G21 = P * A * Z, G22 = P * A * P)
end

# 2D stencil applied matrix-free: every output is the sum of at most five products taken in a fixed
# order, so eq:roundA holds with ε_A = √5 γ_5 and η_A = √n·5η.
function stencil_mul(x, n1, n2, c0, cW, cE, cS, cN)
    y = similar(x)
    @inbounds for i2 in 1:n2, i1 in 1:n1
        idx = i1 + (i2 - 1) * n1
        s = c0 * x[idx]
        i1 > 1 && (s += cW * x[idx - 1])
        i1 < n1 && (s += cE * x[idx + 1])
        i2 > 1 && (s += cS * x[idx - n1])
        i2 < n2 && (s += cN * x[idx + n1])
        y[idx] = s
    end
    return y
end

# ε_A = √w γ_w, rounded upward
epsA_width(w) = mul_up(sqrt_up(Float64(w)), gamma_bound(w, Float64))

# the resolvent floor at one point for a given frame, with no eigenvalue enclosure
function floor_at(A, Q, z)
    sc = certify_split(A, Q; compression = false)
    r = resolvent_floor(sc, z)
    return r.floor, (; r..., split = sc)
end

# the pipeline of §5 on an explicit frame of Definition def:Q: frame, split, cor:agree at R
function leading(Amul, Aadj, n, R; ks, epsA = 0.0, A = nothing, T = Float64)
    Q, _ = two_sided_frame(Amul, Aadj, n, R; ks, T)
    sc = A === nothing ? certify_split(Amul, Aadj, n, Q; epsA, target = R) :
         certify_split(A, Q; target = R)
    ext = certify_exterior(sc, R; nθ = 64, maxdepth = 64)
    return (; ok = ext.ok, R, count = ext.count, discs = ext.discs, exterior = ext.exterior,
        split = sc)
end

# decaying singular values in random frames, plus a low-rank part with a few large eigenvalues
function decaying(n; ρ = 0.7, seed = 1, cplx = false, nbig = 3)
    rng = MersenneTwister(seed)
    T = cplx ? ComplexF64 : Float64
    U = Matrix(qr(randn(rng, T, n, n)).Q)
    V = Matrix(qr(randn(rng, T, n, n)).Q)
    A = U * Diagonal(ρ .^ (0:(n - 1))) * V'
    X = randn(rng, T, n, nbig)
    λ = [1.0, -0.8, 0.6, 0.5, -0.45][1:nbig]
    return A * 0.3 + X * Diagonal(λ) * pinv(X)
end

@testset "RigorousPseudospectra" begin

    @testset "rem:counterexample: no bound where σ11σ22 ≤ γ1γ2" begin
        @test assembly_floor(1, 1, 1, 1) == (0.0, false)
        Q = reshape([1.0, 0], 2, 1)
        A = [1.0 1; 1 1]
        f, info = floor_at(A, Q, 0.0)
        @test f == 0 && !info.ok
        # the paper's data, σ22(0) = σ_min(G22) = 1 exactly
        @test assembly_floor(info.σ11, 1.0, info.γ1, info.γ2)[2] == false
        A2 = [1.0 1; 1 0]
        φ = (1 + sqrt(5)) / 2
        f2, info2 = floor_at(A2, Q, φ)
        @test f2 == 0 && !info2.ok
        @test info2.gbar < 1e-12                      # |z| > ḡ holds: the hypothesis is what fails
        @test svdvals(A2 - φ * I)[end] < 1e-15
    end

    @testset "§2 formulas: rounding direction" begin
        rng = MersenneTwister(11)
        bad = 0
        N = 20000
        for _ in 1:N
            s11, s22, g1, g2 = rand(rng, 4) .* [2, 2, 1, 1]
            f, ok = assembly_floor(s11, s22, g1, g2)
            exact = (big(s11) * s22 - big(g1) * g2) / (big(s11) + s22 + g1 + g2)
            if ok
                bad += (f > exact || exact <= 0)
            else
                bad += (exact > 0 && s11 * s22 - g1 * g2 > 1e-12)
            end
            f2, ok2 = blockdom_floor(s11, s22, g1, g2)
            τ = max(big(g1) / s11, big(g2) / s22)
            ok2 && (bad += (f2 > min(s11, s22) * (1 - τ)))
        end
        @printf("  assembly/blockdom against BigFloat: %d trials, %d violations\n", N, bad)
        @test bad == 0
    end

    @testset "random dense trials: every certified quantity against the truth" begin
        nt = 0
        viol = Dict{Symbol, Int}()
        minmargin = Dict{Symbol, Float64}()
        function rec(key, lo, tru)
            viol[key] = get(viol, key, 0) + !le(lo, tru)
            minmargin[key] = min(get(minmargin, key, Inf), (tru - lo) / max(1.0, abs(tru)))
        end
        counts_checked = 0
        discs_checked = 0
        viol[:enclosure] = 0
        viol[:clustercount] = 0
        viol[:extcount] = 0
        viol[:disccount] = 0
        configs = collect(Iterators.product((40, 90), (false, true), (3, 8), (0.0, 1e-7),
            (false, true)))
        for (trial, (n, cplx, ks, pert, mf)) in enumerate(configs)
            nt += 1
            A = decaying(n; seed = trial, cplx)
            T = cplx ? ComplexF64 : Float64
            Q, _ = two_sided_frame(x -> A * x, x -> A' * x, n, 0.3; ks, T)
            if pert > 0
                Q = Q + pert * randn(MersenneTwister(trial), T, size(Q)...)   # α of order 1e-7
            end
            sc = mf ? certify_split(x -> A * x, x -> A' * x, n, Q; epsA = epsA_width(n)) :
                 certify_split(A, Q)
            tb = truth_blocks(A, Q)
            rec(:α, opnorm(I - Q' * Q), sc.α)
            rec(:γ1, opnorm(tb.G12), sc.γ1)
            rec(:γ2, opnorm(tb.G21), sc.γ2)
            rec(:G11norm, opnorm(tb.G11), sc.snorm)
            rec(:G22norm, opnorm(tb.G22), sc.gbar)
            rec(:Anorm, opnorm(A), sc.abar)
            r = MersenneTwister(100 + trial)
            ev = eigvals(A)
            zs = [ev[1] + 0.05, ev[2] * 1.1, 0.6 * cis(1.0), 0.4 + 0.3im, -0.9,
                (0.2 + rand(r)) * cis(2π * rand(r)), (0.2 + rand(r)) * cis(2π * rand(r))]
            for z in zs
                rf = resolvent_floor(sc, z)
                t11 = svdvals(tb.G11 - z * I)[end]
                rec(:σ11, rf.σ11, t11)
                rec(:floor, rf.floor, svdvals(A - z * I)[end])
            end
            # the inclusions of verifyeigall on G ⊇ Q*AQ: each eigenvalue of Q*AQ in exactly one
            # component, and the counts of the components partition the spectrum
            θ = eigvals(Q' * A * Q)
            comps = RigorousPseudospectra._comps(sc)
            owner = [findall(C -> abs(λ - C.center) <= C.radius + 1e-12, comps) for λ in θ]
            viol[:enclosure] += count(o -> length(o) != 1, owner)
            for (ci, C) in enumerate(comps)
                viol[:clustercount] += (count(o -> o == [ci], owner) != C.count)
            end
            # the exterior (cor:agree) at R in a gap of the spectrum of A
            ma = sort(abs.(ev); rev = true)
            R = (ma[3] + ma[4]) / 2
            ext = certify_exterior(sc, R; nθ = 48)
            if ext.exterior.ok && ext.exterior.region_ok
                counts_checked += 1
                viol[:extcount] += (ext.count != count(>(R), abs.(ev)))
            end
            if ext.ok
                for d in ext.discs
                    discs_checked += 1
                    viol[:disccount] += (d.count != count(λ -> abs(λ - d.center) < d.radius, ev))
                end
            end
        end
        @printf("  %d random trials (n ∈ {40, 90}, real/complex, ks ∈ {3, 8}, α ≈ 0 or 1e-7, stored/matrix-free), 7 points z each\n",
            nt)
        for k in sort(collect(keys(minmargin)))
            @printf("    %-11s violations %d   smallest relative margin (truth − bound)/max(1,truth) %.2e\n",
                k, viol[k], minmargin[k])
        end
        @printf("    enclosure misses %d, cluster-count mismatches %d\n", viol[:enclosure],
            viol[:clustercount])
        @printf("    exterior counts certified in %d of %d trials, mismatches %d; isolation discs %d, mismatches %d\n",
            counts_checked, nt, viol[:extcount], discs_checked, viol[:disccount])
        @test all(v == 0 for v in values(viol))
        @test counts_checked > 0
    end

    @testset "uncertified helpers: predictions against the certificate" begin
        n = 160
        A = decaying(n; ρ = 0.6, seed = 42, nbig = 3)
        op = LinOp(A)
        ev = eigvals(A)
        ma = sort(abs.(ev); rev = true)
        gaps = helper_gaps(op; L = 60)
        @test any(g -> g.N == 3, gaps)
        R = (ma[3] + ma[4]) / 2
        d = helper_diagnose(op, R; ks = 12, verbose = false)
        @test d.nout == count(>(R), abs.(ev))
        @test d.dimQ > 0 && 1 <= d.m <= length(d.gbar_est)
        # the certificate on the same frame, at the order the helper names
        Q, _ = two_sided_frame(x -> A * x, x -> A' * x, n, R; ks = 12, Leig = 200,
            Lsv = min(n, 2 * 12 + 60))
        sc = certify_split(op, Q; order = d.m, target = R)
        ratio = sc.gbar / d.gbar_est[d.m]
        @printf("  helper: predicted ḡ %.3e at order %d, certified %.3e (ratio %.3f); predicted margin %.3e\n",
            d.gbar_est[d.m], d.m, sc.gbar, ratio, d.margin)
        # the prediction from the singular values of the far block, against the certified bound at
        # the same order, and the estimate of ‖G22‖ against the dense one
        @test 0.5 < ratio < 2
        g22 = opnorm(truth_blocks(A, Q).G22)
        @test abs(d.σG22[1] / g22 - 1) < 0.05
        cnt, ok, chk = count_exterior(sc, R; nθ = 64, maxdepth = 64)
        chk.ok && @test d.margin > 0
        @test le(opnorm(truth_blocks(A, Q).G21), 1.0001 * d.γ2 + 1e-12)
    end

    @testset "far block: the sweep certificate against ‖G22‖" begin
        rows = Any[]
        # (a) stored sparse
        rng = MersenneTwister(5)
        n = 1500
        S = sprandn(rng, n, n, 3 / n) + 0.3I
        Q = Matrix(qr(randn(rng, n, 12)).Q)
        sc = certify_split(S, Q; compression = false)
        tru = opnorm(truth_blocks(Matrix(S), Q).G22)
        push!(rows, ("sparse stored", n, 12, sc.far.m, tru, sc.gbar, sc.abar / opnorm(Matrix(S)),
            sc.far.passed))
        @test le(tru, sc.gbar) && le(opnorm(Matrix(S)), sc.abar)
        # (b) stencil, matrix-free
        n1, n2 = 36, 30
        n = n1 * n2
        c = (0.5, 0.2, 0.05, 0.15, 0.1)
        Amul(x) = stencil_mul(x, n1, n2, c...)
        Aadj(x) = stencil_mul(x, n1, n2, c[1], c[3], c[2], c[5], c[4])
        Ad = hcat([Amul(Matrix(1.0I, n, n)[:, j]) for j in 1:n]...)
        Qs, _ = two_sided_frame(Amul, Aadj, n, 0.6; ks = 10)
        trus = opnorm(truth_blocks(Ad, Qs).G22)
        for ord in (2, 6, :auto)
            scs = certify_split(Amul, Aadj, n, Qs; epsA = epsA_width(5),
                etaA = sqrt(n) * 5 * 2.0^-1074 * 2, order = ord, compression = false)
            push!(rows, ("stencil matrix-free, order $ord", n, size(Qs, 2), scs.far.m, trus,
                scs.gbar, scs.abar / opnorm(Ad), scs.far.passed))
            @test le(trus, scs.gbar) && le(opnorm(Ad), scs.abar)
        end
        # (c) dense complex, deep deflation
        A = decaying(300; ρ = 0.8, seed = 9, cplx = true)
        Qd, _ = two_sided_frame(x -> A * x, x -> A' * x, 300, 0.1; ks = 25, T = ComplexF64)
        scd = certify_split(A, Qd; compression = false)
        trud = opnorm(truth_blocks(A, Qd).G22)
        push!(rows, ("dense complex", 300, size(Qd, 2), scd.far.m, trud, scd.gbar,
            scd.abar / opnorm(A), scd.far.passed))
        @test le(trud, scd.gbar) && le(opnorm(A), scd.abar)
        # (d) no frame: the certificate bounds ‖A‖ itself
        A0 = randn(MersenneTwister(2), 200, 200) / 20
        sc0 = certify_split(A0, zeros(200, 0); compression = false)
        push!(rows, ("no frame (k = 0)", 200, 0, sc0.far.m, opnorm(A0), sc0.gbar,
            sc0.abar / opnorm(A0), sc0.far.passed))
        @test le(opnorm(A0), sc0.gbar)
        println("  far block: certified ḡ against the dense ‖G22‖ (threads: $(Threads.nthreads()))")
        @printf("    %-34s %6s %4s %4s %13s %13s %8s %8s\n", "operator", "n", "k", "m", "‖G22‖",
            "ḡ", "ā/‖A‖", "passed")
        for r in rows
            @printf("    %-34s %6d %4d %4d %13.6e %13.6e %8.4f %8s\n", r...)
        end
    end

    @testset "Hermitian: cor:selfadj, cor:weyl, rem:lanczos" begin
        rng = MersenneTwister(21)
        n = 300
        λbig = [10.0, 7.0, 5.0, -8.0, -6.0]
        λ = vcat(λbig, 2rand(rng, n - 5) .- 1)
        U = Matrix(qr(randn(rng, n, n)).Q)
        A = U * Diagonal(λ) * U'
        A = (A + A') / 2
        out = certify_selfadjoint(A; L = 60, tol = 1e-10)
        sc = out.split
        @test sc.hermitian
        tb = truth_blocks(A, sc.Q)
        @test le(opnorm(tb.G22), sc.gbar) && le(opnorm(tb.G21), out.γ̄)
        ev = eigvals(Symmetric(A))
        # (i): every eigenvalue of A with |λ| > ḡ lies in the union of the enclosure intervals
        @test all(x -> abs(x) <= sc.gbar || any(e -> e.lo <= x <= e.hi, out.enclosure), ev)
        ρs = [max(e.ρplus, e.ρminus) for e in out.enclosure if e.J > 0]
        @printf("  Lanczos frame k = %d, ḡ = %.3e, γ̄ = %.3e, largest quadratic radius %.3e (γ̄²/(10 − ḡ) = %.3e)\n",
            sc.k, sc.gbar, out.γ̄, maximum(ρs; init = 0.0), out.γ̄^2 / (10 - sc.gbar))
        # (ii): counts on intervals around each extreme eigenvalue
        for x in λbig
            cnt, ok = selfadj_count(out.θ, sc.gbar, out.γ̄, x - 0.5, x + 0.5)
            @test ok
            @test cnt == count(e -> x - 0.5 <= e <= x + 0.5, ev)
        end
        w = out.weyl
        @test w.ok
        evd = sort(ev; rev = true)
        @test all(t -> t.lo <= evd[t.i] <= t.hi, w.top)
        @test all(t -> t.lo <= evd[t.i] <= t.hi, w.bottom)
        @test evd[w.jplus + 1] <= w.top_rest
        @test evd[n - w.jminus] >= w.bottom_rest
        @test w.jplus == 3 && w.jminus == 2
    end

    @testset "matrix-free pipeline: cor:gap and cor:agree on explicit frames" begin
        n = 160
        A = decaying(n; ρ = 0.6, seed = 42, nbig = 3)
        Amul(x) = A * x
        Aadj(x) = A' * x
        ev = eigvals(A)
        ma = sort(abs.(ev); rev = true)
        Rg = (ma[1] + ma[2]) / 2
        Qg, _ = two_sided_frame(Amul, Aadj, n, Rg; ks = 12)
        scg = certify_split(Amul, Aadj, n, Qg; epsA = epsA_width(n), target = Rg)
        g = certify_gap(scg, Rg; nθ = 64, maxdepth = 64)
        @printf("  certify_gap: certified = %s, R = %.3f, |λ1| = %.3f, |λ2| = %.3f, dim Q = %d, ḡ = %.3e\n",
            g.certified, g.R, ma[1], ma[2], scg.k, scg.gbar)
        @test g.certified
        @test count(>(g.R), abs.(ev)) == 1
        L = leading(Amul, Aadj, n, (ma[3] + ma[4]) / 2; ks = 12, epsA = epsA_width(n))
        @printf("  cor:agree on an explicit frame: ok = %s, R = %.3f, count = %d, disc radii %s\n", L.ok, L.R,
            L.count, string(round.([d.radius for d in L.discs]; sigdigits = 3)))
        @test L.ok
        @test L.count == count(>(L.R), abs.(ev))
        for d in L.discs
            @test d.count == count(λ -> abs(λ - d.center) < d.radius, ev)
        end
        sc = L.split
        # prop:count on a union of overlapping discs strung between two eigenvalues
        λs = ev[sortperm(abs.(ev); rev = true)][1:3]
        for (a, b) in ((λs[1], λs[3]), (λs[1], λs[2]))
            cs = [a + t * (b - a) for t in range(-0.1, 1.1; length = 7)]
            rr = fill(0.12 * abs(b - a), 7)
            U = count_union(sc, cs, rr; nθ = 48)
            inU(λ) = any(i -> abs(λ - cs[i]) < rr[i], 1:7)
            if U.ok
                @test length(U.members) == 1
                @test U.counts[1] == count(inU, ev)
            end
            @printf("  count_union between %.3f%+.3fi and %.3f%+.3fi: ok = %s (region %s, contour %s), count %s, dense count %d\n",
                real(a), imag(a), real(b), imag(b), U.ok, U.region_ok, U.contour_ok,
                U.ok ? string(U.counts) : "-", count(inU, ev))
        end
        for z in (1.5, 1.2im, -1.3 + 0.4im)
            @test le(resolvent_floor(sc, z).floor, svdvals(A - z * I)[end])
        end
        # prop:complete: σ_min(A − z) ≤ δ at |z| > R implies σ_min(G11 − z) ≤ ε
        δ, ok = pseudospectrum_outside(sc, 0.05, L.R)
        @test ok
        tb = truth_blocks(A, sc.Q)
        rng = MersenneTwister(1)
        nhit = 0
        npts = 0
        λout = ev[abs.(ev) .> L.R]
        for _ in 1:400
            # half the points uniform in an annulus, half within 0.08 of an exterior eigenvalue
            z = isodd(npts) ? (L.R * 1.001 + 0.5rand(rng)) * cis(2π * rand(rng)) :
                rand(rng, λout) + 0.08rand(rng) * cis(2π * rand(rng))
            abs(z) > L.R || continue
            npts += 1
            if svdvals(A - z * I)[end] <= δ
                nhit += 1
                @test svdvals(tb.G11 - z * I)[end] <= 0.05 + TOL
            end
        end
        @printf("  prop:complete, ε = 0.05, R = %.3f: δ = %.3e; %d points with |z| > R, %d of them with σ_min(A − zI) ≤ δ\n",
            L.R, δ, npts, nhit)
        @test nhit > 0
    end

    @testset "prop:projector: the spectral projector" begin
        # dense Riesz projector of M for the eigenvalues selected by `sel`
        function riesz(M, sel)
            F = eigen(ComplexF64.(Matrix(M)))
            return F.vectors * Diagonal(ComplexF64.(sel.(F.values))) * inv(F.vectors)
        end
        # an orthonormal basis of the range of a projector of rank k (its leading left singular
        # vectors), and the gap between two subspaces given by bases
        rangebasis(P, k) = k <= 0 ? zeros(ComplexF64, size(P, 1), 0) :
                           svd(ComplexF64.(Matrix(P))).U[:, 1:k]
        function rangegap(B1, B2)
            (size(B1, 2) == 0 || size(B2, 2) == 0) && return 0.0
            Π1 = (U = Matrix(qr(ComplexF64.(B1)).Q)[:, 1:size(B1, 2)]; U * U')
            Π2 = (U = Matrix(qr(ComplexF64.(B2)).Q)[:, 1:size(B2, 2)]; U * U')
            return max(opnorm((I - Π2) * Π1), opnorm((I - Π1) * Π2))
        end
        ratios = Float64[]
        ncontour = 0
        println("  prop:projector: bound r·M·M₀·max(γ1, γ2) against the dense ‖P − P₀‖")
        println("  radius sweep from the radius certify_exterior returns up to the largest the ",
            "hypothesis of prop:count allows (full table for the first operator, best row for the rest)")
        @printf("    %-22s %9s %9s %9s %11s %11s %10s\n", "contour", "r", "M", "M₀", "bound",
            "‖P − P₀‖", "ratio")
        for (trial, (n, cplx)) in enumerate(Iterators.product((60, 120), (false, true)))
            A = decaying(n; ρ = 0.6, seed = 40 + trial, cplx)
            ev = eigvals(A)
            ma = sort(abs.(ev); rev = true)
            R = (ma[3] + ma[4]) / 2
            L = leading(x -> A * x, x -> A' * x, n, R; ks = 12, epsA = epsA_width(n),
                T = cplx ? ComplexF64 : Float64)
            L.ok || continue
            sc = L.split
            Z = sc.Q / sqrt(Hermitian(sc.Q' * sc.Q))
            G11 = Z' * A * Z
            tag = "n = $n $(cplx ? "cplx" : "real")"
            # the exterior circle |z| = R
            pc = certify_projector(sc, 0.0, R; region = :exterior)
            @test pc.ok
            sel = λ -> abs(λ) > R
            P = riesz(A, sel)
            P0 = Z * riesz(G11, sel) * Z'
            tru = opnorm(P - P0)
            @test le(tru, pc.bound)
            @test le(opnorm(P), pc.normP)
            @test le(rangegap(rangebasis(P, pc.count), rangebasis(P0, pc.count)), pc.gap)
            push!(ratios, pc.bound / tru)
            ncontour += 1
            @printf("    %-22s %9.3g %9.3g %9.3g %11.3e %11.3e %10.1f\n", "$tag, exterior R",
                R, pc.M, pc.M0, pc.bound, tru, pc.bound / tru)
            # each isolation disc, swept in the radius
            for d in L.discs
                c = d.center
                far = [abs(λ - c) for λ in ev if abs(λ - c) > 2d.radius]
                rmax = min(abs(c) - R, 0.45 * minimum(far))
                rmax > d.radius || continue
                best = (; ratio = Inf, r = NaN, M = NaN, M0 = NaN, bound = NaN, tru = NaN)
                nrad = 0
                # the full sweep on the first operator, four radii on the others (each contour
                # costs an arc cover); nθ = 64: with fewer samples the tight contours spend the
                # budget on bisection instead
                radii = exp.(range(log(d.radius), log(rmax); length = trial == 1 ? 8 : 4))
                for r in radii
                    pcd = certify_projector(sc, c, r)
                    pcd.ok || continue
                    seld = λ -> abs(λ - c) < r
                    pcd.count < 0 || count(seld, ev) == pcd.count || continue
                    Pd = riesz(A, seld)
                    P0d = Z * riesz(G11, seld) * Z'
                    t = opnorm(Pd - P0d)
                    @test le(t, pcd.bound)
                    if pcd.count > 0
                        @test le(rangegap(rangebasis(Pd, pcd.count), rangebasis(P0d, pcd.count)),
                            pcd.gap)
                    end
                    push!(ratios, pcd.bound / t)
                    ncontour += 1
                    nrad += 1
                    rr = pcd.bound / t
                    rr < best.ratio && (best = (; ratio = rr, r, M = pcd.M, M0 = pcd.M0,
                        bound = pcd.bound, tru = t))
                    trial == 1 && @printf("    %-22s %9.3g %9.3g %9.3g %11.3e %11.3e %10.1f\n",
                        "  disc λ ≈ $(round(real(c); sigdigits = 3))", r, pcd.M, pcd.M0,
                        pcd.bound, t, rr)
                end
                nrad == 0 && continue
                @printf("    %-22s %9.3g %9.3g %9.3g %11.3e %11.3e %10.1f   (best of %d radii, r_max = %.3g)\n",
                    "  best, λ ≈ $(round(real(c); sigdigits = 3))", best.r, best.M, best.M0,
                    best.bound, best.tru, best.ratio, nrad, rmax)
            end
        end
        @printf("    %d contours: ratio bound/truth min %.3g, median %.3g, max %.3g\n",
            length(ratios), minimum(ratios), sort(ratios)[cld(length(ratios), 2)],
            maximum(ratios))
        @test ncontour > 0
        @test minimum(ratios) >= 1
    end
end
