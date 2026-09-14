# Tests of RigorousPseudospectra against dense ground truth. Run with
#     julia --project=. test/runtests.jl            (add -t N to exercise the threaded sweep)
#
# "Truth" is computed densely in Float64 (LAPACK), so it carries its own rounding error, of
# relative size about 1e-14 at these dimensions; comparisons "certified bound vs truth" therefore
# allow a tolerance TOL·max(1, |truth|), and the tests print the smallest margin found and the
# number of trials.

using RigorousPseudospectra
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
epsA_width(w) = RigorousPseudospectra.mulu(RigorousPseudospectra.sqrtu(Float64(w)),
    RigorousPseudospectra.gammau(w))

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
        f, info = krylov_resolvent_floor(A, Q, 0.0)
        @test f == 0 && !info.ok
        # the paper's data, σ22(0) = σ_min(G22) = 1 exactly
        @test assembly_floor(info.σ11, 1.0, info.γ1, info.γ2)[2] == false
        A2 = [1.0 1; 1 0]
        φ = (1 + sqrt(5)) / 2
        f2, info2 = krylov_resolvent_floor(A2, Q, φ)
        @test f2 == 0 && !info2.ok
        @test info2.gbar < 1e-12                      # |z| > ḡ holds: the hypothesis is what fails
        @test svdvals(A2 - φ * I)[end] < 1e-15
    end

    @testset "§2 formulas: rounding direction and eq:farpower" begin
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
        viol = 0
        for seed in 1:20
            r = MersenneTwister(seed)
            B = triu(randn(r, 12, 12), 1) * 0.5 + 0.1 * randn(r, 12, 12)
            gs = [opnorm(B^j) * (1 + 1e-12) for j in 1:6]
            for _ in 1:10
                z = (0.5 + 3rand(r)) * cis(2π * rand(r))
                viol += !le(farpower_lo(z, gs), svdvals(B - z * I)[end])
            end
        end
        @printf("  eq:farpower against dense σ_min(G22 − z) with exact ‖G22^j‖: 200 points, %d violations\n",
            viol)
        @test viol == 0
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
                rec(:σ11_blocks, sigma11_lo(sc, z; method = :blocks), t11)
            end
            # the compression enclosure: each eigenvalue of G11 in exactly one component
            θ = eigvals(tb.G11)
            cc = sc.comp.cert
            ρ = sc.comp.family === :local ? cc.ρlocal : cc.ρglobal
            comps = RigorousPseudospectra._comps(sc)
            owner = [findall(C -> any(b -> abs(λ - cc.centres[b]) <= ρ[b] + 1e-12, C.blocks),
                comps) for λ in θ]
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

    @testset "prop:blockgersg localised radii on random block matrices" begin
        # M = Λ + N with Λ block diagonal and N off-block, exact (β = 0): every eigenvalue must lie in
        # some {σ_min(P_i − λ) ≤ r_i}. The paper's r_i = Σ_{j≠i}‖N_ij‖; baseline: the false form of
        # the old code, r_i = ‖N_{C_i,:}‖.
        function randblock(rng; qmax = 5)
            q = rand(rng, 2:qmax)
            sizes = rand(rng, 1:3, q)
            k = sum(sizes)
            blocks = Vector{Int}[]
            p = 0
            for s in sizes
                push!(blocks, collect((p + 1):(p + s)))
                p += s
            end
            M = randn(rng, ComplexF64, k, k)
            scale = 10.0^(rand(rng) * 3 - 2)
            for (i, bi) in enumerate(blocks), (j, bj) in enumerate(blocks)
                i != j && (M[bi, bj] .*= scale)
            end
            return M, blocks, q, k
        end
        rng = MersenneTwister(1)
        N = 20000
        miss_paper = 0
        miss_false = 0
        for _ in 1:N
            M, blocks, q, k = randblock(rng)
            rp = [sum(opnorm(M[blocks[i], blocks[j]]) for j in 1:q if j != i) for i in 1:q]
            rf = [opnorm(M[blocks[i], setdiff(1:k, blocks[i])]) for i in 1:q]
            for λ in eigvals(M)
                s = [svdvals(M[b, b] - λ * I)[end] for b in blocks]
                miss_paper += !any(s .<= rp .* (1 + 1e-9) .+ 1e-12)
                miss_false += !any(s .<= rf .* (1 + 1e-9) .+ 1e-12)
            end
        end
        @printf("  exact norms, %d random block matrices: eigenvalues outside every region: paper's r_i %d, false r_i (old code) %d\n",
            N, miss_paper, miss_false)
        @test miss_paper == 0
        # the certified radii of certify_compression with the trivial candidate W = Y = I
        miss_cert = 0
        Nc = 1000
        for _ in 1:Nc
            M, blocks, q, k = randblock(rng; qmax = 4)
            Λ = zeros(ComplexF64, k, k)
            for b in blocks
                Λ[b, b] .= M[b, b]
            end
            Id = Matrix{ComplexF64}(I, k, k)
            cc = certify_compression(M, CompressionCandidate(Id, Id, Λ, blocks, :given, (;)))
            for λ in eigvals(M)
                s = [svdvals(M[b, b] - λ * I)[end] for b in blocks]
                inreg = any(s .<= cc.rloc)
                indisc = any(abs(λ - cc.centres[i]) <= cc.ρlocal[i] for i in 1:q)
                miss_cert += !(inreg && indisc)
            end
        end
        @printf("  certified localised radii of certify_compression, %d matrices: misses %d\n", Nc,
            miss_cert)
        @test miss_cert == 0
    end

    @testset "compression: Sylvester and Newton candidates contain the spectrum" begin
        mats = [(e.name, e.A) for e in gallery()]
        push!(mats, ("jordan8+1e-10", jordan_block(8) + 1e-10 * randn(MersenneTwister(3), 8, 8)))
        for seed in 1:3
            rng = MersenneTwister(seed)
            push!(mats, ("nonnormal$seed", triu(randn(rng, ComplexF64, 20, 20)) .*
                                             [2.0^(-abs(i - j) / 2) for i in 1:20, j in 1:20]))
        end
        nok = 0
        for (name, A) in mats, method in (:sylvester, :newton)
            R = miyajima_rump(A; method)
            R.ok || continue
            nok += 1
            ev = eigvals(Matrix(A))
            @test all(λ -> any(C -> abs(λ - C.hull_center) <= C.hull_radius + 1e-12, R.clusters),
                ev)
            @test sum(C -> C.count, R.clusters) == size(A, 1)
        end
        @printf("  %d of %d (matrix, candidate) pairs certified ‖R2‖ < 1\n", nok, 2length(mats))
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

    @testset "matrix-free pipeline: certify_gap and certify_leading_eigs" begin
        n = 160
        A = decaying(n; ρ = 0.6, seed = 42, nbig = 3)
        Amul(x) = A * x
        Aadj(x) = A' * x
        ev = eigvals(A)
        ma = sort(abs.(ev); rev = true)
        g = certify_gap(Amul, Aadj, n; epsA = epsA_width(n), R = (ma[1] + ma[2]) / 2)
        @printf("  certify_gap: certified = %s, R = %.3f, |λ1| = %.3f, |λ2| = %.3f, dim Q = %d, ḡ = %.3e\n",
            g.certified, g.R, ma[1], ma[2], g.split.k, g.split.gbar)
        @test g.certified
        @test count(>(g.R), abs.(ev)) == 1
        L = certify_leading_eigs(Amul, Aadj, n; epsA = epsA_width(n), R = (ma[3] + ma[4]) / 2)
        @printf("  certify_leading_eigs: ok = %s, R = %.3f, count = %d, disc radii %s\n", L.ok, L.R,
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
            L = certify_leading_eigs(x -> A * x, x -> A' * x, n; epsA = epsA_width(n), R)
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
            @test le(rangegap(rangebasis(P0, pc.count), pc.basis), pc.subspace_gap)
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
                for (ir, r) in enumerate(radii)
                    # the compression-level certificate only at the ends of the sweep (it costs a
                    # second arc cover and does not depend on the dense comparison)
                    sub = ir == 1 || ir == length(radii)
                    pcd = certify_projector(sc, c, r; subspace = sub)
                    pcd.ok || continue
                    seld = λ -> abs(λ - c) < r
                    sub && (count(seld, ev) == pcd.count || continue)
                    Pd = riesz(A, seld)
                    P0d = Z * riesz(G11, seld) * Z'
                    t = opnorm(Pd - P0d)
                    @test le(t, pcd.bound)
                    if sub && isfinite(pcd.subspace_gap)
                        @test le(rangegap(rangebasis(P0d, pcd.count), pcd.basis),
                            pcd.subspace_gap)
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

        # the same inside sec:vbd: the invariant subspaces of the compression
        println("  prop:projector inside sec:vbd (bound against ‖P_{M_c,S} − P_{Λ,S}‖)")
        @printf("    %-16s %5s %9s %9s %11s %11s %10s\n", "matrix", "dim", "M", "M₀", "bound",
            "truth", "ratio")
        rat2 = Float64[]
        mats = [("normal16", normal_random(16)), ("grcar32", grcar(32)),
            ("two_clusters", two_clusters()), ("chain", graded_chain(12)),
            ("randn30", randn(MersenneTwister(8), ComplexF64, 30, 30))]
        for (name, A) in mats
            Ac = ComplexF64.(Matrix(A))
            R = miyajima_rump(Ac)
            R.ok || continue
            cc = R.cert
            W = cc.candidate.W
            Mc = W \ (Ac * W)
            i = argmax(abs.(cc.centres))
            c = cc.centres[i]
            others = [j for j in eachindex(cc.centres) if j != i]
            isempty(others) && continue
            rr = 0.45 * minimum(abs.(cc.centres[others] .- c))
            bp = certify_block_projector(cc, c, rr; family = R.family)
            bp.ok || continue
            PΛ = Diagonal([j in bp.cols ? 1.0 + 0im : 0.0 + 0im for j in 1:(cc.k)])
            PM = riesz(Mc, λ -> abs(λ - c) < rr)
            t = opnorm(PM - PΛ)
            @test le(t, bp.bound)
            push!(rat2, bp.bound / max(t, eps()))
            @printf("    %-16s %5d %9.3g %9.3g %11.3e %11.3e %10.1f\n", name, bp.count, bp.M,
                bp.M0, bp.bound, t, bp.bound / max(t, eps()))
        end
        @test !isempty(rat2)
        @test minimum(rat2) >= 1
    end

    @testset "split by modulus: sigma, components, counts and the pipeline" begin
        # (a) the compression alone, against dense truth
        function planted(n, nout; seed = 1, scale = 5.0)
            rng = MersenneTwister(seed)
            A = randn(rng, ComplexF64, n, n) / sqrt(n)
            F = eigen(A)
            d = F.values
            p = sortperm(abs.(d); rev = true)
            d[p[1:nout]] .*= scale
            return F.vectors * Diagonal(d) / F.vectors
        end
        mats = [("planted40", planted(40, 3; seed = 1)), ("planted90", planted(90, 5; seed = 2)),
            ("planted60", planted(60, 2; seed = 3)), ("normal50", normal_random(50; seed = 4)),
            ("chain40", ComplexF64.(graded_chain(40))),
            ("two_clusters", ComplexF64.(two_clusters()))]
        nz = 0
        ncnt = 0
        ndecl = 0
        worst = Inf
        for (name, G) in mats
            a = sort(abs.(eigvals(G)); rev = true)
            js = [j for j in (2, 4) if j < length(a) && a[j] > a[j + 1] * (1 + 1e-8)]
            for j in js
                R = sqrt(a[j] * a[j + 1])
                ms = split_compression(G, R)
                ms.cert_ok || continue
                ms.ok || (ndecl += 1)
                @test ms.reason in (:ok, :r2, :radius, :cost)
                ev = eigvals(G)
                # (i) the resolvent bound is below the truth
                for ρ in (1.0, 1.2, 2.0), t in 0:11
                    z = ρ * R * cis(2π * t / 12)
                    lo = sigma_split_lo(ms, z)
                    tr = minimum(svdvals(G - z * I))
                    @test le(lo, tr)
                    nz += 1
                    lo > 0 && (worst = min(worst, lo / tr))
                end
                # (ii) the certified bound on the powers of the trailing block is above the
                # spectral radius of that block
                @test le(ms.rho_fp, ms.gbarJ)
                # (iii) every eigenvalue lies in some certified component, the counts partition
                # the spectrum, and each component's disc holds at least its own count (the hull
                # disc of a merged component may also cover eigenvalues of a disjoint one)
                comps = split_components(ms)
                @test all(λ -> any(C -> abs(λ - C.center) <= C.radius * (1 + 1e-10), comps), ev)
                @test sum(C -> C.count, comps; init = 0) == size(G, 1)
                for C in comps
                    @test C.count <= count(λ -> abs(λ - C.center) <= C.radius, ev)
                end
                # (iv) the count outside R, when certified, is the dense count
                cnt, okc = eigencount_outside(comps, R)
                if okc
                    ncnt += 1
                    @test cnt == count(λ -> abs(λ) > R, ev)
                end
            end
        end
        @printf("  split: %d points of σ below the truth (smallest positive ratio %.2e), %d certified counts, %d declines (k₊ ceiling or ḡ_J ≥ R)\n",
            nz, worst, ncnt, ndecl)
        # the ceiling is honoured and the decline is reported, never silently exceeded
        for (_, G) in mats
            a = sort(abs.(eigvals(G)); rev = true)
            j = 2
            j < length(a) || continue
            R = sqrt(a[j] * a[j + 1])
            for kmax in (3, 20)
                ms = split_compression(G, R; kmax)
                @test ms.kplus <= kmax
                @test ms.ok == (ms.reason === :ok)
                @test ms.reason in (:ok, :r2, :radius, :cost)
                ms.ok || @test occursin("k₊", split_reason(ms)) || occursin("ḡ_J",
                    split_reason(ms))
                # the stopping rule reports what the condition wanted and what it would cost
                ms.reason === :cost && (@test ms.kwanted > kmax; @test ms.tz > 0)
            end
            # a budget in seconds per z reaches the same decline
            let msb = split_compression(G, R; budget = 1e-12)
                @test msb.reason in (:cost, :r2, :radius)
                msb.reason === :cost && (@test msb.tz > 1e-12)
            end
        end

        # (b) the whole pipeline with the split as the compression layer
        rng = MersenneTwister(11)
        n = 400
        d = [1.0, 0.8, 0.79, 0.6]
        A = triu(randn(rng, n, n) / sqrt(n), 1) * 0.15
        A += Diagonal([i <= 4 ? d[i] : 0.3 * randn(rng) for i in 1:n])
        Q = Matrix(qr(hcat([A^j * randn(rng, n) for j in 0:79]...)).Q)[:, 1:80]
        R = 0.7
        for mode in (:split, :full)
            sc = certify_split(A, Q; target = R, compression = mode)
            @test (mode === :split) == (sc.msplit !== nothing)
            ext = certify_exterior(sc, R)
            if ext.exterior.ok && ext.count > 0
                @test ext.count == count(λ -> abs(λ) > R, eigvals(A))
            end
            @printf("  pipeline %-6s: dim Q = %d, count = %d (ok %s), σ11(R) = %.3e, %.2f s\n",
                mode, sc.k, ext.count, ext.ok, sigma11_lo(sc, R + 0im), sc.time.compression)
        end
    end

    @testset "deep split: the reduced frame, its couplings and its far block" begin
        # A matrix whose frame decouples: a decaying singular tail plus a few outliers.
        function planted(n, nout; seed = 1, scale = 6.0)
            rng = MersenneTwister(seed)
            A = randn(rng, ComplexF64, n, n) / sqrt(n)
            F = eigen(A)
            d = F.values
            p = sortperm(abs.(d); rev = true)
            d[p[1:nout]] .*= scale
            return F.vectors * Diagonal(d) / F.vectors
        end
        # A frame that decouples: A is built block triangular in a random orthonormal basis, with a
        # small far block and small couplings, so that the gate γ1γ2 < R(R − ḡ) holds and the deep
        # cut has something to do. `Q` is the first k columns of the same basis.
        function decoupled(n, k, nout; seed = 1, gfar = 0.15, cpl = 0.02)
            rng = MersenneTwister(seed)
            V = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
            A11 = randn(rng, ComplexF64, k, k) / sqrt(k)
            F = eigen(A11)
            d = F.values
            p = sortperm(abs.(d); rev = true)
            d[p[1:nout]] .*= 6.0
            A11 = F.vectors * Diagonal(d) / F.vectors
            A22 = randn(rng, ComplexF64, n - k, n - k)
            A22 .*= gfar / opnorm(A22)
            B = randn(rng, ComplexF64, n - k, k)
            B .*= cpl / opnorm(B)
            C = randn(rng, ComplexF64, k, n - k)
            C .*= cpl / opnorm(C)
            A = V * [A11 C; B A22] * V'
            return A, V[:, 1:k], sort(abs.(eigvals(A11)); rev = true)
        end
        ncase = 0
        nbadc = 0
        nbadf = 0
        ndecl = 0
        nclose = 0
        worst = Inf
        cases = Any[]
        for (nm, n, k, nout, seed) in [("decoupled200", 200, 60, 3, 11),
            ("decoupled300", 300, 80, 4, 12), ("decoupled150", 150, 50, 2, 13)]
            A, Q, aa = decoupled(n, k, nout; seed)
            push!(cases, (nm, A, Q, sqrt(aa[nout] * aa[nout + 1])))
        end
        # one case that should decline: a Krylov frame on a matrix with no decaying tail
        let A = planted(120, 3; seed = 21), n = 120
            rng = MersenneTwister(5)
            Q = Matrix(qr(hcat([A^j * randn(rng, ComplexF64, n) for j in 0:39]...)).Q)[:, 1:40]
            aa = sort(abs.(eigvals(A)); rev = true)
            push!(cases, ("krylov120", A, Q, sqrt(aa[3] * aa[4])))
        end
        for (nm, A, Q, R) in cases
            sc = certify_split(A, Q; target = R, compression = :none)
            ds = deep_split(sc, R)
            ncase += 1
            if !ds.ok
                ndecl += 1
                @test ds.reason in (:cost, :farblock, :nocut)
                @test !isempty(deep_reason(ds))
                continue
            end
            # the derived bounds against the exact blocks of the reduced frame
            Qn = Q * ds.U
            Zn = Matrix(qr(Qn).Q)[:, 1:size(Qn, 2)]
            Pn = I - Zn * Zn'
            g1t = opnorm(Zn' * A * Pn)
            g2t = opnorm(Pn * A * Zn)
            G22t = Pn * A * Pn
            gt = opnorm(G22t)
            @test le(g1t, ds.gamma1n)
            @test le(g2t, ds.gamma2n)
            @test le(gt, ds.gbars[1])
            for j in 2:length(ds.gbars)
                @test le(opnorm(G22t^j), ds.gbars[j])
            end
            ds.gamma1n >= g1t || (nbadc += 1)
            nclose += 1
            # the reduced certificate is sound where it claims a floor
            sc2 = deep_certificate(sc, ds; compression = :full)
            @test le(opnorm(Zn' * A * Zn), sc2.snorm)
            for ρ in (1.0, 1.3, 2.0), t in 0:7
                z = ρ * R * cis(2π * t / 8)
                f = resolvent_floor(sc2, z; method = :svd).floor
                tr = minimum(svdvals(A - z * I))
                @test le(f, tr)
                f > tr * (1 + TOL) && (nbadf += 1)
                f > 0 && (worst = min(worst, f / tr))
            end
            @printf("  deep %-12s dim Q %d -> %d (E|N|D = %d|%d|%d), ḡ' ≤ %.4g (true %.4g), γ'₁ ≤ %.3e (true %.3e), γ'₂ ≤ %.3e (true %.3e)\n",
                nm, sc.k, sc2.k, ds.kE, ds.kN, ds.kD, ds.gbars[1], gt, ds.gamma1n, g1t,
                ds.gamma2n, g2t)
        end
        @printf("  deep: %d cases, %d closed, %d declined with a named obstruction, %d coupling bounds below the truth, %d floors above the truth (smallest positive ratio %.2e)\n",
            ncase, nclose, ndecl, nbadc, nbadf, worst)
        @test nbadc == 0
        @test nbadf == 0
        @test nclose >= 2

        # the 2×2 monotone bound against a real block matrix
        rng = MersenneTwister(21)
        for _ in 1:20
            n1, n2 = rand(rng, 2:6), rand(rng, 2:6)
            B = randn(rng, n1 + n2, n1 + n2)
            a = opnorm(B[1:n1, 1:n1])
            b = opnorm(B[1:n1, (n1 + 1):end])
            c = opnorm(B[(n1 + 1):end, 1:n1])
            d = opnorm(B[(n1 + 1):end, (n1 + 1):end])
            gs, _ = merged_far_bounds(a, b, c, d, 5)
            for j in 1:5
                @test le(opnorm(B^j), gs[j])
            end
        end
    end

    include("test_poly_range.jl")
end
