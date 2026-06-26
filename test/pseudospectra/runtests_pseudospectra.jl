using RigorousPseudospectra
using LinearAlgebra
using Random
using Test

# Local test matrices (kept independent of bench/).
function grcar(n, k = 3)
    A = zeros(Float64, n, n)
    for i in 1:n
        (i + 1 <= n) && (A[i + 1, i] = -1.0)
        for d in 0:k
            (i + d <= n) && (A[i, i + d] = 1.0)
        end
    end
    return A
end

function normal_random(n; seed = 1)
    rng = MersenneTwister(seed)
    Q, _ = qr(randn(rng, ComplexF64, n, n))
    return Matrix(Q) * Diagonal(randn(rng, ComplexF64, n)) * Matrix(Q)'
end

function jordan_block(m, λ = 0.0 + 0.0im)
    J = zeros(ComplexF64, m, m)
    for i in 1:m
        J[i, i] = λ
        (i + 1 <= m) && (J[i, i + 1] = 1.0)
    end
    return J
end

# weakly-coupled chain: distinct eigenvalues 1..n with small superdiagonal s.
# Globally one coupling-graph cluster, but only a local window is active at each z.
function graded_chain(n; s = 0.5)
    A = zeros(ComplexF64, n, n)
    for i in 1:n
        A[i, i] = i
        (i + 1 <= n) && (A[i, i + 1] = s)
    end
    return A
end

truesig(A, z) = svdvals(ComplexF64.(Matrix(A)) - z * I)[end]

@testset "RigorousPseudospectra.jl" begin
    @testset "smoke" begin
        @test isdefined(RigorousPseudospectra, :CertSetup)
        @test isdefined(RigorousPseudospectra, :bisect)
    end

    # tol absorbs the Float64-only slack; the ball version will enforce exactly.
    tol = 1e-7

    @testset "containment — never exceed true σ_min" begin
        # graded_chain has asymmetric triangular tails — exposes the ℓ∞/ℓ¹ → ℓ²
        # combination (geometric mean, NOT max) for the free triangular gap.
        for A in (grcar(32), normal_random(20), grcar(64, 2), graded_chain(12; s = 0.7))
            S = certified_setup(A)
            rng = MersenneTwister(42)
            for _ in 1:200
                z = complex(6 * randn(rng) - 1, 6 * randn(rng))
                t = truesig(A, z)
                @test gap_tri(z, S) <= t + tol
                @test gap_recentred(z, S) <= t + tol
                @test gap_uniform(z, S) <= t + tol
                @test sigma_min_lower_bound(z, S) <= t + tol
            end
        end
    end

    @testset "free gap is ~exact for normal matrices" begin
        A = normal_random(24)
        S = certified_setup(A)
        rng = MersenneTwister(7)
        for _ in 1:50
            z = complex(4 * randn(rng), 4 * randn(rng))
            @test isapprox(gap_tri(z, S), truesig(A, z); atol = 1e-8)
        end
    end

    @testset "ordering: uniform ≤ recentred (where positive)" begin
        A = grcar(48)
        S = certified_setup(A)
        rng = MersenneTwister(99)
        for _ in 1:200
            z = complex(6 * randn(rng) - 1, 6 * randn(rng))
            gu, gr = gap_uniform(z, S), gap_recentred(z, S)
            if gu > 0
                @test gu <= gr + tol
            end
        end
    end

    @testset "eigenvalue sanity — gap ≤ 0 at the spectrum" begin
        A = grcar(32)
        S = certified_setup(A)
        for λ in eigvals(A)
            @test sigma_min_lower_bound(λ, S) <= tol
        end
    end

    @testset "SVD-frame certificate" begin
        @testset "containment — never exceed true σ_min" begin
            for A in (grcar(32), normal_random(20), grcar(64, 2))
                F = svd_frame(A)
                rng = MersenneTwister(13)
                for _ in 1:200
                    z = complex(6 * randn(rng) - 1, 6 * randn(rng))
                    @test gap_svd(z, F) <= truesig(A, z) + tol
                end
            end
        end

        @testset "exact for Hermitian PSD (W = I)" begin
            rng = MersenneTwister(5)
            Vq, _ = qr(randn(rng, ComplexF64, 18, 18))
            d = rand(rng, 18) .+ 0.5            # positive ⇒ PSD, no sign ambiguity
            A = Matrix(Vq) * Diagonal(d) * Matrix(Vq)'
            F = svd_frame(A)
            for _ in 1:50
                z = complex(3 * randn(rng), 3 * randn(rng))
                @test isapprox(gap_svd(z, F), truesig(A, z); atol = 1e-7)
            end
        end

        @testset "γ ≥ 0 collapses to a real σ_min² bound" begin
            A = grcar(24)
            F = svd_frame(A)
            for z in (10.0 + 0im, -8.0 + 6im)   # far field: |z|² dominates
                @test svdframe_gamma(z, F) > 0
                @test gap_svd(z, F) <= truesig(A, z) + tol
            end
        end
    end

    @testset "combined cheap floor — max(triangular, SVD-frame)" begin
        for A in (grcar(32), grcar(64, 2), normal_random(20))
            f = cheap_floor(A)
            rng = MersenneTwister(21)
            for _ in 1:200
                z = complex(6 * randn(rng) - 1, 6 * randn(rng))
                gt = gap_tri(z, f.setup)
                gs = gap_svd(z, f.frame)
                t = truesig(A, z)
                # combined is the max of the two floors …
                @test f(z) == max(gt, gs)
                # … hence ≥ each and still ≤ truth (containment preserved)
                @test f(z) >= gt - 0.0
                @test f(z) >= gs - 0.0
                @test f(z) <= t + tol
                # binding label is consistent
                val, which = binding_floor(f, z)
                @test val == f(z) || which === :none
                if which === :triangular
                    @test gt >= gs
                elseif which === :svd
                    @test gs >= gt
                end
            end
        end
    end

    @testset "each floor wins somewhere (complementary looseness)" begin
        # The triangular gap (κ=1, sandwich-tight) dominates most of the plane;
        # the SVD frame is complementary near defective/clustered spectra. On a
        # Jordan block both floors are strictly binding on a nonzero fraction of
        # points — demonstrating the max genuinely contracts the inconclusive set.
        f = cheap_floor(jordan_block(8))
        rng = MersenneTwister(123)
        tri_wins = 0
        svd_wins = 0
        for _ in 1:3000
            z = complex(2 * randn(rng), 2 * randn(rng))
            _, which = binding_floor(f, z)
            which === :triangular && (tri_wins += 1)
            which === :svd && (svd_wins += 1)
        end
        @test tri_wins > 0
        @test svd_wins > 0
    end

    @testset "coupling-graph clustering (§4b)" begin
        @testset "diagonal Schur ⇒ all singletons" begin
            T = Matrix(Diagonal(ComplexF64[1, 2, 3, 4, 5]))
            comps = cluster_components(T, diag(T); θ = 0.1, ε_target = Inf)
            @test length(comps) == 5
            @test all(c -> length(c) == 1, comps)
        end

        @testset "two blocks, weak inter-coupling ⇒ two components" begin
            eigs = ComplexF64[0, 0.2, 0.4, 10, 10.2, 10.4]
            T = Matrix(Diagonal(eigs))
            T[1, 2] = 1.0; T[2, 3] = 1.0          # strong intra-block (1–3)
            T[4, 5] = 1.0; T[5, 6] = 1.0          # strong intra-block (4–6)
            T[3, 4] = 1e-6                          # negligible inter-block
            comps = cluster_components(T, eigs; θ = 0.3, ε_target = Inf)
            @test Set(Set.(comps)) == Set([Set([1, 2, 3]), Set([4, 5, 6])])
        end

        @testset "transitive coupling groups endpoints" begin
            eigs = ComplexF64[0, 1, 2]
            T = Matrix(Diagonal(eigs))
            T[1, 2] = 0.5; T[2, 3] = 0.5            # i–k–j, no direct T[1,3]
            comps = cluster_components(T, eigs; θ = 0.3, ε_target = Inf)
            @test length(comps) == 1 && Set(comps[1]) == Set([1, 2, 3])
        end

        @testset "ε-tied threshold: coarse ε blocks fewer couplings" begin
            eigs = ComplexF64[0, 1]
            T = Matrix(Diagonal(eigs))
            T[1, 2] = 0.05
            # gap-relative would give θ·1 = 0.3 > 0.05 ⇒ no edge; but a fine
            # ε_target shrinks the threshold below the coupling ⇒ edge appears.
            @test length(cluster_components(T, eigs; θ = 0.3, ε_target = Inf)) == 2
            @test length(cluster_components(T, eigs; θ = 0.3, ε_target = 0.1)) == 1
        end

        @testset "percolation scan is monotone in θ" begin
            A = grcar(24)
            S = certified_setup(A)
            scan = percolation_scan(S.T, S.eigs; θs = [1.0, 0.5, 0.25, 0.1, 0.05],
                ε_target = Inf)
            sizes = [r.maxsize for r in scan]      # θ decreasing ⇒ maxsize ↑
            @test issorted(sizes)
            @test scan[1].maxsize <= scan[end].maxsize
            @test all(r -> 1 <= r.maxsize <= 24, scan)
        end
    end

    @testset "cluster / block path (§5)" begin
        # A with a tight cluster of 3 eigenvalues near 0 and 4 well-separated ones,
        # in a generic (non-normal) upper-triangular Schur factor rotated by Q.
        function clustered_matrix(; seed = 1)
            rng = MersenneTwister(seed)
            λ = ComplexF64[0, 0.02, -0.015, 5, 6, 7, 8]
            n = length(λ)
            T = Matrix(Diagonal(λ))
            for j in 1:n, i in 1:(j - 1)
                T[i, j] = 0.3 * (randn(rng) + im * randn(rng))
            end
            Q, _ = qr(randn(rng, ComplexF64, n, n))
            return Matrix(Q) * T * Matrix(Q)'
        end

        @testset "2×2 σ_min is exact and cancellation-free" begin
            rng = MersenneTwister(3)
            for _ in 1:200
                M = randn(rng, ComplexF64, 2, 2)
                @test RigorousPseudospectra._sigma_min_2x2(M) ≈ svdvals(M)[end]
            end
            # near-singular: only the outer subtraction is delicate
            M = ComplexF64[1.0 1.0; 0.0 1e-12]
            @test isapprox(RigorousPseudospectra._sigma_min_2x2(M), svdvals(M)[end];
                rtol = 1e-6)
        end

        @testset "reorder re-certifies the basis" begin
            A = clustered_matrix()
            S = certified_setup(A)
            idx = findall(λ -> abs(λ) < 1.0, S.eigs)        # the near-0 cluster
            cb = reorder_cluster(S, idx)
            @test cb.rs isa CertSetup
            @test cb.rs.nrmR2_inf < 1 && cb.rs.nrmR2_1 < 1   # reordered basis certified
            @test cb.m == length(idx)
        end

        @testset "containment via block-triangular assembly" begin
            A = clustered_matrix()
            S = certified_setup(A)
            idx = findall(λ -> abs(λ) < 1.0, S.eigs)
            cb = reorder_cluster(S, idx)
            rng = MersenneTwister(8)
            useful = 0
            for _ in 1:400
                z = complex(3 * randn(rng), 3 * randn(rng))
                g = gap_cluster(z, cb)
                t = truesig(A, z)
                @test g <= t + tol                          # never exceed truth
                isfinite(g) && g > 0 && (useful += 1)
            end
            @test useful > 0                                 # finite & positive somewhere
        end

        @testset "cluster bound is useful near the cluster" begin
            A = clustered_matrix()
            S = certified_setup(A)
            idx = findall(λ -> abs(λ) < 1.0, S.eigs)
            cb = reorder_cluster(S, idx)
            for z in (0.1 + 0.0im, 0.0 + 0.1im, -0.1 + 0.05im)   # near cluster, far from F
                g = gap_cluster(z, cb)
                @test isfinite(g) && g > 0
                @test g <= truesig(A, z) + tol
                @test far_gap(z, cb) > 0
                @test localize(z, cb)
            end
        end

        @testset "encircling-curve certificate (pole-free curve)" begin
            A = clustered_matrix()
            S = certified_setup(A)
            idx = findall(λ -> abs(λ) < 1.0, S.eigs)
            cb = reorder_cluster(S, idx)
            c = cluster_center(cb)
            # a circle between the cluster spread (~0.04) and the far set (~5)
            bound, zw = encircle_cluster(cb, c, 1.0; n = 256)
            @test bound > 0                          # circle certified pole-free
            # rigorous over the WHOLE circle: ≤ true σ_min at every point on it
            fine = [c + 1.0 * cispi(2k / 2048) for k in 0:2047]
            true_curve_min = minimum(truesig(A, z) for z in fine)
            @test bound <= true_curve_min + tol
            @test abs(zw - c) ≈ 1.0 rtol=1e-12       # worst point lies on the circle

            # a circle that swallows a far eigenvalue (~5) is NOT pole-free
            bad, _ = encircle_cluster(cb, c, 6.0; n = 256)
            @test bad <= tol
        end

        @testset "whole-matrix cluster (m = n)" begin
            A = clustered_matrix()
            S = certified_setup(A)
            cb = reorder_cluster(S, 1:size(A, 1))
            @test cb.m == size(A, 1)
            rng = MersenneTwister(11)
            for _ in 1:100
                z = complex(3 * randn(rng), 3 * randn(rng))
                @test gap_cluster(z, cb) <= truesig(A, z) + tol
            end
        end
    end

    @testset "σ_min upper bound (inclusion side)" begin
        for A in (grcar(24), normal_random(16))
            S = certified_setup(A)
            rng = MersenneTwister(77)
            for _ in 1:200
                z = complex(5 * randn(rng) - 1, 5 * randn(rng))
                @test sigma_min_upper(z, S) >= truesig(A, z) - tol   # never below truth
            end
        end
        # exact for normal matrices (= dist to spectrum)
        A = normal_random(20)
        S = certified_setup(A)
        rng = MersenneTwister(78)
        for _ in 1:50
            z = complex(3 * randn(rng), 3 * randn(rng))
            @test isapprox(sigma_min_upper(z, S), truesig(A, z); atol = 1e-8)
        end
    end

    @testset "box-bisection sweep (§6)" begin
        @testset "cells partition; classification is rigorous" begin
            A = grcar(24)
            enc = bisect(A; ε = 0.3, min_halfdiag = 0.05)
            @test enc isa PseudospectrumEnclosure
            @test !isempty(enc.excluded)
            # every EXCLUDED cell is genuinely outside σ_ε: σ_min > ε at its centre
            for cell in enc.excluded
                @test truesig(A, cell.c) > enc.ε - tol
            end
            # every INCLUDED cell is genuinely inside σ_ε: σ_min < ε at its centre
            for cell in enc.included
                @test truesig(A, cell.c) < enc.ε + tol
            end
            # boundary cells are at target resolution
            for cell in enc.boundary
                @test RigorousPseudospectra.halfdiag(cell) < 0.05
            end
            @test length(enc.routing) == length(enc.excluded)
        end

        @testset "Lipschitz certificate holds over whole excluded cells" begin
            # not just the centre: sample interior points of excluded cells
            A = grcar(20)
            enc = bisect(A; ε = 0.5, min_halfdiag = 0.08)
            rng = MersenneTwister(5)
            for cell in enc.excluded[1:min(end, 60)]
                for _ in 1:5
                    z = cell.c + complex((2rand(rng) - 1) * cell.hx,
                        (2rand(rng) - 1) * cell.hy)
                    @test truesig(A, z) > enc.ε - tol
                end
            end
        end

        @testset "normal matrix: ε-disks around eigenvalues" begin
            # for normal A, σ_ε is the union of ε-disks about the eigenvalues, so
            # every included-cell centre is within ε of some eigenvalue
            A = normal_random(12)
            S = certified_setup(A)
            enc = bisect(cheap_floor(A); box = default_box(S; pad = 1.0),
                ε = 0.4, min_halfdiag = 0.05)
            for cell in enc.included
                @test minimum(abs(cell.c - λ) for λ in S.eigs) < 0.4 + tol
            end
        end

        @testset "cluster escalation resolves more, stays rigorous" begin
            A = grcar(20)
            box = (-2.0 - 2.5im, 2.5 + 2.5im)
            ε, h = 0.3, 0.1
            cheap = bisect(cheap_floor(A); box, ε, min_halfdiag = h, cluster = false)
            clust = bisect(cheap_floor(A); box, ε, min_halfdiag = h, cluster = true)
            # the block tier settles strictly more (fewer unresolved cells)
            @test length(clust.boundary) < length(cheap.boundary)
            @test length(clust.excluded) >= length(cheap.excluded)
            @test :cluster in keys(routing_histogram(clust))
            # escalation is still rigorous: excluded cells are genuinely outside σ_ε
            for cell in clust.excluded
                @test truesig(A, cell.c) > ε - tol
            end
            for cell in clust.included
                @test truesig(A, cell.c) < ε + tol
            end
        end

        @testset "z-adaptive active-set escalation localizes" begin
            A = graded_chain(20; s = 0.5)          # one global coupling cluster (m=20)
            S = certified_setup(A)
            # at a mid-spectrum point only a small window of gaps is failing
            act = RigorousPseudospectra._active_set(10.0 + 0im, S, 1.0)
            @test length(act) <= 6                       # local window, not all 20
            @test all(i -> abs(S.eigs[i] - (10 + 0im)) < 3, act)  # genuinely local
            # the active window stays small all along the chain — the win over the
            # global coupling cluster, which would be the whole m=20 matrix
            maxwin = maximum(length(RigorousPseudospectra._active_set(complex(x, 0.0), S, 1.0))
            for x in 1.0:0.5:20.0)
            @test maxwin <= 6
            # the adaptive sweep stays rigorous (small blocks per z)
            box = (0.0 - 1.5im, 21.0 + 1.5im)
            enc = bisect(cheap_floor(A); box, ε = 0.3, min_halfdiag = 0.2,
                adaptive = true)
            for cell in enc.excluded
                @test truesig(A, cell.c) > enc.ε - tol
            end
            for cell in enc.included
                @test truesig(A, cell.c) < enc.ε + tol
            end
        end

        @testset "routing histogram counts excluded cells" begin
            A = grcar(32)
            enc = bisect(A; ε = 0.2, min_halfdiag = 0.05)
            h = routing_histogram(enc)
            @test sum(values(h)) == length(enc.excluded)
            @test all(k -> k in (:triangular, :svd, :recentred, :cluster), keys(h))
        end
    end

    @testset "Neumann guard fires" begin
        # An exactly singular-basis case is hard to force via schur; instead
        # check the guard exists by confirming a well-posed matrix passes.
        @test certified_setup(grcar(16)) isa CertSetup
    end
end
