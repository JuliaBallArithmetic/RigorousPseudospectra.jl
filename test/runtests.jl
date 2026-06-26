using RigorousPseudospectra
using LinearAlgebra
using Random
using Test

# --- test matrices ---------------------------------------------------------------
rand_matrix(n; seed = 1) = (rng = MersenneTwister(seed); randn(rng, ComplexF64, n, n))

# spectrum with prescribed (possibly repeated) eigenvalues in a well-conditioned
# similarity — repeats are EXACT (semisimple), so their discs are coincident
function spectrum(λ; seed = 1)
    rng = MersenneTwister(seed)
    n = length(λ)
    S = randn(rng, ComplexF64, n, n)
    return S * Diagonal(ComplexF64.(λ)) / S
end

function jordan_block(m, λ = 0.0 + 0.0im)
    (J = Matrix{ComplexF64}(λ * I, m, m);
        for i in 1:(m - 1)
            J[i, i + 1] = 1.0
        end;
        J)
end

# every true eigenvalue lies in some certified disc
function encloses_all(B, R; tol = 1e-8)
    for λ in eigvals(ComplexF64.(Matrix(B)))
        any(i -> abs(λ - R.centres[i]) <= R.radii[i] + tol, eachindex(R.centres)) ||
            return false
    end
    return true
end

@testset "Miyajima–Rump eigenvalue enclosure" begin
    @testset "smoke / API" begin
        R = miyajima_rump(rand_matrix(8))
        @test R isa MiyajimaRump
        @test length(R.centres) == 8 == length(R.radii)
        @test R.nrmR2 < 1
        @test R.kappa >= 1 - 1e-12
        @test sum(length(c.idx) for c in R.clusters) == 8
    end

    @testset "containment — every eigenvalue in a certified disc" begin
        for (n, seed) in ((6, 1), (12, 2), (20, 3), (30, 4))
            B = rand_matrix(n; seed)
            @test encloses_all(B, miyajima_rump(B))
        end
    end

    @testset "centres recover the spectrum (well-conditioned)" begin
        B = rand_matrix(16; seed = 7)
        R = miyajima_rump(B)
        # each true eigenvalue matched by a centre to within its radius
        tru = sort(eigvals(ComplexF64.(B)); by = z -> (real(z), imag(z)))
        cen = sort(R.centres; by = z -> (real(z), imag(z)))
        @test maximum(abs.(tru .- cen)) <= maximum(R.radii) + 1e-8
    end

    @testset "normal matrix — tight, κ ≈ 1" begin
        rng = MersenneTwister(5)
        Q, _ = qr(randn(rng, ComplexF64, 14, 14))
        B = Matrix(Q) * Diagonal(randn(rng, ComplexF64, 14)) * Matrix(Q)'
        R = miyajima_rump(B)                      # default: cheap (Collatz) κ₂
        @test maximum(R.radii) <= 1e-8            # negligible residue ⇒ tight discs
        @test all(c -> length(c.idx) == 1, R.clusters)
        @test R.kappa >= 1 - 1e-9                 # cheap κ₂ ≥ true (1), looser (~√n)
        # κ₂≈1 needs the tight svdbox mode (opt-in); cheap overestimates ‖W‖₂ by ~√n
        @test miyajima_rump(B; kappa_mode = :svdbox).kappa <= 1 + 1e-6
        @test R.kappa > 1 + 1e-3                  # default really is the looser bound
    end

    @testset "coincident eigenvalues merge; separable ones stay split" begin
        # A semisimple double eigenvalue (exactly coincident discs) merges into one
        # component; the well-separated ones stay singletons. (Near-defective
        # merging at finite separation is the deferred §4e regime.)
        R = miyajima_rump(Matrix(Diagonal(ComplexF64[1, 1, 2, 3])))
        @test sort([length(c.idx) for c in R.clusters]) == [1, 1, 2]
    end

    @testset "Gershgorin cluster count — each component holds its eigenvalues" begin
        B = spectrum([1, 1, 1, 2, 2, 3, 4, 5]; seed = 3)   # multiplicities 3,2,1,1,1
        R = miyajima_rump(B)
        ev = eigvals(ComplexF64.(B))
        for c in R.clusters
            nin = count(λ -> any(i -> abs(λ - R.centres[i]) <= R.radii[i] + 1e-8, c.idx), ev)
            @test nin == length(c.idx)                     # Gershgorin cluster theorem
        end
        @test sum(length(c.idx) for c in R.clusters) == 8
    end

    @testset "Jordan: exact fails honestly, perturbed is enclosed (finite κ)" begin
        # an EXACT integer Jordan block is maximally degenerate (eig returns a
        # rank-deficient W ⇒ ‖R₂‖ ≥ 1 ⇒ honest Step-4 failure)
        @test_throws Exception miyajima_rump(jordan_block(8, 0.0 + 0im))
        # a perturbed Jordan is regularized by floating eig to finite conditioning;
        # its eigenvalues spread (~ε^{1/m}) and are still rigorously enclosed
        for m in (3, 5, 8)
            rng = MersenneTwister(m)
            B = jordan_block(m, 0.0 + 0im) .+ 1e-6 .* randn(rng, ComplexF64, m, m)
            R = miyajima_rump(B)
            @test R.nrmR2 < 1
            @test isfinite(R.kappa) && R.kappa >= 1 - 1e-9
            @test encloses_all(B, R)
        end
    end

    @testset "Neumann guard / determinism" begin
        @test miyajima_rump([2.0 1.0; 0.0 3.0]) isa MiyajimaRump
        # real input is accepted and complexified
        @test miyajima_rump(Float64[0 -1; 1 0]) isa MiyajimaRump
    end

    @testset "certify a supplied basis W" begin
        B = rand_matrix(10; seed = 9)
        W = eigen(ComplexF64.(B)).vectors
        R = miyajima_rump(B, W)              # certify the externally-produced basis
        @test R isa MiyajimaRump
        @test encloses_all(B, R)
        # a bad basis (identity) is still certified — just loose, never wrong
        Rid = miyajima_rump(Matrix(Diagonal(ComplexF64[1, 2, 3])), Matrix{ComplexF64}(I, 3, 3))
        @test encloses_all(Matrix(Diagonal(ComplexF64[1, 2, 3])), Rid)
    end

    @testset "refinement: Newton steps tighten a rough basis, stay rigorous" begin
        # start from a basis perturbed away from the true eigenvectors: refinement
        # shrinks the off-diagonal residue (hence the certified radii) toward the
        # κ·eps floor, while every intermediate certificate still encloses σ(B).
        rng = MersenneTwister(31)
        B = rand_matrix(12; seed = 31)
        Wgood = eigen(ComplexF64.(B)).vectors
        W = Wgood * (I + 0.02 * randn(rng, ComplexF64, 12, 12))   # rough basis (in basin)
        R0 = miyajima_rump(B, W)
        @test encloses_all(B, R0)
        Wk = W
        for _ in 1:5
            Wk = refine_basis(B, Wk; steps = 1)                   # dynamic clustering
            @test encloses_all(B, miyajima_rump(B, Wk))           # rigorous every step
        end
        @test maximum(miyajima_rump(B, Wk).radii) < maximum(R0.radii) / 10   # tightened
    end

    @testset "dynamic refinement: coincident stays clustered, separable refines" begin
        # semisimple double eigenvalue: the dynamic working clustering neutralizes the
        # un-separable pair (no Newton blow-up, ‖R₂‖<1 holds) — it stays a size-2
        # cluster — while the distinct eigenvalues refine to singletons.
        B = spectrum([1, 1, 2, 3, 4]; seed = 5)
        R = miyajima_rump(B; refine = 6)
        @test encloses_all(B, R)
        @test R.nrmR2 < 1
        @test sort([length(c.idx) for c in R.clusters]) == [1, 1, 1, 2]
    end

    @testset "driver: eigen → dynamic refine → certify stays rigorous, no blow-up" begin
        B = rand_matrix(16; seed = 4)
        R_plain = miyajima_rump(B)
        R_ref = miyajima_rump(B; refine = 6)
        @test encloses_all(B, R_plain)
        @test encloses_all(B, R_ref)
        @test R_ref.nrmR2 < 1
        # eigen is already near-optimal here, so refinement neither helps much nor
        # blows up (radii stay at the κ·eps floor)
        @test maximum(R_ref.radii) <= 1e-6
    end

    @testset "block-Sylvester refinement variant vs entrywise sweep" begin
        # off-diagonal residue of W⁻¹BW (Frobenius) — the quantity refinement shrinks
        offres(B, W) = (A = ComplexF64.(W) \ (ComplexF64.(B) * ComplexF64.(W));
            sqrt(sum(abs2, A - Diagonal(diag(A)))))

        @testset "coincide on a fully-separated spectrum (all singletons)" begin
            # every working block is a singleton ⇒ the block step IS the entrywise
            # step (sylvester is never called), so the refined bases match to fp.
            B = rand_matrix(10; seed = 11)
            rng = MersenneTwister(11)
            W0 = eigen(ComplexF64.(B)).vectors * (I + 0.01 * randn(rng, ComplexF64, 10, 10))
            We = refine_basis(B, W0; steps = 3, mode = :entrywise)
            Wb = refine_basis(B, W0; steps = 3, mode = :block)
            @test maximum(abs.(We .- Wb)) < 1e-9
        end

        @testset "both rigorous + convergent on a near-defective cluster" begin
            # a perturbed 4×4 Jordan block forms a PERSISTENT, internally-coupled
            # working cluster (eigenvalues ~λ±ε^{1/4}, ill-conditioned W) coupled to
            # separated singletons — the regime where the block-Sylvester
            # cross-correction (full M_kk) genuinely differs from the entrywise step.
            rng = MersenneTwister(45)
            J = jordan_block(4, 2.0) .+ 1e-6 .* randn(rng, ComplexF64, 4, 4)
            core = cat(J, fill(8.0 + 0im, 1, 1), fill(14.0 + 0im, 1, 1); dims = (1, 2))
            S = I + 0.3 * randn(rng, ComplexF64, 6, 6)
            B = ComplexF64.(S * core / S)
            W0 = eigen(B).vectors * (I + 0.05 * randn(rng, ComplexF64, 6, 6))
            r0 = offres(B, W0)
            We, Wb = copy(W0), copy(W0)
            for _ in 1:8
                We = refine_basis(B, We; steps = 1, θ = 0.9, mode = :entrywise)
                Wb = refine_basis(B, Wb; steps = 1, θ = 0.9, mode = :block)
                @test encloses_all(B, miyajima_rump(B, We))   # rigorous each step …
                @test encloses_all(B, miyajima_rump(B, Wb))   # … for BOTH variants
            end
            # On THIS (representative hard) basis the entrywise scalar-gap correction
            # STALLS at a non-diagonal fixed point — its cross-cluster step ignores the
            # within-cluster coupling of M_kk — while block-Sylvester converges to the
            # κ·eps floor. (Aggregate over many bases the two are statistically tied
            # with partly-disjoint stalls; see bench/refine_variants.jl.)
            @test offres(B, Wb) < r0 / 1e6                    # block converges deeply
            @test offres(B, Wb) < offres(B, We)               # block wins on this basis
            @test offres(B, We) < r0                          # entrywise at least reduces
            # :auto runs entrywise, detects the stall, retries block from W0 → converges
            Wa = refine_basis(B, W0; steps = 20, mode = :auto)
            @test offres(B, Wa) < r0 / 1e6                    # auto recovers the stall
            @test encloses_all(B, miyajima_rump(B, Wa))
            @info "near-defective refinement (offdiag ‖·‖_F)" start=r0 entrywise=offres(B, We) block=offres(
                B, Wb) auto=offres(B, Wa)
            @info "certified max radius" entrywise=maximum(miyajima_rump(B, We).radii) block=maximum(miyajima_rump(
                B, Wb).radii)
        end

        @testset "driver threads mode, stays rigorous (block + auto)" begin
            B = rand_matrix(14; seed = 7)
            for m in (:block, :auto)
                R = miyajima_rump(B; refine = 4, mode = m)
                @test encloses_all(B, R)
                @test R.nrmR2 < 1
            end
            @test_throws Exception miyajima_rump(B; refine = 1, mode = :nonsense)
        end
    end

    @testset "free resolvent floor (§4f)" begin
        # true σ_min(B − zI) = smallest singular value
        true_smin(B, z) = minimum(svdvals(ComplexF64.(Matrix(B)) - z * I))

        @testset "PRIMARY rigor: floor never exceeds the true σ_min" begin
            for (n, seed) in ((6, 1), (10, 2), (16, 3))
                B = rand_matrix(n; seed)
                fr = free_resolvent(B, miyajima_rump(B; refine = 4))
                ev = eigvals(ComplexF64.(B))
                rng = MersenneTwister(seed)
                zs = vcat(4 .* (randn(rng, ComplexF64, 40)),          # scattered
                    ev .+ 0.3 .* randn(rng, ComplexF64, n),           # near spectrum
                    ev .+ 1e-9)                                       # ~on eigenvalues
                for z in zs
                    @test sigma_min_lower(fr, z) <= true_smin(B, z) + 1e-9
                    @test resolvent_bound(fr, z) >= 1 / (true_smin(B, z) + 1e-12) - 1e-6
                end
            end
        end

        @testset "normal matrix: floor is tight (κ₂≈1, ‖N‖≈0)" begin
            rng = MersenneTwister(5)
            Q, _ = qr(randn(rng, ComplexF64, 12, 12))
            B = Matrix(Q) * Diagonal(randn(rng, ComplexF64, 12)) * Matrix(Q)'
            # tightness needs the svdbox κ₂ (the cheap default loosens the floor by ~√n)
            fr = free_resolvent(B, miyajima_rump(B; kappa_mode = :svdbox))
            @test maximum(fr.rowsum) < 1e-6            # diagonal Ã ⇒ negligible row tails
            @test maximum(fr.colsum) < 1e-6            #                  and column tails
            @test fr.kappa <= 1 + 1e-6
            for z in 4 .* randn(rng, ComplexF64, 30)
                # for a normal matrix σ_min = dist(z, spectrum); floor recovers it
                @test sigma_min_lower(fr, z) >= true_smin(B, z) - 1e-6
                @test sigma_min_lower(fr, z) <= true_smin(B, z) + 1e-9
            end
            # the cheap default stays a VALID (looser) lower bound — never exceeds truth
            frc = free_resolvent(B, miyajima_rump(B))
            @test frc.kappa > 1 + 1e-3                 # cheap κ₂ is the looser bound
            for z in 4 .* randn(rng, ComplexF64, 20)
                @test sigma_min_lower(frc, z) <= true_smin(B, z) + 1e-9
            end
        end

        @testset "near an eigenvalue the floor → 0 / resolvent → ∞" begin
            B = rand_matrix(8; seed = 4)
            fr = free_resolvent(B, miyajima_rump(B; refine = 4))
            λ = eigvals(ComplexF64.(B))[1]
            @test sigma_min_lower(fr, λ) <= 1e-6        # cannot certify σ_min > 0 at a pole
            @test resolvent_bound(fr, λ) > 1e3
            # far away the floor is positive and grows with distance
            far = λ + 50
            @test sigma_min_lower(fr, far) > 1
        end

        @testset "VBD: Schur → Newton-separate → orthogonalize (block-orthonormal, benign κ)" begin
            # two well-separated defective 2×2 Jordan blocks in a similarity. The
            # eigenvector frame is ill-conditioned (near-parallel defective eigenvectors);
            # vbd_solve (schur → Newton-separate → orthogonalize per block) keeps each
            # block orthonormal, so κ is governed by inter-block separation (benign).
            J1 = ComplexF64[0 1; 0 0]
            J2 = ComplexF64[10 1; 0 10]
            core = ComplexF64[J1 zeros(2, 2); zeros(2, 2) J2]
            U = I + 0.2 * randn(MersenneTwister(3), ComplexF64, 4, 4)
            B = ComplexF64.(U * core / U)
            W, clusters = vbd_solve(B)
            @test sort(length.(clusters)) == [2, 2]              # separated the two blocks
            for c in clusters                                    # each block orthonormalized
                @test opnorm(W[:, c]' * W[:, c] - I) < 1e-10
            end
            R = miyajima_rump(B, W; kappa_mode = :svdbox)
            @test R.kappa < miyajima_rump(B; kappa_mode = :svdbox).kappa / 1e5  # vastly better κ
            @test R.kappa < 10                                   # genuinely benign
            @test encloses_all(B, R)
            vr = vbd_resolvent(B, R; blocks = clusters)   # use the vbd_solve block structure
            truesmin(z) = minimum(svdvals(B - z * I))
            rng = MersenneTwister(11)
            for z in vcat(20 .* randn(rng, ComplexF64, 30), R.centres .+ 0.03)
                @test sigma_min_lower(vr, z) <= truesmin(z) + 1e-9               # far-field rigorous
                @test sigma_min_lower(vr, z; near = true) <= truesmin(z) + 1e-9  # per-z block SVD rigorous
            end
            # block-Gershgorin enclosure: every eigenvalue sits in some block disc, and the
            # per-block radii match the block sizes (total multiplicity = n)
            discs = block_enclosure(vr)
            @test sum(d.mult for d in discs) == size(B, 1)
            for λ in eigvals(B)
                @test any(abs(λ - d.center) <= d.radius + 1e-9 for d in discs)
            end
        end

        @testset "non-normality shows in the row/column tail (defective block)" begin
            # A defective 2×2 Jordan block (coincident eigenvalue 5, off-diagonal 1) in the
            # IDENTITY basis: Ã = B, so the recentred disc of row 1 has off-diagonal mass
            # |Ã₁₂| = 1 — the non-normality appears DIRECTLY as a row tail (≥ 1), widening
            # the local grey disc about λ = 5. (No separate ‖N‖₂ block measure needed.)
            B = ComplexF64.(cat(jordan_block(2, 5.0), Diagonal(ComplexF64[10, 20, 30]);
                dims = (1, 2)))
            R = miyajima_rump(B, Matrix{ComplexF64}(I, 5, 5))
            fr = free_resolvent(B, R)
            @test maximum(fr.rowsum) > 0.9                   # the Jordan superdiagonal (=1)
            # still a valid σ_min lower bound everywhere
            rng = MersenneTwister(46)
            for z in vcat(40 .* randn(rng, ComplexF64, 40), ComplexF64[5, 10, 20, 30] .+
                                                            0.3)
                @test sigma_min_lower(fr, z) <= true_smin(B, z) + 1e-8
            end
            # the resolvent is large near the defective eigenvalue (non-normal blow-up)
            @test resolvent_bound(fr, 5.0 + 0.05im) > 10
        end
    end
end

@testset "certified polynomial range (bisection vs critical-point)" begin
    # ascending coeffs ↦ p(x) = Σ cₖ xᵏ ; dense reference range over a fine grid
    pval(c, x) = evalpoly(x, c)
    function ref_range(c, a, b; m = 200_001)
        xs = range(a, b; length = m)
        vs = pval.(Ref(c), xs)
        return minimum(vs), maximum(vs)
    end
    # a certified enclosure must CONTAIN the true range; brackets must be ordered
    function sound(R, tlo, thi)
        @test R.min_lo <= R.min_hi
        @test R.max_lo <= R.max_hi
        @test R.min_lo <= tlo + 1e-7          # encloses the true minimum from below
        @test R.max_hi >= thi - 1e-7          # encloses the true maximum from above
    end

    @testset "smoke / API" begin
        c = [0.0, -1.0, 0.0, 1.0]             # x³ − x
        Rb = range_bisect(c, -2.0, 2.0)
        Rc = range_critical(c, -2.0, 2.0)
        @test Rb isa PolyRange && Rc isa PolyRange
        @test Rb.method === :bisection && Rc.method === :critical
        for R in (Rb, Rc)
            sound(R, -6.0, 6.0)
            @test R.min_lo ≈ -6 atol = 1e-6
            @test R.max_hi ≈ 6 atol = 1e-6
        end
        @test Rc.diag.ncrit == 2              # x = ±1/√3 both interior
    end

    @testset "monotone & extrema at endpoints" begin
        c = [1.0, 2.0]                        # 1 + 2x, linear ⇒ no interior critical pt
        for R in (range_bisect(c, 0.0, 3.0), range_critical(c, 0.0, 3.0))
            sound(R, 1.0, 7.0)
        end
        @test range_critical(c, 0.0, 3.0).diag.ncrit == 0
    end

    @testset "soundness & agreement on random polynomials" begin
        rng = MersenneTwister(7)
        for trial in 1:25
            d = rand(rng, 2:9)
            c = randn(rng, d + 1)
            a = -1.0 - rand(rng);
            b = 1.0 + rand(rng)
            tlo, thi = ref_range(c, a, b)
            Rb = range_bisect(c, a, b; tol = 1e-9)
            Rc = range_critical(c, a, b; refine = 8)
            sound(Rb, tlo, thi)
            sound(Rc, tlo, thi)
            # both pin the extrema to within their reported brackets ⇒ they agree
            @test Rb.min_lo ≈ Rc.min_lo atol = 1e-5
            @test Rb.max_hi ≈ Rc.max_hi atol = 1e-5
        end
    end

    @testset "Chebyshev T_n — known range [-1,1], non-normal companion" begin
        # T_n via recurrence; on [-1,1] the range is exactly [-1,1]
        cheb(n) = (Tm = [1.0]; T = [0.0, 1.0];
            for _ in 2:n
                Tn = zeros(length(T) + 1)
                for i in eachindex(T)
                    Tn[i + 1] += 2T[i]
                end
                for i in eachindex(Tm)
                    Tn[i] -= Tm[i]
                end
                (Tm, T) = (T, Tn)
            end;
            n == 0 ? [1.0] : n == 1 ? [0.0, 1.0] : T)
        for n in (4, 8, 12)
            c = cheb(n)
            Rc = range_critical(c, -1.0, 1.0; refine = 10)
            sound(Rc, -1.0, 1.0)
            @test Rc.diag.ncrit == n - 1      # all n−1 interior extrema found
            @test Rc.max_hi - Rc.min_lo < 2 + 1e-4   # tight about [-1,1]
        end
    end

    @testset "clustered critical points (ill-conditioned companion)" begin
        ε = 1e-4                              # (x²−ε)² : roots of p' are 0, ±√ε (clustered)
        c = [ε^2, 0.0, -2ε, 0.0, 1.0]
        tlo, thi = 0.0, (1 - ε)^2
        for R in (range_bisect(c, -1.0, 1.0; tol = 1e-10), range_critical(c, -1.0, 1.0))
            sound(R, tlo, thi)
        end
    end

    @testset "evaluation never exceeds a finer enclosure (rigor)" begin
        # the critical-point enclosure must contain the bisection enclosure's truth bracket
        rng = MersenneTwister(11)
        for _ in 1:10
            c = randn(rng, 6)
            Rb = range_bisect(c, -1.5, 1.5; tol = 1e-11)
            Rc = range_critical(c, -1.5, 1.5; refine = 8)
            @test Rc.min_lo <= Rb.min_hi + 1e-7   # critical min-bracket below bisection's upper
            @test Rc.max_hi >= Rb.max_lo - 1e-7
        end
    end
end

@testset "matrix-free Krylov resolvent floor on a circle" begin
    # mildly non-normal A with a clear leading eigenvalue λ₁=1 and a spectral gap;
    # the matvec block certificate must (a) be SOUND (floor ≤ true σ_min on Γ) and
    # (b) FIRE (positive whole-circle floor) when the radius stays inside the gap.
    rng = MersenneTwister(7)
    n = 120
    d = vcat(1.0 + 0im, 0.5 .* (2 .* rand(rng, ComplexF64, n - 1) .- (1 + im)))
    A = Matrix(Diagonal(d)) + triu(0.02 .* randn(rng, ComplexF64, n, n), 1)
    F = eigen(A)
    i1 = argmax(abs.(F.values))
    w1 = F.vectors[:, i1]
    c = F.values[i1]

    @testset "soundness + certification at radius ρ=$ρ" for ρ in (0.15, 0.25)
        cert = certify_gap_on_circle(A, w1, c, ρ; nθ = 32)        # default kmax = full basis
        truemin = minimum(minimum(svdvals(A - (c + ρ * cis(2π * t / 32)) * I))
        for t in 0:31)
        @test cert.floor <= truemin + 1e-9                       # rigor: never exceed the truth
        @test cert.certified                                     # fires inside the gap
        @test cert.floor > 0
    end

    @testset "per-z floor: ‖R₂‖<1 (selective reorth) and sound" begin
        fl, info = krylov_resolvent_floor(A, w1, c + 0.2)
        @test info.R2 < 1                                        # orthogonality retained
        @test fl <= minimum(svdvals(A - (c + 0.2) * I)) + 1e-9   # sound
        @test fl > 0
    end

    @testset "radius beyond the gap fails to certify" begin
        # ρ large enough to swallow part of σ(A)∖{λ₁}: the floor must NOT certify a gap
        cert = certify_gap_on_circle(A, w1, c, 0.9; nθ = 32)
        @test !cert.certified
    end
end

@testset "Krylov–Rump–Miyajima eigenvalue decoupling" begin
    # leading eigenvalues of a non-normal matrix certified isolated end-to-end:
    # Arnoldi → miyajima_rump (refined block-diagonal) → per-cluster resolvent circle.
    rng = MersenneTwister(7)
    n = 160
    d = vcat(1.0 + 0im, 0.6 .* (2 .* rand(rng, ComplexF64, n - 1) .- (1 + im)))
    A = Matrix(Diagonal(d)) + triu(0.03 .* randn(rng, ComplexF64, n, n), 1)
    out = certify_leading_eigs(A; nblock = 16, neig = 4, refine = 6)
    @test length(out.results) >= 1
    @testset "result $k" for (k, r) in enumerate(out.results)
        @test r.cert.certified                                   # isolated
        @test r.cert.floor > 0
        truemin = minimum(minimum(svdvals(A -
                                          (r.λ + r.cert.radius * cis(2π * t / 24)) * I))
        for t in 0:23)
        @test r.cert.floor <= truemin + 1e-9                     # sound
    end
end

@testset "spectral-gap certificate via far-block Gram (certify_gap)" begin
    # genuinely LOW-RANK non-normal operator: a 6×6 non-normal block (λ₁=1 + fast modes)
    # embedded by a random isometry, the rest exactly 0 ⇒ G₂₂ (deflate λ₁) is rank 5.
    rng = MersenneTwister(11)
    n = 200
    B = [1.0+0im 0.3 0.2 0.1 0.05 0.05; 0 0.45 0.2 0.1 0.05 0.05;
         0 0 0.40 0.15 0.05 0.05; 0 0 0 -0.35 0.1 0.05;
         0 0 0 0 0.30+0.1im 0.1; 0 0 0 0 0 0.30-0.1im]
    Q = Matrix(qr(randn(rng, ComplexF64, n, 6)).Q)
    A = Q * B * Q'
    F = eigen(A)
    i1 = argmax(abs.(F.values))
    Wslow = reshape(F.vectors[:, i1] / norm(F.vectors[:, i1]), n, 1)
    rgap = 0.7
    fl, info = certify_gap(A, Wslow, rgap; kgram = 30, nθ = 48)
    @test info.certified                                  # gap certified
    @test fl > 0
    @test info.ksing == 5                                 # captured the rank-5 far block
    @test info.Enorm < 1e-8                               # residual ≈ 0 (genuinely low rank)
    truemin = minimum(minimum(svdvals(A - rgap * cis(2π * t / 48) * I)) for t in 0:47)
    @test fl <= truemin + 1e-9                            # sound (floor ≤ true σ_min on Γ_gap)
end
