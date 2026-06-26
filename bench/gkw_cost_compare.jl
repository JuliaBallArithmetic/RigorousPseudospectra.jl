# Cost comparison on the K=256 GKW matrix: the FREE resolvent estimate (one enclosure,
# O(#clusters) per z, NO SVD) vs the FULL per-z verified resolvent (a svdbox of (zI−A) at
# each contour sample — what CertifScripts does). Same rigorous output kind (a lower bound
# on σ_min(zI−A) / upper bound on ‖R(z)‖); the free route trades a small constant of
# tightness for skipping the per-z verified SVD entirely.
#
# Float64 here; for BigFloat / larger K run on a bigger box (e.g. ibis over ssh) — same
# script, just point at the bigfloat matrix and setprecision.
#
# Run:  julia --project=bench bench/gkw_cost_compare.jl

using RigorousPseudospectra
using LinearAlgebra
using BallArithmetic
using Serialization
using Printf

const DATA = "/home/isaia/Code/GKWExperiments.jl/data"
eps_K(::Type{T}; K = 256, N = 5000) where {T} =
    T(sum(sqrt(2n + 1) / (n - 0.5)^2 for n in 1:(N - 1))) * (T(2) / 3)^(K + 1)

function main()
    A = ComplexF64.(BallArithmetic.mid(deserialize(joinpath(DATA, "ball_matrix_K256.jls"))))
    n = size(A, 1)
    ε = eps_K(Float64)
    AB = BallMatrix(A)
    Id = Matrix{ComplexF64}(I, n, n)

    # ---- FREE route: one enclosure + per-z O(#clusters) evaluation -----------------
    miyajima_rump(A)                                            # warmup/compile
    t0 = time()
    R = miyajima_rump(A)
    fr = free_resolvent(A, R)
    tfree_setup = time() - t0
    λs = R.centres[sortperm(R.centres; by = abs, rev = true)]
    dnear(j) = minimum(abs(λs[j] - λs[k]) for k in eachindex(λs) if k != j)
    # cost of one circle scan (the per-z work the bridge needs)
    nsamp = 256
    tfree_perz = @elapsed for q in 0:(nsamp - 1)
        resolvent_bound(fr, λs[1] + (dnear(1) / 3) * cis(2π * q / nsamp))
    end

    # ---- FULL route: verified 1/σ_min(zI−A) via svdbox, per z ----------------------
    z0 = λs[1] + 0.01
    BallArithmetic.svd_bound_L2_opnorm_inverse(BallMatrix(z0 * Id - A))   # warmup
    tunit = minimum(@elapsed(BallArithmetic.svd_bound_L2_opnorm_inverse(BallMatrix(z * Id - A)))
    for z in (z0, z0 + 0.001im, z0 - 0.002))                              # per-z svdbox cost

    @printf("\nGKW K=256 (n=%d), Float64\n", n)
    @printf("FREE   : setup (1 enclosure) = %.3f s ;  per-circle scan (%d samples) = %.4f s\n",
        tfree_setup, nsamp, tfree_perz)
    @printf("FULL   : one verified 1/σ_min(zI−A) via svdbox = %.3f s  (per z)\n", tunit)
    @printf("\nFor a bridge with C circles × %d samples each, the per-z verified scan costs\n", nsamp)
    for C in (20, 50)
        tfull = C * nsamp * tunit
        @printf("  C=%-3d circles : FULL ≈ %.0f s   vs   FREE ≈ %.2f s (setup) + %.2f s (scan) → speedup ×%.0f\n",
            C, tfull, tfree_setup, C * tfree_perz, tfull / (tfree_setup + C * tfree_perz))
    end
    println("\n(FREE pays ONE O(n³) enclosure then O(#clusters)/z; FULL pays an O(n³) verified")
    println(" SVD at EVERY z. The free estimate is looser by ~κ₂ but the GKW small-gain has")
    println(" ~30 orders of ε-headroom, so the looseness is free — see gkw_free_resolvent.jl.)")
end

main()
