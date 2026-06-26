# Compare the free-eigenvalue count from the row/col interpolation bound (free_resolvent)
# vs the VBD block-σ_min Weyl bound (vbd_resolvent) on the cached GKW BigFloat pipeline.
# Requires bench/gkw_cache.jl to have been run.  julia --project=bench bench/gkw_resolvent_test.jl [prec]
using RigorousPseudospectra, LinearAlgebra, BallArithmetic, Serialization, Printf

cachepath(prec) = joinpath(@__DIR__, "gkw_vbd_bf$(prec).jls")

function run(prec::Int = 256)
    setprecision(BigFloat, prec)
    c = deserialize(cachepath(prec))
    A, clusters, R = c.A, c.clusters, c.R
    n = size(A, 1)
    # GKW small-gain perturbation radius ε_K (arXiv:2602.19435), K=256
    ε = sum(sqrt(2k + 1) / (k - 0.5)^2 for k in 1:4999) * (big(2) / 3)^257
    @printf("n=%d  prec=%d  κ₂=%.3e  ‖R₂‖∞=%.2e  ε_K=%.2e\n",
        n, prec, Float64(R.kappa), Float64(R.nrmR2), Float64(ε))

    fr = free_resolvent(A, R)
    t0 = time()
    vr = vbd_resolvent(A, R; blocks = clusters)        # block structure from vbd_solve
    @printf("vbd_resolvent build %.0fs  #blocks=%d  ‖N‖₂(inter-block)=%.3e\n",
        time() - t0, length(clusters), Float64(vr.coupling))

    λs = R.centres[sortperm(R.centres; by = abs, rev = true)]
    dn(j) = minimum(abs(λs[j] - λs[k]) for k in eachindex(λs) if k != j)

    # count eigenvalues whose ε·sup_Γ ‖R(z)‖ < 1 on a circle Γ of radius (nearest-gap)/3
    function countfree(bound)
        free = 0
        for j in eachindex(λs)
            rj = dn(j) / 3
            rj <= 0 && continue
            M = big(0.0); ok = true
            for q in 0:63
                rb = bound(λs[j] + rj * cis(2 * big(π) * q / 64))
                isfinite(rb) || (ok = false; break)
                M = max(M, rb)
            end
            ok && ε * M < 1 && (free += 1)
        end
        free
    end

    @printf("free_resolvent (row/col interp)     : %d / %d\n",
        countfree(z -> resolvent_bound(fr, z)), n)
    @printf("vbd_resolvent  (block Weyl, far)    : %d / %d\n",
        countfree(z -> resolvent_bound(vr, z; near = false)), n)
end

run(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 256)
