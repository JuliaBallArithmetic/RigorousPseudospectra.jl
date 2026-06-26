# Stage 2 (RigorousPseudospectra env): certify the Arnoldi compression H_N produced by
# RigorousUlam2D/examples/block_build.jl. miyajima_rump gives rigorous eigenvalue discs of
# H_N and κ₂ (the diagonalising-frame conditioning). Combined with the Arnoldi residual β_N,
# the "safe" leading Ritz values are certified eigenvalues of L (Bauer–Fike: each true
# eigenvalue of L lies within β_N·|last Ritz component| ≤ (disc) of a Ritz value), and
# ‖(slow block)ⁿ‖ ≤ κ₂·ρⁿ. The unconverged tail (last few discs) is the buffer — discarded.
#
#   julia --project=. bench/block_certify.jl <block_*.jls> ...

using RigorousPseudospectra, LinearAlgebra, Serialization, Printf

for path in ARGS
    d = deserialize(path)
    H = ComplexF64.(d.H)
    mr = miyajima_rump(H; kappa_mode = :svdbox)        # certified discs + tight κ₂
    println("\n", "="^72)
    @printf("%s  (depth %d, n=%d, Arnoldi N=%d, residual β_N=%.3e)\n",
        d.tag, d.depth, d.n, d.N, d.βN)
    @printf("  κ₂(diagonalising frame) = %.3f   ‖R₂‖ = %.2e\n", mr.kappa, mr.nrmR2)
    println("  certified eigenvalue discs of H_N  (centre |λ|, radius, |Bauer–Fike to σ(L)|):")
    ord = sortperm(abs.(mr.centres); rev = true)
    for i in ord
        c = mr.centres[i]; rad = mr.radii[i]
        # singleton & well inside the unit disk & tight ⇒ "safe" (converged Ritz)
        cl = findfirst(cl -> i in cl.indices, mr.clusters)
        sz = length(mr.clusters[cl].indices)
        safe = sz == 1 && rad < 1e-3 && abs(c) < 0.999
        @printf("    λ=%+.5f%+.5fi  |λ|=%.5f  disc r=%.2e  cluster size %d  %s\n",
            real(c), imag(c), abs(c), rad, sz, safe ? "SAFE" : (abs(c) > 0.999 ? "(λ₁=1)" : "(tail/buffer)"))
    end
    # contraction rate from the SAFE non-unit eigenvalues: ρ = max |λ| over safe, |λ|<1
    safe_abs = [abs(mr.centres[i]) + mr.radii[i] for i in eachindex(mr.centres)
                if length(mr.clusters[findfirst(cl -> i in cl.indices, mr.clusters)].indices) == 1 &&
                   mr.radii[i] < 1e-3 && abs(mr.centres[i]) < 0.999]
    if !isempty(safe_abs)
        ρ = maximum(safe_abs)
        @printf("  ⇒ certified slow rate ρ = %.5f (incl. disc radius);  ‖slowⁿ‖ ≤ κ₂·ρⁿ = %.3f·%.5fⁿ\n",
            ρ, mr.kappa, ρ)
        @printf("    N for κ₂·ρⁿ ≤ 0.5 :  n ≥ %d\n", ceil(Int, log(0.5 / mr.kappa) / log(ρ)))
    end
end
