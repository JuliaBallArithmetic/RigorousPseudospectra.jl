# Does capturing the FULL isolated block open an ℓ² resolvent gap in the far block?
# Dense diagnostic (no Lanczos), N=1024 Ulam (Lanford map): ordered-Schur capture of the top-m
# eigenvalues (right-invariant ⇒ far block = trailing Schur block, G21=0 exactly), then compare
# the TRUE far-block σ_min(G22−zI) on the separating circle to the full σ_min(P−zI). If
# capturing more isolated eigenvalues makes min_{|z|=ρ} σ_min(G22−zI) grow to something decently
# positive, the user's idea works (the non-normality was in the isolated coupling); if it stays
# ~1e-4 no matter how much we capture, the essential-spectrum bulk is intrinsically non-normal
# in ℓ² and no circle works.
using RigorousInvariantMeasures, LinearAlgebra, SparseArrays, Printf
import IntervalArithmetic

N = 1024
P = Matrix(sparse(map(IntervalArithmetic.mid,
    DiscretizedOperator(Ulam(N), mod1_dynamic(x -> 2 * x + 0.5 * x * (1 - x))).L)))

F = schur(ComplexF64.(P))
absλ = abs.(F.values)
ord = sortperm(absλ; rev = true)
sorted = absλ[ord]
@printf("N=%d  top-30 |λ|:\n", N)
for r in 1:5
    @printf("  %s\n", join(round.(sorted[(6*(r-1)+1):min(6r, 30)]; digits = 4), "  "))
end
println()

circ_min(M, ρ; np = 16) = minimum(minimum(svdvals(M - ρ * cis(θ) * I))
                                  for θ in range(0, 2π; length = np + 1)[1:np])

@printf("%-4s  %-8s  %-12s  %-12s  %-12s  %-10s\n",
    "m", "ρ", "min σ22(circ)", "min σP(circ)", "ratio σ22/σP", "tri floor")
for m in (2, 4, 8, 16, 32, 64)
    m >= N && continue
    sel = falses(N)
    sel[ord[1:m]] .= true
    Fo = ordschur(F, sel)
    T22 = Fo.T[(m + 1):end, (m + 1):end]          # far block (trailing), G21 = 0 exactly
    G12 = Fo.T[1:m, (m + 1):end]
    γ = opnorm(G12, 2)
    G11 = Fo.T[1:m, 1:m]
    ρ = (sorted[m] + sorted[m + 1]) / 2           # circle between λ_m (kept) and λ_{m+1} (far)
    s22 = circ_min(T22, ρ)
    sP = circ_min(P, ρ)
    # triangular assembled floor at the worst circle point (use the per-circle minima as a proxy)
    s11 = circ_min(G11, ρ)
    tri = (s11 <= 0 || s22 <= 0) ? 0.0 : s11 * s22 / (s11 + s22 + γ)
    @printf("%-4d  %-8.4f  %-12.4e  %-12.4e  %-12.3g  %-10.4e  (γ=%.2e)\n",
        m, ρ, s22, sP, s22 / sP, tri, γ)
end
