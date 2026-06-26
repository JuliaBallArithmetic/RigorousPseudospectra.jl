# Certified gap functions (implementation.md §4, paper §2–3). Each returns a
# LOWER bound on σ_min(A − zI) at a single point z. Float64 first cut.
#
# All are O(n) per point: the z-independent centres/radii/tails live in CertSetup,
# and only |z − centre| changes with z. Order by sharpness/cost and take the
# best (largest) available lower bound.
#
# Two distinct gap families:
#  • Free triangular (Lemma `triag`/`sandwich`): Gerschgorin on T̃ − zI. Because
#    Z is unitary, σ_min(T̃ − zI) = σ_min(A − zI), so this is a DIRECT σ_min lower
#    bound with no conditioning factor — tight for (near-)normal A.
#  • Recentred (Thm `resolvent-row`): Gerschgorin on the collapsed M₀ − zI, giving
#    ℓ∞/ℓ¹ resolvent bounds that interpolate (Riesz–Thorin) to an ℓ² σ_min bound,
#    paying the κ(Z) conditioning prefactor. Sharper where the free gap is loose.

"""
    gap_tri_schur(z, S::CertSetup) -> Float64

Free triangular gap on the Schur factor (Lemma `triag`, cost O(n)): a lower bound
on σ_min(**T̃** − zI) from diag(T̃) and the fixed off-diagonal tails. The ℓ∞ and ℓ¹
diagonal-dominance gaps bound `‖(T̃−zI)⁻¹‖_∞ ≤ 1/g∞` and `‖·‖_1 ≤ 1/g₁`; the
σ_min (= 1/‖·‖_2) lower bound is their **geometric mean** `√(g₁·g∞)` via
`‖M⁻¹‖_2 ≤ √(‖M⁻¹‖_1‖M⁻¹‖_∞)` — NOT `max(g∞,g₁)`, which bounds neither 2-norm and
can exceed σ_min. Returns `√(g₁g∞)` when both gaps are positive, else `min(g∞,g₁)`
(≤ 0, inconclusive). This is the genuinely free screen on the triangular factor;
it does NOT yet carry the transport to A.
"""
function gap_tri_schur(z, S::CertSetup)
    ginf = Inf
    g1 = Inf
    @inbounds for i in eachindex(S.eigs)
        d = abs(z - S.eigs[i])
        ginf = min(ginf, d - S.tailrow[i])
        g1 = min(g1, d - S.tailcol[i])
    end
    return (ginf > 0 && g1 > 0) ? sqrt(ginf * g1) : min(ginf, g1)
end

"""
    gap_tri(z, S::CertSetup) -> Float64

Free triangular gap transported to a lower bound on σ_min(**A** − zI):
`(σ_min(T̃−zI) − ‖S‖₂) / κ₂(X̃)` (paper §3.3, the collapse
(A−zI)⁻¹ = X̃(T̃−S−zI)⁻¹X̃⁻¹). The transport is carried honestly — `‖S‖₂` for the
reduced residual and `κ₂ = ‖Z‖₂‖Y‖₂/(1−‖R₂‖₂)` for the basis conditioning, both
computed (not assumed 1). For a normal/unitary Schur basis `‖S‖₂≈0`, `κ₂≈1`, so
this recovers `dist(z, σ)` exactly. May be negative (inconclusive).
"""
function gap_tri(z, S::CertSetup)
    return (gap_tri_schur(z, S) - S.normS_2) / S.kappa_2
end

# recentred ℓp gap values ĝ_p(z) on the collapsed matrix M₀ − zI (Thm
# `resolvent-row`, computed form): centre at diag(M0), radius the off-diagonal
# abs row/col sum, minus the residual slack β_p.
function _recentred_gap_inf(z, S::CertSetup)
    g = Inf
    @inbounds for i in eachindex(S.centres)
        g = min(g, abs(z - S.centres[i]) - S.radrow[i])
    end
    return g - S.beta_inf
end

function _recentred_gap_1(z, S::CertSetup)
    g = Inf
    @inbounds for j in eachindex(S.centres)
        g = min(g, abs(z - S.centres[j]) - S.radcol[j])
    end
    return g - S.beta_1
end

"""
    gap_recentred(z, S::CertSetup) -> Float64

Recentred per-row gap interpolated to ℓ² (Thm `resolvent-row` + `resolvent2`,
cost O(n)) — the sharpest scalar layer. Returns the certified σ_min lower bound
`sqrt(ĝ₁·ĝ∞ / (κ₁·κ∞))`, or `-Inf` if either endpoint gap is non-positive
(inconclusive — guard before the sqrt).
"""
function gap_recentred(z, S::CertSetup)
    ginf = _recentred_gap_inf(z, S)
    g1 = _recentred_gap_1(z, S)
    (ginf > 0 && g1 > 0) || return -Inf
    return sqrt(g1 * ginf / (S.kappa_1 * S.kappa_inf))
end

"""
    gap_uniform(z, S::CertSetup) -> Float64

Uniform gap (Thm `resolvent`): the recentred bound with per-row radii replaced by
their maximum — a valid but looser lower bound (`gap_uniform ≤ gap_recentred`).
Kept for the cost ladder / comparison.
"""
function gap_uniform(z, S::CertSetup)
    maxrow = maximum(S.radrow)
    maxcol = maximum(S.radcol)
    mininf = Inf
    min1 = Inf
    @inbounds for i in eachindex(S.centres)
        mininf = min(mininf, abs(z - S.centres[i]))
        min1 = min(min1, abs(z - S.centres[i]))
    end
    uinf = mininf - maxrow - S.beta_inf
    u1 = min1 - maxcol - S.beta_1
    (uinf > 0 && u1 > 0) || return -Inf
    return sqrt(u1 * uinf / (S.kappa_1 * S.kappa_inf))
end

"""
    sigma_min_lower_bound(z, S::CertSetup) -> Float64

Top-level certified lower bound on σ_min(A − zI): the best (largest) of the free
triangular and recentred-ℓ² gaps. The cluster/block path (implementation.md §5),
which handles the case where both scalar layers go non-positive near clustered
spectra, is not yet implemented.
"""
sigma_min_lower_bound(z, S::CertSetup) = max(gap_tri(z, S), gap_recentred(z, S))

"""
    sigma_min_upper(z, S::CertSetup) -> Float64

Certified UPPER bound on σ_min(A − zI) (the inclusion side): `σ_min ≤ ‖(A−zI)q‖₂/‖q‖₂`
for any `q`, minimized over the Schur basis vectors `q = Z[:,i]`. Cost O(n²) per z
from the precomputed `A·Z`. Exact (`= dist(z,σ)`) for normal matrices. When
`sigma_min_upper(z) + r < ε` on a cell of half-diagonal `r`, the whole cell is
certified INSIDE the ε-pseudospectrum.
"""
function sigma_min_upper(z, S::CertSetup)
    n = size(S.AZ, 1)
    best = Inf
    @inbounds for i in 1:n
        s = 0.0
        for k in 1:n
            s += abs2(S.AZ[k, i] - z * S.Z[k, i])
        end
        v = sqrt(s) / S.Zcolnorm[i]
        v < best && (best = v)
    end
    return best
end
