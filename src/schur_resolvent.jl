# Schur-frame resolvent floor — the z-local, κ≈1 alternative to the eigenvector-frame
# `free_resolvent` (draft §resolvent; implementation.md §4.1/§4.4 gap ladder, revived in
# the unitary frame). Instead of transporting σ_min(M−zI) by a GLOBAL κ₂(W) (the
# eigenvector-basis conditioning — large and z-uniform), we use the UNITARY Schur basis
# A = ZTZ': σ_min(A−zI) = σ_min(Z⁻¹(A−zI)Z) up to κ₂(Z) ≈ 1, and bound σ_min of the
# (near-triangular) collapsed Ã = Z⁻¹AZ by the per-row (ℓ∞) and per-column (ℓ¹)
# diagonal-dominance gaps, INTERPOLATED to ℓ² (Riesz–Thorin):
#     σ_min(A−zI) ≥ √( g₁(z)·g_∞(z) ) / κ₂(Z),
#     g_∞(z) = minᵢ(|z−Ãᵢᵢ| − Σ_{j≠i}|Ãᵢⱼ| − β),   g₁(z) = minⱼ(|z−Ãⱼⱼ| − Σ_{i≠j}|Ãᵢⱼ| − β).
# Both gaps are z-LOCAL (the binding row/column changes with z), so the bound adapts to z
# instead of paying the worst-case κ₂ everywhere. And κ₂(Z) is ≈1 WITHOUT an SVD: Z is
# near-unitary, so Z'Z = I+R₂ with ‖R₂‖ tiny ⇒ κ₂(Z) ≤ √((1+‖R₂‖₂)/(1−‖R₂‖₂)). The grey
# region is then ⋃ᵢ disc(Ãᵢᵢ, tailᵢ+β) — a LOCAL exclusion around each Schur eigenvalue
# set by that row/column's tail, not a global κ(W)-radius.
#
# Trade-off: the triangular tails Σ|Ãᵢⱼ| are the resolvent looseness here; they are O(1)
# for a globally non-normal matrix (Grcar) but small for mildly non-normal / transfer
# operators. The recentred (S(z)) sharpening of implementation.md §4.3 tightens them where
# a raw tail is loose. Generic in the float type; BigFloat `schur` needs GenericSchur or
# GenericLinearAlgebra loaded (Float64 fallback otherwise).

using LinearAlgebra
using BallArithmetic

"""
    SchurResolvent

Precomputed Schur-frame resolvent floor (see [`schur_resolvent`](@ref)): the Schur
diagonal `diag` (= eigenvalues), the per-row (`rowtail`, ℓ∞) and per-column (`coltail`,
ℓ¹) off-diagonal masses (incl. the slack `β`), and `kappaZ` = κ₂(Z) ≈ 1. Evaluate with
[`sigma_min_lower`](@ref) / [`resolvent_bound`](@ref).
"""
struct SchurResolvent{T <: AbstractFloat}
    diag::Vector{Complex{T}}
    rowtail::Vector{T}
    coltail::Vector{T}
    kappaZ::T
    nrmR2::T
    beta::T
end

"""
    schur_resolvent(B) -> SchurResolvent

Build the Schur-frame `z`-local resolvent floor of `B`. One Schur factorization
`A = ZTZ'` (the candidate; Float64 or, in extended precision, GenericSchur), then ONE
Miyajima-style certification with `Y = Z'` (the near-inverse of the near-unitary `Z`):
`R₂ = Z'Z − I`, `Ã = Z'BZ`, slack `β`. No SVD anywhere — `κ₂(Z)` comes from `‖R₂‖₂`.
"""
function schur_resolvent(B::AbstractMatrix)
    CT = _cfloat(B)
    Bc = CT.(Matrix(B))
    n = size(Bc, 1)
    Z = try
        Matrix(schur(Bc).Z)
    catch
        CT.(Matrix(schur(ComplexF64.(Matrix(B))).Z))
    end
    BB = BallMatrix(Bc)
    ZB = BallMatrix(Z)
    YB = BallMatrix(Matrix(Z'))                # Y = Z' (Z near-unitary ⇒ Z' ≈ Z⁻¹)
    R2 = YB * ZB - I
    nrmR2 = upper_bound_L_inf_opnorm(R2)
    nrmR2 < 1 ||
        error("Schur basis not certifiably nonsingular: ‖R₂‖_∞ = $nrmR2 ≥ 1")
    A = YB * BB * ZB                           # rigorous Z⁻¹BZ proxy (≈ upper-triangular)
    normA = upper_bound_L_inf_opnorm(A)
    beta = nrmR2 * normA / (1 - nrmR2)
    Ac = mid(A)
    Ar = rad(A)
    T = real(CT)
    dg = [Ac[i, i] for i in 1:n]
    rowtail = zeros(T, n)
    coltail = zeros(T, n)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        m = abs(Ac[i, j]) + Ar[i, j]
        rowtail[i] += m
        coltail[j] += m
    end
    @inbounds for i in 1:n
        rowtail[i] += Ar[i, i] + beta          # diagonal rounding + Y-vs-Z⁻¹ slack
        coltail[i] += Ar[i, i] + beta
    end
    r2₂ = upper_bound_L2_opnorm(R2)
    kappaZ = sqrt((1 + r2₂) / (1 - r2₂))       # κ₂(Z) ≈ 1, from the unitarity residual
    return SchurResolvent(dg, rowtail, coltail, kappaZ, nrmR2, beta)
end

"""
    sigma_min_lower(sr::SchurResolvent, z) -> Float64

Certified lower bound on `σ_min(B − zI)` via the interpolated row/column gaps:
`√(g₁(z)·g_∞(z))/κ₂(Z)`, or `0` where either gap is non-positive (`z` inside a Schur
diagonal's tail disc — the `z`-local grey region).
"""
function sigma_min_lower(sr::SchurResolvent, z)
    g∞ = minimum(abs(z - sr.diag[i]) - sr.rowtail[i] for i in eachindex(sr.diag))
    g1 = minimum(abs(z - sr.diag[j]) - sr.coltail[j] for j in eachindex(sr.diag))
    (g∞ > 0 && g1 > 0) || return zero(real(eltype(sr.diag)))
    return sqrt(g1 * g∞) / sr.kappaZ
end

"""
    resolvent_bound(sr::SchurResolvent, z) -> certified upper bound on ‖(B − zI)⁻¹‖₂
"""
function resolvent_bound(sr::SchurResolvent, z)
    s = sigma_min_lower(sr, z)
    return s > 0 ? inv(s) : oftype(s, Inf)
end

(sr::SchurResolvent)(z) = sigma_min_lower(sr, z)
