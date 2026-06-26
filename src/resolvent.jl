# Free resolvent floor from a Miyajima–Rump certificate (implementation.md §4.3 recentred
# Gershgorin + §4.4 ℓ¹/ℓ∞→ℓ² interpolation), in the EIGENVECTOR frame. Given the certified
# collapsed Ã = W⁻¹BW (≈ diagonal), the recentred Gershgorin discs give a z-local σ_min
# lower bound in BOTH norms — the per-ROW gap (ℓ∞) and per-COLUMN gap (ℓ¹):
#     g_∞(z) = minᵢ( |z − Ãᵢᵢ| − Σ_{j≠i}|Ãᵢⱼ| − β_∞ ),   (ℓ∞, row sums)
#     g_1(z) = minⱼ( |z − Ãⱼⱼ| − Σ_{i≠j}|Ãᵢⱼ| − β_1 ),   (ℓ¹, column sums)
# Riesz–Thorin interpolates them to ℓ²: σ_min(M−zI) ≥ √(g_1(z)·g_∞(z)). Transporting to A
# (the eigenvector basis W is not unitary) by the 2-norm condition number,
#     σ_min(B − zI) ≥ √( g_1(z)·g_∞(z) ) / κ₂(W).
# The z-DEPENDENCE lives in g_1(z), g_∞(z) — the bound ADAPTS to z (tight near a
# well-separated eigenvalue, loose only near genuinely coupled ones) — instead of the
# block 2-norm far-field's single global subtraction. κ₂(W) is the only global factor;
# it is the certificate's reported κ₂ (cheap Collatz by default). A non-normal/coupled
# cluster shows up directly as a large row/column off-diagonal sum (a wide recentred
# disc), so no separate block-SVD machinery is needed. NO Schur frame, NO per-block SVD.

using LinearAlgebra
using BallArithmetic

"""
    FreeResolvent

Precomputed Miyajima–Rump resolvent floor for `B` from a [`MiyajimaRump`](@ref)
certificate (see [`free_resolvent`](@ref)): the recentred disc centres `diag` (= `Ãᵢᵢ`),
the per-row (`rowsum`, ℓ∞) and per-column (`colsum`, ℓ¹) off-diagonal masses (incl. the
slacks `β_∞`, `β_1`), and the transport `kappa` (`κ₂(W)`). Evaluate with
[`sigma_min_lower`](@ref) / [`resolvent_bound`](@ref).
"""
struct FreeResolvent{T <: AbstractFloat}
    diag::Vector{Complex{T}}    # Ãᵢᵢ  (recentred disc centres = eigenvalue estimates)
    rowsum::Vector{T}           # Σ_{j≠i}|Ãᵢⱼ| + β_∞   (ℓ∞ recentred Gershgorin radius)
    colsum::Vector{T}           # Σ_{i≠j}|Ãᵢⱼ| + β_1   (ℓ¹ recentred Gershgorin radius)
    kappa::T                    # κ₂(W)  (resolvent transport)
end

# max absolute column sum (ℓ¹ operator norm) of a ball matrix from its mid/rad
function _l1_from(Am, Ar)
    n = size(Am, 2)
    s = zero(real(eltype(Am)))
    @inbounds for j in 1:n
        c = zero(s)
        for i in 1:n
            c += abs(Am[i, j]) + Ar[i, j]
        end
        s = max(s, c)
    end
    return s
end

"""
    free_resolvent(B, R::MiyajimaRump) -> FreeResolvent

Build the Miyajima–Rump row/column resolvent floor of `B` from the certificate `R`. The
collapsed `Ã = YBW` and residual `R₂ = YW−I` are formed once in ball arithmetic; the
ℓ∞ and ℓ¹ slacks `β_p = ‖R₂‖_p/(1−‖R₂‖_p)·‖Ã‖_p` and the per-row / per-column
off-diagonal sums are read off (`O(n²)`, no SVD). Reuses `R.kappa` (the transport `κ₂`).
"""
function free_resolvent(B::AbstractMatrix, R::MiyajimaRump{T}) where {T}
    CT = Complex{T}
    Bc = CT.(Matrix(B))
    W = R.W
    Y = inv(W)
    n = size(Bc, 1)
    Atil = BallMatrix(Y) * BallMatrix(Bc) * BallMatrix(W)     # rigorous Ã = YBW
    R2 = BallMatrix(Y) * BallMatrix(W) - I
    Am = mid(Atil)
    Ar = rad(Atil)
    # slacks ‖M − Ã‖_p ≤ β_p,  M = (I+R₂)⁻¹Ã
    r2inf = upper_bound_L_inf_opnorm(R2)
    βinf = r2inf < 1 ? r2inf / (1 - r2inf) * upper_bound_L_inf_opnorm(Atil) : T(Inf)
    r2one = _l1_from(mid(R2), rad(R2))
    βone = r2one < 1 ? r2one / (1 - r2one) * _l1_from(Am, Ar) : T(Inf)

    diag = [Am[i, i] for i in 1:n]
    rowsum = zeros(T, n)
    colsum = zeros(T, n)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        m = abs(Am[i, j]) + Ar[i, j]
        rowsum[i] += m       # ℓ∞: off-diagonal of row i
        colsum[j] += m       # ℓ¹: off-diagonal of column j
    end
    @inbounds for i in 1:n
        rowsum[i] += Ar[i, i] + βinf    # diagonal rounding + ℓ∞ slack
        colsum[i] += Ar[i, i] + βone    #                    + ℓ¹ slack
    end
    return FreeResolvent(diag, rowsum, colsum, R.kappa)
end

"""
    row_col_gaps(fr::FreeResolvent, z) -> (g_∞, g_1)

The per-`z` recentred Gershgorin gaps: `g_∞` (binding row, ℓ∞) and `g_1` (binding
column, ℓ¹). Either `≤ 0` ⇒ `z` is inside a recentred disc (the local grey region).
"""
function row_col_gaps(fr::FreeResolvent, z)
    g∞ = minimum(abs(z - fr.diag[i]) - fr.rowsum[i] for i in eachindex(fr.diag))
    g1 = minimum(abs(z - fr.diag[j]) - fr.colsum[j] for j in eachindex(fr.diag))
    return g∞, g1
end

"""
    sigma_min_lower(fr::FreeResolvent, z) -> Float64

Certified lower bound on `σ_min(B − zI)` via the interpolated row/column Gershgorin gaps:
`√(g_1(z)·g_∞(z))/κ₂`, or `0` where either gap is non-positive (`z` inside a recentred
disc — the `z`-local grey region).
"""
function sigma_min_lower(fr::FreeResolvent, z)
    g∞, g1 = row_col_gaps(fr, z)
    (g∞ > 0 && g1 > 0) || return zero(real(eltype(fr.diag)))
    return sqrt(g1 * g∞) / fr.kappa
end

"""
    resolvent_bound(fr::FreeResolvent, z) -> certified upper bound on `‖(B − zI)⁻¹‖₂`
"""
function resolvent_bound(fr::FreeResolvent, z)
    s = sigma_min_lower(fr, z)
    return s > 0 ? inv(s) : oftype(s, Inf)
end

(fr::FreeResolvent)(z) = sigma_min_lower(fr, z)
