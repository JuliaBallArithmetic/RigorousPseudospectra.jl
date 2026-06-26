# Cluster / block path (implementation.md §5, paper §4 "Block enclosures").
#
# When approximate eigenvalues cluster, the scalar gaps go non-positive and the
# cluster must be treated as a unit. The path:
#   1. reorder the Schur form (ordschur) so the cluster leads, and RE-CERTIFY the
#      reordered basis by residuals (`_certify_from_basis`) — the reorder is unitary
#      only approximately, so its R₂/S/β/κ are recomputed, not assumed;
#   2. partition the collapsed matrix M₀ = Z⁻¹AZ = (T̃ − S) into cluster C (leading
#      m×m) and far F blocks, M₀ − zI = [[P, Q],[R, W]];
#   3. bound the far block W cheaply (triangular Gershgorin, carrying β);
#   4. bound the small cluster block P by a certified σ_min (SVD-frame for m≥3,
#      closed form for m≤2);
#   5. assemble (Thm `assembly`), carrying κ₂(X̃) and the lower-left block R = −S_FC
#      via the Schur-complement correction (Remark `schurcomp`).
#
# FLOAT64 first cut; residuals are tiny here but computed and carried in the same
# structure, so the BallArithmetic version is a drop-in.

"""
    ClusterBlock

Re-certified, reordered setup for one cluster: `rs` is the residual record of the
reordered basis, `m` the cluster size (leading block of `rs.M0`), and the rest is
the z-independent partition data — coupling norms `‖Q‖₂`/`‖R‖₂`, far-block
Gershgorin centres/radii, and the reusable SVD-frame of the cluster block.
"""
struct ClusterBlock{M}
    rs::CertSetup{M}
    m::Int
    idx::Vector{Int}
    normQ::Float64               # ‖M₀_CF‖₂ (cluster→far coupling)
    normR::Float64               # ‖M₀_FC‖₂ = ‖S_FC‖₂ (dropped lower-left block)
    far_centres::Vector{ComplexF64}
    far_radrow::Vector{Float64}
    far_radcol::Vector{Float64}
    Mcc::M                       # cluster block M₀_CC (the m×m hard object)
end

# exact σ_min of a 2×2 (complex) matrix, cancellation-free Kahan form:
# σ_min = |det M| / σ_max, with σ_max from ‖M‖_F² and |det M| (no √-of-difference).
function _sigma_min_2x2(M)
    d = abs(M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
    f = abs2(M[1, 1]) + abs2(M[1, 2]) + abs2(M[2, 1]) + abs2(M[2, 2])  # ‖M‖_F²
    disc = sqrt(max(0.0, f * f - 4 * d * d))
    σmax = sqrt((f + disc) / 2)
    return σmax == 0 ? 0.0 : d / σmax
end

"""
    reorder_cluster(S::CertSetup, idx) -> ClusterBlock

Reorder the Schur form so the indices `idx` (a coupling-graph component, §4b) lead,
re-certify the reordered basis, and build the partition data. `idx` is realized as
the `ordschur` `select` mask.
"""
function reorder_cluster(S::CertSetup, idx)
    n = size(S.A, 1)
    cidx = collect(idx)
    select = falses(n)
    select[cidx] .= true
    Fr = ordschur(schur(S.A), select)
    rs = _certify_from_basis(S.A, Matrix(Fr.Z), Matrix(Fr.T))
    m = count(select)

    M0 = rs.M0
    Mcc = M0[1:m, 1:m]
    if m < n
        Mcf = M0[1:m, (m + 1):n]
        Mfc = M0[(m + 1):n, 1:m]
        Mff = M0[(m + 1):n, (m + 1):n]
        normQ = opnorm(Mcf, 2)
        normR = opnorm(Mfc, 2)
        far_centres = diag(Mff)
        far_radrow, far_radcol = _offdiag_sums(Mff)
    else
        normQ = 0.0
        normR = 0.0
        far_centres = ComplexF64[]
        far_radrow = Float64[]
        far_radcol = Float64[]
    end
    return ClusterBlock{typeof(M0)}(rs, m, cidx, normQ, normR,
        far_centres, far_radrow, far_radcol, Mcc)
end

"""
    sigma_min_block(z, cb::ClusterBlock) -> Float64

Certified lower bound on σ_min of the cluster block `P = M₀_CC − zI` — the lone
genuinely hard ℓ² object, now confined to the `m×m` cluster: `abs` for `m=1`, the
cancellation-free 2×2 closed form for `m=2`, and a small **verified SVD** for
`m≥3`. A tight per-z SVD is used here (not the SVD-frame surface bound, which
overshoots precisely near the cluster spectrum where this is evaluated). Float64
cut: `svdvals(P)[end]`; the BallArithmetic drop-in is a verified σ_min on the
`m×m` `BallMatrix` (e.g. `svdbox` / `rigorous_svd` / `oishi_2023_sigma_min_bound`).
"""
function sigma_min_block(z, cb::ClusterBlock)
    m = cb.m
    if m == 1
        return abs(cb.Mcc[1, 1] - z)
    elseif m == 2
        return _sigma_min_2x2(cb.Mcc - z * I)
    else
        return svdvals(cb.Mcc - z * I)[end]
    end
end

# far-block ℓ∞ and ℓ¹ Gershgorin gaps on M₀_FF − zI, carrying the β slack.
function _far_gaps(z, cb::ClusterBlock)
    ginf = Inf
    g1 = Inf
    @inbounds for j in eachindex(cb.far_centres)
        d = abs(z - cb.far_centres[j])
        ginf = min(ginf, d - cb.far_radrow[j])
        g1 = min(g1, d - cb.far_radcol[j])
    end
    return ginf - cb.rs.beta_inf, g1 - cb.rs.beta_1
end

"""
    far_gap(z, cb::ClusterBlock) -> Float64

The ℓ∞ far gap (Lemma `fargap`): a lower bound on σ_min of the far block, used by
the localization check. Negative ⇒ the far block is not certified nonsingular.
"""
far_gap(z, cb::ClusterBlock) = first(_far_gaps(z, cb))

"""
    localize(z, cb::ClusterBlock) -> Bool

Certified localization (Lemma `localize`): `(‖S‖_∞ + β_∞)/g_F(z) < 1`, i.e. the
dominant eigenvector component provably sits in the cluster. `gap_cluster`'s
positivity discharges this automatically; provided for diagnostics.
"""
function localize(z, cb::ClusterBlock)
    gF = far_gap(z, cb)
    gF > 0 || return false
    normS_inf = opnorm(cb.rs.T - cb.rs.M0, Inf)
    return (normS_inf + cb.rs.beta_inf) / gF < 1
end

"""
    gap_cluster(z, cb::ClusterBlock) -> Float64

Certified lower bound on σ_min(A − zI) by block-triangular assembly (Thm
`assembly`), carrying every residual:

    σ_min(A − zI) ≥ 1 / ( κ₂(X̃) · [ (1 + ‖Q‖₂‖W⁻¹‖₂) ‖P_SC⁻¹‖₂ + ‖W⁻¹‖₂ ] )

with `‖P_SC⁻¹‖₂ = 1/σ_min(P_SC)`, `σ_min(P_SC) ≥ σ_min(P) − ‖Q‖₂‖W⁻¹‖₂‖R‖₂`
(the Schur-complement correction for the lower-left block `R = −S_FC`), and
`‖W⁻¹‖₂ ≤ 1/√(g^W_1 g^W_∞)` from the far Gershgorin gaps. Returns `-Inf` when the
far block or cluster block is not certified nonsingular (inconclusive — escalate
the clustering threshold, §4b).
"""
function gap_cluster(z, cb::ClusterBlock)
    n = size(cb.rs.M0, 1)
    κ2 = cb.rs.kappa_2
    σP = sigma_min_block(z, cb)
    σP > 0 || return -Inf
    if cb.m == n
        return σP / κ2                         # no far block
    end
    ginf, g1 = _far_gaps(z, cb)
    (ginf > 0 && g1 > 0) || return -Inf
    Winv = 1 / sqrt(ginf * g1)
    σPsc = σP - cb.normQ * Winv * cb.normR     # Schur-complement correction
    σPsc > 0 || return -Inf
    Pinv = 1 / σPsc
    resbound = (1 + cb.normQ * Winv) * Pinv + Winv
    return 1 / (κ2 * resbound)
end

# -- Encircling-curve certificate ------------------------------------------------
#
# An eigenvalue λ is a POLE of the resolvent: σ_min(A−zI) → 0 as z → λ. Hence any
# cell of positive radius r reaching λ contains a point with σ_min = 0, and the
# Lipschitz cell certificate σ_min(z) ≥ σ_lo(z₀) − r is necessarily ≤ 0 there — no
# positive lower bound on σ_min is certifiable at interior points near a cluster.
# The only thing one CAN do is bound σ_min from below on an ENCIRCLING CURVE that
# stays pole-free (between the cluster and the far set), using the small certified
# SVD (gap_cluster). A positive on-curve bound encloses the cluster: the resolvent
# is controlled on the curve, so the cluster's eigenvalues / its ε-pseudospectral
# component sit inside it.

"""
    curve_lower_bound(f, center, radius; n = 64) -> (bound, zworst)

Rigorous lower bound on σ_min(A − zI) over the ENTIRE circle `|z − center| = radius`
from `n` samples and the 1-Lipschitz cell certificate (Lemma `lipschitz`):
`σ_min(z) ≥ min_i f(z_i) − r_arc`, where `r_arc = 2·radius·sin(π/2n)` is the
greatest distance from any curve point to its nearest sample. `f` is any certified
σ_min lower bound (e.g. `z -> gap_cluster(z, cb)`). A positive return value
certifies the circle is pole-free.
"""
function curve_lower_bound(f, center, radius::Real; n::Integer = 64)
    r_arc = 2 * radius * sinpi(1 / (2n))
    best = Inf
    zbest = complex(float(center))
    for k in 0:(n - 1)
        z = center + radius * cispi(2k / n)
        v = f(z)
        if v < best
            best = v
            zbest = z
        end
    end
    return best - r_arc, zbest
end

"""
    cluster_center(cb::ClusterBlock) -> ComplexF64

Centroid of the cluster's approximate eigenvalues (diagonal of the cluster block) —
a natural centre for an encircling curve.
"""
cluster_center(cb::ClusterBlock) = sum(diag(cb.Mcc)) / cb.m

"""
    encircle_cluster(cb::ClusterBlock, center, radius; n = 64) -> (bound, zworst)

Rigorous lower bound on σ_min(A − zI) over a circle encircling the cluster, using
the small certified SVD through `gap_cluster`. Pick `radius` between the cluster
spread and the far gap so the circle is pole-free; `bound > 0` then certifies the
enclosure, and `bound > ε` places the cluster's ε-pseudospectral component strictly
inside the circle.
"""
encircle_cluster(cb::ClusterBlock, center, radius; n::Integer = 64) =
    curve_lower_bound(z -> gap_cluster(z, cb), center, radius; n)
