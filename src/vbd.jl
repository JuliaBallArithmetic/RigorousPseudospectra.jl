# Verified block-diagonalization (VBD) resolvent — the guarded-trevc / Miyajima-VBD
# architecture (algorithm_spec.md §5). The certificate (from `miyajima_rump(B; frame=:schur)`)
# validates B ≈ W·blockdiag(P_j)·W⁻¹ with W block-orthonormal (Schur within each kept
# cluster), so the collapsed Ã = W⁻¹BW is block-diagonal + small inter-block coupling N.
# Block σ_min by Weyl perturbation: A = blockdiag(P_j) + N (P_j = Ã[Cⱼ,Cⱼ], N = off-block),
#   σ_min(blockdiag − zI) = minⱼ σ_min(P_j − zI)  (MᴴM block-diagonal decouples exactly),
#   |σ_min(A−zI) − σ_min(blockdiag−zI)| ≤ ‖N‖₂      (Weyl, singular values 1-Lipschitz),
# transported to B by κ₂ = ‖W‖₂‖W⁻¹‖₂ (benign in the Schur frame):
#   σ_min(B − zI)  ≥  ( minⱼ σ_min(P_j − zI) − ‖N‖₂ ) / κ₂.
# Per block, σ_min(P_j − zI) is the FAR-FIELD floor |z − cⱼ| − ‖Nⱼ‖₂ (z-uniform, cheap,
# `near=false`) or, inside the block's disc, the per-z verified SVD of P_j − zI
# (`near=true`, tight). The intra-cluster non-normality lives in ‖Nⱼ‖ / the block SVD —
# NOT in κ₂ — which is the whole point of the VBD frame (cf. the eigenvector frame, where
# it inflated κ₂ non-monotonically with precision).

using LinearAlgebra
using BallArithmetic

"""
    vbd_solve(B; sep = -1, maxsteps = 6) -> (W, clusters)

Verified block-diagonalizing frame for `B`: the block-orthonormal Schur+Newton basis `W`
(orthonormal within each spectral cluster ⇒ `κ(W)` governed by benign inter-block angles)
and its `clusters` (index sets). THIN WRAPPER over `BallArithmetic.schur_newton_vbd` — the VBD
algorithm (Schur, distance+column-norm clustering, between-block Newton, per-block QR) lives in
BallArithmetic; this returns just `(W, clusters)` for the manual certify→resolvent workflow.
Certify with `miyajima_rump(B, W)`; bound the resolvent with [`vbd_resolvent`](@ref) (or build
it directly from the `SchurNewtonVBDResult`).
"""
function vbd_solve(B::AbstractMatrix; sep::Real = -1, maxsteps::Integer = 6)
    CT = _cfloat(B)
    res = BallArithmetic.schur_newton_vbd(BallMatrix(CT.(Matrix(B))); sep, maxsteps)
    return mid(res.basis), [collect(c) for c in res.clusters]
end

"""
    VBDResolvent

Block-σ_min resolvent floor (see [`vbd_resolvent`](@ref)): per cluster the barycentre
`centres`, the within-block non-normality `blocknn` (`‖Nⱼ‖₂`) and the block ball
(`Pmid`,`Prad`) for the per-`z` SVD; the GLOBAL inter-block coupling `coupling` (`‖N‖₂`,
the off-block of `Ã`) and the single Miyajima slack `beta` (`β₂ = ‖R₂‖₂‖Ã‖₂/(1−‖R₂‖₂)`,
the `Y`-vs-`W⁻¹` correction `‖M − Ã‖₂`); the PER-BLOCK coupling radii `blockrad`
(`rⱼ = ‖N[Cⱼ,:]‖₂ + β₂ ≥ ‖(M−Λ)[Cⱼ,:]‖₂`, the off-block-row mass of block `j`); and the
transport `kappa` (`κ₂`). Evaluate with [`sigma_min_lower`](@ref) / [`resolvent_bound`](@ref).

The block radii give the Feingold–Varga block-Gershgorin localization: each block is
charged only its *own* off-block-row coupling `rⱼ` instead of the global `‖N‖₂`, so the
many-small-blocks regime (e.g. the GKW singleton tail) is no longer dragged down by the
worst block. See [`block_enclosure`](@ref) for the resulting σ(B) discs.
"""
struct VBDResolvent{T <: AbstractFloat}
    centres::Vector{Complex{T}}
    blocknn::Vector{T}
    Pmid::Vector{Matrix{Complex{T}}}
    Prad::Vector{Matrix{T}}
    coupling::T
    beta::T
    blockrad::Vector{T}
    kappa::T
end

# certified σ_min lower bound of a ball matrix (verified SVD; svdbox sorted descending)
function _sigmin_lower(M::BallMatrix)
    sv = BallArithmetic.svdbox(M)
    return minimum(mid(s) - rad(s) for s in sv)
end

"""
    vbd_resolvent(B, R::MiyajimaRump; tight = false) -> VBDResolvent

Build the block-σ_min Weyl resolvent floor from the certificate `R` (best with
`R = miyajima_rump(B; frame=:schur, refine=…)`). The collapsed `Ã = YBW`, the per-block
non-normalities `‖Nⱼ‖₂`, the global inter-block coupling `‖N‖₂`, the Miyajima slack
`β₂ = ‖R₂‖₂‖Ã‖₂/(1−‖R₂‖₂)` (the `Y`-vs-`W⁻¹` correction) and the per-block off-block-row
radii `rⱼ = ‖N[Cⱼ,:]‖₂ + β₂` are formed once (`tight` selects svdbox vs cheap Collatz for
those norms; the per-`z` block SVD is always verified).

The deviation identity (draft eq. blockdev) `M − Λ = Ñ − (I+R₂)⁻¹R₂Ã` (with `Ñ` the
off-block of `Ã`, `Λ` its block diagonal, `M = W⁻¹BW = (I+R₂)⁻¹Ã`) gives, per block,
`‖(M−Λ)[Cⱼ,:]‖₂ ≤ ‖Ñ[Cⱼ,:]‖₂ + β₂ = rⱼ`: the off-block coupling is LOCAL to the block,
only the small `β₂` correction stays global. `κ₂(W)` is taken from the certificate
(`R.kappa`, the explicit residual bound `‖W‖₂‖Y‖₂/(1−‖R₂‖₂)`, or svdbox if so certified) —
no extra factorization here.
"""
function vbd_resolvent(B::AbstractMatrix, R::MiyajimaRump{T};
        blocks::Vector{Vector{Int}} = [c.idx for c in R.clusters],
        tight::Bool = false) where {T}
    CT = Complex{T}
    Bc = CT.(Matrix(B))
    W = R.W
    Y = inv(W)
    n = size(Bc, 1)
    Wb = BallMatrix(W)
    Yb = BallMatrix(Y)
    Atil = Yb * BallMatrix(Bc) * Wb
    Am = mid(Atil)
    Ar = rad(Atil)
    opn = tight ? BallArithmetic.svd_bound_L2_opnorm : upper_bound_L2_opnorm
    # β₂ = ‖R₂‖₂‖Ã‖₂/(1−‖R₂‖₂) — the single Miyajima slack bounding ‖M − Ã‖₂, the only
    # part of the deviation that is NOT captured by the off-block of the ball Ã (Ã encloses
    # rounding, but Y≠W⁻¹ is a systematic error charged here, exactly as in `_certify`).
    nR2 = opn(Yb * Wb - I)
    beta = nR2 < 1 ? nR2 * opn(Atil) / (1 - nR2) : T(Inf)
    centres = Complex{T}[]
    blocknn = T[]
    Pmid = Matrix{CT}[]
    Prad = Matrix{T}[]
    inblock = falses(n, n)
    # `blocks` is the partition used for the block-σ_min split — default the certificate's
    # disc-overlap clusters, but pass the `vbd_solve` clusters to fix the block structure
    # at the chosen separation level rather than re-inferring it.
    for idx in blocks
        cj = sum(Am[i, i] for i in idx) / length(idx)         # block barycentre
        pm = Matrix{CT}(Am[idx, idx])
        pr = Matrix{T}(Ar[idx, idx])
        push!(centres, cj)
        push!(Pmid, pm)
        push!(Prad, pr)
        push!(blocknn, opn(BallMatrix(pm - cj * I, pr)))      # ‖Nⱼ‖₂ = ‖P_j − cⱼI‖₂
        for j in idx, i in idx

            inblock[i, j] = true
        end
    end
    Em = copy(Am)
    Erad = copy(Ar)
    @inbounds for j in 1:n, i in 1:n

        if inblock[i, j]
            Em[i, j] = zero(CT)
            Erad[i, j] = zero(T)
        end
    end
    coupling = opn(BallMatrix(Em, Erad))                       # global ‖N‖₂ inter-block
    # per-block off-block-row radii rⱼ = ‖Ñ[Cⱼ,:]‖₂ + β₂ (Feingold–Varga block-Gershgorin):
    # the spectral norm of the block ROWS of the off-block matrix, plus the global slack.
    blockrad = T[opn(BallMatrix(Em[idx, :], Erad[idx, :])) + beta for idx in blocks]
    return VBDResolvent(centres, blocknn, Pmid, Prad, coupling, beta, blockrad, R.kappa)
end

"""
    vbd_resolvent(res::SchurNewtonVBDResult) -> VBDResolvent

Build the block-σ_min resolvent floor directly from a BallArithmetic verified block
diagonalization (`schur_newton_vbd`). The VBD itself — the block-orthonormal Schur+Newton
frame, the Miyajima certification, and the per-block coupling/centre/non-normality data —
lives in BallArithmetic; RigorousPseudospectra only repackages it for the resolvent bound
(no recomputation). Every field maps straight across: `block_centers`→centres,
`block_nonnormality`→‖Nⱼ‖₂, `block_coupling`→rⱼ, `remainder_norm`→‖N‖₂, `beta`, `kappa`; the
per-block `Pⱼ` is the diagonal block of `block_diagonal`.
"""
function vbd_resolvent(res::SchurNewtonVBDResult)
    bd = res.block_diagonal
    Am = mid(bd)
    Ar = rad(bd)
    CT = eltype(Am)
    Tr = real(CT)
    Pmid = [Matrix{CT}(Am[c, c]) for c in res.clusters]
    Prad = [Matrix{Tr}(Ar[c, c]) for c in res.clusters]
    return VBDResolvent(collect(res.block_centers), collect(res.block_nonnormality),
        Pmid, Prad, res.remainder_norm, res.beta, collect(res.block_coupling), res.kappa)
end

"""
    sigma_min_lower(vr::VBDResolvent, z; near = false) -> Float64

Certified lower bound on `σ_min(B − zI)`, the maximum of two rigorous floors (both
transported by `κ₂`), where `sⱼ = σ_min(Pⱼ − zI)` is `near = false` (default): the cheap
z-uniform far-field `|z−cⱼ| − ‖Nⱼ‖₂`; or `near = true`: the per-`z` verified SVD of
`Pⱼ − zI` (tight inside a cluster's disc, where the far-field floor goes non-positive):

  * **global Weyl** — `(minⱼ sⱼ − ‖N‖₂ − β₂)/κ₂` (`‖M−Λ‖₂ ≤ ‖N‖₂ + β₂`);
  * **block dominance** — `minⱼ sⱼ · (1 − √Σⱼ(rⱼ/sⱼ)²)/κ₂` when `Σⱼ(rⱼ/sⱼ)² < 1`
    (factor `M−zI = (Λ−zI)(I + (Λ−zI)⁻¹(M−Λ))`; `‖(Λ−zI)⁻¹(M−Λ)‖₂ ≤ √Σⱼ(rⱼ/sⱼ)²` since
    its block rows are `(Pⱼ−zI)⁻¹(M−Λ)[Cⱼ,:]`, of norm `≤ rⱼ/sⱼ`). The block floor uses
    each block's OWN coupling `rⱼ`, so it dominates wherever the coupling is uneven across
    blocks (the localized win); the global floor wins when one block's `sⱼ → 0`.
"""
function sigma_min_lower(vr::VBDResolvent{T}, z; near::Bool = false) where {T}
    p = length(vr.centres)
    s = Vector{T}(undef, p)
    @inbounds for j in 1:p
        s[j] = near ? _sigmin_lower(BallMatrix(vr.Pmid[j] - z * I, vr.Prad[j])) :
               abs(z - vr.centres[j]) - vr.blocknn[j]
    end
    gmin = minimum(s)
    fglob = (gmin - vr.coupling - vr.beta) / vr.kappa          # global Weyl floor
    fdom = typemin(T)                                          # block-dominance floor
    if all(>(zero(T)), s)
        τ2 = sum((vr.blockrad[j] / s[j])^2 for j in 1:p)       # ‖(Λ−zI)⁻¹(M−Λ)‖₂² bound
        τ2 < 1 && (fdom = gmin * (1 - sqrt(τ2)) / vr.kappa)
    end
    return max(zero(T), fglob, fdom)
end

"""
    block_enclosure(vr::VBDResolvent) -> Vector{NamedTuple}

The block-Gershgorin (Feingold–Varga) enclosure of `σ(B)`: one disc per block,
`(center = cⱼ, radius = ‖Nⱼ‖₂ + rⱼ, mult = mⱼ)`. Since `σ_min(Pⱼ − zI) ≥ |z−cⱼ| − ‖Nⱼ‖₂`,
the block-Gershgorin region `{z : σ_min(Pⱼ − zI) ≤ rⱼ}` (which holds exactly `mⱼ`
eigenvalues of `Ã`, hence of `B`) is contained in the disc of radius `‖Nⱼ‖₂ + rⱼ` about the
barycentre `cⱼ`. A connected component of `k` overlapping discs holds the summed
multiplicities. This is sharper than the scalar Gershgorin discs of [`miyajima_rump`](@ref)
wherever the off-block coupling is concentrated (the radius is the block's OWN `rⱼ`, not the
global slack); the within-block spread stays in `‖Nⱼ‖₂` rather than inflating every disc.
"""
function block_enclosure(vr::VBDResolvent{T}) where {T}
    return [(center = vr.centres[j], radius = vr.blocknn[j] + vr.blockrad[j],
                mult = size(vr.Pmid[j], 1)) for j in eachindex(vr.centres)]
end

"""
    resolvent_bound(vr::VBDResolvent, z; near = false) -> upper bound on `‖(B − zI)⁻¹‖₂`
"""
function resolvent_bound(vr::VBDResolvent, z; near::Bool = false)
    s = sigma_min_lower(vr, z; near)
    return s > 0 ? inv(s) : oftype(s, Inf)
end
