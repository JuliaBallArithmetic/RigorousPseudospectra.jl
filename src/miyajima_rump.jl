# Miyajima–Rump certified eigenvalue enclosure (implementation.md §3, §4c–4e).
#
# CERTIFY a candidate eigenvector basis W of B:
#   1. transform to the almost-diagonal collapsed matrix  Ã = W⁻¹ B W = D + E;
#   2. ONE Miyajima two-residual certification. Y is an APPROXIMATE inverse of W,
#      Y ≈ W⁻¹, in floating point — the exact inverse W⁻¹ is NEVER formed. The
#      certificate is valid for ANY Y, its quality entering only through the single
#      residual R₂ = YW − I ("Y is free", Miyajima Rem.): ‖R₂‖ < 1 proves W (hence
#      Y) nonsingular and bounds ‖W⁻¹‖ ≤ ‖Y‖/(1−‖R₂‖), with no inverse computed.
#   3. recentred Gershgorin discs of Ã, inflated by the certification slack
#      β = ‖R₂ Ã‖/(1−‖R₂‖), rigorously enclose σ(B): centre dᵢ = Ãᵢᵢ, radius
#      Σ_{j≠i}|Ãᵢⱼ| + β. NOTE the enclosure depends ONLY on the basis W — the
#      candidate eigenvalues "D" drop out (β is independent of them, and the centres
#      are read off as diag(Ã)); so the certifier takes W, not (W, D).
#   4. separation clustering: connected components of OVERLAPPING discs — a
#      connected component of k discs holds exactly k eigenvalues (Gershgorin).
#
# REFINE a candidate basis (§4e): float Newton steps shrink the off-diagonal E by
# the reciprocal-distance (Sylvester) correction, neutralized WITHIN clusters (where
# eigenvalues are coincident and 1/(dᵢ−dⱼ) blows up). The refined W is then re-
# certified — tighter discs may SPLIT a cluster, coincident ones MERGE. Refinement
# is a pure float candidate producer; correctness rests solely on the final certify
# (the asymmetry of §4e), so it never needs to be itself verified.
#
# RIGOROUS: candidate production (eigen, inv, refinement) is Float64, but the
# CERTIFICATION runs in midpoint-radius ball arithmetic (BallArithmetic.jl), so the
# discs account for every floating-point rounding error in the products YW, YBW, plus
# the approximate-inverse correction (Y vs W⁻¹) via β. The pseudospectrum engine is
# quarantined in src/pseudospectra/ (see CLAUDE.md) and not loaded here.

using LinearAlgebra
using BallArithmetic

"""
    EigCluster

A certified eigenvalue cluster: the candidate indices `idx`, the recentred disc
`centres` (= `Ãᵢᵢ`) and common `radius`, and a single enclosing disc
(`hull_center`, `hull_radius`). Provably contains exactly `length(idx)` eigenvalues
of `B` (with multiplicity).
"""
struct EigCluster{T <: AbstractFloat}
    idx::Vector{Int}
    centres::Vector{Complex{T}}
    radius::T
    hull_center::Complex{T}
    hull_radius::T
end

"""
    MiyajimaRump

Result of [`miyajima_rump`](@ref). Every eigenvalue of `B` lies in
`⋃ᵢ disc(centres[i], radii[i])`; `clusters` partition the discs into connected
(jointly-enclosed) groups. `W`/`Atil` are the (refined) candidate basis and its
collapsed matrix `W⁻¹BW`; `nrmR2`, `beta`, `kappa` are the certification record
(`κ₂` bounded from the residuals, never `cond`).
"""
struct MiyajimaRump{T <: AbstractFloat}
    centres::Vector{Complex{T}}   # recentred centres dᵢ = Ãᵢᵢ
    radii::Vector{T}              # Σ_{j≠i}|Ãᵢⱼ| + β
    clusters::Vector{EigCluster{T}}
    W::Matrix{Complex{T}}
    Atil::Matrix{Complex{T}}
    nrmR2::T
    beta::T
    kappa::T
end

# common complex float element type for the candidate flow: Float64 for real/Float64
# input, Complex{BigFloat} preserved for BigFloat input (the whole pipeline is generic
# in the float type — BallArithmetic's products/svdbox/opnorms work transparently in
# BigFloat, so a high-precision candidate certifies in high precision)
_cfloat(Ms...) = Complex{mapreduce(M -> real(float(eltype(M))), promote_type, Ms)}

# total off-diagonal mass (refinement convergence monitor)
function _offdiag_norm(M)
    s = zero(real(float(eltype(M))))
    n = size(M, 1)
    @inbounds for j in 1:n, i in 1:n
        i != j && (s += abs2(M[i, j]))
    end
    return sqrt(s)
end

# off-diagonal residue of the collapsed W⁻¹BW (the quantity refinement shrinks) — the
# stall monitor for the :auto mode
_offres(Bc, Wc) = _offdiag_norm(Wc \ (Bc * Wc))

# connected components of an undirected boolean adjacency, via union–find
function _components(adj::AbstractMatrix{Bool})
    n = size(adj, 1)
    parent = collect(1:n)
    find(x) = (r = x; while parent[r] != r; r = parent[r]; end;
    while parent[x] != r; (parent[x], x) = (r, parent[x]); end; r)
    @inbounds for j in 1:n, i in 1:(j - 1)
        (adj[i, j] || adj[j, i]) && (ri = find(i); rj = find(j); ri != rj && (parent[ri] = rj))
    end
    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, find(i), Int[]), i)
    end
    comps = collect(values(groups))
    sort!(comps; by = first)
    return comps
end

# index → cluster id
function _membership(clusters, n)
    cof = zeros(Int, n)
    for (k, c) in enumerate(clusters), i in c.idx
        cof[i] = k
    end
    return cof
end

# core certification of a candidate basis W (B already complexified), in ball
# arithmetic so every product rounding is rigorously bounded. The true collapsed
# matrix is M₀ = W⁻¹BW = (I+R₂)⁻¹·(YBW); we enclose YBW rigorously as the ball Ã and
# charge the Y-vs-W⁻¹ correction ‖M₀−Ã‖_∞ ≤ β = ‖R₂‖_∞‖Ã‖_∞/(1−‖R₂‖_∞) per row.
function _certify(Bc::Matrix{Complex{T}}, W::Matrix{Complex{T}};
        kappa_mode::Symbol = :cheap) where {T <: AbstractFloat}
    n = size(Bc, 1)
    Y = inv(W)               # APPROXIMATE (floating-point) inverse; W⁻¹ never formed.
    BB = BallMatrix(Bc)
    WB = BallMatrix(W)
    YB = BallMatrix(Y)

    R2 = YB * WB - I         # single residual certifying the approximate inverse
    nrmR2 = upper_bound_L_inf_opnorm(R2)            # certified ‖R₂‖_∞
    nrmR2 < 1 ||
        error("Neumann condition ‖R₂‖_∞ = $nrmR2 ≥ 1: basis not certifiably " *
              "nonsingular (numerically defective at this precision).")

    A = YB * BB * WB                                # rigorous enclosure of Y·B·W
    normA = upper_bound_L_inf_opnorm(A)
    beta = nrmR2 * normA / (1 - nrmR2)              # ‖M₀ − Ã‖_∞ bound (D-free)

    Ac = mid(A)
    Ar = rad(A)
    centres = [Ac[i, i] for i in 1:n]
    radii = Vector{T}(undef, n)
    @inbounds for i in 1:n
        s = Ar[i, i]                                # rounding in the computed Ãᵢᵢ
        for j in 1:n
            j == i && continue
            s += abs(Ac[i, j]) + Ar[i, j]           # |M₀ᵢⱼ| ≤ |Ãᵢⱼ| (+ ball radius)
        end
        radii[i] = s + beta                         # + Y-vs-W⁻¹ correction (whole row)
    end

    # κ₂ = ‖W‖₂·‖W⁻¹‖₂ — a reported diagnostic AND the free-resolvent prefactor, but NOT
    # used in the eigenvalue enclosure (scale-invariant). The verified `svdbox(W)` is the
    # ONLY O(n³) step beyond the ball products, so `kappa_mode` trades its cost:
    #   :cheap (DEFAULT) — NO SVD: Collatz/interpolation ‖W‖₂,‖Y‖₂ (`upper_bound_L2_opnorm`,
    #     O(n²)) + residual ‖W⁻¹‖₂ ≤ ‖Y‖₂/(1−‖R₂‖₂). Overestimates by ~√n on orthonormal
    #     bases (so κ₂≈√n·… for a normal matrix, true 1), but skips the verified SVD —
    #     and κ₂ rarely needs to be tight (it is a scale-invariant diagnostic, and the
    #     resolvent small-gain typically has orders of headroom, e.g. GKW's ε~1e-45). This
    #     keeps the whole pipeline O(n³ matmul) — cheap even in BigFloat.
    #   :svdbox (opt-in, tight) — one svdbox of W: σ_max(W) = ‖W‖₂, and 1/σ_min(W) = ‖W⁻¹‖₂
    #     when svdbox certifies σ_min>0, else the residual fallback (W badly ill-conditioned,
    #     the Float64 SVD can't certify σ_min>0 — e.g. GKW cond(W)~1e13; BigFloat certifies
    #     it directly). Use when a tight κ₂ is actually needed.
    kappa = if kappa_mode === :cheap
        r2₂ = upper_bound_L2_opnorm(R2)
        r2₂ < 1 ? upper_bound_L2_opnorm(WB) * upper_bound_L2_opnorm(YB) / (1 - r2₂) : T(Inf)
    else
        sv = BallArithmetic.svdbox(WB)
        σmax = maximum(mid(s) + rad(s) for s in sv)
        σmin = minimum(mid(s) - rad(s) for s in sv)
        invW = if σmin > 0
            1 / σmin
        else
            r2₂ = upper_bound_L2_opnorm(R2)
            r2₂ < 1 ? BallArithmetic.svd_bound_L2_opnorm(YB) / (1 - r2₂) : T(Inf)
        end
        σmax * invW
    end
    # κ₂ must never escape as NaN. When the candidate basis is numerically singular (a defective
    # spectrum: cond(W) reaches 1e303 for a Jordan block, 1e177 for Demmel's counterexample), the
    # approximate inverse overflows and the product above can evaluate to NaN. The ENCLOSURE is
    # unaffected — Theorem `enclosure` reads σ_min off the diagonal blocks and never uses κ₂ — but
    # the resolvent transport of `vbd_resolvent`/`sigma_min_lower` divides by it, and a NaN would
    # silently poison every comparison (NaN < x and NaN > x are both false) instead of failing
    # loudly. Inf is the honest value: no transport bound is available, so that floor is vacuous.
    isfinite(kappa) || (kappa = T(Inf))

    adj = falses(n, n)
    @inbounds for j in 1:n, i in 1:(j - 1)
        if abs(centres[i] - centres[j]) <= radii[i] + radii[j]
            adj[i, j] = adj[j, i] = true
        end
    end
    clusters = EigCluster{T}[]
    for c in _components(adj)
        cen = centres[c]
        radc = maximum(radii[c])
        hc = sum(cen) / length(c)
        hr = maximum(abs(cen[k] - hc) + radii[c[k]] for k in eachindex(c))
        push!(clusters, EigCluster(c, cen, radc, hc, hr))
    end

    return MiyajimaRump(centres, radii, clusters, W, Ac, nrmR2, beta, kappa)
end

"""
    miyajima_rump(B, W) -> MiyajimaRump

Certify the candidate eigenvector basis `W` of `B` (Miyajima two-residual,
diagonal-residue frame). The enclosure depends only on `W`; the eigenvalue
estimates are `diag(W⁻¹BW)`. Throws if `W` fails the Neumann condition `‖R₂‖_∞ < 1`.
"""
function miyajima_rump(B::AbstractMatrix, W::AbstractMatrix; kappa_mode::Symbol = :cheap)
    CT = _cfloat(B, W)
    return _certify(CT.(Matrix(B)), CT.(Matrix(W)); kappa_mode)
end

# one Newton step from the already-formed Ã = W⁻¹BW (d = diag Ã): cancel the
# off-diagonal where `cof` says i,j are separable, Xᵢⱼ = −Ãᵢⱼ/(dᵢ−dⱼ) (the minus
# cancels: Ã' = (I−X)Ã(I+X) ≈ Ã + (dᵢ−dⱼ)Xᵢⱼ), and W ← W(I+X). Pairs sharing a
# working cluster (cof) are neutralized.
function _newton_from(A, d, Wc, cof)
    n = size(A, 1)
    X = zeros(eltype(A), n, n)
    moved = false
    @inbounds for j in 1:n, i in 1:n
        if i != j && cof[i] != cof[j]
            X[i, j] = -A[i, j] / (d[i] - d[j])
            moved = true
        end
    end
    return (moved ? Wc * (I + X) : Wc), moved
end

# BLOCK-Sylvester Newton step (the block variant of `_newton_from`, draft Remark
# `rem:block`): decouple ACROSS working clusters by solving, for each ordered pair of
# blocks (k,l), the Sylvester equation  M_kk X − X M_ll = −M_kl  with the FULL diagonal
# blocks M_kk = Ã[Iₖ,Iₖ], rather than the diagonal approximation dᵢ−dⱼ of the entrywise
# step (which drops the within-block coupling of M_kk). Within-block entries stay
# neutralized; W ← W(I+X). A singleton/singleton pair reduces EXACTLY to the scalar
# Xᵢⱼ = −Ãᵢⱼ/(dᵢ−dⱼ), so the two variants coincide on a fully-separated spectrum.
#
# COST: the LAPACK Sylvester back end is `trsyl` (triangular inputs); the high-level
# `sylvester` re-Schurs its arguments on EVERY call, so a naive per-pair loop would
# re-factorize each cluster once per partner (O(n·m³)). Instead we Schur-factor each
# cluster ONCE (M_kk = Q T Qᴴ) and reuse it: a cluster–singleton pair is then a single
# triangular back-substitution against the cached factor ((M_kk − dⱼI)X = −M_kj, no
# trsyl needed), and only the rare cluster–cluster pair calls the full Sylvester. Total
# O(m³ + n·m²) per cluster. The cross-block spectra are separated, so each solve is
# well-posed; any accidental ill-conditioning is caught by the final certification.
function _newton_block_from(A, d, Wc, cof)
    n = size(A, 1)
    blocks = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(blocks, cof[i], Int[]), i)
    end
    ids = sort!(collect(keys(blocks)))
    # nothing to decouple if there is only one working cluster (e.g. a globally-coupled
    # Schur frame) — no cross-block pairs, so no step (avoids Schur-factoring the whole block)
    length(ids) <= 1 && return (Wc, false)
    # fast path: every working block is a singleton ⇒ identical to the entrywise step
    all(id -> length(blocks[id]) == 1, ids) && return _newton_from(A, d, Wc, cof)

    # Schur-factor each multi-element cluster ONCE; singletons carry no factor.
    sch = Dict{Int, Tuple{Matrix{eltype(A)}, Matrix{eltype(A)}}}()
    for id in ids
        I = blocks[id]
        if length(I) > 1
            F = schur(A[I, I])           # A[I,I] = F.Z F.T F.Zᴴ, F.T upper-triangular
            sch[id] = (Matrix(F.Z), Matrix(F.T))
        end
    end

    X = zeros(eltype(A), n, n)
    @inbounds for a in eachindex(ids), b in eachindex(ids)
        a == b && continue
        Ik = blocks[ids[a]]
        Il = blocks[ids[b]]
        mk = length(Ik)
        ml = length(Il)
        if mk == 1 && ml == 1
            i = Ik[1]
            j = Il[1]
            X[i, j] = -A[i, j] / (d[i] - d[j])
        elseif ml == 1                   # cluster k vs singleton j: (M_kk − dⱼI)x = −M_kj
            Q, T = sch[ids[a]]
            dj = d[Il[1]]
            w = UpperTriangular(T - dj * I) \ (Q' * (-A[Ik, Il[1]]))
            X[Ik, Il[1]] = Q * w
        elseif mk == 1                   # singleton i vs cluster l: dᵢx − x M_ll = −M_il
            Q, T = sch[ids[b]]           # solve the adjoint: (d̄ᵢI − Tᴴ) xᴴ = Qᴴ(−M_ilᴴ)
            di = d[Ik[1]]
            w = LowerTriangular(conj(di) * I - T') \ (Q' * (-conj.(A[Ik[1], Il])))
            X[Ik[1], Il] = conj.(Q * w)  # x = (Q w)ᴴ ⇒ xₖ = conj((Q w)ₖ)
        else                             # cluster–cluster (rare): full Bartels–Stewart
            X[Ik, Il] = sylvester(A[Ik, Ik], -A[Il, Il], A[Ik, Il])
        end
    end
    return Wc * (I + X), true
end

# DYNAMIC working clustering (§4e merge-detection): pair (i,j) is non-contracting —
# Newton would diverge, |Ãᵢⱼ| ≥ θ|dᵢ−dⱼ| (θ<1) — hence kept together (neutralized);
# contracting pairs are separated. Connected components. As off-diagonals shrink,
# pairs become contracting and split; genuinely coincident/too-coupled stay merged.
function _dynamic_membership(A, d, θ)
    n = size(A, 1)
    adj = falses(n, n)
    @inbounds for j in 1:n, i in 1:(j - 1)
        gap = abs(d[i] - d[j])
        if abs(A[i, j]) >= θ * gap || abs(A[j, i]) >= θ * gap
            adj[i, j] = adj[j, i] = true
        end
    end
    cof = zeros(Int, n)
    for (k, c) in enumerate(_components(adj)), i in c
        cof[i] = k
    end
    return cof
end

# the single-correction dynamic refinement loop (block selects entrywise vs Sylvester)
function _refine_loop(Bc, Wc, θ, steps, block::Bool)
    step = block ? _newton_block_from : _newton_from
    for _ in 1:steps
        A = Wc \ (Bc * Wc)
        d = diag(A)
        cof = _dynamic_membership(A, d, θ)
        Wc, moved = step(A, d, Wc, cof)
        moved || break
    end
    return Wc
end

"""
    refine_basis(B, W; steps = 1, θ = 0.9, mode = :entrywise, tol = 1e-8) -> W′

Float-Newton refinement of the candidate basis (§4e), with the working clustering
recomputed **dynamically** each step by the contraction criterion: a pair with
`|Ãᵢⱼ| ≥ θ|dᵢ−dⱼ|` is non-contracting (Newton would diverge) so it is neutralized,
the rest are separated and `W ← W(I+X)`. Shrinks the off-diagonal residue of `W⁻¹BW`
(so the re-certified discs tighten); coincident / too-coupled eigenvalues stay
together, separable ones split as the residue falls. Pure float candidate producer —
correctness rests on the final certification. Stops early when no cross-cluster pair
remains.

`mode` selects the cross-cluster correction (draft Remarks `rem:block`, `rem:exp`):

  * `:entrywise` (default): the sweep `Xᵢⱼ = −Ãᵢⱼ/(dᵢ−dⱼ)` (scalar reciprocal gaps;
    the within-cluster coupling of `M_kk` is ignored in the cross-cluster step).
  * `:block`: for each pair of working clusters `(k,l)` solve the Sylvester equation
    `M_kk X − X M_ll = −M_kl` with the full diagonal blocks, so the cross-cluster
    correction accounts for the clusters' internal structure. Identical to
    `:entrywise` on a fully-separated spectrum (all singletons); differs only where a
    cluster has internal coupling (the near-defective / tight-cluster regime).
  * `:auto`: run `:entrywise`; if the off-diagonal residue of `W⁻¹BW` stays above
    `tol` (a stall — the two corrections have partly-disjoint, robust-attractor
    stalls, `rem:exp`), retry `:block` from the original `W` and keep whichever basis
    has the smaller residue. Recovers the entrywise-only stalls without losing its
    successes, parameter-free (the `4/40` that stall for both need extended precision).
"""
function refine_basis(B::AbstractMatrix, W::AbstractMatrix; steps::Integer = 1,
        θ::Real = 0.9, mode::Symbol = :entrywise, tol::Real = 1e-8)
    mode in (:entrywise, :block, :auto) ||
        error("mode must be :entrywise, :block, or :auto (got :$mode)")
    CT = _cfloat(B, W)
    Bc = CT.(Matrix(B))
    W0 = CT.(Matrix(W))
    if mode === :auto
        We = _refine_loop(Bc, W0, θ, steps, false)        # cheap entrywise first
        _offres(Bc, We) < tol && return We                # converged — done
        Wb = _refine_loop(Bc, W0, θ, steps, true)         # stalled — retry block from W₀
        return _offres(Bc, Wb) < _offres(Bc, We) ? Wb : We
    end
    return _refine_loop(Bc, W0, θ, steps, mode === :block)
end

"""
    refine_basis(B, W, clusters; steps = 1) -> W′

As above but with a **fixed** working clustering supplied (e.g. the certified
clusters) — for the manual certify→refine→re-certify workflow.
"""
function refine_basis(B::AbstractMatrix, W::AbstractMatrix, clusters; steps::Integer = 1)
    CT = _cfloat(B, W)
    Bc = CT.(Matrix(B))
    Wc = CT.(Matrix(W))
    cof = _membership(clusters, size(Bc, 1))
    for _ in 1:steps
        A = Wc \ (Bc * Wc)
        Wc, moved = _newton_from(A, diag(A), Wc, cof)
        moved || break
    end
    return Wc
end

"""
    miyajima_rump(B; refine = 0, θ = 0.9, mode = :entrywise) -> MiyajimaRump

Driver: numerically diagonalize (`eigen`), run up to `refine` dynamic float-Newton
refinement steps (§4e — separates everything separable, keeps coincident/too-coupled
together; `mode ∈ (:entrywise, :block, :auto)` selects the cross-cluster correction,
see [`refine_basis`](@ref)), then a SINGLE rigorous certification. `refine = 0` gives
plain `eigen` + certify. The final disc-overlap clustering is the rigorous arbiter (an
over-optimistic split shows overlapping discs → re-merged; a divergent split
ill-conditions `W` → Neumann fail, caught).
"""
function miyajima_rump(B::AbstractMatrix; refine::Integer = 0, θ::Real = 0.9,
        mode::Symbol = :entrywise, kappa_mode::Symbol = :cheap, frame::Symbol = :eigen)
    CT = _cfloat(B)
    Bc = CT.(Matrix(B))
    # candidate basis. `frame = :eigen` (default): the eigenvector basis — fully
    # diagonalizes, but a near-degenerate cluster yields near-parallel columns and κ(W)
    # blows up. `frame = :schur` (the guarded-VBD frame, algorithm_spec.md §0/§2): start
    # from the UNITARY Schur vectors, so the columns within a kept-together cluster stay
    # orthonormal — refinement then decouples only BETWEEN clusters, leaving κ(W) governed
    # by inter-cluster separation (benign), with the intra-cluster non-normality confined
    # to the diagonal block (handled by its block σ_min, see `vbd_resolvent`). For BigFloat
    # `:schur` needs GenericSchur loaded (else the Float64 fallback).
    cand(M) = frame === :schur ? Matrix(schur(M).vectors) : Matrix(eigen(M).vectors)
    W = try
        cand(Bc)
    catch
        CT.(cand(ComplexF64.(Matrix(B))))
    end
    refine > 0 && (W = refine_basis(Bc, W; steps = refine, θ, mode))
    return _certify(Bc, W; kappa_mode)
end
