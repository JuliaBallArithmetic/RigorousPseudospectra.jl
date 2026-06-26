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

The guarded-`trevc` solve in the correct order (algorithm_spec.md §0/§2/§3):

  1. **Schur** `B = QTQ'` (`Q` unitary; GenericSchur for BigFloat) — the orthonormal frame;
  2. **cluster** the eigenvalues at a FIXED separation level `sep` (default `√u·‖B‖`):
     group `i,j` if `|t_ii−t_jj| < sep` (too close) or `|T_ij| ≥ |t_ii−t_jj|` (Newton would
     diverge — `r₀ ≥ 1`), connected components;
  3. **Rump–Newton** steps `X_ij = −A_ij/(d_i−d_j)` for `i,j` in DIFFERENT clusters,
     `W ← W(I+X)`, `A ← (I+X)⁻¹A(I+X)` — separating the blocks (between-cluster only;
     within-cluster kept);
  4. **orthogonalize** each block's columns (`QR` within each cluster) ⇒ a genuinely
     **block-orthonormal** `W`: orthonormal within each invariant subspace, so `κ(W)` is
     governed by the inter-block subspace angles `1/sinθ_inter` (benign), NOT by
     intra-cluster eigenvector near-parallelism.

Certify the result with `miyajima_rump(B, W)` and bound the resolvent with
[`vbd_resolvent`](@ref).
"""
function vbd_solve(B::AbstractMatrix; sep::Real = -1, maxsteps::Integer = 6)
    CT = _cfloat(B)
    Bc = CT.(Matrix(B))
    n = size(Bc, 1)
    Tr = real(CT)
    # 1. Schur (orthonormal frame); BigFloat via GenericSchur, Float64 fallback
    Q, A = try
        F = schur(Bc)
        Matrix(F.vectors), Matrix(F.Schur)
    catch
        F = schur(ComplexF64.(Matrix(B)))
        CT.(Matrix(F.vectors)), CT.(Matrix(F.Schur))
    end
    d = diag(A)
    # 2. initial FINE clustering — distance only, like Rump's `conncomp` on |dᵢ−dⱼ| < τ
    # (verifyeigall step 2). NO coupling term: Rump's clusters only ever GROW, decided by
    # the verification below, never by a pairwise coupling heuristic (which is blind to the
    # cumulative coupling of a near-defective chain — exactly the GKW tail).
    τ = sep > 0 ? Tr(sep) : sqrt(eps(Tr)) * opnorm(Bc, Inf)
    parent = collect(1:n)
    rt(x) = parent[x] == x ? x : (parent[x] = rt(parent[x]))
    uni(a, b) = (ra = rt(a); rb = rt(b); ra == rb ? false : (parent[ra] = rb; true))
    @inbounds for j in 1:n, i in 1:(j - 1)
        abs(d[i] - d[j]) < τ && uni(i, j)
    end
    W = copy(Q)
    # Newton transform decoupling the current cross-cluster pairs: Xᵢⱼ = −Aᵢⱼ/(dᵢ−dⱼ)
    function newtonX()
        cof = Int[rt(i) for i in 1:n]
        X = zeros(CT, n, n)
        @inbounds for j in 1:n, i in 1:n
            cof[i] != cof[j] && (X[i, j] = -A[i, j] / (d[i] - d[j]))
        end
        return X
    end
    # 3a. SELF-CORRECTION = Rump's merge-and-recurse (verifyeigall step 6), in float. A
    # column j whose decoupling transform is O(1) (‖X[:,j]‖₂ ≥ 1) "fails" — the float proxy
    # for the rigorous fixed-point (2.10) NOT contracting — so MERGE it into its dominant
    # coupling partner and re-cluster. The column norm sees the *cumulative* coupling a
    # per-pair r₀ misses, so the near-defective GKW tail coalesces into one block instead of
    # collapsing the whole matrix. Merge-only: clusters grow to a fixpoint, never split.
    Xcap = one(Tr)
    for _ in 1:n
        X = newtonX()
        fail = findall(j -> norm(view(X, :, j)) > Xcap, 1:n)
        isempty(fail) && break
        merged = false
        for j in fail
            uni(argmax(abs.(view(X, :, j))), j) && (merged = true)
        end
        merged || break
    end
    # 3b. Rump–Newton refinement of the now-separable structure (between-cluster only).
    for _ in 1:maxsteps
        X = newtonX()
        iszero(X) && break
        IpX = I + X
        A = IpX \ (A * IpX)
        W = W * IpX
        d = diag(A)
    end
    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, rt(i), Int[]), i)
    end
    clusters = collect(values(groups))
    # 4. orthogonalize each block ⇒ block-orthonormal W
    for c in clusters
        W[:, c] = Matrix(qr(W[:, c]).Q)
    end
    return W, clusters
end

# Re-round a frame to the CURRENT float precision element-wise. Needed because
# Complex{BigFloat}(z::Complex{BigFloat}) is the identity (no re-rounding), so a frame
# deserialized / computed at a lower precision keeps its old limb count — and mixing limb
# counts segfaults MPFR inside `inv`/`lu`. Reconstructing through the real components forces
# the current precision (and casts a Float64 warm-start frame to BigFloat).
_to_prec(::Type{Complex{T}}, M) where {T <: AbstractFloat} =
    Complex{T}[Complex{T}(T(real(x)), T(imag(x))) for x in M]

# union–find helpers (shared by the warm-start refinement)
_uf_find(p, x) = p[x] == x ? x : (p[x] = _uf_find(p, p[x]))
function _uf_union!(p, a, b)
    ra = _uf_find(p, a)
    rb = _uf_find(p, b)
    ra == rb ? false : (p[ra] = rb; true)
end
function _uf_clusters(p, n)
    g = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(g, _uf_find(p, i), Int[]), i)
    end
    collect(values(g))
end
# Newton transform Xᵢⱼ = −Aᵢⱼ/(dᵢ−dⱼ) decoupling the current cross-cluster pairs
function _newton_decouple(A, d, parent, n, CT)
    X = zeros(CT, n, n)
    @inbounds for j in 1:n, i in 1:n
        (i != j && _uf_find(parent, i) != _uf_find(parent, j)) &&
            (X[i, j] = -A[i, j] / (d[i] - d[j]))
    end
    return X
end

"""
    vbd_refine(B, W0; outer = 4, inner = 3) -> (W, clusters)

Warm-start precision lift: take a decomposition `W0` computed at a LOWER precision and
refine it at the CURRENT precision (set by the caller, e.g. `setprecision(BigFloat, 512)`)
WITHOUT a fresh Schur. Each outer step **re-grounds against the original `B`**
`A = W⁻¹BW` (the residue/`S` correction — recentres the Gershgorin discs onto the
residual-minimizing centres, spec §4/§6), then re-clusters from the refined spectrum
(distance `τ` + the column-norm merge of [`vbd_solve`](@ref)) and Newton-refines the
between-block coupling. As precision rises the recentred discs shrink, so the block
structure splits finer (e.g. GKW 256-bit's 168-block → 512-bit's 100-block) — quadratic
Newton–Kantorovich convergence to the working-precision floor. Certify with
`miyajima_rump(B, W)` as usual.

The re-grounding inverse `Y` need only PRECONDITION (`A = Y·B·W`; the residual `B·W` is
full-precision, so the recentred diagonal keeps its precision — an approximate `Y` only
adds a `~κ·u_Y·‖B‖` noise floor to the off-block), so a cheap **Float64** `inv(W)` is used
in the loop and the single exact BigFloat inverse is deferred to the certification.

!!! warning "Blocked on an upstream Julia bug (further optimization)"
    This is a not-yet-usable precision-lift optimization. On Julia 1.12.6 it segfaults
    inside the GC mark loop under the sustained `Complex{BigFloat}` matmul allocation
    (reproducible with a bare `inv`/`*` loop — not our code; serial GC does not help).
    Run it only on a Julia without that regression. Until then use `vbd_solve` (a single
    high-precision Schur), which does few enough BigFloat ops to survive. See the
    `julia-1-12-bigfloat-gc-segfault` note.

    **Also TO TEST** (could not be validated because of the bug above): whether the
    cheap Float64-`inv(W)` preconditioner is good enough — i.e. whether its
    `~κ·u_Y·‖B‖` off-block noise floor still lets the refinement converge to the
    working-precision decomposition (matching a cold high-precision Schur), or whether
    `Y` must be lifted (incremental BigFloat `Y ← (I−X)Y`, or an occasional exact
    inverse) to reach the floor. The `outer`/`inner` counts are likewise untuned.
"""
function vbd_refine(B::AbstractMatrix, W0::AbstractMatrix; outer::Integer = 4,
        inner::Integer = 3)
    CT = _cfloat(B)
    Bc = CT.(Matrix(B))
    n = size(Bc, 1)
    Tr = real(CT)
    W = _to_prec(CT, Matrix(W0))          # re-round warm-start frame to current precision
    τ = sqrt(eps(Tr)) * opnorm(Bc, Inf)
    Xcap = one(Tr)
    parent = collect(1:n)
    local A, d
    for _ in 1:outer
        # re-ground the working matrix + diagonal against the ORIGINAL B (the S correction).
        # Y need only PRECONDITION: the residual B·W is full-precision, so the recentred
        # diagonal keeps its precision; an approximate Y just adds a ~κ·u_Y·‖B‖ noise floor
        # to the off-block. Use a FLOAT64 inverse (LAPACK — no MPFR `inv`, so no Julia-1.12
        # BigFloat-LU GC segfault); the single exact BigFloat inverse happens only in the
        # final `miyajima_rump` certification. NOTE: this Float64-Y preconditioner is an
        # UNVERIFIED choice — whether its κ·u_Y·‖B‖ noise floor still lets the refinement
        # reach the working-precision floor is one of the things to test once the upstream
        # GC bug is fixed (see the docstring warning).
        Y = _to_prec(CT, inv(ComplexF64.(Matrix(W))))
        A = Y * Bc * W
        d = diag(A)
        # re-cluster from scratch at the refined spectrum: distance, then column-norm merge
        parent = collect(1:n)
        @inbounds for j in 1:n, i in 1:(j - 1)
            abs(d[i] - d[j]) < τ && _uf_union!(parent, i, j)
        end
        for _ in 1:n
            X = _newton_decouple(A, d, parent, n, CT)
            fail = findall(j -> norm(view(X, :, j)) > Xcap, 1:n)
            isempty(fail) && break
            merged = false
            for j in fail
                _uf_union!(parent, argmax(abs.(view(X, :, j))), j) && (merged = true)
            end
            merged || break
        end
        # Newton refinement of the between-block coupling: the EXACT similarity
        # (I+X)⁻¹A(I+X) via the explicit inverse — no first-order (I−X) approximation, so no
        # O(‖X‖²) drift from the true spectrum (the approximation would cap the attainable
        # precision; the exact inverse reaches the working-precision floor). The BigFloat
        # `inv(I+X)` is subject to the same upstream GC bug as the rest of this routine —
        # see the docstring warning.
        for _ in 1:inner
            X = _newton_decouple(A, d, parent, n, CT)
            iszero(X) && break
            IpX = I + X
            A = inv(IpX) * (A * IpX)
            W = W * IpX
            d = diag(A)
        end
    end
    clusters = _uf_clusters(parent, n)
    for c in clusters
        W[:, c] = Matrix(qr(W[:, c]).Q)
    end
    return W, clusters
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
