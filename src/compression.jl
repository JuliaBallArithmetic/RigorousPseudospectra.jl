# §3.2 (sec:vbd): the compression. A floating-point candidate similarity (sec:candidate), certified
# a posteriori in ball arithmetic (sec:certification), and transported to a lower bound on
# σ_min(G11 − zI) (sec:transport).
#
# The compression may be a pencil: for a frame Q that is only almost orthonormal the compression is
# read as Q*AQ − zQ*Q (Proposition prop:almostortho). With G̃ = Q*AQ and M = Q*Q, the eigenvalues
# of the isometry's compression Z*AZ = S⁻¹G̃S⁻¹ (Q = ZS the polar decomposition) are those of
# M⁻¹G̃, and everything below is written for the pair (G̃, M), M = I being the orthonormal case:
#
#   R2 = Y(MW) − I,   R1 = Y(G̃W − MWΛ),   M_c := (MW)⁻¹G̃W  (similar to M⁻¹G̃),
#   M_c − Λ = (I + R2)⁻¹ R1                                  (lem:deviation with W → MW on the left),
#   G̃ − zM = MW (M_c − zI) W⁻¹,  so  σ_min(G̃ − zM) ≥ σ_min(M_c − zI) / (‖(MW)⁻¹‖ ‖W‖),
#   ‖(MW)⁻¹‖ ≤ ‖Y‖/(1 − ‖R2‖)                                (lem:neumann).
#
# With M = I these are exactly eq:residuals, lem:neumann, lem:deviation and eq:blockres.

# ------------------------------------------------------------------------------------------------
# The candidate (sec:candidate)
# ------------------------------------------------------------------------------------------------

"""
    CompressionCandidate

A floating-point candidate: a frame `W`, an approximate inverse `Y ≈ (MW)⁻¹` (≈ `W⁻¹` when
`M = I`), a partition `blocks` of `1:k`, and the block-diagonal `Λ` (stored densely), the block
diagonal of `Y G̃ W`. `method` is `:sylvester` or `:newton`; `info` records iterations, merges,
failures and time. Nothing here is certified.
"""
struct CompressionCandidate
    W::Matrix{ComplexF64}
    Y::Matrix{ComplexF64}
    Λ::Matrix{ComplexF64}
    blocks::Vector{Vector{Int}}
    method::Symbol
    info::NamedTuple
end

"Connected components of the graph `|d_i − d_j| ≤ τ`, largest modulus first."
function cluster_by_distance(d::AbstractVector, τ::Real)
    k = length(d)
    parent = collect(1:k)
    find(i) = (while parent[i] != i
        parent[i] = parent[parent[i]]
        i = parent[i]
    end; i)
    for i in 1:k, j in (i + 1):k
        abs(d[i] - d[j]) <= τ && (parent[find(i)] = find(j))
    end
    groups = Dict{Int, Vector{Int}}()
    for i in 1:k
        push!(get!(groups, find(i), Int[]), i)
    end
    cl = collect(values(groups))
    sort!(cl; by = c -> (-maximum(abs, d[c]), minimum(c)))
    return cl
end

_default_cluster_tol(G) = sqrt(unit_roundoff(Float64)) * max(norm(G), floatmin(Float64))

function _offblock_norm(G, label)
    s = 0.0
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        label[i] != label[j] && (s += abs2(G[i, j]))
    end
    return sqrt(s)
end

function _blockdiag(G, blocks)
    Λ = zeros(ComplexF64, size(G))
    for b in blocks
        Λ[b, b] .= G[b, b]
    end
    return Λ
end

# Orthonormalise inside each block and not across (sec:candidate, step 4): W_C = Q_C R_C,
# W_C ← Q_C and Y_C ← R_C Y_C, so that Y stays the inverse of W up to rounding with no inverse
# formed.
function _orthonormalise_blocks!(W, Y, blocks)
    for b in blocks
        F = qr(W[:, b])
        Qb = Matrix(F.Q)[:, 1:length(b)]
        Rb = F.R
        W[:, b] .= Qb
        Y[b, :] .= Rb * Y[b, :]
    end
    return W, Y
end

# LAPACK trsen on a complex Schur pair (T, Z), in place: the selected eigenvalues are moved to the
# leading positions, keeping the relative order of the selected and of the unselected ones.
function _ordschur!(T::Matrix{ComplexF64}, Z::Matrix{ComplexF64}, sel)
    LinearAlgebra.LAPACK.trsen!('N', 'V', convert(Vector{LinearAlgebra.BlasInt}, sel), T, Z)
    return T, Z
end

# Place the clusters contiguously, in the given order, by LAPACK trsen (`ordschur!`), which keeps
# the relative order of the selected and of the unselected eigenvalues. Returns the blocks as
# ranges of positions.
function _order_clusters!(T, V, clusters)
    k = size(T, 1)
    perm = collect(1:k)                 # perm[pos] = original index now at pos
    d0 = diag(T)
    pos = 0
    for c in clusters
        sel = falses(k)
        sel[1:pos] .= true
        for i in c
            sel[findfirst(==(i), perm)] = true
        end
        if !all(sel[1:(pos + length(c))])
            _ordschur!(T, V, sel)
            perm = vcat(perm[sel], perm[.!sel])
        end
        pos += length(c)
    end
    # check that the diagonal went where the permutation says
    scale = max(maximum(abs, d0), floatmin(Float64))
    good = all(abs(T[i, i] - d0[perm[i]]) <= 1e-6 * scale for i in 1:k)
    blocks = UnitRange{Int}[]
    p = 0
    for c in clusters
        push!(blocks, (p + 1):(p + length(c)))
        p += length(c)
    end
    return blocks, good
end

"""
    candidate_sylvester(G; M = nothing, cluster_tol, xmax = 1e8) -> CompressionCandidate

The candidate of sec:candidate by block diagonalisation of the Schur form, the algorithm of
Bavely and Stewart (SIAM J. Numer. Anal. 16 (1979) 359–367):

1. complex Schur form `M⁻¹G̃ = V T V*` (`M = I` unless given);
2. clusters: connected components of `|t_ii − t_jj| ≤ cluster_tol` (default `√u ‖G‖_F`), placed
   contiguously by `ordschur!`, largest modulus first;
3. the leading cluster is split from the trailing block by the triangular Sylvester equation
   `T_cc X − X T_rr = −T_cr` (LAPACK `trsyl`), applied as `W ← W [I X; 0 I]`,
   `Y ← [I −X; 0 I] Y` (the exact inverse), `T_cr ← 0`; if `‖X‖_F > xmax` the cluster of the
   trailing block nearest in eigenvalue is moved next to the current one (`ordschur!` on the
   trailing block) and the two are merged, and the equation is solved again;
4. orthonormalisation inside each block (QR), not across;
5. `Λ` = block diagonal of `Y G̃ W` in floating point.

The default `xmax = 1e8` is of the order of `u^{-1/2}`, the scale of the clustering threshold:
the rounding of the transform then enters β_Λ at about `√u ‖G‖`. With `xmax = 1e4` the companion
matrix of the derivative of the Chebyshev polynomial T₂₀ is merged into a single block (see
bench/candidates_compare.jl). The merge on `‖X‖` is the brief's request and departs from the
sentence of sec:candidate that "blocks are not merged on the size of a decoupling transform"; the
certification does not depend on it.
"""
function candidate_sylvester(G::AbstractMatrix; M = nothing, cluster_tol = nothing,
        xmax::Real = 1e8)
    t0 = time()
    Gm = ComplexF64.(Matrix(mid(G)))
    Mm = M === nothing ? nothing : ComplexF64.(Matrix(mid(M)))
    Gc = Mm === nothing ? Gm : Mm \ Gm
    k = size(Gc, 1)
    τ = cluster_tol === nothing ? _default_cluster_tol(Gc) : Float64(cluster_tol)
    S = schur(Gc)
    T = Matrix(S.T)
    V = Matrix(S.Z)
    clusters = cluster_by_distance(diag(T), τ)
    ranges, ordered_ok = _order_clusters!(T, V, clusters)
    W = copy(V)
    Y = Matrix(V')
    nmerge = 0
    nsolve = 0
    maxX = 0.0
    blocks = Vector{Int}[]
    # `pending` are the ranges still to be split off, in order along the diagonal
    pending = copy(ranges)
    while !isempty(pending)
        cur = popfirst!(pending)
        while true
            isempty(pending) && break
            rest = (last(cur) + 1):k
            C = -T[cur, rest]
            local X, sc
            try
                X, sc = LinearAlgebra.LAPACK.trsyl!('N', 'N', T[cur, cur], T[rest, rest], C, -1)
            catch
                X, sc = fill(ComplexF64(NaN), length(cur), length(rest)), 1.0
            end
            X = X ./ sc
            nsolve += 1
            # the spectral norm, not the Frobenius one: it is ‖X‖₂ that sets
            # κ([I X; 0 I]) = ((‖X‖₂+√(‖X‖₂²+4))/2)² ≈ ‖X‖₂² and the rounding of the transform
            # in β_Λ. ‖X‖_F overestimates it by up to √(min(size(X)...)), measured at 1.4 to 2.1
            # on the shapes trsyl returns here, so the merge fired earlier than it needed to and
            # coarsened the blocks for nothing.
            nx = isempty(X) ? 0.0 : opnorm(X)
            if isfinite(nx) && nx <= xmax
                maxX = max(maxX, nx)
                W[:, rest] .+= W[:, cur] * X
                Y[cur, :] .-= X * Y[rest, :]
                T[cur, rest] .= 0
                break
            end
            # merge: bring the trailing cluster nearest to `cur` next to it
            dcur = diag(T)[cur]
            dist(r) = minimum(abs(a - b) for a in diag(T)[r] for b in dcur)
            j = argmin(dist.(pending))
            target = pending[j]
            if j != 1
                sel = falses(length(rest))
                sel[(first(target) - first(rest) + 1):(last(target) - first(rest) + 1)] .= true
                Trr = T[rest, rest]
                U = Matrix{ComplexF64}(I, length(rest), length(rest))
                _ordschur!(Trr, U, sel)
                T[rest, rest] .= Trr
                T[1:(first(rest) - 1), rest] .= T[1:(first(rest) - 1), rest] * U
                W[:, rest] .= W[:, rest] * U
                Y[rest, :] .= U' * Y[rest, :]
                # the moved cluster now occupies the first positions of `rest`; shift the others
                len = length(target)
                newpending = UnitRange{Int}[]
                p = first(rest) + len
                for (i, r) in enumerate(pending)
                    i == j && continue
                    push!(newpending, p:(p + length(r) - 1))
                    p += length(r)
                end
                pending = newpending
                cur = first(cur):(last(cur) + len)
            else
                popfirst!(pending)
                cur = first(cur):last(target)
            end
            nmerge += 1
        end
        push!(blocks, collect(cur))
    end
    _orthonormalise_blocks!(W, Y, blocks)
    Yp = Mm === nothing ? Y : Y / Mm
    Gt = Yp * (Gm * W)
    Λ = _blockdiag(Gt, blocks)
    return CompressionCandidate(W, Yp, Λ, blocks, :sylvester,
        (; nsolve, nmerge, maxX, nclusters0 = length(clusters), ordered_ok, τ,
            time = time() - t0))
end

"""
    candidate_newton(G; M = nothing, cluster_tol, maxiter = 20, tol = 1e-13) -> CompressionCandidate

The alternative candidate of sec:candidate, step 2: from the Schur vectors, clusters by diagonal
distance (connected components, no merging), then the first-order step
`X_ij = −G̃_ij/(d_i − d_j)` for `i`, `j` in different clusters, `W ← W(I + X)`,
`Y ← (I + X)⁻¹Y` (a floating-point solve), iterated up to `maxiter` times or until the off-block
Frobenius norm is below `tol ‖M⁻¹G̃‖_F` (the norm of the matrix being decoupled, fixed). The
off-block norm is not required to decrease at every step: in 9 of the 13 runs of
bench/candidates_compare.jl it grew before decreasing or before the step limit (up to 6.6e13 from
6.0 on grcar32, which then converged at step 9). The iterate with the smallest off-block norm is
kept. `info.stop` is `:converged`, `:maxiter` or `:nonfinite`; `info.iters` the number of
steps taken, `info.best` the step whose iterate was kept.
"""
function candidate_newton(G::AbstractMatrix; M = nothing, cluster_tol = nothing,
        maxiter::Integer = 20, tol::Real = 1e-13)
    t0 = time()
    Gm = ComplexF64.(Matrix(mid(G)))
    Mm = M === nothing ? nothing : ComplexF64.(Matrix(mid(M)))
    Gc = Mm === nothing ? Gm : Mm \ Gm
    k = size(Gc, 1)
    nG = norm(Gc)
    τ = cluster_tol === nothing ? _default_cluster_tol(Gc) : Float64(cluster_tol)
    S = schur(Gc)
    V = Matrix(S.Z)
    clusters = cluster_by_distance(diag(S.T), τ)
    label = zeros(Int, k)
    for (c, idx) in enumerate(clusters)
        label[idx] .= c
    end
    W = copy(V)
    Y = Matrix(V')
    Gt = Y * Gc * W
    off = _offblock_norm(Gt, label)
    off0 = off
    best = (W = copy(W), Y = copy(Y), off = off, it = 0)
    stop = :maxiter
    iters = 0
    maxoff = off
    for it in 1:maxiter
        if off <= tol * nG
            stop = :converged
            break
        end
        d = diag(Gt)
        X = zeros(ComplexF64, k, k)
        @inbounds for j in 1:k, i in 1:k
            label[i] != label[j] && (X[i, j] = -Gt[i, j] / (d[i] - d[j]))
        end
        if !all(isfinite, X)
            stop = :nonfinite
            break
        end
        IX = I + X
        W = W * IX
        Y = IX \ Y
        Gt = Y * Gc * W
        iters = it
        if !all(isfinite, Gt)
            stop = :nonfinite
            break
        end
        off = _offblock_norm(Gt, label)
        maxoff = max(maxoff, off)
        off < best.off && (best = (W = copy(W), Y = copy(Y), off = off, it = it))
    end
    stop === :maxiter && best.off <= tol * nG && (stop = :converged)
    W, Y = best.W, best.Y
    blocks = [sort(c) for c in clusters]
    _orthonormalise_blocks!(W, Y, blocks)
    Yp = Mm === nothing ? Y : Y / Mm
    Λ = _blockdiag(Yp * (Gm * W), blocks)
    return CompressionCandidate(W, Yp, Λ, blocks, :newton,
        (; iters, best = best.it, stop, off0, off = best.off, maxoff,
            nclusters0 = length(clusters), τ, time = time() - t0))
end

# ------------------------------------------------------------------------------------------------
# The certification (sec:certification)
# ------------------------------------------------------------------------------------------------

"""
    EigComponent

A union of blocks whose outer discs are certified disjoint from those of every other block
(Corollary cor:clusters): it holds exactly `count` eigenvalues of the compression, counted with
algebraic multiplicity, all inside the disc `D(center, radius)`.
"""
struct EigComponent
    blocks::Vector{Int}
    idx::Vector{Int}
    count::Int
    center::ComplexF64
    radius::Float64
end

"""
    CompressionCertificate

Output of [`certify_compression`](@ref). Certified: `nrmR2 ≥ ‖R2‖ < 1`, `nrmR1 ≥ ‖R1‖`,
`betaΛ ≥ β_Λ` (lem:deviation), `beta2 ≥ β₂` and the localised radii `rloc` (prop:blockgersg),
`kappa ≥ ‖W‖‖(MW)⁻¹‖` (lem:neumann), the outer radii `ρglobal[i]` of
`Ω_i = {σ_min(P_i − zI) ≤ β_Λ}` (thm:enclosure) and `ρlocal[i]` of `{σ_min(P_i − zI) ≤ r_i}`
about `centres[i]`, and the components of each family (cor:clusters). `ok = false` when
`‖R2‖ < 1` could not be certified; then nothing else is meaningful.
"""
struct CompressionCertificate
    ok::Bool
    k::Int
    blocks::Vector{Vector{Int}}
    P::Vector{Matrix{ComplexF64}}
    centres::Vector{ComplexF64}
    nrmR2::Float64
    nrmR1::Float64
    betaΛ::Float64
    beta2::Float64
    blocknorms::Matrix{Float64}
    rloc::Vector{Float64}
    ρglobal::Vector{Float64}
    ρlocal::Vector{Float64}
    nrmW::Float64
    nrmY::Float64
    kappa::Float64
    components_global::Vector{EigComponent}
    components_local::Vector{EigComponent}
    candidate::CompressionCandidate
    time::Float64
end

# Ball product with a complex/real mixture handled by BallArithmetic's real kernels.
_ball(A::BallMatrix) = A
_ball(A::AbstractMatrix) = BallMatrix(Matrix(A))
function _bmul(A, B)
    A = _ball(A)
    B = _ball(B)
    TA = eltype(mid(A))
    TB = eltype(mid(B))
    if TA <: Real && TB <: Real
        return A * B
    elseif TA <: Complex && TB <: Complex
        return A * B
    elseif TA <: Complex
        return A * BallMatrix(ComplexF64.(mid(B)), rad(B))
    else
        return BallMatrix(ComplexF64.(mid(A)), rad(A)) * B
    end
end
_bsub(A, B) = _promote_ball(_ball(A)) - _promote_ball(_ball(B))
_promote_ball(A::BallMatrix) = eltype(mid(A)) <: Complex ? A :
                               BallMatrix(ComplexF64.(mid(A)), rad(A))

"""
    outer_radius(P, c, β; mmax = 8) -> ρ

A certified radius `ρ` with `{z : σ_min(P − zI) ≤ β} ⊆ D(c, ρ)`. With `N = P − cI` and
`g_j ≥ ‖N^j‖` (ball powers), Proposition prop:power applied to `N` gives, for `|z − c| = r` with
`r^m > g_m`, `σ_min(P − zI) ≥ f_m(r) = (r^m − g_m)/Σ_{j<m} r^{m−1−j} g_j`, and `f_m` is increasing
for `r > g_m^{1/m}` (its logarithmic derivative is at least `1/r` there). So any such `r` with
`f_m(r) ≥ β` bounds the region; `m = 1` gives `ρ = ‖P − cI‖ + β`, the disc of the brief. We take
the least over `m ≤ mmax` of the radii found by bisection, each checked with directed rounding.
"""
function outer_radius(P::AbstractMatrix, c::Number, β::Real; mmax::Int = 8)
    ℓ = size(P, 1)
    β = Float64(β)
    N = _bsub(BallMatrix(ComplexF64.(P)), BallMatrix(Matrix{ComplexF64}(c * I, ℓ, ℓ)))
    mm = min(ℓ, mmax)
    g = zeros(mm)
    Nj = N
    g[1] = opnorm_up(N)
    for j in 2:mm
        Nj = _bmul(Nj, N)
        g[j] = min(opnorm_up(Nj), mulu(g[j - 1], g[1]))
    end
    return _radius_from_powers(g, β)
end

"""
    _radius_from_powers(g, β) -> ρ

The radius search of [`outer_radius`](@ref) from the certified power bounds `g[j] ≥ ‖N^j‖`
alone: the least `ρ` over `m = 1:length(g)` for which `f_m(ρ) ≥ β` is certified, so that
`{z : σ_min(N − zI) ≤ β} ⊆ D(0, ρ)` (Proposition prop:power applied to `N`, whose lower bound
`f_m` increases for `r > g_m^{1/m}`). `m = 1` gives `ρ = g_1 + β`.
"""
function _radius_from_powers(g::AbstractVector{<:Real}, β::Real)
    β = Float64(β)
    mm = length(g)
    mm == 0 && return Inf
    g = Float64.(g)
    ρ = addu(g[1], β)
    function flo(r, m)                       # certified lower bound of f_m(r)
        num = subd(powd(r, m), g[m])
        num > 0 || return -1.0
        den = powu(r, m - 1)
        for j in 1:(m - 1)
            den = addu(den, mulu(powu(r, m - 1 - j), g[j]))
        end
        return divd(num, den)
    end
    for m in 2:mm
        flo(ρ, m) >= β || continue
        lo = g[m] > 0 ? rootu(g[m], m) : 0.0
        hi = ρ
        for _ in 1:80
            mid_ = (lo + hi) / 2
            (mid_ == lo || mid_ == hi) && break
            flo(mid_, m) >= β ? (hi = mid_) : (lo = mid_)
        end
        flo(hi, m) >= β && (ρ = min(ρ, hi))
    end
    return ρ
end

# Components of a family of discs D(c_i, ρ_i): union-find on "not certified disjoint".
function _components(centres, ρ, blocks)
    q = length(centres)
    parent = collect(1:q)
    find(i) = (while parent[i] != i
        parent[i] = parent[parent[i]]
        i = parent[i]
    end; i)
    onepu = nextfloat(1.0)
    for i in 1:q, j in (i + 1):q
        dij = divd(absd(centres[i] - centres[j]), onepu)
        (dij > addu(ρ[i], ρ[j])) || (parent[find(i)] = find(j))
    end
    groups = Dict{Int, Vector{Int}}()
    for i in 1:q
        push!(get!(groups, find(i), Int[]), i)
    end
    out = EigComponent[]
    for bs in values(groups)
        sort!(bs)
        c = sum(centres[bs]) / length(bs)
        r = 0.0
        for i in bs
            r = max(r, addu(mulu(absu(c - centres[i]), onepu), ρ[i]))
        end
        idx = sort(vcat(blocks[bs]...))
        push!(out, EigComponent(bs, idx, length(idx), c, r))
    end
    sort!(out; by = C -> (-abs(C.center), C.blocks[1]))
    return out
end

"""
    certify_compression(G̃, cand; M = nothing, kappa_mode = :svdbox) -> CompressionCertificate

The certification of sec:certification for the candidate `cand` of the compression `G̃` (a ball
matrix enclosing Q*AQ, or a plain matrix, taken as exact), with `M` the ball enclosure of Q*Q for
an almost-orthonormal frame (`nothing` for M = I):

* `R2 = Y(MW) − I`, `R1 = Y(G̃W − MWΛ)` as ball products (eq:residuals); spectral norms bounded
  above by [`opnorm_up`](@ref); refuses unless `‖R2‖ < 1` (lem:neumann);
* `β_Λ = ‖R1‖/(1 − ‖R2‖)` (lem:deviation); outer discs of `Ω_i` (thm:enclosure) by
  [`outer_radius`](@ref); components and counts (cor:clusters);
* `M̃ = Y G̃ W` (ball), `β₂ = ‖R2‖‖M̃‖/(1 − ‖R2‖)`, `Ñ = M̃ − Λ`, and the localised radii
  `r_i = Σ_j ‖Ñ_{C_iC_j}‖ + qβ₂` of prop:blockgersg. The sum runs over every `j` including
  `j = i`: in the paper the diagonal blocks of `Ñ` vanish, while here `Λ` is a floating-point
  matrix and `M̃` a ball, so `Ñ_{C_iC_i}` is a ball about a rounding-level midpoint and is kept;
* `κ = ‖Y‖‖W‖/(1 − ‖R2‖) ≥ ‖(MW)⁻¹‖‖W‖`, for eq:blockres.

`kappa_mode = :svdbox` bounds ‖W‖, ‖Y‖ and the residual norms also through `svdbox`;
`:cheap` uses only the Frobenius and interpolation bounds.
"""
function certify_compression(G, cand::CompressionCandidate; M = nothing,
        kappa_mode::Symbol = :svdbox)
    t0 = time()
    sv = kappa_mode === :svdbox
    k = size(cand.W, 1)
    Gb = _promote_ball(_ball(G))
    Wb = BallMatrix(cand.W)
    Yb = BallMatrix(cand.Y)
    MW = M === nothing ? Wb : _bmul(M, Wb)
    Λ = cand.Λ
    R2 = _bsub(_bmul(Yb, MW), Matrix{ComplexF64}(I, k, k))
    nR2 = opnorm_up(R2; svd = sv)
    blocks = cand.blocks
    q = length(blocks)
    P = [Matrix(Λ[b, b]) for b in blocks]
    centres = ComplexF64[sum(diag(p)) / size(p, 1) for p in P]
    bad() = CompressionCertificate(false, k, blocks, P, centres, nR2, Inf, Inf, Inf,
        fill(Inf, q, q), fill(Inf, q), fill(Inf, q), fill(Inf, q), Inf, Inf, Inf,
        EigComponent[], EigComponent[], cand, time() - t0)
    (isfinite(nR2) && nR2 < 1) || return bad()
    GW = _bmul(Gb, Wb)
    R1 = _bmul(Yb, _bsub(GW, _bmul(MW, BallMatrix(Λ))))
    nR1 = opnorm_up(R1; svd = sv)
    omr = subd(1.0, nR2)
    βΛ = divu(nR1, omr)
    Mt = _bmul(Yb, GW)
    nMt = opnorm_up(Mt; svd = sv)
    β2 = divu(mulu(nR2, nMt), omr)
    Nt = _bsub(Mt, BallMatrix(Λ))
    bn = zeros(q, q)
    for i in 1:q, j in 1:q
        bn[i, j] = opnorm_up(Nt[blocks[i], blocks[j]]; svd = sv && length(blocks[i]) *
                                                              length(blocks[j]) > 1)
    end
    rloc = [addu(sumu(bn[i, :]), mulu(Float64(q), β2)) for i in 1:q]
    ρg = [outer_radius(P[i], centres[i], βΛ) for i in 1:q]
    ρl = [outer_radius(P[i], centres[i], rloc[i]) for i in 1:q]
    nW = opnorm_up(Wb; svd = sv)
    nY = opnorm_up(Yb; svd = sv)
    κ = divu(mulu(nY, nW), omr)
    cg = _components(centres, ρg, blocks)
    cl = _components(centres, ρl, blocks)
    return CompressionCertificate(true, k, blocks, P, centres, nR2, nR1, βΛ, β2, bn, rloc,
        ρg, ρl, nW, nY, κ, cg, cl, cand, time() - t0)
end

"""
    sigma_blocks_lo(cc, z) -> Float64

eq:blockres: `σ_min(G̃ − zM) ≥ max(0, min_i σ_min(P_i − zI) − β_Λ)/κ`, with `σ_min(P_i − zI)`
in closed form for a 1×1 block (`|p − z|`, bounded below through the rounding of the
subtraction) and by `svdbox` otherwise. For `M = I` this is a lower bound on `σ_min(G11 − zI)`.
"""
function sigma_blocks_lo(cc::CompressionCertificate, z::Number)
    cc.ok || return 0.0
    v = subd(_sigmin_blocks_lo(cc, z), cc.betaΛ)
    return v > 0 ? divd(v, cc.kappa) : 0.0
end

"`σ_min(P_i − zI)` lower bound for one block: closed form for a 1×1 block, `svdbox` otherwise."
function _sigmin_block_lo(P::AbstractMatrix, z::Number)
    z = ComplexF64(z)
    if size(P, 1) == 1
        return divd(absd(P[1, 1] - z), nextfloat(1.0))
    end
    return sigmin_lo(_bsub(BallMatrix(ComplexF64.(P)),
        Matrix{ComplexF64}(z * I, size(P)...)))
end

"`min_i σ_min(P_i − zI)`, certified from below; 1-Lipschitz in z."
_sigmin_blocks_lo(cc::CompressionCertificate, z::Number) =
    minimum(_sigmin_block_lo(P, z) for P in cc.P; init = Inf)

"""
    BlockProjectorCertificate

Proposition prop:projector applied inside sec:vbd, on a circle `|z − c| = r` separating the regions
of thm:enclosure: `bound ≥ ‖P_{M_c,S} − P_{Λ,S}‖`, where `M_c = (MW)⁻¹G̃W` is the collapsed matrix
of lem:deviation, `Λ` the block-diagonal candidate, `S` the blocks inside the circle and `P_{Λ,S}`
the projector onto their columns (along the others). `M ≥ sup_Γ ‖(M_c − zI)⁻¹‖` and `M₀ ≥ sup_Γ ‖(Λ − zI)⁻¹‖` are the two resolvent bounds of
eq:projector, and `minval` is the certified minimum over the circle of
`min_i σ_min(P_i − zI) − β_Λ`.
"""
struct BlockProjectorCertificate
    ok::Bool
    center::ComplexF64
    radius::Float64
    S::Vector{Int}
    cols::Vector{Int}
    count::Int
    bound::Float64
    M::Float64
    M0::Float64
    minval::Float64
    nevals::Int
end

"""
    certify_block_projector(cc, c, r; family = :global, nθ = 64, maxdepth = 6)
        -> BlockProjectorCertificate

The projector bound of prop:projector for the compression itself, the roles of the blocks being
those of `Λ` and of the deviation `M_c − Λ`. The blocks are classified by their outer discs of the
family given (`:global`, the regions of thm:enclosure with β_Λ; `:local`, those of
prop:blockgersg): each must be certified inside the open disc `|z − c| < r` or outside its closure,
else the call declines. `min_i σ_min(P_i − zI)` is bounded below over the circle by an arc cover
([`circle_min`](@ref), the function being 1-Lipschitz in z): it gives `M₀`, the resolvent bound of
`Λ`, and, after subtracting `β_Λ` (`‖M_c − Λ‖ ≤ β_Λ`, lem:deviation), `M`, the resolvent bound of
`M_c`. Hence `‖P_{M_c,S} − P_{Λ,S}‖ ≤ r M M₀ β_Λ`. The same number bounds the difference of the
complementary projectors, the identity cancelling in `P_out = I − P_in`.
"""
function certify_block_projector(cc::CompressionCertificate, c::Number, r::Real;
        family::Symbol = :global, nθ::Int = 64, maxdepth::Int = 6, blockfun = nothing,
        discs = nothing)
    c = ComplexF64(c)
    r = Float64(r)
    σblocks = blockfun === nothing ? (z -> _sigmin_blocks_lo(cc, z)) : blockfun
    bad = BlockProjectorCertificate(false, c, r, Int[], Int[], 0, Inf, Inf, Inf, 0.0, 0)
    cc.ok || return bad
    ρ = discs !== nothing ? discs[2] : (family === :local ? cc.ρlocal : cc.ρglobal)
    centres = discs !== nothing ? discs[1] : cc.centres
    onepu = nextfloat(1.0)
    S = Int[]
    for i in eachindex(centres)
        d = centres[i] - c
        if addu(mulu(absu(d), onepu), ρ[i]) < r
            push!(S, i)
        elseif divd(absd(d), onepu) > addu(r, ρ[i])
            continue
        else
            return bad
        end
    end
    ok, m0, nev = circle_min(σblocks, c, r; nθ, maxdepth)
    m = subd(m0, cc.betaΛ)                 # min over Γ of σ_min(Λ_t − zI) ≥ min_i σ_min(P_i) − β_Λ
    b, M, M0, okb = projector_bound(m, m0, r, cc.betaΛ)
    cols = isempty(S) ? Int[] : sort(vcat(cc.blocks[S]...))
    return BlockProjectorCertificate(ok && okb, c, r, S, cols, length(cols), b, M, M0, m, nev)
end

"""
    eigencount_in_disc(comps, c, r) -> (count, ok)

Number of eigenvalues of the compression in the closed disc `D(c, r)`, from a family of
components: every component must lie inside the disc or be certified disjoint from it.
"""
function eigencount_in_disc(comps::Vector{EigComponent}, c::Number, r::Real)
    onepu = nextfloat(1.0)
    cnt = 0
    for C in comps
        d = mulu(absu(C.center - c), onepu)
        dl = divd(absd(C.center - c), onepu)
        if addu(d, C.radius) < r                  # inside the open disc
            cnt += C.count
        elseif dl > addu(Float64(r), C.radius)    # disjoint
            continue
        else
            return (0, false)
        end
    end
    return (cnt, true)
end

"""
    eigencount_outside(comps, R) -> (count, ok)

Number of eigenvalues of the compression in `{|z| > R}`: every component must lie in it or in
`{|z| < R}`.
"""
function eigencount_outside(comps::Vector{EigComponent}, R::Real)
    cnt = 0
    for C in comps
        if subd(absd(C.center), C.radius) > R
            cnt += C.count
        elseif addu(absu(C.center), C.radius) < R
            continue
        else
            return (0, false)
        end
    end
    return (cnt, true)
end

# ------------------------------------------------------------------------------------------------
# The exported certificate of a dense matrix (kept name: miyajima_rump)
# ------------------------------------------------------------------------------------------------

"""
    EigCluster

A certified cluster of eigenvalues: `count` eigenvalues (with multiplicity) in the disc
`D(hull_center, hull_radius)`, occupying the columns `idx` of the certified frame.
"""
struct EigCluster
    idx::Vector{Int}
    count::Int
    hull_center::ComplexF64
    hull_radius::Float64
end

"""
    MiyajimaRump

Result of [`miyajima_rump`](@ref): the compression certificate `cert`, the enclosure family used
for `clusters` (`:global`, the regions of thm:enclosure, or `:local`, those of prop:blockgersg; the
two are not mixed block by block), `centres`/`radii` of the outer discs of that family, `kappa`,
`nrmR2`, `betaΛ`. Every eigenvalue lies in some cluster disc, and each cluster disc holds exactly
`count` eigenvalues (cor:clusters).
"""
struct MiyajimaRump
    ok::Bool
    cert::CompressionCertificate
    family::Symbol
    clusters::Vector{EigCluster}
    centres::Vector{ComplexF64}
    radii::Vector{Float64}
    kappa::Float64
    nrmR2::Float64
    betaΛ::Float64
end

"""
    miyajima_rump(G; M = nothing, method = :sylvester, refine = 20, kappa_mode = :svdbox,
                  cluster_tol = nothing, xmax = 1e8, family = :auto) -> MiyajimaRump

Certified eigenvalue enclosure of the compression (sec:vbd): candidate by `method`
(`:sylvester`, `candidate_sylvester`; `:newton`, `candidate_newton` with `refine` iterations;
`:best`, both, keeping the one whose chosen family has the smaller largest cluster radius),
certified by [`certify_compression`](@ref). `G` may be a plain matrix (exact) or a ball matrix;
`M` the ball enclosure of Q*Q for a pencil. `family = :auto` picks, as a whole, the enclosure family
whose largest cluster radius is smaller. When `‖R2‖ < 1` is not certified the result has
`ok = false` and a single cluster of infinite radius, which is still a valid statement.
"""
function miyajima_rump(G::AbstractMatrix; M = nothing, method::Symbol = :sylvester,
        refine::Integer = 20, kappa_mode::Symbol = :svdbox, cluster_tol = nothing,
        xmax::Real = 1e8, family::Symbol = :auto)
    if method === :best
        a = miyajima_rump(G; M, method = :sylvester, refine, kappa_mode, cluster_tol, xmax,
            family)
        b = miyajima_rump(G; M, method = :newton, refine, kappa_mode, cluster_tol, xmax, family)
        ra = a.ok ? maximum(c -> c.hull_radius, a.clusters) : Inf
        rb = b.ok ? maximum(c -> c.hull_radius, b.clusters) : Inf
        return rb < ra ? b : a
    end
    cand = method === :newton ? candidate_newton(G; M, cluster_tol, maxiter = refine) :
           candidate_sylvester(G; M, cluster_tol, xmax)
    cc = certify_compression(G, cand; M, kappa_mode)
    return MiyajimaRump(cc; family)
end

function MiyajimaRump(cc::CompressionCertificate; family::Symbol = :auto)
    k = cc.k
    if !cc.ok
        return MiyajimaRump(false, cc, :none, [EigCluster(collect(1:k), k, 0.0im, Inf)],
            cc.centres, fill(Inf, length(cc.centres)), Inf, cc.nrmR2, Inf)
    end
    maxr(cs) = maximum(C -> C.radius, cs)
    fam = family === :auto ?
          (maxr(cc.components_local) < maxr(cc.components_global) ? :local : :global) : family
    comps = fam === :local ? cc.components_local : cc.components_global
    radii = fam === :local ? cc.ρlocal : cc.ρglobal
    clusters = [EigCluster(C.idx, C.count, C.center, C.radius) for C in comps]
    return MiyajimaRump(true, cc, fam, clusters, cc.centres, radii, cc.kappa, cc.nrmR2,
        cc.betaΛ)
end

Base.show(io::IO, R::MiyajimaRump) = print(io, "MiyajimaRump(ok = ", R.ok, ", ",
    length(R.clusters), " clusters, family = ", R.family, ", κ ≤ ", R.kappa, ", β_Λ ≤ ",
    R.betaΛ, ")")

# ------------------------------------------------------------------------------------------------
# The split by modulus (sec:candidate with two blocks, certified by thm:assembly)
# ------------------------------------------------------------------------------------------------
#
# The full block diagonalisation of sec:candidate separates every cluster of eigenvalues of the
# compression from every other one. Only the eigenvalues outside a circle are wanted, and the cost
# of separating the eigenvalues inside the circle from one another is what destroys the certificate
# on the Belousov-Zhabotinsky compressions (see the report 20260912_compression_split.md): κ(W)
# reaches 1.2e9, ‖R2‖ 2e-2 and the localised radii of prop:blockgersg merge every component into
# one.
#
# Here the candidate has two blocks: the leading block E, carrying every eigenvalue of modulus
# above a threshold R' below R, and the trailing block J, carrying all the rest. J is simply the
# part of the compression's spectrum inside the circle, which the certificate never reads: it may
# hold genuine eigenvalues of A below R, whatever approximates the essential spectrum, and
# artefacts of the discretisation, and at this level nothing distinguishes them and nothing needs
# to. The only property required of J is the certified bound ‖J^m‖^{1/m} < R, which keeps it from
# contributing to the count outside. Write k₊ for the size of E, Λ = E ⊕ J, and, as in the header
# of this file,
#
#   R2 = Y(MW) − I,   R1 = Y(G̃W − MWΛ),   M_c = (MW)⁻¹G̃W,   M_c − Λ = (I + R2)⁻¹R1 .
#
# Split M_c into the 2×2 blocks of the splitting ℂ^{k₊} ⊕ ℂ^{k−k₊}:
#
#   M_c = [ E + Δ₁₁   Δ₁₂ ;  Δ₂₁   J + Δ₂₂ ],       Δ = M_c − Λ .
#
# Write (I + R2)⁻¹ = I + S with ‖S‖ ≤ ‖R2‖/(1 − ‖R2‖) (lem:neumann). Then Δ = R1 + S R1, and the
# rows of R1 in a block row are the rows of that block row of R1 while the second term is bounded
# in norm by ‖S‖‖R1‖ whatever the rows, so
#
#   ‖Δ₁₁‖, ‖Δ₁₂‖ ≤ β₁ := ‖R1[1:k₊, :]‖ + ‖R2‖·β_Λ,
#   ‖Δ₂₁‖, ‖Δ₂₂‖ ≤ β₂ := ‖R1[k₊+1:k, :]‖ + ‖R2‖·β_Λ,
#
# both at most β_Λ = ‖R1‖/(1 − ‖R2‖) of lem:deviation up to the same second term. (Taking instead
# ‖R1[rows]‖/(1 − ‖R2‖) would be WRONG: the rows of (I + R2)⁻¹R1 are the rows of (I + R2)⁻¹ times
# the WHOLE of R1, not the inverse times those rows of R1.) Theorem thm:assembly applied to M_c with
# this splitting, σ_min and 1/‖(·)⁻¹‖ being 1-Lipschitz in the matrix, gives at every z
#
#   σ_min(M_c − zI) ≥ floor( σ_E(z) − β₁,  σ_J(z) − β₂,  β₁,  β₂ ),                      (*)
#
# `floor` being the larger of eq:assembly and eq:blockdom, with
#
#   σ_E(z) = σ_min(E − zI)                     one svdbox of the SMALL k₊×k₊ block, per z,
#   σ_J(z) = 1/‖(J − zI)⁻¹‖ ≥ eq:farpower      from ḡ_j ≥ ‖J^j‖, scalars, per z,
#
# and the transport of eq:blockres closes it:
#
#   σ_min(G̃ − zM) ≥ σ_min(M_c − zI)/κ,         κ ≥ ‖(MW)⁻¹‖‖W‖  (lem:neumann).
#
# So the deviation of the collapsed matrix from the candidate is accounted for three times over:
# subtracted from each diagonal quantity and used as both couplings. Nothing is certified about
# the candidate; the certified object is G̃ (the pencil (G̃, M)).
#
# The norms of the powers of J are computed on cJ with c a power of two (rem:impl), and the
# region radius and σ_J are evaluated in the scaled variable, which keeps |z|^m in range at every
# scale of the operator.

"""
    ModulusSplit

Output of [`split_compression`](@ref): the candidate split by modulus at the threshold `Rp`
(`= frac·R`) and its certificate. Certified: `nrmR2 ≥ ‖R2‖ < 1`, `nrmR1 ≥ ‖R1‖`,
`betaΛ ≥ ‖M_c − Λ‖` (lem:deviation), the block-row deviations `beta1`, `beta2`,
`kappa ≥ ‖W‖‖(MW)⁻¹‖` (lem:neumann), `gbars[j] ≥ ‖(cJ)^j‖` (`c = scale`, a power of two) and
`rhoJ`, a radius with `{z : σ_min(J − zI) ≤ β_Λ} ⊆ D(0, rhoJ)` for the TWO-BLOCK `β_Λ` below
(the regions of the nested certificate carry its own deviation and are recomputed in
[`split_block_discs`](@ref)). `E` and `J` are the two diagonal
blocks of `Λ` (floating point), `kplus = size(E, 1)`, `nrmX` the norm of the Sylvester solution.
`gbar_order[j] = ḡ_j^{1/j}` is the certified bound at each order and `gbarJ` its least value;
`rho_fp` is the largest modulus of an eigenvalue of the trailing block and `nout_fp` the number of
eigenvalues of the compression outside the circle, both in floating point and neither certified,
so that the gap between the certified bound and the block's own spectral radius is visible.
`tries` is how many values of `k₊` were tried by the adaptive rule; `reason` says whether both
the condition on the trailing block held (`:ok`) and, if not, what happened: `:r2` (‖R2‖ < 1 not
certified), `:radius` (no `k₊` at all brings `ḡ_J` below the circle) or `:cost` (the condition asks
for a `k₊` past the stopping rule). `kwanted` is the `k₊` the condition asks for, `tz` the measured
seconds per `z` that `k₊` implies, and `kcap` the stopping rule in force. `cert_ok` is the certification
alone (`‖R2‖ < 1`): every number below is sound whenever it holds, `ok` being the narrower claim
that the split delivers the exterior. `inner` is the certificate of
the nested candidate, in which `E` is block diagonalised further and `J` is left alone; it carries
the discs of thm:enclosure and the counts of cor:clusters. `ok = false` when `‖R2‖ < 1` could not
be certified.
"""
struct ModulusSplit
    ok::Bool
    cert_ok::Bool
    reason::Symbol
    kcap::Int
    kwanted::Int
    tz::Float64
    k::Int
    kplus::Int
    R::Float64
    Rp::Float64
    W::Matrix{ComplexF64}
    Y::Matrix{ComplexF64}
    E::Matrix{ComplexF64}
    J::Matrix{ComplexF64}
    nrmX::Float64
    nrmR2::Float64
    nrmR1::Float64
    betaΛ::Float64
    beta1::Float64
    beta2::Float64
    nrmW::Float64
    nrmY::Float64
    kappa::Float64
    scale::Float64
    gbars::Vector{Float64}
    gbarJ::Float64
    gbar_order::Vector{Float64}
    rhoJ::Float64
    rho_fp::Float64
    nout_fp::Int
    tries::Int
    inner::Union{Nothing, CompressionCertificate}
    time::NamedTuple
end

function Base.show(io::IO, ms::ModulusSplit)
    print(io, "ModulusSplit(ok = ", ms.ok, " (", ms.reason, "), k₊ = ", ms.kplus, "/", ms.k,
        ", ‖X‖ = ", ms.nrmX, ", ‖R2‖ ≤ ", ms.nrmR2, ", β_Λ ≤ ", ms.betaΛ, ", κ ≤ ", ms.kappa,
        ", ḡ_J ≤ ", ms.gbarJ, " against R = ", ms.R, ")")
end

"""
    split_reason(ms) -> String

One line saying why [`split_compression`](@ref) declined, with the numbers: whether the condition
on the trailing block failed, or the stopping rule bit, and by how much.
"""
function split_reason(ms::ModulusSplit)
    if ms.reason === :ok
        return "the condition holds at k₊ = $(ms.kplus), within the stopping rule k₊ ≤ $(ms.kcap)"
    elseif ms.reason === :r2
        return "‖R2‖ < 1 was not certified (‖R2‖ ≤ $(ms.nrmR2))"
    elseif ms.reason === :cost
        return "a cost decline, not a failure of the method: the condition asks for " *
               "k₊ = $(ms.kwanted) of $(ms.k), which costs $(ms.tz) s per z for one verified " *
               "SVD of that size, and that is past the stopping rule; at k₊ = $(ms.kplus) the " *
               "certified bound is ḡ_J ≤ $(ms.gbarJ) against R = $(ms.R)"
    elseif ms.reason === :radius
        return "no k₊ brings the trailing block below the circle: at the largest admissible " *
               "one, k₊ = $(ms.kplus) of $(ms.k), the certified bound is ḡ_J ≤ $(ms.gbarJ) " *
               "against R = $(ms.R)"
    end
    return string(ms.reason)
end

# The power of two c with c·x in [1, 2), and 1 when x is not finite and positive.
function _pow2_scale(x::Real)
    (isfinite(x) && x > 0) || return 1.0
    p = -exponent(Float64(x))
    e = clamp(p, -1000, 1000)
    return 2.0^e
end

"""
    _power_bounds(J, m) -> (c, g)

`c` a power of two and `g[j] ≥ ‖(cJ)^j‖` for `j = 1:m`, certified (ball powers, the entrywise
norm bounds of `src/normbounds.jl`), with `c` chosen so that `‖cJ‖` lies near 1 (rem:impl); the
scaling is exact, and every use of `g` below is in the scaled variable, so no power of `|z|`
leaves the range of Float64. `g[j]` is the least of the ball bound and `g[j−1]g[1]`.
"""
function _power_bounds(J::AbstractMatrix, m::Integer)
    ℓ = size(J, 1)
    (ℓ == 0 || m < 1) && return (1.0, Float64[])
    c = _pow2_scale(interp_up(BallMatrix(ComplexF64.(J))))
    Jb = BallMatrix(ComplexF64.(c .* J))
    mm = min(ℓ, Int(m))
    g = zeros(mm)
    Pj = Jb
    g[1] = opnorm_up(Jb)
    for j in 2:mm
        Pj = _bmul(Pj, Jb)
        g[j] = min(opnorm_up(Pj), mulu(g[j - 1], g[1]))
        isfinite(g[j]) || (g = g[1:(j - 1)]; break)
    end
    return (c, g)
end

# The candidate of the nested certificate: the columns of the leading block further transformed by
# a block diagonalisation of E, the trailing block left alone.
function _nested_candidate(Gm, Mm, W, Y, kplus, E; xmax, cluster_tol)
    k = size(W, 1)
    candE = candidate_sylvester(E; cluster_tol, xmax)
    Wt = copy(W)
    Yt = copy(Y)
    if kplus > 0
        Wt[:, 1:kplus] = W[:, 1:kplus] * candE.W
        Yt[1:kplus, :] = candE.Y * Y[1:kplus, :]
    end
    blocks = Vector{Int}[copy(b) for b in candE.blocks]
    kplus < k && push!(blocks, collect((kplus + 1):k))
    Ypn = Mm === nothing ? Yt : Yt / Mm
    Λ = _blockdiag(Ypn * (Gm * Wt), blocks)
    return CompressionCandidate(Wt, Ypn, Λ, blocks, :modulus_split,
        (; kplus, nblocksE = length(candE.blocks), candE.info...))
end

"""
    split_compression(G, R; M = nothing, frac = 0.9, m = 8, margin = 1.02, kmax = nothing,
                      budget = nothing, kappa_mode = :svdbox, inner = true, xmax = 1e8,
                      cluster_tol = nothing) -> ModulusSplit

The candidate of sec:candidate stopped after one Sylvester solve, splitting the compression by
modulus, and its certification:

1. complex Schur form of `M⁻¹G̃` (`M = I` unless given), reordered by `ordschur!` so that the `k₊`
   eigenvalues of largest modulus come first, giving the leading block `E` and the trailing block
   `J`; one triangular Sylvester equation `E X − X J = −T₁₂` by LAPACK `trsyl`, applied as
   `W ← V[I X; 0 I]`, `Y ← [I −X; 0 I]V*` (the exact inverse); QR inside each of the two blocks
   and not across (sec:candidate step 4); `Λ = E ⊕ J` the block diagonal of `YG̃W`.
   `k₊` is chosen by one condition: `ḡ_J·margin < R` with `ḡ_J = min_{j ≤ m} ‖J^j‖^{1/j}`
   certified. It starts at `#{|λ| > frac·R}` and grows, one admissible value at a time (a value
   that does not split a pair of equal moduli), until that holds. There is no cap on `k₊`: the
   circle fixes which eigenvalues must be resolved and the condition fixes the rest, so the size
   of the resolved tier is an outcome, and its only consequence is the per-`z` cost of one
   verified SVD of that size. The search is capped only by a STOPPING RULE, `kmax` (a ceiling, by
   default `k`, so none) or `budget` (seconds per `z`, measured by
   [`_split_cost_per_z`](@ref)). If the condition asks for more than the rule allows the call
   declines with `reason = :cost`, `kwanted` and `tz` recording what it asked for and what it
   would cost; if no `k₊` at all works the reason is `:radius`. Either way the certificate is
   still computed at the largest `k₊` the rule allows and is still sound (`cert_ok = true`), and
   what declines is the claim that the trailing block contributes nothing outside the circle.
   [`split_reason`](@ref) prints the line.

   `J` is simply the part of the compression's spectrum inside the circle, which the certificate
   never reads: it may hold genuine eigenvalues of `A` below `R`, whatever approximates the
   essential spectrum, and artefacts of the discretisation, and at this level nothing
   distinguishes them and nothing needs to. The only property required of `J` is the certified
   bound `‖J^m‖^{1/m} < R`, which keeps it from contributing to the count outside;
2. the residuals eq:residuals in ball arithmetic, `β_Λ` of lem:deviation, the two block-row
   deviations `β₁`, `β₂`, and `κ` of lem:neumann;
3. `ḡ_j ≥ ‖(cJ)^j‖` for `j ≤ m` and the radius `rhoJ` of `{σ_min(J − zI) ≤ β_Λ}` about 0
   (prop:power applied to `J`), which is what says that the trailing block keeps its spectrum
   inside the circle;
4. with `inner = true`, the nested candidate in which the leading block alone is block
   diagonalised by [`candidate_sylvester`](@ref) and the trailing block is left as one block,
   certified by [`certify_compression`](@ref): the discs of thm:enclosure and the counts of
   cor:clusters come from it.

Nothing in step 1 is certified; steps 2 to 4 are. [`sigma_split_lo`](@ref) is the resolvent bound
this produces.
"""
function split_compression(G::AbstractMatrix, R::Real; M = nothing, frac::Real = 0.9,
        m::Integer = 8, margin::Real = 1.02, kmax = nothing, budget = nothing,
        kappa_mode::Symbol = :svdbox, inner::Bool = true, xmax::Real = 1e8, cluster_tol = nothing)
    t00 = time()
    sv = kappa_mode === :svdbox
    Gb = _promote_ball(_ball(G))
    Gm = ComplexF64.(Matrix(mid(G)))
    Mm = M === nothing ? nothing : ComplexF64.(Matrix(mid(M)))
    Gc = Mm === nothing ? Gm : Mm \ Gm
    k = size(Gc, 1)
    R = Float64(R)
    S = schur(Gc)
    T0 = Matrix(S.T)
    V0 = Matrix(S.Z)
    d0 = diag(T0)
    a = sort(abs.(d0); rev = true)
    nout_fp = count(>(R), a)
    cands = _split_candidates(a, R, Float64(frac))
    isempty(cands) && (cands = [clamp(count(>(Float64(frac) * R), a), 0, k)])
    kcap = kmax === nothing ? k : clamp(Int(kmax), 0, k)
    # The search is not capped: k₊ is whatever the condition on the trailing block asks for, and
    # the size of the resolved tier is an outcome. The stopping rule is applied afterwards and is
    # a statement about the per-z cost, not about the method.
    tries = 0
    chosen = nothing
    passed = false
    for kp in cands
        tries += 1
        cand = _split_candidate(T0, V0, Gm, Mm, kp)
        # floating-point proxy for ‖J^m‖^{1/m}: an estimate, used only to choose k₊
        est = _rho_estimate_fp(cand.J, m)
        chosen = (; cand..., est, kplus = kp)
        if est * Float64(margin) < R
            passed = true
            break
        end
    end
    kneed = passed ? chosen.kplus : -1
    tz = passed ? _split_cost_per_z(chosen.E, R) : NaN
    over = passed && (chosen.kplus > kcap || (budget !== nothing && tz > Float64(budget)))
    if over || !passed
        # certify at the largest admissible k₊ the stopping rule allows, so that every number is
        # still reported, and decline
        allow = [j for j in cands if j <= kcap]
        kp = isempty(allow) ? clamp(kcap, 0, k) : last(allow)
        tries += 1
        cand = _split_candidate(T0, V0, Gm, Mm, kp)
        chosen = (; cand..., est = _rho_estimate_fp(cand.J, m), kplus = kp)
    end
    W, Y, E, J, Λ, nX, kplus = chosen.W, chosen.Y, chosen.E, chosen.J, chosen.Λ, chosen.nX,
        chosen.kplus
    Rp = kplus == 0 ? Inf : (kplus == k ? 0.0 : sqrt(a[kplus] * a[kplus + 1]))
    rho_fp = kplus < k ? maximum(abs, @view a[(kplus + 1):k]) : 0.0
    t_cand = time() - t00

    # the certification
    t1 = time()
    Wb = BallMatrix(W)
    Yb = BallMatrix(Y)
    MW = M === nothing ? Wb : _bmul(M, Wb)
    R2 = _bsub(_bmul(Yb, MW), Matrix{ComplexF64}(I, k, k))
    nR2 = opnorm_up(R2; svd = sv)
    bad() = ModulusSplit(false, false, :r2, kcap, kneed, tz, k, kplus, R, Rp, W, Y, E, J, nX,
        nR2, Inf, Inf, Inf, Inf, Inf, Inf, Inf, 1.0, Float64[], Inf, Float64[], Inf, rho_fp,
        nout_fp, tries, nothing,
        (; candidate = t_cand, certificate = time() - t1, powers = 0.0, inner = 0.0))
    (isfinite(nR2) && nR2 < 1) || return bad()
    GW = _bmul(Gb, Wb)
    R1 = _bmul(Yb, _bsub(GW, _bmul(MW, BallMatrix(Λ))))
    nR1 = opnorm_up(R1; svd = sv)
    omr = subd(1.0, nR2)
    βΛ = divu(nR1, omr)
    # Δ = R1 + S R1 with ‖S‖ ≤ ‖R2‖/(1 − ‖R2‖), so a block row of Δ is bounded by the norm of
    # that block row of R1 plus ‖S‖‖R1‖ = ‖R2‖β_Λ (see the header of this section)
    βS = mulu(nR2, βΛ)
    β1 = kplus > 0 ? addu(opnorm_up(R1[1:kplus, :]; svd = sv), βS) : 0.0
    β2 = kplus < k ? addu(opnorm_up(R1[(kplus + 1):k, :]; svd = sv), βS) : 0.0
    nW = opnorm_up(Wb; svd = sv)
    nY = opnorm_up(Yb; svd = sv)
    κ = divu(mulu(nY, nW), omr)
    t_cert = time() - t1

    # the trailing block through its powers
    t2 = time()
    c, g = _power_bounds(J, m)
    gorder = [divu(rootu(g[j], j), c) for j in eachindex(g)]
    gbarJ = isempty(gorder) ? 0.0 : minimum(gorder)
    ρJ = isempty(g) ? 0.0 : divu(_radius_from_powers(g, mulu(c, βΛ)), c)
    t_pow = time() - t2

    # the nested certificate for the discs and the counts
    t3 = time()
    icc = nothing
    if inner && kplus > 0
        cand = _nested_candidate(Gm, Mm, W, Y, kplus, E; xmax, cluster_tol)
        icc = certify_compression(G, cand; M, kappa_mode)
    end
    t_inner = time() - t3
    # the two conditions on the leading block: the ceiling, and the certified bound on the
    # trailing block's powers below the circle with the same margin used to choose k₊
    radius_ok = kplus == k || mulu(gbarJ, Float64(margin)) < R
    reason = over ? :cost : (radius_ok ? :ok : :radius)
    return ModulusSplit(reason === :ok, true, reason, kcap, kneed, tz, k, kplus, R, Rp, W, Y, E, J,
        nX, nR2, nR1, βΛ, β1, β2, nW, nY, κ, c, g, gbarJ, gorder, ρJ, rho_fp, nout_fp, tries,
        icc, (; candidate = t_cand, certificate = t_cert, powers = t_pow, inner = t_inner))
end

"""
    _split_candidates(a, R, frac) -> Vector{Int}

The sizes `k₊` the leading block may take, in increasing order: the moduli `a` of the eigenvalues
of the Schur form sorted downwards, and `k₊ = j` admissible when `a[j] > a[j+1]` strictly (a
relative gap above `1e-12`), so that the threshold `√(a[j]a[j+1])` selects exactly the top `j` and
no conjugate pair, whose two members have the same modulus, is split. The list starts at the
smallest admissible `j ≥ #{|λ| > frac·R}` and ends at `k`.
"""
function _split_candidates(a::AbstractVector{<:Real}, R::Real, frac::Real)
    k = length(a)
    j0 = isfinite(frac) ? count(>(frac * R), a) : 0
    out = Int[]
    for j in max(j0, 0):k
        j == 0 && (push!(out, 0); continue)
        j == k && (push!(out, k); continue)
        a[j] > a[j + 1] * (1 + 1e-12) && push!(out, j)
    end
    isempty(out) && push!(out, min(j0, k))
    return out
end

# One candidate of the split at a given k₊: reorder a copy of the Schur pair so that the k₊
# eigenvalues of largest modulus come first, decouple by one triangular Sylvester solve,
# orthonormalise inside each of the two blocks, and read off Λ = E ⊕ J.
function _split_candidate(T0, V0, Gm, Mm, kplus)
    k = size(T0, 1)
    T = copy(T0)
    V = copy(V0)
    if 0 < kplus < k
        a = sort(abs.(diag(T)); rev = true)
        thr = sqrt(a[kplus] * a[kplus + 1])
        sel = abs.(diag(T)) .> thr
        count(sel) == kplus || (sel = _topk_mask(abs.(diag(T)), kplus))
        _ordschur!(T, V, sel)
    end
    W = copy(V)
    Y = Matrix(V')
    nX = 0.0
    if 0 < kplus < k
        lead = 1:kplus
        rest = (kplus + 1):k
        C = -T[lead, rest]
        local X, scl
        try
            X, scl = LinearAlgebra.LAPACK.trsyl!('N', 'N', T[lead, lead], T[rest, rest], C, -1)
        catch
            X, scl = fill(ComplexF64(NaN), kplus, k - kplus), 1.0
        end
        X = X ./ scl
        nX = norm(X)
        if isfinite(nX)
            W[:, rest] .+= W[:, lead] * X
            Y[lead, :] .-= X * Y[rest, :]
        else
            nX = Inf
        end
    end
    blocks2 = Vector{Int}[]
    kplus > 0 && push!(blocks2, collect(1:kplus))
    kplus < k && push!(blocks2, collect((kplus + 1):k))
    _orthonormalise_blocks!(W, Y, blocks2)
    Yp = Mm === nothing ? Y : Y / Mm
    Λf = Yp * (Gm * W)
    E = kplus > 0 ? Matrix(Λf[1:kplus, 1:kplus]) : zeros(ComplexF64, 0, 0)
    J = kplus < k ? Matrix(Λf[(kplus + 1):k, (kplus + 1):k]) : zeros(ComplexF64, 0, 0)
    return (; W, Y = Yp, E, J, Λ = _blockdiag(Λf, blocks2), nX)
end

# Mask selecting the `kplus` largest entries of `v` (ties broken by position).
function _topk_mask(v::AbstractVector{<:Real}, kplus::Integer)
    p = sortperm(v; rev = true)
    sel = falses(length(v))
    sel[p[1:kplus]] .= true
    return sel
end

"""
    _split_cost_per_z(E, R) -> seconds

The measured cost of one `z` for the split: one `svdbox` of the `k₊ × k₊` leading block `E − zI`,
timed (three repetitions, the least kept). That is the only per-`z` work whose cost depends on
`k₊`, so the stopping rule is expressed in it.
"""
function _split_cost_per_z(E::AbstractMatrix, R::Real)
    size(E, 1) == 0 && return 0.0
    z = ComplexF64(R)
    best = Inf
    for _ in 1:3
        t = time()
        _sigmin_block_lo(E, z)
        best = min(best, time() - t)
    end
    return best
end

# min_j ‖J^j‖^{1/j} for j ≤ m in plain Float64: an ESTIMATE, used only to choose k₊.
function _rho_estimate_fp(J::AbstractMatrix, m::Integer)
    ℓ = size(J, 1)
    ℓ == 0 && return 0.0
    c = _pow2_scale(opnorm(J, 1))
    P = ComplexF64.(c .* J)
    Jc = copy(P)
    best = Inf
    for j in 1:min(ℓ, Int(m))
        j > 1 && (P = P * Jc)
        n = opnorm(P)
        best = min(best, n^(1 / j) / c)
    end
    return best
end

"""
    sigma_J_lo(ms, z) -> Float64

Certified lower bound on `1/‖(J − zI)⁻¹‖` for the trailing block, by eq:farpower in the scaled
variable: `σ_J(z) ≥ farpower_lo(cz, ḡ)/c` with `ḡ_j ≥ ‖(cJ)^j‖`, which is prop:power applied to
`cJ` and divided by the power of two `c`. Returns `Inf` when the trailing block is empty.
"""
function sigma_J_lo(ms::ModulusSplit, z::Number)
    ms.kplus == ms.k && return Inf
    isempty(ms.gbars) && return 0.0
    v = farpower_lo(ms.scale * ComplexF64(z), ms.gbars)
    return v > 0 ? divd(v, ms.scale) : 0.0
end

"""
    sigma_split_lo(ms, z) -> Float64

Certified lower bound on `σ_min(G̃ − zM)` (on `σ_min(G11 − zI)` when `M = I`) through the split:
`(*)` of the header of this section, that is the larger of eq:assembly and eq:blockdom applied to
the collapsed matrix `M_c` with the splitting `ℂ^{k₊} ⊕ ℂ^{k−k₊}`,

    σ11 ≥ σ_min(E − zI) − β₁,   σ22 ≥ σ_J(z) − β₂,   γ1 ≤ β₁,   γ2 ≤ β₂,

divided by `κ` (eq:blockres). One `svdbox` of the k₊×k₊ block `E` and a handful of scalars per
`z`. When one of the two blocks is empty this reduces to eq:blockres with a single block.
"""
function sigma_split_lo(ms::ModulusSplit, z::Number)
    ms.cert_ok || return 0.0
    if ms.kplus == 0
        v = subd(sigma_J_lo(ms, z), ms.betaΛ)
        return v > 0 ? divd(v, ms.kappa) : 0.0
    elseif ms.kplus == ms.k
        v = subd(_sigmin_block_lo(ms.E, z), ms.betaΛ)
        return v > 0 ? divd(v, ms.kappa) : 0.0
    end
    s11 = subd(_sigmin_block_lo(ms.E, z), ms.beta1)
    s22 = subd(sigma_J_lo(ms, z), ms.beta2)
    f1, _ = assembly_floor(s11, s22, ms.beta1, ms.beta2)
    f2, _ = blockdom_floor(s11, s22, ms.beta1, ms.beta2)
    v = max(f1, f2)
    return v > 0 ? divd(v, ms.kappa) : 0.0
end

"""
    split_sigma_blocks_lo(ms, z) -> Float64

`min_i σ_min(P_i − zI)` over the blocks of the NESTED candidate, certified from below, the
trailing block being handled by eq:farpower instead of a `svdbox` of its `k−k₊` rows. This is the
1-Lipschitz function `certify_block_projector` covers on a contour.
"""
function split_sigma_blocks_lo(ms::ModulusSplit, z::Number)
    ms.inner === nothing && return 0.0
    cc = ms.inner
    v = sigma_J_lo(ms, z)
    nb = length(cc.P)
    last_is_J = ms.kplus < ms.k
    for i in 1:(last_is_J ? nb - 1 : nb)
        v = min(v, _sigmin_block_lo(cc.P[i], z))
    end
    return v
end

"""
    split_components(ms; family = :auto) -> Vector{EigComponent}

The certified components of the nested certificate, with the trailing block's own component
replaced by the disc `D(0, rhoJ)` computed about the origin (thm:enclosure applied to `J` through
prop:power in the scaled variable), which is the disc the exterior count needs.
"""
function split_components(ms::ModulusSplit; family::Symbol = :auto)
    (ms.cert_ok && ms.inner !== nothing && ms.inner.ok) || return EigComponent[]
    centres, ρ, _ = split_block_discs(ms; family)
    return _components(centres, ρ, ms.inner.blocks)
end

"""
    split_block_discs(ms; family = :auto) -> (centres, radii, family)

The outer discs of the blocks of the nested certificate, with the trailing block's disc replaced
by `D(0, rhoJ)`: about the origin, and from prop:power in the scaled variable rather than from the
ball powers of `J − c_JI`. `family = :auto` takes, as a whole, the family (thm:enclosure with β_Λ,
or prop:blockgersg with the localised radii) whose largest component radius is smaller.
"""
function split_block_discs(ms::ModulusSplit; family::Symbol = :auto)
    cc = ms.inner
    fam = family
    if fam === :auto
        maxr(cs) = isempty(cs) ? Inf : maximum(C -> C.radius, cs)
        fam = maxr(cc.components_local) < maxr(cc.components_global) ? :local : :global
    end
    ρ = fam === :local ? copy(cc.ρlocal) : copy(cc.ρglobal)
    centres = copy(cc.centres)
    if ms.kplus < ms.k
        # The last block is J. Two discs enclose its region: the one `certify_compression` builds
        # about the mean of its diagonal, and the one prop:power builds about the ORIGIN in the
        # scaled variable. Their centres differ, so the two radii cannot be combined; we keep the
        # disc that reaches less far from the origin, which is what the exterior count reads.
        # the deviation of the NESTED candidate, which is the one thm:enclosure is applied with
        # here, and not the two-block `ms.betaΛ`
        ρ1 = _rho_J(ms, fam === :local ? cc.rloc[end] : cc.betaΛ)
        if ρ1 < addu(absu(centres[end]), ρ[end])
            centres[end] = 0.0 + 0.0im
            ρ[end] = ρ1
        end
    end
    return centres, ρ, fam
end

"""
    _rho_J(ms, β) -> ρ

A radius with `{z : σ_min(J − zI) ≤ β} ⊆ D(0, ρ)`, from prop:power applied to `cJ` in the scaled
variable (`c = ms.scale`, a power of two) and divided by `c`. The caller supplies the deviation β
that belongs to the certificate the region is read from: `ms.betaΛ` for the two-block candidate,
and the nested certificate's own `β_Λ` or localised radius when the region is one of its blocks.
"""
function _rho_J(ms::ModulusSplit, β::Real)
    isempty(ms.gbars) && return Inf
    return divu(_radius_from_powers(ms.gbars, mulu(ms.scale, Float64(β))), ms.scale)
end
