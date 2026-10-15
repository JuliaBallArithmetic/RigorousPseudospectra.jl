# §5 (sec:split): the certified splitting. The four certified numbers, ḡ ≥ ‖G22‖ (§4), γ1, γ2 from
# eq:gram (§3.3) and σ11 from the compression (§3.2), fed to §2: the resolvent bound
# (thm:assembly with eq:farshort), the inclusion (prop:complete), the counts (prop:count), the
# exterior with isolation discs (cor:agree) and the gap (cor:gap), with the frame only almost
# orthonormal (prop:almostortho, sec:almostortho).
#
# The eigenvalues of the compression. The frame Q is orthonormal only to rounding, Q = ZS its polar
# decomposition, and the compression the theorems are about is X = Z*AZ, while what is formed is the
# ball G ⊇ Q*AQ = SXS. Its eigenvalues are enclosed by BallArithmetic's `verifyeigall` (Rump 2022,
# Theorem 2.2; where that theorem does not cover the whole spectrum, by the Gershgorin discs of the
# transformed matrix, `inclusion_discs`), and they are transferred to X by a homotopy: ‖S − I‖ ≤ 1 − √(1 − α) ≤ α,
# ‖S‖ ≤ √(1 + α), ‖X‖ ≤ ‖G‖/(1 − α), so
#
#     ‖G − X‖ = ‖(S − I)XS + X(S − I)‖ ≤ η_α := α(1 + √(1 + α))‖G‖/(1 − α),
#
# and along X + t(G − X), t ∈ [0, 1], no eigenvalue crosses a contour Γ on which
# σ_min(X − zI) > η_α. The cover of the contour certifies σ11 = σ_min(X − zI) from below on all of
# Γ (`check.minσ11`).
#
# The eigenvalues are enclosed for the midpoint Ĝ = mid(G), not for the ball: for an operator
# applied matrix-free the radius of G is a norm bound on the rounding of fl(AQ) spread over every
# entry, and carried through the eigenvector matrix it can exceed the separation of eigenvalues the
# count never needs to separate (on the BZ compression of experiment 5, k = 94: radius 1.9e-9 per
# entry, 4e-2 per row after the transformation, 7 of 72 clusters certified on G against all 94 on
# Ĝ). Since ‖Ĝ − Q*AQ‖ ≤ ‖rad(G)‖₂ =: ρ, the count read off Ĝ is a count for X whenever
# `check.minσ11 > η_α + ρ`, which is `etaG`.

"""
    SplitCertificate

Everything certified for one operator and one frame `Q` (n×k):

* `α ≥ ‖I − Q*Q‖ < 1`, `G ⊇ Q*AQ`, `M ⊇ Q*Q` (ball matrices);
* `gbar > ‖G22‖` and `abar > ‖A‖` (prop:fpchecks), or supplied by the caller (`far = nothing`);
* `γ1 ≥ ‖G12‖`, `γ2 ≥ ‖G21‖`, `snorm ≥ ‖G11‖` for the isometry Z of the polar decomposition
  `Q = ZS` (the factors of prop:almostortho already applied);
* `comp`, the inclusions of the eigenvalues of the midpoint of G by `verifyeigall`, or `nothing`;
* `etaG ≥ ‖mid(G) − Z*AZ‖`, the homotopy bound of the header.
"""
struct SplitCertificate
    n::Int
    k::Int
    Q::Matrix
    α::Float64
    G::BallMatrix
    M::BallMatrix
    gbar::Float64
    abar::Float64
    γ1::Float64
    γ2::Float64
    snorm::Float64
    far::Union{Nothing, FarBlockCertificate}
    fd::FrameData
    comp::Union{Nothing, VerifyEigAllResult}
    etaG::Float64
    hermitian::Bool
    time::NamedTuple
end

function Base.show(io::IO, sc::SplitCertificate)
    print(io, "SplitCertificate(n = $(sc.n), k = $(sc.k), α ≤ $(sc.α), ‖G22‖ < $(sc.gbar), ",
        "‖A‖ < $(sc.abar), γ1 ≤ $(sc.γ1), γ2 ≤ $(sc.γ2))")
end

"""
    certify_split(op::LinOp, Q; order = :auto, compression = :full, target = nothing,
                  coupling = :gram, hermitian = false, far_bound = nothing, kwargs...)
        -> SplitCertificate
    certify_split(A::AbstractMatrix, Q; kwargs...)
    certify_split(Amul, Aadj, n, Q; epsA, etaA = 0, kwargs...)

Certify the four numbers for the frame `Q`: frame data (couplings.jl), the far block by the
floating-point sweep and prop:fpchecks (farblock.jl; keyword arguments are passed to
[`certify_far_block`](@ref)), and the eigenvalues of the compression by `verifyeigall` when
`compression = :full` (or `true`); `:none` (or `false`) skips them, which is all a resolvent bound
needs. `far_bound = (ḡ, ā)` skips the sweep and uses the caller's bounds `‖G22‖ ≤ ḡ`, `‖A‖ ≤ ā`
instead. `hermitian = true` asserts `A = A*` (checked exactly for a stored matrix). `target` is the
radius `R` the certificate is aimed at, passed to the sweep.
"""
function certify_split(op::LinOp, Q::AbstractMatrix; order = :auto, compression = :full,
        target = nothing, coupling::Symbol = :gram, hermitian::Bool = false,
        far_bound = nothing, kwargs...)
    t0 = time()
    if hermitian && op.stored !== nothing
        ishermitian(op.stored) || error("hermitian = true but the stored matrix is not Hermitian")
    end
    full = compression === true || compression === :full
    full || compression === false || compression === :none ||
        error("compression must be :full or :none (or a Bool)")
    fd = frame_data(op, Q; coupling)
    t1 = time()
    if far_bound === nothing
        far = certify_far_block(op, Q, fd; order, target, kwargs...)
        gbar, abar = far.gbar, far.abar
    else
        far = nothing
        gbar, abar = Float64(far_bound[1]), Float64(far_bound[2])
    end
    t2 = time()
    s, γ1, γ2 = isfinite(abar) ? isometry_bounds(fd, abar) : (Inf, Inf, Inf)
    G = isfinite(abar) ? compression_ball(fd, abar) : fd.Gt
    comp = nothing
    etaG = Inf
    if full && isfinite(abar) && size(Q, 2) > 0
        comp = verifyeigall(_complex_ball(BallMatrix(Matrix(mid(G)))))
        α = fd.α
        ηα = div_up(mul_up(mul_up(α, add_up(1.0, sqrt_up(add_up(1.0, α)))),
                upper_bound_L2_opnorm(G; svd = true)), sub_down(1.0, α))
        ρ = upper_bound_L2_opnorm(BallMatrix(Matrix{Float64}(rad(G))); svd = true)
        etaG = add_up(ηα, ρ)
    end
    t3 = time()
    return SplitCertificate(size(Q, 1), size(Q, 2), Matrix(Q), fd.α, G, fd.Mq, gbar, abar,
        γ1, γ2, s, far, fd, comp, etaG, hermitian,
        (; frame = t1 - t0, far = t2 - t1, compression = t3 - t2, total = t3 - t0))
end

# verifyeigall works on a complex ball matrix: a real compression has complex eigenvectors
_complex_ball(G::BallMatrix) = eltype(mid(G)) <: Complex ? G :
                               BallMatrix(ComplexF64.(mid(G)), rad(G))

certify_split(A::AbstractMatrix, Q::AbstractMatrix; kwargs...) = certify_split(LinOp(A), Q;
    kwargs...)
function certify_split(Amul, Aadj, n::Integer, Q::AbstractMatrix; epsA::Real, etaA::Real = 0.0,
        kwargs...)
    certify_split(LinOp(Amul, Aadj, n; epsA, etaA), Q; kwargs...)
end

"""
    sigma11_lo(sc, z) -> Float64

Certified lower bound on `σ11(z) = σ_min(G11 − zI)`, G11 = Z*AZ the compression of the isometry:
`σ_min(Q*AQ − zQ*Q)/(1 + α)` (prop:almostortho), with `σ_min(Q*AQ − zQ*Q)` bounded below by one
`svdbox` of the whole ball pencil (thm:svdbox) at `O(k³)` per `z`. The keyword `method = :svd` is
accepted for the callers that name it; it is the only method.
"""
function sigma11_lo(sc::SplitCertificate, z::Number; method::Symbol = :svd)
    method === :svd || method === :auto ||
        error("sigma11_lo: the only method is :svd (got $(repr(method)))")
    sc.k == 0 && return Inf
    v = svd_lower_bound_sigma_min(sc.G - ComplexF64(z) * sc.M; method = SVD_METHOD)
    return div_down(v, add_up(1.0, sc.α))
end

"""
    sigma22_lo(sc, z) -> Float64

Certified lower bound on `σ22(z) = 1/‖(G22 − zI)⁻¹‖` by eq:farshort, `|z| − ḡ`.
"""
sigma22_lo(sc::SplitCertificate, z::Number) = farshort_lo(z, sc.gbar)

"""
    resolvent_floor(sc, z; method = :svd) -> NamedTuple

Certified lower bound on `σ_min(A − zI) = 1/‖(A − zI)⁻¹‖` at one point: Theorem thm:assembly
(`assembly`) and Proposition prop:blockdom (`blockdom`), each only under its own hypothesis, and
`floor` their maximum (both are lower bounds on the same number). `ok = false` and `floor = 0`
mean "no bound".
"""
function resolvent_floor(sc::SplitCertificate, z::Number; method::Symbol = :svd)
    s11 = sigma11_lo(sc, z; method)
    s22 = sigma22_lo(sc, z)
    f1, ok1 = assembly_floor(s11, s22, sc.γ1, sc.γ2)
    f2, ok2 = blockdom_floor(s11, s22, sc.γ1, sc.γ2)
    return (; floor = max(f1, f2), ok = ok1 || ok2, assembly = f1, blockdom = f2, σ11 = s11,
        σ22 = s22, γ1 = sc.γ1, γ2 = sc.γ2, gbar = sc.gbar)
end

"""
    pseudospectrum_outside(sc, z) -> NamedTuple
    pseudospectrum_outside(sc, ε, R) -> (δ, ok)

At a point: the certified floor of [`resolvent_floor`](@ref), so `z ∉ Λ_ε(A)` for every
`ε < floor`. For a level: Proposition prop:complete,
`Λ_δ(A) ∩ {|z| > R} ⊆ Λ_ε(G11) ∩ {|z| > R}`, with `δ` from [`inclusion_delta`](@ref).
"""
pseudospectrum_outside(sc::SplitCertificate, z::Number; kwargs...) = resolvent_floor(sc, z;
    kwargs...)
pseudospectrum_outside(sc::SplitCertificate, ε::Real, R::Real) = inclusion_delta(ε, R, sc.gbar,
    sc.γ1, sc.γ2)

"""
    check_circle(sc, c, r; region = :interior, kwargs...) -> CircleCheck

[`check_circle`](@ref) with the certificate's σ11 and σ22.
"""
function check_circle(sc::SplitCertificate, c::Number, r::Real; region::Symbol = :interior,
        method::Symbol = :svd, kwargs...)
    check_circle(z -> sigma11_lo(sc, z; method), sc.gbar, sc.γ1, sc.γ2, c, r; region,
        σ22lo = z -> sigma22_lo(sc, z), kwargs...)
end

"Whether the certificate carries discs holding all the eigenvalues of the compression, with counts."
_has_counts(sc::SplitCertificate) = sc.comp !== nothing && inclusion_discs(sc.comp)[4]

# the homotopy of the header: a count read off G holds for Z*AZ when σ11 > η_α on the contour
_transfers(sc::SplitCertificate, chk::CircleCheck) = chk.ok && chk.minσ11 > sc.etaG

"""
The components of the inclusions of the eigenvalues of G: the discs of `inclusion_discs` (those of
Theorem 2.2, or the Gershgorin discs of the transformed matrix) grouped by overlap
(`overlap_components`), each with the hull of its discs (`ball_hull`) and the number of eigenvalues
of G it holds, with multiplicity.
"""
function _comps(sc::SplitCertificate)
    _has_counts(sc) || return NamedTuple[]
    ctr, rad_, cnt_, _ = inclusion_discs(sc.comp)
    balls = [Ball(ComplexF64(ctr[i]), Float64(rad_[i])) for i in eachindex(ctr)]
    out = NamedTuple[]
    for idx in overlap_components(balls)
        h = foldl(ball_hull, balls[idx])
        push!(out, (; center = mid(h), radius = rad(h), count = sum(cnt_[idx])))
    end
    sort!(out; by = C -> -abs(C.center))
    return out
end

"""
    count_disc(sc, c, r; kwargs...) -> (count, ok, check)

Proposition prop:count for the open disc `D(c, r)`: if the hypothesis holds on its boundary and
`D ∪ Γ ⊆ {|z| > ḡ}`, A and G11 have the same number of eigenvalues in `D`; that number is read off
the inclusions of `verifyeigall` on G (no disc meets the circle) and transferred to G11 by the
homotopy of the header.
"""
function count_disc(sc::SplitCertificate, c::Number, r::Real; kwargs...)
    chk = check_circle(sc, c, r; region = :interior, kwargs...)
    (chk.ok && chk.region_ok && _has_counts(sc) && _transfers(sc, chk)) || return (0, false, chk)
    cnt, okc = eigencount_in_disc(sc.comp, ComplexF64(c), Float64(r))
    return (cnt, okc, chk)
end

"""
    count_union(sc, centres, radii; nθ = 64, maxdepth = 6) -> NamedTuple

Proposition prop:count for the union `U` of the closed discs `D(c_i, r_i)`: the hypothesis is
certified on every arc of every circle that can belong to `∂U` ([`union_boundary_arcs`](@ref)),
`U ⊆ {|z| > ḡ}` is checked disc by disc, and the connected components of the interior of `U`
(discs joined when certified to overlap; a pair neither certified overlapping nor certified apart
makes the call fail) each receive the number of eigenvalues of G11 they contain, read off the
components of the inclusions (each inside one disc or disjoint from all) and transferred by the
homotopy of the header. On `ok`, `counts[c]` is also the number of eigenvalues of A in the
component `members[c]`.
"""
function count_union(sc::SplitCertificate, centres::AbstractVector, radii::AbstractVector;
        nθ::Int = 64, maxdepth::Int = 6)
    p = length(centres)
    cs = ComplexF64.(centres)
    rs = Float64.(radii)
    region_ok = all(i -> sub_down(abs_down(cs[i]), rs[i]) > sc.gbar, 1:p)
    arcs = union_boundary_arcs(cs, rs)
    checks = [check_circle(sc, cs[i], rs[i]; region = :interior, arcs = arcs[i], nθ, maxdepth)
              for i in 1:p]
    contour_ok = all(ck -> ck.ok, checks)
    # every pair is either certified to overlap or certified apart; a pair that is neither could
    # make the interior of U split where the discs only touch
    geom_ok = true
    for i in 1:p, j in (i + 1):p
        sr = add_up(rs[i], rs[j])
        dist_up(cs[i], cs[j]) < add_down(rs[i], rs[j]) && continue
        dist_down(cs[i], cs[j]) > sr && continue
        geom_ok = false
    end
    members = overlap_components([Ball(cs[i], rs[i]) for i in 1:p])
    host_of = Dict(i => c for (c, m) in enumerate(members) for i in m)
    counts = zeros(Int, length(members))
    count_ok = _has_counts(sc) && all(ck -> _transfers(sc, ck), checks)
    if count_ok
        for C in _comps(sc)
            host = findfirst(i -> add_up(dist_up(C.center, cs[i]), C.radius) < rs[i], 1:p)
            if host !== nothing
                counts[host_of[host]] += C.count
            elseif all(i -> sub_down(dist_down(C.center, cs[i]), C.radius) > rs[i], 1:p)
                continue
            else
                count_ok = false
            end
        end
    end
    ok = region_ok && contour_ok && geom_ok && count_ok
    return (; ok, members, counts, checks, region_ok, contour_ok, geom_ok, count_ok)
end

"""
    count_exterior(sc, R; kwargs...) -> (count, ok, check)

Proposition prop:count for `D = {|z| > R}`, `R > ḡ`: the hypothesis on `|z| = R`, and the number of
eigenvalues of G11 of modulus `> R`, read off the inclusions of `verifyeigall` on G and transferred
by the homotopy of the header.
"""
function count_exterior(sc::SplitCertificate, R::Real; kwargs...)
    chk = check_circle(sc, 0.0, R; region = :exterior, kwargs...)
    (chk.ok && chk.region_ok && _has_counts(sc) && _transfers(sc, chk)) || return (0, false, chk)
    cnt, okc = eigencount_outside(sc.comp, Float64(R))
    return (cnt, okc, chk)
end

"""
    ProjectorCertificate

Proposition prop:projector on a circle: `bound ≥ ‖P − P₀‖`, `P` the Riesz projector of A for its
spectrum in the region, `P₀` the spectral projector of G11 for its eigenvalues there, padded by
zero on `ran P⊥`. `M ≥ sup_Γ ‖(A − zI)⁻¹‖` and `M0 ≥ sup_Γ max(1/σ11, 1/σ22)`, the resolvent bound
of the block-diagonal part, both taken from the same certified arc cover as the counts
(`check.floor`, `check.minσ11`, `check.minσ22`); `arclen = |Γ|`; `normP ≥ ‖P‖`;
`gap ≥ gap(ran P, ran P₀)`, which is `bound`. `count` is the dimension of `ran P₀`, read off the
inclusions as in [`count_disc`](@ref), or `-1` when no count is certified.
"""
struct ProjectorCertificate
    ok::Bool
    region::Symbol
    center::ComplexF64
    radius::Float64
    bound::Float64
    M::Float64
    M0::Float64
    arclen::Float64
    normP::Float64
    gap::Float64
    count::Int
    check::CircleCheck
end

function Base.show(io::IO, pc::ProjectorCertificate)
    print(io, "ProjectorCertificate(", pc.region, ", |z − ", pc.center, "| = ", pc.radius,
        ", ok = ", pc.ok, ", ‖P − P₀‖ ≤ ", pc.bound, ", M ≤ ", pc.M, ", M₀ ≤ ", pc.M0,
        ", dim = ", pc.count, ")")
end

"""
    certify_projector(sc, c, r; region = :interior, nθ = 64, maxdepth = 6) -> ProjectorCertificate

Proposition prop:projector for the circle `Γ = {|z − c| = r}` and the region named by `region`
(`:interior`, the open disc; `:exterior`, `{|z| > r}` with `c = 0`), under the hypotheses of
prop:count, which are certified on the whole contour by the arc cover of [`check_circle`](@ref):
with `M = 1/min_Γ (right-hand side of eq:assembly)` and `M0 = 1/min_Γ min(σ11, σ22)`,

    ‖P − P₀‖ ≤ (|Γ|/2π) M M₀ max(γ1, γ2) = r M M₀ max(γ1, γ2).

For the exterior the contour of the projector is the circle together with a circle of radius
`M' → ∞`, on which the integrand is `O(1/M'²)`, so its contribution vanishes and the bound is
unchanged; `P = I − P_in` there, and `‖P‖ ≤ 1 + rM` instead of `rM`.
"""
function certify_projector(sc::SplitCertificate, c::Number, r::Real; region::Symbol = :interior,
        nθ::Int = 64, maxdepth::Int = 6, check = nothing)
    c = ComplexF64(c)
    r = Float64(r)
    chk = check === nothing ? check_circle(sc, c, r; region, nθ, maxdepth) : check
    e = max(sc.γ1, sc.γ2)
    f0 = chk.ok ? min(chk.minσ11, chk.minσ22) : 0.0
    b, M, M0, okb = projector_bound(chk.ok ? chk.floor : 0.0, f0, r, e)
    ok = chk.ok && chk.region_ok && okb
    normP = region === :exterior ? add_up(1.0, mul_up(r, M)) : mul_up(r, M)
    arclen = mul_up(mul_up(2.0, nextfloat(Float64(π))), r)
    cnt = -1
    if ok && _has_counts(sc) && _transfers(sc, chk)
        k, okc = region === :exterior ? eigencount_outside(sc.comp, r) :
                 eigencount_in_disc(sc.comp, c, r)
        okc && (cnt = k)
    end
    return ProjectorCertificate(ok, region, c, r, b, M, M0, arclen, normP, b, cnt, chk)
end

"""
    certify_exterior(sc, R; nθ = 64, nradii = 12, kwargs...) -> NamedTuple

Corollary cor:agree. Certifies the hypothesis on `|z| = R` and counts the eigenvalues outside;
then, for every component of the inclusions of G outside `R`, looks for the smallest radius (in a
geometric sequence between the component's radius and the largest admissible one) at which the
boundary of the isolation disc passes the hypothesis of prop:count, keeping the discs pairwise
disjoint and inside `{|z| > R}`. On success every eigenvalue of A of modulus `> R` lies in the
union of the discs, and each disc holds as many eigenvalues of A as of G11 (`discs[i].count`).
"""
function certify_exterior(sc::SplitCertificate, R::Real; nθ::Int = 64, nradii::Int = 12,
        maxdepth::Int = 6, method::Symbol = :svd)
    cnt, okext, chk = count_exterior(sc, R; nθ, maxdepth, method)
    discs = NamedTuple[]
    okext || return (; ok = false, R = Float64(R), count = cnt, exterior = chk, discs)
    comps = [C for C in _comps(sc) if sub_down(abs_down(C.center), C.radius) > R]
    ok = true
    for (i, C) in enumerate(comps)
        rmax = sub_down(abs_down(C.center), Float64(R))
        for (j, D) in enumerate(comps)
            j == i && continue
            rmax = min(rmax, div_down(dist_down(C.center, D.center), 2.0))
        end
        rmin = C.radius * (1 + 1e-6) + 1e-300
        found = false
        if rmin < rmax
            for r in exp.(range(log(rmin), log(rmax * (1 - 1e-6)); length = nradii))
                c2, okc, ck = count_disc(sc, C.center, r; nθ, maxdepth, method)
                if okc && c2 == C.count
                    push!(discs, (; center = C.center, radius = r, count = C.count, check = ck))
                    found = true
                    break
                end
            end
        end
        found || (ok = false; push!(discs, (; center = C.center, radius = NaN,
            count = C.count, check = nothing)))
    end
    # the discs are pairwise disjoint by construction (r_i ≤ |c_i − c_j|/2, rounded down); recheck
    for i in eachindex(discs), j in (i + 1):length(discs)
        (isfinite(discs[i].radius) && isfinite(discs[j].radius)) || continue
        dist_down(discs[i].center, discs[j].center) > add_up(discs[i].radius, discs[j].radius) ||
            (ok = false)
    end
    ok &= sum(d -> d.count, discs; init = 0) == cnt
    return (; ok, R = Float64(R), count = cnt, exterior = chk, discs)
end

"""
    certify_gap(sc, R; kwargs...) -> NamedTuple

Corollary cor:gap: if G11 has exactly one eigenvalue of modulus `> R` and the exterior count is
certified, so does A, and every other eigenvalue of A satisfies `|λ| ≤ R`.
"""
function certify_gap(sc::SplitCertificate, R::Real; kwargs...)
    ext = certify_exterior(sc, R; kwargs...)
    return (; certified = ext.exterior.ok && ext.exterior.region_ok && ext.count == 1,
        R = Float64(R), count = ext.count, exterior = ext.exterior, discs = ext.discs,
        isolated = ext.ok)
end

"Leading Ritz values of a short Arnoldi, by decreasing modulus (floating point)."
function _ritz_leading(Amul, n, T; L = 60)
    V, H = arnoldi(Amul, n, min(L, n); T)
    Lh = size(H, 2)
    λ = eigvals(H[1:Lh, 1:Lh])
    return λ[sortperm(abs.(λ); rev = true)]
end

# ------------------------------------------------------------------------------------------------
# Self-adjoint operators (cor:selfadj, cor:weyl, rem:lanczos)
# ------------------------------------------------------------------------------------------------

# σ11 > etaG certified on the whole circle |z − c| = r by the cover of `check_circle`; then, by the
# homotopy of the header, the open disc holds as many eigenvalues of G11 = Z*AZ as of mid(G)
function _transfer_circle(sc::SplitCertificate, c, r; nθ::Int, maxdepth::Int)
    ok, _ = _check_arcs(z -> sigma11_lo(sc, z), z -> 0.0, 0.0, 0.0, 0.0, ComplexF64(c),
        Float64(r), [(0.0, 2π)]; nθ, maxdepth, pass = (l11, _) -> (l11, l11 > sc.etaG))
    return ok
end

"""
    eig_intervals(sc; nθ = 64, maxdepth = 6) -> Vector{EigInterval}

For a certificate made with `hermitian = true` (A = A*): disjoint real intervals, each holding
`count` eigenvalues of G11 = Z*AZ with multiplicity. The eigenvalues are enclosed for the midpoint
Ĝ of G by `verifyeigall`, which does not assume Ĝ Hermitian (it is not: it is a floating-point
compression). Each component of the inclusions, of centre `c` and radius `r`, is surrounded by the
circle of radius `r + μ`, `μ` half the distance to the nearest other component (`‖G11‖` bound
`snorm` if there is none), on which `σ_min(G11 − zI) > etaG ≥ ‖Ĝ − G11‖` is certified
([`check_circle`](@ref)'s cover); by the homotopy of the header the disc holds as many eigenvalues
of G11 as the component holds of Ĝ. G11 is Hermitian, so they are real and lie in
`[Re c − r − μ, Re c + r + μ]`; overlapping intervals are merged with their counts added. Empty
when the inclusions carry no counts or a circle fails. Throws for a certificate that is not
Hermitian.
"""
function eig_intervals(sc::SplitCertificate; nθ::Int = 64, maxdepth::Int = 6)
    sc.hermitian || error("eig_intervals: not a Hermitian certificate")
    _has_counts(sc) || return EigInterval[]
    comps = _comps(sc)
    J = EigInterval[]
    for (i, C) in enumerate(comps)
        gap = Inf
        for (j, D) in enumerate(comps)
            j == i && continue
            gap = min(gap, sub_down(sub_down(dist_down(C.center, D.center), C.radius), D.radius))
        end
        μ = isfinite(gap) ? div_down(gap, 2.0) : max(sc.snorm, C.radius)
        R = add_down(C.radius, μ)
        (μ > 0 && R > C.radius && _transfer_circle(sc, C.center, R; nθ, maxdepth)) ||
            return EigInterval[]
        push!(J, EigInterval(sub_down(real(C.center), add_up(C.radius, μ)),
            add_up(real(C.center), add_up(C.radius, μ)), C.count))
    end
    sort!(J; by = x -> x.lo)
    out = EigInterval[]
    for x in J
        if !isempty(out) && x.lo <= out[end].hi
            y = out[end]
            out[end] = EigInterval(y.lo, max(y.hi, x.hi), y.count + x.count)
        else
            push!(out, x)
        end
    end
    return out
end

"""
    coupling_hermitian(sc::SplitCertificate) -> γ̄

For a certificate made with `hermitian = true`, `γ̄ = min(γ1, γ2) ≥ ‖P⊥AZ‖₂`, `Z` the isometry of
the frame: for a Hermitian `A`, `G21 = G12*`, so the two couplings have the same norm and each of
the two certified bounds bounds it (proof of cor:selfadj). Throws for a certificate that is not
Hermitian.
"""
coupling_hermitian(sc::SplitCertificate) = (sc.hermitian || error("not a Hermitian certificate");
    min(sc.γ1, sc.γ2))

"""
    certify_selfadjoint(Amul, n; L = 80, tol = 1e-10, kmax = L, epsA, etaA = 0, kwargs...)
    certify_selfadjoint(A::AbstractMatrix; kwargs...)

Remark rem:lanczos and Corollaries cor:selfadj, cor:weyl for `A = A*`: a Lanczos frame of
converged Ritz vectors ([`lanczos_frame`](@ref)), one certified split (`hermitian = true`), and
the eigenvalue intervals of G11. Returns `(; split, θ, γ̄, ḡ, enclosure, weyl, frame)` with
`enclosure = selfadj_enclosure(θ, ḡ, γ̄)` and `weyl = weyl_pairing(θ, ḡ, γ̄, n)`; counts on
intervals by [`selfadj_count`](@ref).
"""
function certify_selfadjoint(Amul, n::Integer; L::Integer = 80, tol::Real = 1e-10,
        kmax::Integer = L, epsA::Real, etaA::Real = 0.0, T = Float64, kwargs...)
    Q, finfo = lanczos_frame(Amul, n, L; tol, kmax, T)
    sc = certify_split(Amul, Amul, n, Q; epsA, etaA, hermitian = true, kwargs...)
    return _selfadj_out(sc, finfo)
end
function certify_selfadjoint(A::AbstractMatrix; L::Integer = 80, tol::Real = 1e-10,
        kmax::Integer = L, kwargs...)
    T = eltype(A) <: Complex ? ComplexF64 : Float64
    Q, finfo = lanczos_frame(x -> A * x, size(A, 1), L; tol, kmax, T)
    sc = certify_split(A, Q; hermitian = true, kwargs...)
    return _selfadj_out(sc, finfo)
end
function _selfadj_out(sc, finfo)
    θ = eig_intervals(sc)
    γ̄ = coupling_hermitian(sc)
    return (; split = sc, θ, γ̄, ḡ = sc.gbar, enclosure = selfadj_enclosure(θ, sc.gbar, γ̄),
        weyl = weyl_pairing(θ, sc.gbar, γ̄, sc.n), frame = finfo)
end
