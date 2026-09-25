# §5 (sec:split): the certified splitting. The four certified numbers, ḡ ≥ ‖G22‖ (§4), γ1, γ2 from
# eq:gram (§3.3) and σ11 from the compression (§3.2), fed to §2: the resolvent bound
# (thm:assembly with eq:farshort or eq:farpower), the inclusion (prop:complete), the counts
# (prop:count), the exterior with isolation discs (cor:agree) and the gap (cor:gap), with the frame
# only almost orthonormal (prop:almostortho, sec:almostortho).

"""
    SplitCertificate

Everything certified for one operator and one frame `Q` (n×k):

* `α ≥ ‖I − Q*Q‖ < 1`, `G ⊇ Q*AQ`, `M ⊇ Q*Q` (ball matrices);
* `gbar > ‖G22‖` and `abar > ‖A‖` (prop:fpchecks), or supplied by the caller (`far = nothing`);
* `γ1 ≥ ‖G12‖`, `γ2 ≥ ‖G21‖`, `snorm ≥ ‖G11‖` for the isometry Z of the polar decomposition
  `Q = ZS` (the factors of prop:almostortho already applied);
* `comp`, the certified eigenvalue enclosure of the compression G11 = Z*AZ by the full block
  diagonalisation (sec:vbd, on the pencil `(Q*AQ, Q*Q)`), or `nothing`;
* `msplit`, the same compression split by modulus at the target radius and certified with the
  two-block floor ([`split_compression`](@ref)), or `nothing`. Exactly one of the two is built,
  according to the `compression` keyword of [`certify_split`](@ref);
* `gbars_power`, bounds `ḡ_j ≥ ‖G22^j‖` supplied by the caller for eq:farpower (empty by default:
  the package does not certify powers of the far block, see the report).
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
    comp::Union{Nothing, MiyajimaRump}
    msplit::Union{Nothing, ModulusSplit}
    hermitian::Bool
    gbars_power::Vector{Float64}
    time::NamedTuple
end

function Base.show(io::IO, sc::SplitCertificate)
    print(io, "SplitCertificate(n = $(sc.n), k = $(sc.k), α ≤ $(sc.α), ‖G22‖ < $(sc.gbar), ",
        "‖A‖ < $(sc.abar), γ1 ≤ $(sc.γ1), γ2 ≤ $(sc.γ2))")
end

"""
    certify_split(op::LinOp, Q; order = :auto, compression = :auto, target = nothing,
                  split_size = 64, split_frac = 0.9, split_m = 8, split_kmax = nothing,
                  split_budget = nothing,
                  vbd_method = :sylvester,
                  coupling = :gram, hermitian = false, far_bound = nothing,
                  gbars_power = Float64[], kwargs...) -> SplitCertificate
    certify_split(A::AbstractMatrix, Q; kwargs...)
    certify_split(Amul, Aadj, n, Q; epsA, etaA = 0, kwargs...)

Certify the four numbers for the frame `Q`: frame data (couplings.jl), the far block by the
floating-point sweep and prop:fpchecks (farblock.jl; keyword arguments are passed to
[`certify_far_block`](@ref)), and the compression (compression.jl). `far_bound = (ḡ, ā)` skips
the sweep and uses the caller's bounds `‖G22‖ ≤ ḡ`, `‖A‖ ≤ ā` instead ("any other route to a bound
on the far block feeds the same theorem"). `hermitian = true` asserts `A = A*` (checked exactly
for a stored matrix).

`compression` chooses the layer of §3.2:

* `:split` — the split by modulus at `R' = split_frac·target` ([`split_compression`](@ref)), which
  separates the eigenvalues outside the circle from all the rest and leaves the rest alone; the
  size of the leading block is an outcome of the condition on the trailing one, capped only by
  the stopping rule `split_kmax`/`split_budget` (`sc.msplit.ok`, `split_reason`);
* `:full` (or `true`) — the full block diagonalisation ([`miyajima_rump`](@ref));
* `:auto` (the default) — `:split` when `target` is given and `size(Q, 2) > split_size`, `:full`
  otherwise;
* `false` or `:none` — neither.

`target` is the radius `R` of the circle the certificate is aimed at; the matrix-free entry points
pass it themselves.
"""
function certify_split(op::LinOp, Q::AbstractMatrix; order = :auto, compression = :auto,
        target = nothing, split_size::Integer = 64, split_frac::Real = 0.9,
        split_m::Integer = 8, split_kmax = nothing, split_budget = nothing,
        vbd_method::Symbol = :sylvester, coupling::Symbol = :gram,
        hermitian::Bool = false, far_bound = nothing, gbars_power = Float64[],
        xmax::Real = 1e8, kappa_mode::Symbol = :svdbox, kwargs...)
    t0 = time()
    if hermitian && op.stored !== nothing
        ishermitian(op.stored) || error("hermitian = true but the stored matrix is not Hermitian")
    end
    fd = frame_data(op, Q; coupling)
    t1 = time()
    if far_bound === nothing
        far = certify_far_block(op, Q, fd; order, target, kwargs...)
        gbar, abar = far.gbar, far.abar
        # the sweep's own bounds on ‖G22^j‖, so that eq:farpower has something to work with;
        # without them σ22 is always the order-one form
        isempty(gbars_power) && (gbars_power = far.gbars_power)
    else
        far = nothing
        gbar, abar = Float64(far_bound[1]), Float64(far_bound[2])
    end
    t2 = time()
    s, γ1, γ2 = isfinite(abar) ? isometry_bounds(fd, abar) : (Inf, Inf, Inf)
    G = isfinite(abar) ? compression_ball(fd, abar) : fd.Gt
    mode = _compression_mode(compression, target, size(Q, 2), split_size)
    comp = nothing
    ms = nothing
    if isfinite(abar) && size(Q, 2) > 0
        if mode === :split
            ms = split_compression(G, Float64(target); M = fd.Mq, frac = split_frac,
                m = split_m, kmax = split_kmax, budget = split_budget, kappa_mode, xmax)
        elseif mode === :full
            comp = miyajima_rump(G; M = fd.Mq, method = vbd_method, xmax, kappa_mode)
        end
    end
    t3 = time()
    return SplitCertificate(size(Q, 1), size(Q, 2), Matrix(Q), fd.α, G, fd.Mq, gbar, abar,
        γ1, γ2, s, far, fd, comp, ms, hermitian, Float64.(collect(gbars_power)),
        (; frame = t1 - t0, far = t2 - t1, compression = t3 - t2, total = t3 - t0,
            mode))
end

function _compression_mode(compression, target, k, split_size)
    compression === false && return :none
    compression === :none && return :none
    compression === true && return :full
    compression === :full && return :full
    if compression === :split
        target === nothing && error("compression = :split needs `target` (the radius R)")
        return :split
    end
    compression === :auto ||
        error("compression must be :auto, :split, :full, :none (or a Bool)")
    return (target !== nothing && k > split_size) ? :split : :full
end
certify_split(A::AbstractMatrix, Q::AbstractMatrix; kwargs...) = certify_split(LinOp(A), Q;
    kwargs...)
function certify_split(Amul, Aadj, n::Integer, Q::AbstractMatrix; epsA::Real, etaA::Real = 0.0,
        kwargs...)
    certify_split(LinOp(Amul, Aadj, n; epsA, etaA), Q; kwargs...)
end

"""
    sigma11_lo(sc, z; method = :auto) -> Float64

Certified lower bound on `σ11(z) = σ_min(G11 − zI)`, G11 = Z*AZ the compression of the isometry:
`σ_min(Q*AQ − zQ*Q)/(1 + α)` (prop:almostortho), with `σ_min(Q*AQ − zQ*Q)` bounded below by

* `:split` — the two-block floor of the split by modulus ([`sigma_split_lo`](@ref)), one `svdbox`
  of the small leading block and a handful of scalars per `z`;
* `:svd` — one `svdbox` of the whole ball pencil, the exact-SVD escape of sec:transport, at
  `O(k³)` per `z`;
* `:blocks` — the transport eq:blockres of the full block-diagonal certificate;
* `:auto` (the default) — `:split` when the certificate carries one, `:svd` otherwise.
"""
function sigma11_lo(sc::SplitCertificate, z::Number; method::Symbol = :auto)
    sc.k == 0 && return Inf
    meth = method === :auto ? (sc.msplit !== nothing && sc.msplit.ok ? :split : :svd) : method
    # an explicit `:split` uses the split whenever its certification passed, even where the split
    # declined the exterior (`ok = false`): the bound is sound, it is the claim about the trailing
    # block that failed
    if meth === :blocks
        sc.comp === nothing && return 0.0
        v = sigma_blocks_lo(sc.comp.cert, z)
    elseif meth === :split
        sc.msplit === nothing && return 0.0
        v = sigma_split_lo(sc.msplit, z)
    else
        v = sigmin_lo(_bsub(sc.G, _zm_ball(sc.M, z)))
    end
    return divd(v, addu(1.0, sc.α))
end

# the ball z·M, with the rounding of the product in the radius
function _zm_ball(M::BallMatrix, z::Number)
    z = ComplexF64(z)
    C = z .* ComplexF64.(mid(M))
    u = unit_roundoff(Float64)
    az = absu(z)
    R = [addu(mulu(az, Float64(r)), mulu(4u, absu(c))) for (c, r) in zip(C, rad(M))]
    return BallMatrix(C, reshape(R, size(C)))
end

"""
    sigma22_lo(sc, z) -> Float64

Certified lower bound on `σ22(z) = 1/‖(G22 − zI)⁻¹‖`: eq:farshort `|z| − ḡ`, and eq:farpower with
the caller's `ḡ_j` when supplied; the larger.
"""
sigma22_lo(sc::SplitCertificate, z::Number) = max(farshort_lo(z, sc.gbar),
    isempty(sc.gbars_power) ? 0.0 : farpower_lo(z, sc.gbars_power))

"""
    resolvent_floor(sc, z; method = :auto) -> NamedTuple

Certified lower bound on `σ_min(A − zI) = 1/‖(A − zI)⁻¹‖` at one point: Theorem thm:assembly
(`assembly`) and Proposition prop:blockdom (`blockdom`), each only under its own hypothesis, and
`floor` their maximum (both are lower bounds on the same number). `ok = false` and `floor = 0`
mean "no bound".
"""
function resolvent_floor(sc::SplitCertificate, z::Number; method::Symbol = :auto)
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
    check_circle(sc, c, r; region = :interior, method = :auto, kwargs...) -> CircleCheck

[`check_circle`](@ref) with the certificate's σ11 and σ22.
"""
function check_circle(sc::SplitCertificate, c::Number, r::Real; region::Symbol = :interior,
        method::Symbol = :auto, kwargs...)
    check_circle(z -> sigma11_lo(sc, z; method), sc.gbar, sc.γ1, sc.γ2, c, r; region,
        σ22lo = z -> sigma22_lo(sc, z), kwargs...)
end

"""
    count_disc(sc, c, r; kwargs...) -> (count, ok, check)

Proposition prop:count for the open disc `D(c, r)`: if the hypothesis holds on its boundary and
`D ∪ Γ ⊆ {|z| > ḡ}`, A and G11 have the same number of eigenvalues in `D`; that number is read off
the certified enclosure of G11 (every component inside `D` or disjoint from its closure).
"""
function count_disc(sc::SplitCertificate, c::Number, r::Real; kwargs...)
    chk = check_circle(sc, c, r; region = :interior, kwargs...)
    (chk.ok && chk.region_ok && _has_counts(sc)) || return (0, false, chk)
    comps = _comps(sc)
    cnt, okc = eigencount_in_disc(comps, c, r)
    return (cnt, okc, chk)
end

"""
    count_union(sc, centres, radii; nθ = 64, maxdepth = 6) -> NamedTuple

Proposition prop:count for the union `U` of the closed discs `D(c_i, r_i)`: the hypothesis is
certified on every arc of every circle that can belong to `∂U` ([`union_boundary_arcs`](@ref)),
`U ⊆ {|z| > ḡ}` is checked disc by disc, and the connected components of the interior of `U`
(discs joined when certified to overlap; a pair neither certified overlapping nor certified apart
makes the call fail) each receive the number of eigenvalues of G11 they contain, read off the
certified enclosure (each of its components inside one disc or disjoint from all). On `ok`,
`counts[c]` is also the number of eigenvalues of A in the component `members[c]`.
"""
function count_union(sc::SplitCertificate, centres::AbstractVector, radii::AbstractVector;
        nθ::Int = 64, maxdepth::Int = 6, method::Symbol = :auto)
    p = length(centres)
    region_ok = all(i -> subd(absd(ComplexF64(centres[i])), Float64(radii[i])) > sc.gbar, 1:p)
    arcs = union_boundary_arcs(centres, radii)
    checks = [check_circle(sc, centres[i], radii[i]; region = :interior, arcs = arcs[i], nθ,
                  maxdepth, method) for i in 1:p]
    contour_ok = all(ck -> ck.ok, checks)
    onepu = nextfloat(1.0)
    parent = collect(1:p)
    find(i) = (while parent[i] != i
        i = parent[i]
    end; i)
    geom_ok = true
    for i in 1:p, j in (i + 1):p
        d = ComplexF64(centres[i]) - ComplexF64(centres[j])
        rs = Float64(radii[i]) + Float64(radii[j])
        if mulu(absu(d), onepu) < prevfloat(rs)
            parent[find(i)] = find(j)
        elseif divd(absd(d), onepu) > nextfloat(rs)
            continue
        else
            geom_ok = false
        end
    end
    roots = unique(find.(1:p))
    members = [findall(i -> find(i) == r, 1:p) for r in roots]
    counts = zeros(Int, length(roots))
    count_ok = _has_counts(sc)
    if count_ok
        for C in _comps(sc)
            host = findfirst(i -> addu(mulu(absu(C.center - centres[i]), onepu), C.radius) <
                                  radii[i], 1:p)
            if host !== nothing
                counts[findfirst(==(find(host)), roots)] += C.count
            elseif all(i -> divd(absd(C.center - centres[i]), onepu) >
                            addu(Float64(radii[i]), C.radius), 1:p)
                continue
            else
                count_ok = false
            end
        end
    end
    ok = region_ok && contour_ok && geom_ok && count_ok
    return (; ok, members, counts, checks, region_ok, contour_ok, geom_ok, count_ok)
end

"Whether the certificate carries an enclosure of the compression from which a count can be read."
_has_counts(sc::SplitCertificate) = (sc.comp !== nothing && sc.comp.ok) ||
                                    (sc.msplit !== nothing && sc.msplit.cert_ok &&
                                     sc.msplit.inner !== nothing && sc.msplit.inner.ok)

"""
The certified components of the enclosure of the compression: those of the full block
diagonalisation (cor:clusters through [`miyajima_rump`](@ref)) or, when the compression was split
by modulus, those of [`split_components`](@ref), whose trailing component is the disc about the
origin that prop:power gives for the block left undiagonalised.
"""
function _comps(sc::SplitCertificate)
    if sc.msplit !== nothing && sc.msplit.cert_ok
        return split_components(sc.msplit)
    end
    sc.comp === nothing && return EigComponent[]
    return sc.comp.family === :local ? sc.comp.cert.components_local :
           sc.comp.cert.components_global
end

"""
    count_exterior(sc, R; kwargs...) -> (count, ok, check)

Proposition prop:count for `D = {|z| > R}`, `R > ḡ`: the hypothesis on `|z| = R`, and the number of
eigenvalues of G11 of modulus `> R` from its certified enclosure.
"""
function count_exterior(sc::SplitCertificate, R::Real; kwargs...)
    chk = check_circle(sc, 0.0, R; region = :exterior, kwargs...)
    (chk.ok && chk.region_ok && _has_counts(sc)) || return (0, false, chk)
    cnt, okc = eigencount_outside(_comps(sc), R)
    return (cnt, okc, chk)
end

"""
    ProjectorCertificate

Proposition prop:projector on a circle: `bound ≥ ‖P − P₀‖`, `P` the Riesz projector of A for its
spectrum in the region, `P₀` the spectral projector of G11 for its eigenvalues there, padded by
zero on `ran P⊥`. `M ≥ sup_Γ ‖(A − zI)⁻¹‖` and `M0 ≥ sup_Γ max(1/σ11, 1/σ22)`, the resolvent bound
of the block-diagonal part, both taken from the same certified arc cover as the counts
(`check.floor`, `check.minσ11`, `check.minσ22`); `arclen = |Γ|`; `normP ≥ ‖P‖`;
`gap ≥ gap(ran P, ran P₀)`, which is `bound`. `count` is the dimension of `ran P₀` from the
certified enclosure of G11, `basis` a floating-point `Q·W_S` spanning the candidate for `ran P₀`,
and `subspace_gap ≥ gap(ran P₀, ran basis)` from the compression-level certificate `block`
(prop:projector inside sec:vbd), transported by `κ(QW) ≤ √((1+α)/(1−α))(1+α)κ`.
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
    basis::Matrix
    subspace_gap::Float64
    block::Union{Nothing, BlockProjectorCertificate}
    check::CircleCheck
end

function Base.show(io::IO, pc::ProjectorCertificate)
    print(io, "ProjectorCertificate(", pc.region, ", |z − ", pc.center, "| = ", pc.radius,
        ", ok = ", pc.ok, ", ‖P − P₀‖ ≤ ", pc.bound, ", M ≤ ", pc.M, ", M₀ ≤ ", pc.M0,
        ", dim = ", pc.count, ")")
end

"""
    certify_projector(sc, c, r; region = :interior, nθ = 64, maxdepth = 6, method = :auto,
                      subspace = true) -> ProjectorCertificate

Proposition prop:projector for the circle `Γ = {|z − c| = r}` and the region named by `region`
(`:interior`, the open disc; `:exterior`, `{|z| > r}` with `c = 0`), under the hypotheses of
prop:count, which are certified on the whole contour by the arc cover of [`check_circle`](@ref):
with `M = 1/min_Γ (right-hand side of eq:assembly)` and `M0 = 1/min_Γ min(σ11, σ22)`,

    ‖P − P₀‖ ≤ (|Γ|/2π) M M₀ max(γ1, γ2) = r M M₀ max(γ1, γ2).

For the exterior the contour of the projector is the circle together with a circle of radius
`M' → ∞`, on which the integrand is `O(1/M'²)`, so its contribution vanishes and the bound is
unchanged; `P = I − P_in` there, and `‖P‖ ≤ 1 + rM` instead of `rM`. The contour should be taken
as large as the hypothesis of prop:count allows: what is bounded does not depend on it, while `M`
and `M₀` grow like the reciprocal of the distance from Γ to the spectrum.

With `subspace = true` the compression's own projector is certified too
([`certify_block_projector`](@ref)) and transported, giving `subspace_gap`, a bound on the gap
between `ran P₀` and the span of the returned `basis = Q·W_S`. The basis itself is a
floating-point object: what is certified about it is that bound.
"""
function certify_projector(sc::SplitCertificate, c::Number, r::Real; region::Symbol = :interior,
        nθ::Int = 64, maxdepth::Int = 6, method::Symbol = :auto, subspace::Bool = true,
        check = nothing)
    c = ComplexF64(c)
    r = Float64(r)
    chk = check === nothing ? check_circle(sc, c, r; region, nθ, maxdepth, method) : check
    e = max(sc.γ1, sc.γ2)
    f0 = chk.ok ? min(chk.minσ11, chk.minσ22) : 0.0
    b, M, M0, okb = projector_bound(chk.ok ? chk.floor : 0.0, f0, r, e)
    ok = chk.ok && chk.region_ok && okb
    normP = region === :exterior ? addu(1.0, mulu(r, M)) : mulu(r, M)
    arclen = mulu(mulu(2.0, nextfloat(Float64(π))), r)
    cnt = -1
    basis = zeros(eltype(sc.Q), sc.n, 0)
    sgap = NaN
    blk = nothing
    if ok && subspace && _has_counts(sc)
        if sc.msplit !== nothing && sc.msplit.cert_ok
            ms = sc.msplit
            cc = ms.inner
            ctr, rad_, fam = split_block_discs(ms)
            blk = certify_block_projector(cc, c, r; family = fam, nθ, maxdepth,
                blockfun = z -> split_sigma_blocks_lo(ms, z), discs = (ctr, rad_))
        else
            cc = sc.comp.cert
            blk = certify_block_projector(cc, c, r; family = sc.comp.family, nθ, maxdepth)
        end
        if blk !== nothing && blk.ok
            cols = region === :exterior ? setdiff(1:(cc.k), blk.cols) : blk.cols
            cnt = length(cols)
            basis = sc.Q * cc.candidate.W[:, cols]
            oma = subd(1.0, sc.α)
            opa = addu(1.0, sc.α)
            κQW = mulu(mulu(sqrtu(divu(opa, oma)), opa), cc.kappa)
            sgap = mulu(κQW, blk.bound)
        end
    end
    return ProjectorCertificate(ok, region, c, r, b, M, M0, arclen, normP, b, cnt,
        basis, sgap, blk, chk)
end

"""
    certify_exterior(sc, R; nθ = 64, nradii = 12, kwargs...) -> NamedTuple

Corollary cor:agree. Certifies the hypothesis on `|z| = R` and counts the eigenvalues outside;
then, for every component of the certified enclosure of G11 outside `R`, looks for the smallest
radius (in a geometric sequence between the component's radius and the largest admissible one)
at which the boundary of the isolation disc passes the hypothesis of prop:count, keeping the discs
pairwise disjoint and inside `{|z| > R}`. On success every eigenvalue of A of modulus `> R` lies in
the union of the discs, and each disc holds as many eigenvalues of A as of G11 (`discs[i].count`).
"""
function certify_exterior(sc::SplitCertificate, R::Real; nθ::Int = 64, nradii::Int = 12,
        maxdepth::Int = 6, method::Symbol = :auto)
    cnt, okext, chk = count_exterior(sc, R; nθ, maxdepth, method)
    discs = NamedTuple[]
    okext || return (; ok = false, R = Float64(R), count = cnt, exterior = chk, discs)
    comps = [C for C in _comps(sc) if subd(absd(C.center), C.radius) > R]
    onepu = nextfloat(1.0)
    ok = true
    for (i, C) in enumerate(comps)
        rmax = subd(absd(C.center), Float64(R))
        for (j, D) in enumerate(comps)
            j == i && continue
            dij = divd(absd(C.center - D.center), onepu)
            rmax = min(rmax, dij / 2)
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
        dij = divd(absd(discs[i].center - discs[j].center), onepu)
        dij > addu(discs[i].radius, discs[j].radius) || (ok = false)
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

# ------------------------------------------------------------------------------------------------
# Matrix-free entry points (`(Amul, Aadj, n; kwargs...)`) and dense wrappers
# ------------------------------------------------------------------------------------------------

"Leading Ritz values of a short Arnoldi, by decreasing modulus (floating point)."
function _ritz_leading(Amul, n, T; L = 60)
    V, H = arnoldi(Amul, n, min(L, n); T)
    Lh = size(H, 2)
    λ = eigvals(H[1:Lh, 1:Lh])
    return λ[sortperm(abs.(λ); rev = true)]
end

function _radius_between(λ, neig)
    length(λ) > neig || return abs(λ[end]) / 2
    return sqrt(abs(λ[neig]) * abs(λ[neig + 1]))
end

"""
    certify_leading_eigs(Amul, Aadj, n; R = nothing, neig = 5, epsA, etaA = 0, kmax = 200,
                         frame_kwargs = (;), split_kwargs = (;), nθ = 64) -> NamedTuple
    certify_leading_eigs(A::AbstractMatrix; kwargs...)

The whole pipeline of §5: a frame of Definition def:Q chosen by the depth rule eq:depthrule in
floating point ([`choose_frame`](@ref)), one certified split ([`certify_split`](@ref)), and
[`certify_exterior`](@ref) at `R`. If `R` is not given it is placed between the moduli of the
`neig`-th and `(neig+1)`-th Ritz values of a short Arnoldi (geometric mean). Returns
`(; ok, R, count, discs, exterior, split, frame)`: on `ok`, A has exactly `count` eigenvalues of
modulus `> R`, all in the union of `discs`, `discs[i].count` of them in the i-th.
"""
function certify_leading_eigs(Amul, Aadj, n::Integer; R = nothing, neig::Integer = 5,
        epsA::Real, etaA::Real = 0.0, kmax::Integer = min(n - 1, 200), frame_kwargs = (;),
        split_kwargs = (;), nθ::Int = 64, Q = nothing)
    op = LinOp(Amul, Aadj, n; epsA, etaA)
    return _certify_leading(op, Amul, Aadj; R, neig, kmax, frame_kwargs, split_kwargs, nθ, Q)
end
function certify_leading_eigs(A::AbstractMatrix; R = nothing, neig::Integer = 5,
        kmax::Integer = min(size(A, 1) - 1, 200), frame_kwargs = (;), split_kwargs = (;),
        nθ::Int = 64, Q = nothing)
    op = LinOp(A)
    return _certify_leading(op, x -> A * x, x -> A' * x; R, neig, kmax, frame_kwargs,
        split_kwargs, nθ, Q)
end

function _certify_leading(op, Amul, Aadj; R, neig, kmax, frame_kwargs, split_kwargs, nθ, Q)
    t0 = time()
    if R === nothing
        R = _radius_between(_ritz_leading(Amul, op.n, op.T), neig)
    end
    if Q === nothing
        Q, finfo = choose_frame(op, Amul, Aadj, R; kmax, frame_kwargs...)
    else
        finfo = (; given = true, dimQ = size(Q, 2))
    end
    t1 = time()
    sc = certify_split(op, Q; target = R, split_kwargs...)
    t2 = time()
    ext = certify_exterior(sc, R; nθ)
    t3 = time()
    return (; ok = ext.ok, R = Float64(R), count = ext.count, discs = ext.discs,
        exterior = ext.exterior, split = sc, frame = finfo,
        time = (; frame = t1 - t0, split = t2 - t1, exterior = t3 - t2))
end

"""
    certify_gap(Amul, Aadj, n; R = nothing, epsA, etaA = 0, kwargs...) -> NamedTuple
    certify_gap(A::AbstractMatrix; kwargs...)

Corollary cor:gap through the pipeline of [`certify_leading_eigs`](@ref) with `neig = 1`:
`certified = true` means that A has exactly one eigenvalue of modulus `> R` and every other
eigenvalue has `|λ| ≤ R`; for a stochastic A, `1 − |λ₂| ≥ 1 − R`.
"""
function certify_gap(Amul, Aadj, n::Integer; R = nothing, kwargs...)
    out = certify_leading_eigs(Amul, Aadj, n; R, neig = 1, kwargs...)
    return (; certified = out.exterior.ok && out.exterior.region_ok && out.count == 1,
        out...)
end
function certify_gap(A::AbstractMatrix; R = nothing, kwargs...)
    out = certify_leading_eigs(A; R, neig = 1, kwargs...)
    return (; certified = out.exterior.ok && out.exterior.region_ok && out.count == 1,
        out...)
end

"""
    krylov_resolvent_floor(Amul, Aadj, n, Q, z; epsA, etaA = 0, kwargs...) -> (floor, info)
    krylov_resolvent_floor(A::AbstractMatrix, Q, z; kwargs...)

Certified lower bound on `σ_min(A − zI)` at one `z` for a given frame `Q`: [`certify_split`](@ref)
then [`resolvent_floor`](@ref). `floor = 0` means "no bound" (for instance when `|z| ≤ ḡ`).
"""
function krylov_resolvent_floor(Amul, Aadj, n::Integer, Q::AbstractMatrix, z::Number;
        epsA::Real, etaA::Real = 0.0, kwargs...)
    sc = certify_split(Amul, Aadj, n, Q; epsA, etaA, compression = false, kwargs...)
    r = resolvent_floor(sc, z)
    return r.floor, (; r..., split = sc)
end
function krylov_resolvent_floor(A::AbstractMatrix, Q::AbstractMatrix, z::Number; kwargs...)
    sc = certify_split(A, Q; compression = false, kwargs...)
    r = resolvent_floor(sc, z)
    return r.floor, (; r..., split = sc)
end
krylov_resolvent_floor(A::AbstractMatrix, Q::AbstractVector, z::Number; kwargs...) =
    krylov_resolvent_floor(A, reshape(Q, :, 1), z; kwargs...)

# ------------------------------------------------------------------------------------------------
# Self-adjoint operators (cor:selfadj, cor:weyl, rem:lanczos)
# ------------------------------------------------------------------------------------------------

"""
    eig_intervals(sc) -> Vector{EigInterval}

For a Hermitian A: real intervals `[Re c − r, Re c + r]` from the certified components of the
enclosure of G11 (its eigenvalues are real, and each component disc `D(c, r)` holds `count` of
them).
"""
function eig_intervals(sc::SplitCertificate)
    _has_counts(sc) || return EigInterval[]
    return [EigInterval(subd(real(C.center), C.radius), addu(real(C.center), C.radius),
        C.count) for C in _comps(sc)]
end

"`γ̄ ≥ ‖P⊥AZ‖` for Hermitian A: both couplings bound the same number, so the smaller is kept."
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
