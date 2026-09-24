# The triangular route for the compression: the two-block floor in the unitary Schur basis.
#
# The compression is put in Schur form, ordered by modulus, and cut once. In that basis the matrix
# is block triangular, so the coupling below the diagonal is zero up to the enclosure, the
# hypothesis of Theorem thm:assembly is satisfied wherever both diagonal blocks are invertible, and
# the whole price of the cut is the coupling above the diagonal in the denominator:
#
#     σ_min(G̃ − zM) ≥ (σ₁₁σ₂₂ − γ₁γ₂)/(σ₁₁ + σ₂₂ + γ₁ + γ₂) / D,
#     σ₁₁(z) = σ_min(T₁₁ − z)  by one verified SVD of the leading block,
#     σ₂₂(z)                   from eq:farpower on the norms ‖T₂₂^j‖ (Proposition prop:power),
#     D = ‖Y‖‖W‖(1 + α)/(1 − ‖R₂‖)  the certified cost of the change of basis.
#
# Two traps, both of which cost a round of wrong numbers. σ₂₂ is bounded through the norms of the
# powers, never through min_j ‖T₂₂^j‖^{1/j}, which bounds the spectral radius and is smaller by
# orders on a non-normal block. And the change of basis is chosen in floating point, so ‖Y‖, ‖W‖
# and ‖R₂‖ have to be certified rather than taken to be one.
#
# The basis is unitary, so nothing here carries the conditioning of a block diagonalisation: there
# is no κ and no β_Λ. The count follows from the same quantity, because the right-hand side is
# decreasing in the couplings, so a contour on which it is positive stays non-singular along the
# homotopy t·T₁₂, t: 1 → 0, and the count is that of diag(T₁₁, T₂₂), read off the diagonal.
#
# The cut is chosen by a floating-point scan: the shallowest cut whose predicted floor is within
# `factor` of the best over the scan, which trades a little quality for a per-point cost that grows
# like k_EN³.

"""
    TriangularSplit

The Schur basis, the cut and the certified quantities of the triangular route. Fields: `k` the
size, `c` the cut `k_EN`, `R` the radius, `V` and `T` the floating-point Schur factor and basis,
`That` the ball `V*M⁻¹G̃V`, `P11` and `Pdeep` its diagonal blocks as balls, `γ1`, `γ2` the two
couplings, `gj[j] ≥ ‖T₂₂^j‖` the certified norms of the powers of the trailing block and `gbar`
the least of their `j`-th roots (a bound on its spectral radius, reported but never used in a
floor), `corr` the correction from `‖R₂‖`, `α` the frame defect, `divisor` the certified cost of
the change of basis, and `info` the scan.
"""
struct TriangularSplit
    k::Int
    c::Int
    R::Float64
    V::Matrix{ComplexF64}
    T::Matrix{ComplexF64}
    That::BallMatrix
    P11::BallMatrix
    Pdeep::BallMatrix
    γ1::Float64
    γ2::Float64
    gbar::Float64
    gj::Vector{Float64}
    corr::Float64
    nR2::Float64
    α::Float64
    divisor::Float64
    info::NamedTuple
end

function Base.show(io::IO, ts::TriangularSplit)
    print(io, "TriangularSplit(k = $(ts.k), k_EN = $(ts.c), R = $(ts.R), γ1 = $(ts.γ1), ",
        "γ2 = $(ts.γ2), ḡ₂₂ = $(ts.gbar))")
end

_tri_blockball(B, r1, r2) = BallMatrix(Matrix(mid(B))[r1, r2], Matrix(rad(B))[r1, r2])

"""
Sort a complex Schur factor by decreasing modulus of the diagonal, in place, by moving each
largest remaining entry into position with one LAPACK reordering. At most `k − 1` moves, and every
cut is then a prefix, so the scan never reorders again.
"""
function _tri_sort_schur!(T::Matrix{ComplexF64}, V::Matrix{ComplexF64})
    k = size(T, 1)
    for i in 1:(k - 1)
        j = argmax(m -> abs(T[m, m]), i:k)
        j == i && continue
        LinearAlgebra.LAPACK.trexc!('V', j, i, T, V)
    end
    return T, V
end

"""
`‖T[1:c, c+1:k]‖_F` at every cut `c` in `cuts`, in one pass. Moving the cut from `c` to `c+1`
removes the entries of column `c+1` above row `c+1` and adds the entries of row `c+1` to the right
of it, so the whole sweep costs `O(k²)` rather than `O(k²)` a cut.
"""
function _tri_coupling_fro(T::AbstractMatrix, cuts::AbstractVector{<:Integer})
    k = size(T, 1)
    isempty(cuts) && return Float64[]
    c0 = first(cuts)
    s = 0.0
    for j in (c0 + 1):k, i in 1:c0
        s += abs2(T[i, j])
    end
    out = zeros(Float64, length(cuts))
    c = c0
    for (idx, cc) in enumerate(cuts)
        while c < cc
            for i in 1:c
                s -= abs2(T[i, c + 1])
            end
            for j in (c + 2):k
                s += abs2(T[c + 1, j])
            end
            c += 1
        end
        out[idx] = sqrt(max(s, 0.0))
    end
    return out
end

"""
Floating-point `‖B^j‖` for `j = 1, 2, …`, stopping as soon as eq:farpower gives a positive bound
on `σ_min(B − z)` at `|z| = R`, or at `mmax`. These are norms of powers, which is what
[`farpower_lo`](@ref) consumes; the `j`-th roots bound the spectral radius and are not usable in
their place.
"""
function _tri_fp_powers(B::AbstractMatrix, R::Real, mmax::Integer)
    isempty(B) && return Float64[0.0]
    P = Matrix(B)
    Pj = P
    gs = Float64[]
    for j in 1:mmax
        push!(gs, opnorm(Pj))
        farpower_lo(ComplexF64(R), gs) > 0 && break
        j < mmax && (Pj = Pj * P)
    end
    return gs
end

"""
    triangular_split(G, M, R; α = 0.0, factor = 2.0, nshort = 24, mmax = 8, nθ = 8)
        -> TriangularSplit

The scan, the cut and the one ball evaluation. `G` and `M` are the ball compression and its Gram,
`R` the radius, `α ≥ ‖I − Q*Q‖`.

The Schur factor is sorted by decreasing modulus once, so that every cut is a prefix of the same
`T` and no candidate costs a reordering. The scan then runs in two stages, both in floating point
and neither certified. Every admissible cut, one that does not separate a pair of equal moduli, is
ranked by what is free to compute, the coupling `‖T₁₂‖_F` above the diagonal and the modulus of the
largest eigenvalue left behind, which caps `σ₂₂`. The best `nshort` of them, and the shallowest, are
then measured properly: the power bound on the trailing block and `σ₁₁` on the circle at `nθ` angles
by inverse iteration on the leading triangular block. The cut kept is the shallowest whose predicted
floor is within `factor` of the best.

Returns `nothing` when no cut brings the trailing block below `R`.
"""
function triangular_split(G, M, R::Real; α::Real = 0.0, factor::Real = 2.0,
        nshort::Integer = 24, mmax::Integer = 8, nθ::Integer = 8)
    Gb = _promote_ball(_ball(G))
    Mb = _promote_ball(_ball(M))
    Gm = ComplexF64.(Matrix(mid(Gb)))
    Mm = ComplexF64.(Matrix(mid(Mb)))
    k = size(Gm, 1)
    k == 0 && return nothing
    Gc = Mm \ Gm
    R = Float64(R)
    S = schur(Gc)
    T0 = Matrix(S.T)
    V0 = Matrix(S.Z)
    # sort the Schur factor by decreasing modulus once, so that every cut is a prefix of the same
    # T and the scan costs no further reordering
    T, V = _tri_sort_schur!(T0, V0)
    a = abs.(diag(T))
    kE = count(>(R), a)
    kE == 0 && return nothing
    adm = [c for c in kE:(k - 1) if a[c] > a[c + 1] * (1 + 1e-10)]
    isempty(adm) && return nothing

    # stage one, every admissible cut. The Frobenius norm of the coupling is used here and only
    # here, because it is the one quantity that updates in O(k) as the cut moves by one: the block
    # loses the part of column c+1 above the diagonal and gains the part of row c+1 to the right of
    # it. It overestimates the spectral norm by at most the square root of the rank, so the ranking
    # is pessimistic, which is the safe direction for a shortlist.
    fro = _tri_coupling_fro(T, adm)
    prox = [(; c = adm[i], n12f = fro[i], s22p = R - a[adm[i] + 1]) for i in eachindex(adm)]
    rank = sort(prox; by = p -> -(p.s22p > 0 ? p.s22p / (p.s22p + p.n12f) : -Inf))
    short = unique(vcat(first(rank, min(nshort, length(rank))), [prox[1]]))

    # stage two, the shortlist only: the spectral norm of the coupling by an economy SVD, the power
    # bound on the trailing block, and σ11 on the circle by inverse iteration on the leading
    # triangular block. The block is being formed here anyway, and the Frobenius overestimate would
    # otherwise choose the cut.
    rows = NamedTuple[]
    for p in short
        c = p.c
        gs = _tri_fp_powers(Matrix(view(T, (c + 1):k, (c + 1):k)), R, mmax)
        s22 = farpower_lo(ComplexF64(R), gs)
        s22 > 0 || continue
        s11 = minimum(sigma11_estimate(view(T, 1:c, 1:c), R * cis(2π * t / nθ)) for t in 0:(nθ - 1))
        s11 > 0 || continue
        n12 = opnorm(Matrix(view(T, 1:c, (c + 1):k)))
        g = minimum(gs[j]^(1 / j) for j in eachindex(gs))
        push!(rows, (; c, score = s11 * s22 / (s11 + s22 + n12), n12, n12f = p.n12f, g, s11))
    end
    isempty(rows) && return nothing
    bestscore = maximum(r.score for r in rows)
    elig = [r for r in rows if r.score >= bestscore / Float64(factor)]
    best = elig[argmin([r.c for r in elig])]
    c = best.c
    cur, rest = 1:c, (c + 1):k

    # one ball evaluation: T̂ = V*M⁻¹G̃V, with the residual of the basis carried as a correction
    Wb = BallMatrix(V)
    Yb = BallMatrix(Matrix(V') * inv(Mm))
    That = _bmul(Yb, _bmul(Gb, Wb))
    R2 = _bsub(_bmul(Yb, _bmul(Mb, Wb)), Matrix{ComplexF64}(I, k, k))
    nR2 = opnorm_up(R2)
    nR2 < 1 || return nothing
    corr = divu(mulu(nR2, opnorm_up(That)), subd(1.0, nR2))
    # The change of basis is chosen in floating point, so its cost has to be certified rather than
    # assumed. With Y and W as here, Y(G̃ − zM)W = T̂ − z(I + R₂), and multiplying by (I + R₂)⁻¹
    #     σ_min(G̃ − zM) ≥ (1 − ‖R₂‖) · σ_min(T̂ + Δ − zI) / (‖Y‖ ‖W‖),   ‖Δ‖ ≤ corr,
    # so the floor is divided by ‖Y‖‖W‖/(1 − ‖R₂‖), and by (1 + α) for the frame as before. V is
    # unitary only up to rounding and M⁻¹ is a floating-point inverse, so neither norm is one.
    divisor = divu(mulu(mulu(opnorm_up(Yb), opnorm_up(Wb)), addu(1.0, Float64(α))),
        subd(1.0, nR2))
    γ1 = addu(opnorm_up(_tri_blockball(That, cur, rest)), corr)
    γ2 = addu(opnorm_up(_tri_blockball(That, rest, cur)), corr)
    P11 = _tri_blockball(That, cur, cur)
    Pdeep = _tri_blockball(That, rest, rest)
    # the certified norms of the powers of the trailing block; the loop stops as soon as
    # eq:farpower gives a positive σ22 on |z| = R, since further powers cannot be needed
    gj = Float64[]
    Nj = Pdeep
    gbar = Inf
    for j in 1:min(mmax, length(rest))
        j > 1 && (Nj = _bmul(Nj, Pdeep))
        push!(gj, opnorm_up(Nj; svd = false))
        gbar = min(gbar, nextfloat(gj[end]^(1 / j)))
        g1c = addu(gj[1], corr)
        gc = [addu(gj[i], subu(powu(g1c, i), powd(gj[1], i))) for i in eachindex(gj)]
        farpower_lo(ComplexF64(R), gc) > 0 && break
    end
    gbar = addu(gbar, corr)
    return TriangularSplit(k, c, R, V, T, That, P11, Pdeep, γ1, γ2, gbar, gj, corr, nR2,
        Float64(α), divisor,
        (; scan = [(; r.c, r.score, r.n12, r.n12f, r.g) for r in rows], factor = Float64(factor),
            bestscore))
end

"""
    _tri_far_lo(ts, z) -> Float64

A certified lower bound on `σ_min(T₂₂ − z)` for the trailing block. `ts.gj[j]` bounds `‖N^j‖` for
the computed block `N`; the true block is `N + Δ` with `‖Δ‖ ≤ ts.corr`, and expanding the product
gives `‖(N+Δ)^j‖ ≤ ‖N^j‖ + (‖N‖+corr)^j − ‖N‖^j`. The larger of eq:farshort at order one and
eq:farpower over the orders available is returned.

`ts.gbar`, the least of the `‖N^j‖^{1/j}`, is a bound on the spectral radius and **not** on a norm,
so it must not be used here; doing so overstated `σ₂₂` by five orders on a strongly non-normal
trailing block.
"""
function _tri_far_lo(ts::TriangularSplit, z::Number)
    isempty(ts.gj) && return 0.0
    g1 = addu(ts.gj[1], ts.corr)
    gc = [addu(ts.gj[j], subu(powu(g1, j), powd(ts.gj[1], j))) for j in eachindex(ts.gj)]
    return max(farshort_lo(z, g1), farpower_lo(z, gc))
end

"""
    sigma_tri_lo(ts, z) -> Float64

The certified lower bound on `σ_min(G̃ − zM)` of the triangular route at `z`: one verified SVD of
the leading block, [`_tri_far_lo`](@ref) for the trailing one, and \\eqref{eq:assembly}. Returns 0
when the hypothesis fails at `z`.
"""
function sigma_tri_lo(ts::TriangularSplit, z::Number)
    s11 = subd(sigmin_lo(_bsub(ts.P11, _zm_ball_id(ts.c, z))), ts.corr)
    s11 > 0 || return 0.0
    s22 = _tri_far_lo(ts, z)
    s22 > 0 || return 0.0
    num = subd(muld(s11, s22), mulu(ts.γ1, ts.γ2))
    num > 0 || return 0.0
    den = sumu((s11, s22, ts.γ1, ts.γ2))
    return divd(divd(num, den), ts.divisor)
end

_zm_ball_id(n::Integer, z::Number) = BallMatrix(ComplexF64(z) * Matrix{ComplexF64}(I, n, n))

"""
    count_tri(ts) -> Int

The number of eigenvalues of the compression outside `|z| = R`, valid once
[`cover_tri`](@ref) has certified the circle: the floor of [`sigma_tri_lo`](@ref) is decreasing in
the couplings, so positivity on the contour persists along the homotopy `t·T₁₂`, and the count is
that of the block diagonal, read off the Schur diagonal.
"""
count_tri(ts::TriangularSplit) = count(z -> abs(z) > ts.R, diag(ts.T))

"""
    _cover_kernel(floorfun, c, r; η, samples, maxarcs) -> NamedTuple

One certified cover of the circle `|z − c| = r`, through
`BallArithmetic.CertifScripts.adaptive_arcs!`. `floorfun(z)` returns a certified lower bound on
`σ_min(A − zI)`; an arc is accepted when its chord is at most `η` times the bound at one endpoint,
and bisected otherwise.

`σ_min` is 1-Lipschitz and on an arc of at most half a circle the farthest point from an endpoint
is the other endpoint, so the chord is the exact allowance and an accepted arc carries
`σ ≥ σ_a − ℓ ≥ (1 − η)σ_a`. Bisection takes the midpoint of the chord, so the contour certified is
the polygon inscribed in the circle, within the sagitta `r(1 − cos(π/samples))` of it.

Returns `(; ok, floor, npoints, nbisect, sagitta, rmin, reason)`, `rmin` being the least `|z|` over
the samples, which is what a region condition has to be tested against rather than `r`.
"""
function _cover_kernel(floorfun, c::Number, r::Real; η::Real = 0.5,
        samples::Integer = 256, maxarcs::Integer = 200_000)
    CS = BallArithmetic.CertifScripts
    r = Float64(r)
    circle = CS.CertificationCircle(ComplexF64(c), r; samples)
    arcs = collect(CS._initial_arcs(circle))
    cache = Dict{ComplexF64, Any}()
    evaluator = z -> (; z = ComplexF64(z), val = Ball(floorfun(z), 0.0),
        second_val = Ball(0.0, 0.0))
    res = CS.adaptive_arcs!(arcs, cache, Dict{Int, Any}(), Float64(η); evaluator, maxarcs)
    sag = mulu(r, subd(1.0, cos(π / samples)))
    npoints = length(cache)
    if !res.ok
        return (; ok = false, floor = 0.0, npoints, nbisect = res.processed, sagitta = sag,
            rmin = 0.0, reason = res.reason)
    end
    σmin = minimum(Float64(v[1].c) - Float64(v[1].r) for v in values(cache))
    rmin = minimum(absd(z) for z in keys(cache))
    return (; ok = true, floor = muld(subd(1.0, Float64(η)), σmin), npoints,
        nbisect = res.processed, sagitta = sag, rmin, reason = :complete)
end

"""
    certify_exterior_tri(ts, gbar, γ1, γ2; R = ts.R, σ22lo = nothing, nθ = 64, maxdepth = 6,
                         nradii = 12, tol = 1e-8) -> NamedTuple

The exterior certificate of Corollary cor:agree with σ11 from the triangular route. `gbar`, `γ1`
and `γ2` are the frame's own certified quantities, those of the far block and the two couplings,
and `σ22lo` an optional sharper lower bound on σ22 than `|z| − gbar`.

Three statements, in order. On `|z| = R` the hypothesis of Theorem thm:assembly is certified by
the kernel's adaptive refinement, which makes the floor a bound over
the whole circle and not only at the samples. The count outside is then that of the compression,
[`count_tri`](@ref), by the homotopy in the coupling. And around each eigenvalue of the Schur
factor lying outside `R` a disc is certified by the same test, at the first radius on a geometric
ladder for which it closes, the ladder stopping at half the distance to the nearest neighbour so
that the discs are disjoint.

Returns `(; ok, R, count, exterior, discs)`, with `ok` true when the circle and every disc closed.
"""
function certify_exterior_tri(ts::TriangularSplit, gbar::Real, γ1::Real, γ2::Real;
        R::Real = ts.R, σ22lo = nothing, η::Real = 0.5, samples::Integer = 256,
        maxarcs::Integer = 200_000, nradii::Int = 12, tol::Real = 1e-8,
        nθ::Int = 0, maxdepth::Int = 0)
    R = Float64(R)
    far = σ22lo === nothing ? (z -> farshort_lo(z, gbar)) : σ22lo
    function floorfun(z)
        s11 = sigma_tri_lo(ts, z)
        s11 > 0 || return 0.0
        s22 = far(z)
        s22 > 0 || return 0.0
        f1, _ = assembly_floor(s11, s22, γ1, γ2)
        f2, _ = blockdom_floor(s11, s22, γ1, γ2)
        return max(f1, f2)
    end
    chk = _cover_kernel(floorfun, 0, R; η, samples, maxarcs)
    discs = NamedTuple[]
    # the region condition of prop:count, tested against the polygon and not the circle
    region_ok = chk.ok && chk.rmin > Float64(gbar)
    # The contour certified is the polygon and the count is taken over the circle, so the two agree
    # unless an eigenvalue lies between them, that is within the sagitta of the radius. Since R is
    # placed in a gap between moduli that is what has to be checked, and it is far weaker than
    # asking the floor to exceed the sagitta.
    count_ok = chk.ok &&
               !any(z -> abs(abs(z) - R) <= chk.sagitta, diag(ts.T))
    (chk.ok && region_ok && count_ok) ||
        return (; ok = false, R, count = -1, exterior = chk, discs, region_ok, count_ok)
    λ = diag(ts.T)
    out = [z for z in λ if abs(z) > R]
    centres = ComplexF64[]
    for z in out
        any(c -> abs(c - z) <= tol * max(1.0, abs(z)), centres) || push!(centres, z)
    end
    ok = true
    for c in centres
        rmax = subd(absd(c), R)
        for d in centres
            d == c && continue
            rmax = min(rmax, absd(c - d) / 2)
        end
        rmin = max(tol * max(1.0, abs(c)), 1e-300)
        found = false
        if rmin < rmax
            for r in exp.(range(log(rmin), log(rmax * (1 - 1e-6)); length = nradii))
                ck = _cover_kernel(floorfun, c, r; η, samples, maxarcs)
                if ck.ok && subd(absd(c), r) > Float64(gbar) &&
                   !any(z -> abs(abs(z - c) - r) <= ck.sagitta, diag(ts.T))
                    push!(discs, (; center = c, radius = r,
                        count = count(z -> abs(z - c) < r, λ), check = ck))
                    found = true
                    break
                end
            end
        end
        found || (ok = false; push!(discs, (; center = c, radius = NaN, count = -1,
            check = nothing)))
    end
    return (; ok, R, count = count_tri(ts), exterior = chk, discs, region_ok, count_ok)
end

"""
    cover_tri(ts, R; η = 0.5, samples = 256, maxarcs = 200_000)

A certified cover of the contour `|z| = R` for the compression alone, through the kernel's
adaptive refinement, `BallArithmetic.CertifScripts.adaptive_arcs!`: the circle is discretised,
each arc is accepted when its chord is at most `η` times the bound at one endpoint, and bisected
otherwise. The bisection takes the midpoint of the chord, so what is certified is the polygon
inscribed in the circle; starting from a fine discretisation is what keeps the two together, and
the deviation is the sagitta `R(1 − cos(π/samples))`.

`σ_min` is 1-Lipschitz and on an arc of at most half a circle the farthest point from an endpoint
is the other endpoint, so the chord is the right allowance and an accepted arc carries
`σ ≥ σ_a − ℓ ≥ (1 − η)σ_a`. The floor returned is `(1 − η)` times the least bound seen.

The count is that of the compression outside `R`, withheld only when an eigenvalue of the Schur
factor lies within the sagitta of the radius: such a one could sit between the polygon and the
circle, outside the contour certified and inside the circle the count is taken over. Since `R` is
placed in a gap between moduli this rarely bites.

Returns `(; ok, floor, count, npoints, nbisect, sagitta, reason, time)`.
"""
function cover_tri(ts::TriangularSplit, R::Real = ts.R; η::Real = 0.5,
        samples::Integer = 256, maxarcs::Integer = 200_000, escalate = nothing)
    t0 = time()
    R = Float64(R)
    CS = BallArithmetic.CertifScripts
    circle = CS.CertificationCircle(0.0, R; samples)
    arcs = collect(CS._initial_arcs(circle))
    cache = Dict{ComplexF64, Any}()
    evaluator = function (z)
        v = sigma_tri_lo(ts, z)
        escalate === nothing || (v = max(v, Float64(escalate(z))))
        return (; z = ComplexF64(z), val = Ball(v, 0.0), second_val = Ball(0.0, 0.0))
    end
    res = CS.adaptive_arcs!(arcs, cache, Dict{Int, Any}(), Float64(η); evaluator, maxarcs)
    npoints = length(cache)
    sag = mulu(R, subd(1.0, cos(π / samples)))
    if !res.ok
        return (; ok = false, floor = 0.0, count = -1, npoints, nbisect = res.processed,
            sagitta = sag, reason = res.reason, time = time() - t0, zfail = res.z)
    end
    σmin = minimum(Float64(v[1].c) - Float64(v[1].r) for v in values(cache))
    fl = muld(subd(1.0, Float64(η)), σmin)
    near = any(z -> abs(abs(z) - R) <= sag, diag(ts.T))
    return (; ok = true, floor = fl, count = near ? -1 : count_tri(ts), npoints,
        nbisect = res.processed, sagitta = sag,
        reason = near ? :eigenvalue_within_the_sagitta : :complete, time = time() - t0,
        zfail = ComplexF64(NaN))
end
