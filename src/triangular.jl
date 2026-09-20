# The triangular route for the compression: the two-block floor in the unitary Schur basis.
#
# The compression is put in Schur form, ordered by modulus, and cut once. In that basis the matrix
# is block triangular, so the coupling below the diagonal is zero up to the enclosure, the
# hypothesis of Theorem thm:assembly is satisfied wherever both diagonal blocks are invertible, and
# the whole price of the cut is the coupling above the diagonal in the denominator:
#
#     σ_min(G̃ − zM) ≥ (σ₁₁σ₂₂ − γ₁γ₂)/(σ₁₁ + σ₂₂ + γ₁ + γ₂) / (1 + α),
#     σ₁₁(z) = σ_min(T₁₁ − z)  by one verified SVD of the leading block,
#     σ₂₂(z) ≥ |z| − ḡ₂₂       from the powers of the trailing block (Proposition prop:power).
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
couplings, `gbar` the bound on the deep tier with `gj` the power bounds it came from, `corr` the
correction from `‖R₂‖`, `α` the frame defect, and `info` the scan.
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

"Floating-point `min_{j ≤ mmax} ‖B^j‖^{1/j}`, stopping as soon as it falls below `R`."
function _tri_fp_power(B::AbstractMatrix, R::Real, mmax::Integer)
    isempty(B) && return 0.0
    g = Inf
    P = Matrix(B)
    Pj = P
    for j in 1:mmax
        g = min(g, opnorm(Pj)^(1 / j))
        g < R && break
        j < mmax && (Pj = Pj * P)
    end
    return g
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

    # stage one, every admissible cut: the coupling above the diagonal and the modulus of the
    # largest eigenvalue left behind, which is what σ22 can at best be. Both are free here.
    prox = [(; c, n12 = norm(view(T, 1:c, (c + 1):k)), s22p = R - a[c + 1]) for c in adm]
    rank = sort(prox; by = p -> -(p.s22p > 0 ? p.s22p / (p.s22p + p.n12) : -Inf))
    short = unique(vcat(first(rank, min(nshort, length(rank))), [prox[1]]))

    # stage two, the shortlist: the power bound on the trailing block and σ11 on the circle by
    # inverse iteration on the leading triangular block, O(c²) a step
    rows = NamedTuple[]
    for p in short
        c = p.c
        g = _tri_fp_power(Matrix(view(T, (c + 1):k, (c + 1):k)), R, mmax)
        g < R || continue
        s11 = minimum(sigma11_estimate(view(T, 1:c, 1:c), R * cis(2π * t / nθ)) for t in 0:(nθ - 1))
        s11 > 0 || continue
        s22 = R - g
        push!(rows, (; c, score = s11 * s22 / (s11 + s22 + p.n12), n12 = p.n12, g, s11))
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
    γ1 = addu(opnorm_up(_tri_blockball(That, cur, rest)), corr)
    γ2 = addu(opnorm_up(_tri_blockball(That, rest, cur)), corr)
    P11 = _tri_blockball(That, cur, cur)
    Pdeep = _tri_blockball(That, rest, rest)
    gj = Float64[]
    Nj = Pdeep
    gbar = Inf
    for j in 1:min(mmax, length(rest))
        j > 1 && (Nj = _bmul(Nj, Pdeep))
        push!(gj, opnorm_up(Nj; svd = false))
        gbar = min(gbar, nextfloat(gj[end]^(1 / j)))
        gbar < R && break
    end
    gbar = addu(gbar, corr)
    return TriangularSplit(k, c, R, V, T, That, P11, Pdeep, γ1, γ2, gbar, gj, corr, nR2,
        Float64(α), (; scan = [(; r.c, r.score, r.n12, r.g) for r in rows], factor = Float64(factor),
            bestscore))
end

"""
    sigma_tri_lo(ts, z) -> Float64

The certified lower bound on `σ_min(G̃ − zM)` of the triangular route at `z`: one verified SVD of
the leading block, the Neumann bound for the trailing one, and \\eqref{eq:assembly}. Returns 0
when the hypothesis fails at `z`.
"""
function sigma_tri_lo(ts::TriangularSplit, z::Number)
    s11 = subd(sigmin_lo(_bsub(ts.P11, _zm_ball_id(ts.c, z))), ts.corr)
    s11 > 0 || return 0.0
    s22 = subd(absu(ComplexF64(z)), ts.gbar)
    s22 > 0 || return 0.0
    num = subd(muld(s11, s22), mulu(ts.γ1, ts.γ2))
    num > 0 || return 0.0
    den = sumu((s11, s22, ts.γ1, ts.γ2))
    return divd(divd(num, den), addu(1.0, ts.α))
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
    certify_exterior_tri(ts, gbar, γ1, γ2; R = ts.R, σ22lo = nothing, nθ = 64, maxdepth = 6,
                         nradii = 12, tol = 1e-8) -> NamedTuple

The exterior certificate of Corollary cor:agree with σ11 from the triangular route. `gbar`, `γ1`
and `γ2` are the frame's own certified quantities, those of the far block and the two couplings,
and `σ22lo` an optional sharper lower bound on σ22 than `|z| − gbar`.

Three statements, in order. On `|z| = R` the hypothesis of Theorem thm:assembly is certified by
[`check_circle`](@ref) with the arc-wise Lipschitz refinement, which makes the floor a bound over
the whole circle and not only at the samples. The count outside is then that of the compression,
[`count_tri`](@ref), by the homotopy in the coupling. And around each eigenvalue of the Schur
factor lying outside `R` a disc is certified by the same test, at the first radius on a geometric
ladder for which it closes, the ladder stopping at half the distance to the nearest neighbour so
that the discs are disjoint.

Returns `(; ok, R, count, exterior, discs)`, with `ok` true when the circle and every disc closed.
"""
function certify_exterior_tri(ts::TriangularSplit, gbar::Real, γ1::Real, γ2::Real;
        R::Real = ts.R, σ22lo = nothing, nθ::Int = 64, maxdepth::Int = 6, nradii::Int = 12,
        tol::Real = 1e-8)
    R = Float64(R)
    σ11 = z -> sigma_tri_lo(ts, z)
    chk = check_circle(σ11, gbar, γ1, γ2, 0, R; region = :exterior, nθ, maxdepth, σ22lo)
    discs = NamedTuple[]
    (chk.ok && chk.region_ok) || return (; ok = false, R, count = -1, exterior = chk, discs)
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
                ck = check_circle(σ11, gbar, γ1, γ2, c, r; region = :interior, nθ, maxdepth,
                    σ22lo)
                if ck.ok && ck.region_ok
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
    return (; ok, R, count = count_tri(ts), exterior = chk, discs)
end

"""
    cover_tri(ts, R; nθ = 32, η = 0.5, maxpoints = 20000, escalate = nothing)

An adaptive cover of `|z| = R`. At a sample the bound is `sigma_tri_lo` (or the larger of it and
`escalate(z)`, when an escalation is supplied); an arc of length `ℓ` is accepted when the bound
exceeds `ℓ/η`, since the true `σ_min` is 1-Lipschitz. A positive bound that is too small for the
arc is answered by bisecting, which costs two cheap evaluations; a zero bound is answered by
escalating, since no refinement helps there.

Returns `(; ok, floor, count, npoints, nbisect, nescalate, time)`.
"""
function cover_tri(ts::TriangularSplit, R::Real = ts.R; nθ::Integer = 32, η::Real = 0.5,
        maxpoints::Integer = 20000, escalate = nothing)
    t0 = time()
    R = Float64(R)
    pts = [R * cis(2π * t / nθ) for t in 0:(nθ - 1)]
    arcs = [(pts[i], pts[mod1(i + 1, nθ)]) for i in 1:nθ]
    cache = Dict{ComplexF64, Float64}()
    esc = Set{ComplexF64}()
    fl = Inf
    nb = 0
    while !isempty(arcs)
        za, zb = pop!(arcs)
        b = get!(() -> sigma_tri_lo(ts, za), cache, za)
        ℓ = abs(zb - za)
        if b <= 0 && escalate !== nothing && !(za in esc)
            push!(esc, za)
            b = max(b, Float64(escalate(za)))
            cache[za] = b
        end
        if b > ℓ / η
            fl = min(fl, b)
        elseif b > 0
            zm = (za + zb) / 2
            zm = R * zm / abs(zm)
            push!(arcs, (za, zm))
            push!(arcs, (zm, zb))
            nb += 1
        else
            return (; ok = false, floor = 0.0, count = -1, npoints = length(cache), nbisect = nb,
                nescalate = length(esc), time = time() - t0, zfail = za)
        end
        length(cache) > maxpoints &&
            return (; ok = false, floor = 0.0, count = -1, npoints = length(cache), nbisect = nb,
                nescalate = length(esc), time = time() - t0, zfail = za)
    end
    return (; ok = true, floor = fl, count = count_tri(ts), npoints = length(cache), nbisect = nb,
        nescalate = length(esc), time = time() - t0, zfail = ComplexF64(NaN))
end
