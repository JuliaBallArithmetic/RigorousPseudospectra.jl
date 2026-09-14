# §2 (sec:bound): the two-block bound, the far block through its norm and its powers, the
# exterior inclusion and count, and the self-adjoint corollaries.
#
# Inputs are certified numbers: σ11 and σ22 LOWER bounds, γ1, γ2, ḡ UPPER bounds. Every formula
# below is monotone in them in the direction that keeps the output a bound (the monotonicity is
# stated in thm:assembly and checked in the comments), and is evaluated with directed rounding.
# A function that cannot certify returns `ok = false` and a zero bound; it never returns a
# positive number without its hypothesis checked.

"""
    assembly_floor(σ11, σ22, γ1, γ2) -> (floor, ok)

Theorem thm:assembly, eq:assembly: if `σ11 σ22 > γ1 γ2` (certified with the rounding directed
against it) then `1/‖(A − zI)⁻¹‖ ≥ (σ11σ22 − γ1γ2)/(σ11 + σ22 + γ1 + γ2) = floor > 0`.
Arguments: lower bounds `σ11 ≤ σ_min(G11 − zI)`, `σ22 ≤ 1/‖(G22 − zI)⁻¹‖`, upper bounds
`γ1 ≥ ‖G12‖`, `γ2 ≥ ‖G21‖`, kept separate. Otherwise returns `(0.0, false)`: "no bound".
The right-hand side increases in σ11, σ22 and decreases in γ1, γ2, so evaluating it at the
bounds gives a lower bound on its value at the true quantities.
"""
function assembly_floor(s11::Real, s22::Real, g1::Real, g2::Real)
    s11, s22, g1, g2 = Float64(s11), Float64(s22), Float64(g1), Float64(g2)
    (s11 > 0 && s22 > 0 && g1 >= 0 && g2 >= 0) || return (0.0, false)
    num = subd(muld(s11, s22), mulu(g1, g2))
    (num > 0 && isfinite(num)) || return (0.0, false)
    den = addu(addu(s11, s22), addu(g1, g2))
    f = divd(num, den)
    return f > 0 ? (f, true) : (0.0, false)
end

"""
    blockdom_floor(σ11, σ22, γ1, γ2) -> (floor, ok)

Proposition prop:blockdom, eq:blockdom: if `γ1 < σ11` and `γ2 < σ22`, then
`1/‖(A − zI)⁻¹‖ ≥ min(σ11, σ22)(1 − τ)`, `τ = max(γ1/σ11, γ2/σ22)`. Same argument conventions as
[`assembly_floor`](@ref); the right-hand side increases in σ11, σ22.
"""
function blockdom_floor(s11::Real, s22::Real, g1::Real, g2::Real)
    s11, s22, g1, g2 = Float64(s11), Float64(s22), Float64(g1), Float64(g2)
    (s11 > 0 && s22 > 0) || return (0.0, false)
    τ = max(divu(g1, s11), divu(g2, s22))
    τ < 1 || return (0.0, false)
    f = muld(min(s11, s22), subd(1.0, τ))
    return f > 0 ? (f, true) : (0.0, false)
end

"""
    farshort_lo(z, ḡ) -> Float64

eq:farshort: `σ22(z) ≥ |z| − ḡ` when `ḡ ≥ ‖G22‖` and `|z| > ḡ`; returns a certified lower bound,
or `0.0` when `|z| ≤ ḡ` cannot be excluded.
"""
function farshort_lo(z::Number, gbar::Real)
    v = subd(absd(z), Float64(gbar))
    return v > 0 ? v : 0.0
end

"""
    farpower_lo(z, ḡs) -> Float64

eq:farpower (Proposition prop:power applied to G22): with `ḡs[j] ≥ ‖G22^j‖` for `j = 1:m`
(and `ḡ_0 = 1`), `σ22(z) ≥ (|z|^m − ḡ_m)/Σ_{j<m} |z|^{m−1−j} ḡ_j` when `|z|^m > ḡ_m`. Returns the
largest of these lower bounds over the orders supplied (each order is a valid bound on its own),
or `0.0`. The numerator increases and the denominator increases in |z|, so the lower bound of |z|
is used in the numerator and the upper bound in the denominator.
"""
function farpower_lo(z::Number, gbars::AbstractVector{<:Real})
    zl, zu = absd(z), absu(z)
    best = 0.0
    for m in eachindex(gbars)
        num = subd(powd(zl, m), Float64(gbars[m]))
        num > 0 || continue
        den = powu(zu, m - 1)                     # j = 0 term, ḡ_0 = 1
        for j in 1:(m - 1)
            den = addu(den, mulu(powu(zu, m - 1 - j), Float64(gbars[j])))
        end
        best = max(best, divd(num, den))
    end
    return best
end

"""
    inclusion_delta(ε, R, ḡ, γ1, γ2) -> (δ, ok)

Proposition prop:complete, eq:incl: if `R > ḡ ≥ ‖G22‖` and `ε(R − ḡ) > γ1γ2`, then
`Λ_δ(A) ∩ {|z| > R} ⊆ Λ_ε(G11) ∩ {|z| > R}` with
`δ = (ε(R − ḡ) − γ1γ2)/(ε + (R − ḡ) + γ1 + γ2)`. δ increases in `s = R − ḡ`, so a lower bound of
`s` is used throughout.
"""
function inclusion_delta(ε::Real, R::Real, gbar::Real, g1::Real, g2::Real)
    s = subd(Float64(R), Float64(gbar))
    s > 0 || return (0.0, false)
    return assembly_floor(Float64(ε), s, g1, g2)    # same expression with σ11 → ε, σ22 → s
end

# ------------------------------------------------------------------------------------------------
# Contours: the sound cover of a circle (Proposition prop:count)
# ------------------------------------------------------------------------------------------------

"""
    CircleCheck

Result of [`check_circle`](@ref). `ok` means that `σ11(z)σ22(z) > γ1γ2` was certified at every
point of the circle (not only at the samples); `floor` is then a certified lower bound on
`σ_min(A − zI)` over the whole circle (Theorem thm:assembly); `region_ok` records the hypothesis
`D ∪ Γ ⊆ {|z| > ḡ}` of prop:count for the component `D` named by `region`.
"""
struct CircleCheck
    center::ComplexF64
    radius::Float64
    ok::Bool
    region_ok::Bool
    region::Symbol          # :interior (the open disc) or :exterior ({|z − c| > r}, c = 0)
    floor::Float64
    nevals::Int
    minσ11::Float64
    minσ22::Float64
end

"""
    check_circle(σ11lo, ḡ, γ1, γ2, c, r; region, nθ = 64, maxdepth = 6, σ22lo = nothing,
                 arcs = nothing)

Certify the hypothesis of prop:count on the circle `Γ = {|z − c| = r}` (or on the sub-arcs
`arcs`, a list of angle intervals `(θa, θb)`). `σ11lo(z)` must return a certified lower bound on
`σ11(z) = σ_min(G11 − zI)`; `σ22` is bounded below by eq:farshort, `|z| − ḡ`, or by `σ22lo(z)` if
given. The true σ11 and σ22 are 1-Lipschitz in `z` (a smallest singular value, and the reciprocal
of a resolvent norm, move by at most `|Δz|`), which is what makes a finite sample a cover: the
circle is cut into `nθ` arcs; every point of an arc between two samples is within `h = r·Δθ/2`
(arc length) of the nearer sample, plus `δ = 8u(|c| + r)`, a bound on the distance between a
computed sample point and the exact point of the circle at that angle. Hence on the arc
`σ11 ≥ min(σ11lo(z_a), σ11lo(z_b)) − h` and likewise for σ22, and thm:assembly's hypothesis is
tested with these arc-wise lower bounds; an arc that fails is bisected, up to `maxdepth` times.

`region = :interior` also checks `D ∪ Γ ⊆ {|z| > ḡ}` for the open disc `D`, i.e. `|c| − r > ḡ`;
`region = :exterior` (only for `c = 0`) checks it for `D = {|z| > r}`, i.e. `r > ḡ`.
"""
function check_circle(σ11lo, gbar::Real, g1::Real, g2::Real, c::Number, r::Real;
        region::Symbol = :interior, nθ::Int = 64, maxdepth::Int = 6, σ22lo = nothing,
        arcs = nothing)
    c = ComplexF64(c)
    r = Float64(r)
    gbar = Float64(gbar)
    region_ok = if region === :interior
        subd(absd(c), r) > gbar
    elseif region === :exterior
        c == 0 && r > gbar
    else
        error("region must be :interior or :exterior")
    end
    arcs0 = arcs === nothing ? [(0.0, 2π)] : arcs
    ok, floor, nev, m11, m22 = _check_arcs(σ11lo, σ22lo, gbar, g1, g2, c, r, arcs0; nθ,
        maxdepth)
    return CircleCheck(c, r, ok, region_ok, region, ok ? floor : 0.0, nev, m11, m22)
end

function _check_arcs(σ11lo, σ22lo, gbar, g1, g2, c::ComplexF64, r::Float64, arcs0; nθ, maxdepth)
    u = unit_roundoff(Float64)
    δ = mulu(8u, addu(absu(c), r))
    s22(z) = σ22lo === nothing ? subd(absd(z), gbar) : Float64(σ22lo(z))
    nev = 0
    vals = Dict{Float64, Tuple{Float64, Float64}}()
    ev(θ) = get!(vals, θ) do
        nev += 1
        z = c + r * cis(θ)
        (Float64(σ11lo(z)), s22(z))
    end
    arcs = Tuple{Float64, Float64, Int}[]
    step = 2π / nθ
    for (θa, θb) in arcs0
        np = max(1, ceil(Int, (θb - θa) / step))
        for s in 0:(np - 1)
            push!(arcs, (θa + (θb - θa) * s / np, s == np - 1 ? θb : θa + (θb - θa) * (s + 1) / np,
                0))
        end
    end
    ok = true
    floor = Inf
    m11 = Inf
    m22 = Inf
    while !isempty(arcs)
        θa, θb, d = pop!(arcs)
        a = ev(θa)
        b = ev(θb)
        h = addu(mulu(r, divu(subu(θb, θa), 2.0)), δ)
        l11 = subd(min(a[1], b[1]), h)
        l22 = subd(min(a[2], b[2]), h)
        f, fok = assembly_floor(l11, l22, g1, g2)
        if fok
            # only the accepted arcs enter the minima: they cover Γ, and a bisected arc's own
            # (coarser) values would only make σ11, σ22 and the floor look worse than certified
            floor = min(floor, f)
            m11 = min(m11, l11)
            m22 = min(m22, l22)
        elseif d < maxdepth
            θm = (θa + θb) / 2
            push!(arcs, (θa, θm, d + 1))
            push!(arcs, (θm, θb, d + 1))
        else
            ok = false
            m11 = min(m11, l11)
            m22 = min(m22, l22)
            break
        end
    end
    return ok, (ok ? floor : 0.0), nev, m11, m22
end

"""
    union_boundary_arcs(centres, radii; μ = 1e-6) -> Vector{Vector{Tuple{Float64,Float64}}}

For the union U of the closed discs `D(c_i, r_i)`, the arcs of each circle that must be checked:
the whole circle minus the arcs lying strictly inside another disc. The skipped arcs are shrunk by
the angle `μ` at both ends and kept only when that margin moves the points inside the other disc
by far more than the rounding of κ = cos ψ (`μ sin ψ > max(1e-9, 10·err(κ))`, ψ the half-angle of
the covered arc), so every point of ∂U lies on a checked arc.
"""
function union_boundary_arcs(centres::AbstractVector, radii::AbstractVector; μ::Real = 1e-6)
    p = length(centres)
    out = Vector{Vector{Tuple{Float64, Float64}}}(undef, p)
    for i in 1:p
        skips = Tuple{Float64, Float64}[]
        ci, ri = ComplexF64(centres[i]), Float64(radii[i])
        for j in 1:p
            j == i && continue
            d = ComplexF64(centres[j]) - ci
            D = abs(d)
            rj = Float64(radii[j])
            D == 0 && continue
            κ = (ri^2 + D^2 - rj^2) / (2ri * D)
            κ >= 1 && continue
            if κ <= -1
                # the whole circle i inside disc j, with a margin
                (D + ri) < rj * (1 - 1e-12) && push!(skips, (0.0, 2π))
                continue
            end
            ψ = acos(κ)
            errκ = 4eps() * (ri^2 + D^2 + rj^2) / (2ri * D)     # rounding in κ, generously
            μ * sin(ψ) > max(1e-9, 10errκ) || continue
            ψ > μ || continue
            φ = angle(d)
            push!(skips, (φ - ψ + μ, φ + ψ - μ))
        end
        # normalise to [0, 2π), merge, complement
        segs = Tuple{Float64, Float64}[]
        for (a, b) in skips
            if b - a >= 2π
                push!(segs, (0.0, 2π))
                continue
            end
            a2 = mod(a, 2π)
            b2 = a2 + (b - a)
            if b2 <= 2π
                push!(segs, (a2, b2))
            else
                push!(segs, (a2, 2π))
                push!(segs, (0.0, b2 - 2π))
            end
        end
        sort!(segs)
        merged = Tuple{Float64, Float64}[]
        for sgm in segs
            if !isempty(merged) && sgm[1] <= merged[end][2]
                merged[end] = (merged[end][1], max(merged[end][2], sgm[2]))
            else
                push!(merged, sgm)
            end
        end
        comp = Tuple{Float64, Float64}[]
        t = 0.0
        for (a, b) in merged
            a > t && push!(comp, (t, a))
            t = max(t, b)
        end
        t < 2π && push!(comp, (t, 2π))
        out[i] = comp
    end
    return out
end

# ------------------------------------------------------------------------------------------------
# The spectral projector (Proposition prop:projector)
# ------------------------------------------------------------------------------------------------

"""
    circle_min(flo, c, r; nθ = 64, maxdepth = 6) -> (ok, minimum, nevals)

Certified lower bound on the minimum over the circle `|z − c| = r` of a function that is
1-Lipschitz in `z`, from a certified lower bound `flo(z)` at sample points: the same arc cover as
[`check_circle`](@ref), with the slack `h = rΔθ/2 + δ` on each arc, and bisection (up to
`maxdepth` times) of the arcs whose bound is not positive. `ok` means that the returned minimum is
positive.
"""
function circle_min(flo, c::Number, r::Real; nθ::Int = 64, maxdepth::Int = 6)
    c = ComplexF64(c)
    r = Float64(r)
    δ = mulu(8 * unit_roundoff(Float64), addu(absu(c), r))
    nev = 0
    vals = Dict{Float64, Float64}()
    ev(θ) = get!(vals, θ) do
        nev += 1
        Float64(flo(c + r * cis(θ)))
    end
    arcs = [(2π * s / nθ, 2π * (s + 1) / nθ, 0) for s in 0:(nθ - 1)]
    ok = true
    m = Inf
    while !isempty(arcs)
        θa, θb, d = pop!(arcs)
        h = addu(mulu(r, divu(subu(θb, θa), 2.0)), δ)
        v = subd(min(ev(θa), ev(θb)), h)
        if v > 0
            m = min(m, v)
        elseif d < maxdepth
            θm = (θa + θb) / 2
            push!(arcs, (θa, θm, d + 1))
            push!(arcs, (θm, θb, d + 1))
        else
            ok = false
            m = min(m, v)
            break
        end
    end
    return ok, (ok ? m : min(m, 0.0)), nev
end

"""
    projector_bound(floorΓ, floor0, r, e) -> (bound, M, M0, ok)

Proposition prop:projector on a circle of radius `r`: with `floorΓ` a certified lower bound over
the contour on the right-hand side of eq:assembly, so that `M = 1/floorΓ ≥ sup_Γ ‖(A − zI)⁻¹‖`,
`floor0` a certified lower bound over the contour on `min(σ11(z), σ22(z))`, so that
`M0 = 1/floor0 ≥ sup_Γ max(1/σ11, 1/σ22)`, the resolvent bound of the block-diagonal part, and
`e ≥ ‖E‖` the norm of the off-diagonal part,

    ‖P − P₀‖ ≤ (|Γ|/2π) M M₀ e = r M M₀ e,

the length of the circle being `2πr`. The proof is the resolvent identity
`(zI−A)⁻¹ − (zI−A₀)⁻¹ = (zI−A)⁻¹E(zI−A₀)⁻¹` integrated over Γ. For the two-block splitting
`e = max(γ1, γ2)`, since `E*E` is block diagonal; inside sec:vbd, `A₀ = Λ`, `E = M_c − Λ` and
`e = β_Λ`.
"""
function projector_bound(floorΓ::Real, floor0::Real, r::Real, e::Real)
    floorΓ = Float64(floorΓ)
    floor0 = Float64(floor0)
    (floorΓ > 0 && floor0 > 0) || return (Inf, Inf, Inf, false)
    M = divu(1.0, floorΓ)
    M0 = divu(1.0, floor0)
    return (mulu(mulu(Float64(r), mulu(M, M0)), Float64(e)), M, M0, true)
end

# ------------------------------------------------------------------------------------------------
# Self-adjoint operators: Corollaries cor:selfadj and cor:weyl
# ------------------------------------------------------------------------------------------------

"""
    EigInterval(lo, hi, count)

A real interval `[lo, hi]` certified to contain exactly `count` eigenvalues of the compression
G11, counted with multiplicity; the intervals of one certificate are disjoint and together hold
all `k` eigenvalues.
"""
struct EigInterval
    lo::Float64
    hi::Float64
    count::Int
end

"""
    selfadj_enclosure(θ, ḡ, γ̄) -> Vector{NamedTuple}

Corollary cor:selfadj (i) for `A = A*`: every point λ of `spec(A)` with `|λ| > ḡ` satisfies
`min_i |λ − θ_i| (|λ| − ḡ) ≤ γ̄²`, `θ_i` the eigenvalues of G11 (enclosed by the intervals `θ`).
Returns intervals whose union contains `spec(A) ∩ {|λ| > ḡ}`, one per side and per interval `J`
of `θ` (`side = +1` for λ > ḡ, `-1` for λ < −ḡ), with the quadratic radii `ρ⁻`, `ρ⁺` about `J`:

* around `J = [lo, hi]` the interval `[r₂, r⁺]`, `r⁺ = ((hi + ḡ) + √((hi − ḡ)² + 4γ̄²))/2`
  (so `r⁺ − hi ≤ γ̄²/(hi − ḡ)`), `r₂ = ((lo + ḡ) + √((lo − ḡ)² − 4γ̄²))/2` when the
  discriminant is nonnegative, else `r₂ = ḡ`;
* next to the disc, `(ḡ, r₁]` with `r₁ = ((lo + ḡ) − √((lo − ḡ)² − 4γ̄²))/2` for the interval
  nearest to ḡ, reported with `J = 0`: that region is not excluded.

Endpoints are rounded outward (r⁺ increases in hi, ḡ, γ̄; r₂ increases in lo and decreases in ḡ,
γ̄; r₁ decreases in lo and increases in ḡ, γ̄).
"""
function selfadj_enclosure(θ::Vector{EigInterval}, gbar::Real, γbar::Real)
    gbar, γbar = Float64(gbar), Float64(γbar)
    g2 = mulu(γbar, γbar)
    absdiffu(a, b) = a >= b ? subu(a, b) : subu(b, a)
    out = NamedTuple[]
    for side in (1, -1)
        near = gbar                                  # the region (ḡ, near] is not excluded
        for (i, J) in enumerate(θ)
            lo, hi = side == 1 ? (J.lo, J.hi) : (-J.hi, -J.lo)
            d = absdiffu(hi, gbar)
            rp = divu(addu(addu(hi, gbar), sqrtu(addu(mulu(d, d), mulu(4.0, g2)))), 2.0)
            dl = subd(lo, gbar)
            disc = dl > 0 ? subd(muld(dl, dl), mulu(4.0, g2)) : -1.0
            if disc >= 0
                sq = sqrtd(disc)
                r2 = divd(addd(addd(lo, gbar), sq), 2.0)
                r1 = divu(subu(addu(lo, gbar), sq), 2.0)
                near = max(near, r1)
                push!(out, (; side, J = i, lo = side == 1 ? r2 : -rp,
                    hi = side == 1 ? rp : -r2, count = J.count,
                    ρminus = subu(lo, r2), ρplus = subu(rp, hi)))
            else
                near = max(near, rp)
            end
        end
        push!(out, (; side, J = 0, lo = side == 1 ? gbar : -near,
            hi = side == 1 ? near : -gbar, count = -1, ρminus = NaN, ρplus = NaN))
    end
    return out
end

"""
    selfadj_count(θ, ḡ, γ̄, x1, x2) -> (count, ok)

Corollary cor:selfadj (ii): if `ḡ < x1 < x2` and `x1`, `x2` are at distance more than
`γ̄²/(x1 − ḡ)` from every eigenvalue of G11, then A and G11 have the same number of eigenvalues
in `[x1, x2]`. For `x2 < x1 < −ḡ`… use negative arguments: the interval `[x1, x2]` with
`x1 < x2 < −ḡ` is handled by applying the statement to `−A`. The distance from `x` to `θ_i` is
bounded below by the distance to the enclosing interval.
"""
function selfadj_count(θ::Vector{EigInterval}, gbar::Real, γbar::Real, x1::Real, x2::Real)
    x1, x2 = Float64(x1), Float64(x2)
    x1 < x2 || return (0, false)
    if x2 < 0
        θn = [EigInterval(-J.hi, -J.lo, J.count) for J in θ]
        return selfadj_count(θn, gbar, γbar, -x2, -x1)
    end
    s = subd(x1, Float64(gbar))
    s > 0 || return (0, false)
    thr = divu(mulu(Float64(γbar), Float64(γbar)), s)
    cnt = 0
    for J in θ
        for x in (x1, x2)
            d = x < J.lo ? subd(J.lo, x) : x > J.hi ? subd(x, J.hi) : 0.0
            d > thr || return (0, false)
        end
        (J.lo > x1 && J.hi < x2) && (cnt += J.count)
    end
    return (cnt, true)
end

"""
    weyl_pairing(θ, ḡ, γ̄, n) -> NamedTuple

Corollary cor:weyl for Hermitian `A ∈ ℂ^{n×n}`: with `j₊ = #{θ_i > ḡ}`,
`|λ_i − θ_i| ≤ γ̄` for `i ≤ j₊` and `λ_{j₊+1} ≤ ḡ + γ̄` (eigenvalues in decreasing order), and
symmetrically at the bottom. Needs every interval of `θ` to lie on one side of `ḡ` (and of `−ḡ`);
the `i`-th largest θ lies in the interval reached by the cumulative counts from the top, which
gives `λ_i ∈ [lo − γ̄, hi + γ̄]` for that interval. Returns `top` and `bottom` lists of
`(i, lo, hi)` and the bounds `λ_{j₊+1} ≤ top_rest`, `λ_{n−j₋} ≥ bottom_rest`, or `ok = false`.
"""
function weyl_pairing(θ::Vector{EigInterval}, gbar::Real, γbar::Real, n::Integer)
    gbar, γbar = Float64(gbar), Float64(γbar)
    any(J -> J.lo <= gbar < J.hi || J.lo < -gbar <= J.hi, θ) &&
        return (; ok = false, top = [], bottom = [], jplus = 0, jminus = 0,
            top_rest = Inf, bottom_rest = -Inf)
    desc = sort(θ; by = J -> -J.hi)
    top = NamedTuple[]
    i = 0
    for J in desc
        J.lo > gbar || break
        for _ in 1:(J.count)
            i += 1
            push!(top, (; i, lo = subd(J.lo, γbar), hi = addu(J.hi, γbar)))
        end
    end
    jplus = i
    asc = sort(θ; by = J -> J.lo)
    bottom = NamedTuple[]
    i = 0
    for J in asc
        J.hi < -gbar || break
        for _ in 1:(J.count)
            i += 1
            push!(bottom, (; i = n + 1 - i, lo = subd(J.lo, γbar), hi = addu(J.hi, γbar)))
        end
    end
    jminus = i
    return (; ok = true, top, bottom, jplus, jminus,
        top_rest = addu(gbar, γbar), bottom_rest = subd(-gbar, γbar))
end
