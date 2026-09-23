# The planning phase: floating point, nothing certified, the operator only through matvecs.
#
# Everything the certificate of §5 does is decided by four numbers, and three of the four are
# settled before any ball arithmetic is touched. The planner estimates them over a grid of
# (R, m, dim Q) and reports where the two conditions that decide the outcome can hold:
#
#     the gate         γ1γ2 < R·σ22(R)                        (prop:absorb, with D empty)
#     the assembly     σ11(R)·σ22(R) > amargin·γ1γ2           (thm:assembly on |z| = R)
#
# with σ22(R) the larger of `R − ḡ_1` and the eq:farpower expression at order m. Nothing here is a
# bound: `estimate_orders` is a stochastic trace estimate, σ11 is a dense SVD of the midpoint
# pencil, γ1 and γ2 are floating-point norms of computed blocks. The certified run afterwards is
# what decides; the planner only says where to start it, and reports the feasible region rather
# than a single point.
#
# Cost. One Arnoldi of length `Leig` and one Golub-Kahan-Lanczos of length `Lsv`, shared by the
# whole grid, and then per grid point `2·dim Q` applications of A and A* for the couplings and
# `mmax·nsamples` applications of `B = P⊥AP⊥` for the far block. The frames cost no matvec, being
# assembled from the shared pool, and each is exactly the frame
# `two_sided_frame(Amul, Aadj, n, R; ks, Leig, Lsv)` returns, so the plan transfers to the
# certified run column for column.

"""
    SplitPlan

Output of [`plan_split`](@ref). Two ways to ask it a question: give `Rs` to fix the radius outside
which everything is to be bounded, and the grid ranges over frames and orders at that radius alone;
or give `nmin` to fix how many eigenvalues are to be certified, and the radius is chosen. With
neither, the cheapest feasible point is taken. `rows` is the grid, one entry per `(R, ks, m)` with the estimated
`gbar[j]`, `σ22`, `γ1`, `γ2`, `σ11`, and the two conditions; `feasible` the sub-list on which both
hold; `best` the feasible row with the largest margin (fewest frame columns among ties), or
`nothing`. `predicted` records, for each `(R, m)` on which no frame on the grid is feasible, the
`dim Q` at which the margin would reach `amargin` if it keeps growing at the rate the grid shows.
`nmatvec` counts applications of `A`, `A*` and `B`; nothing in here is certified.
"""
struct SplitPlan
    ok::Bool
    rows::Vector{NamedTuple}
    feasible::Vector{NamedTuple}
    best::Union{Nothing, NamedTuple}
    predicted::Vector{NamedTuple}
    lambda::Vector{ComplexF64}
    Rs::Vector{Float64}
    params::NamedTuple
    nmatvec::Int
    time::NamedTuple
end

function Base.show(io::IO, p::SplitPlan)
    print(io, "SplitPlan(", length(p.rows), " grid points, ", length(p.feasible), " feasible")
    p.best === nothing || print(io, ", best R = ", p.best.R, ", m = ", p.best.m, ", dim Q = ",
        p.best.dimQ, ", margin = ", p.best.margin)
    print(io, ", ", p.nmatvec, " matvecs)")
end

"""
    candidate_radii(λ, nmax = 8; tol = 1e-9, jmax = 40) -> Vector{Float64}

Candidate radii: the geometric means between consecutive distinct moduli of the Ritz values, the
`nmax` with the largest relative gap `a_j/a_{j+1}` kept and returned in decreasing order. Taking the
widest gaps rather than the first few puts the candidates where `R` has room, which is what the two
conditions are sensitive to.
"""
function candidate_radii(λ::AbstractVector, nmax::Integer = 8; tol::Real = 1e-9,
        jmax::Integer = 40, ntop::Integer = 4, rmin::Real = 0.0)
    a = sort(abs.(λ); rev = true)
    cand = Tuple{Float64, Float64, Int}[]
    for j in 1:min(length(a) - 1, Int(jmax))
        a[j] > a[j + 1] * (1 + tol) || continue
        r = sqrt(a[j] * a[j + 1])
        # A gap below the floor the singular values allow is in the shadow of the far block: by
        # Proposition prop:frame every frame has ḡ ≥ σ_{2k+1}(A), so no radius there can ever
        # satisfy ḡ < R and a grid point spent on it is wasted. The spectrum is free to thin in
        # there and it does not help.
        r > Float64(rmin) || continue
        push!(cand, (a[j] / max(a[j + 1], floatmin(Float64)), r, j))
    end
    isempty(cand) && return Float64[]
    # above the floor there are usually few gaps, so every one of them is tried; `nmax` caps the
    # budget when the spectrum is thin everywhere, keeping the widest relative gaps and the ones
    # at the very top, where a count of a few eigenvalues sits
    length(cand) <= Int(nmax) + Int(ntop) &&
        return sort(unique([t[2] for t in cand]); rev = true)
    top = [t[2] for t in cand[1:min(Int(ntop), length(cand))]]
    wide = [t[2] for t in sort(cand; by = t -> -t[1])[1:min(Int(nmax), length(cand))]]
    return sort(unique(vcat(top, wide)); rev = true)
end

# `estimate_orders` returns v_t ≈ (tr H^t)^{1/2t} with H = B*B, which BOUNDS ‖B‖ itself and falls
# to it as t grows (eq:mincert takes the least over the orders); it is not ‖B^t‖^{1/t}. So the
# order-m estimate of the far block is ḡ_m = min_{t ≤ m} v_t and σ22(R) = R − ḡ_m, which is
# eq:farshort with the bound the sweep will actually return at order m. That is the only place the
# order enters the two conditions.
_gbar_est(g::AbstractVector{<:Real}, m::Integer) = minimum(Float64.(g[1:min(Int(m), length(g))]))
_sigma22_est(R::Real, g::AbstractVector{<:Real}, m::Integer) = Float64(R) - _gbar_est(g, m)

"""
    plan_split(op::LinOp, Amul, Aadj; Rs = nothing, mmax = 12, kss, Leig = 120, Lsv, nθ = 24,
               amargin = 1.5, nsamples = 4, nR = 6, T = Float64) -> SplitPlan
    plan_split(A::AbstractMatrix; kwargs...)

The planning phase. `Rs` are the candidate radii (by default the geometric means between the
moduli of the leading Ritz values, so the gaps of the spectrum choose them); `kss` the numbers of
singular directions per side; `mmax` the highest order of the far-block sweep to estimate, the
estimate at order `m` being `ḡ_m = min_{t ≤ m} v_t` with `v` from [`estimate_orders`](@ref). Returns the
whole grid and the feasible region; see [`SplitPlan`](@ref).
"""
function plan_split(op::LinOp, Amul, Aadj; Rs = nothing, mmax::Integer = 12,
        kss = nothing, Leig::Integer = 120, Lsv = nothing, nθ::Integer = 24,
        amargin::Real = 1.5, nsamples::Integer = 4, nR::Integer = 6, droptol::Real = 1e-8,
        η::Real = 0.5, objective::Symbol = :cost, nmin::Integer = 1,
        budget::Integer = 40_000, T = nothing)
    t0 = time()
    n = op.n
    TT = T === nothing ? op.T : T
    nmv = Ref(0)
    # --- the leading eigenvalues, one Arnoldi
    V, H = arnoldi(Amul, n, min(Leig, n - 1); T = TT)
    nmv[] += size(H, 2)
    Lh = size(H, 2)
    F = eigen(H[1:Lh, 1:Lh])
    λ = F.values[sortperm(abs.(F.values); rev = true)]
    t_arn = time() - t0

    # --- the singular directions, one Golub-Kahan-Lanczos, shared by every grid point
    #
    # How far it has to run is not a matter of taste. The length L fixes the deepest singular value
    # that can be seen, hence the floor σ_{2k+1} below which no radius is admissible, so a ladder
    # chosen in advance also decides which radii may be considered. Instead the run is doubled
    # until the floor drops below the smallest gap between Ritz moduli that is worth having, or
    # until `budget` applications of A and A* are spent, whichever comes first. What is reported
    # is how it stopped.
    t1 = time()
    gaps = Float64[]
    let a = sort(abs.(λ); rev = true)
        for j in 1:(length(a) - 1)
            a[j] > a[j + 1] * (1 + 1e-9) && push!(gaps, sqrt(a[j] * a[j + 1]))
        end
    end
    # how deep the singular values have to be seen: down to the radius the caller asked for, or,
    # when the caller asked for none, down to the smallest gap between Ritz moduli, since a radius
    # can only be placed in a gap
    target = Rs !== nothing ? minimum(Float64.(collect(Rs))) :
             isempty(gaps) ? 0.0 : minimum(gaps)
    local Vr, Ul, σ, L
    if kss !== nothing || Lsv !== nothing
        kmax0 = kss === nothing ? [16, 32, 64] : sort(collect(Int.(kss)))
        L = Lsv === nothing ? min(n, 2 * maximum(kmax0) + 40) : Int(Lsv)
        Vr, Ul, σ = gkl(Amul, Aadj, n, L; T = TT)
        nmv[] += 2L
        grew = "fixed"
    else
        L = min(n, 128)
        grew = "budget"
        while true
            Vr, Ul, σ = gkl(Amul, Aadj, n, L; T = TT)
            nmv[] += 2L
            sv0 = Float64[abs(x) for x in σ]
            (isempty(sv0) || last(sv0) < target) && (grew = "floor reached"; break)
            L >= n && (grew = "the whole space"; break)
            2L > Int(budget) - nmv[] && (grew = "budget spent"; break)
            L = min(n, 2L)
        end
    end
    kmax = kss === nothing ? unique(clamp.(round.(Int,
            exp.(range(log(8), log(max(div(size(Vr, 2) - 1, 2), 8)); length = 7))), 1,
            max(div(size(Vr, 2) - 1, 2), 1))) : sort(collect(Int.(kss)))
    ksmax = min(maximum(kmax), size(Vr, 2))
    # The floor, and the frame size each radius needs, both from the singular values the
    # Golub-Kahan-Lanczos has just returned, so neither costs a matvec.
    #
    # prop:frame gives ḡ ≥ σ_{2k+1}(A), so a radius below σ_{2k+1} is out of reach of a frame of k
    # columns. The floor that discards only what no frame on the grid can reach is therefore the
    # one at the largest frame, not the smallest; and for a radius above it, the smallest frame
    # worth trying is the first k with σ_{2k+1} < R. These σ are Lanczos estimates and the interior
    # ones are not accurate, which is why they are used to order the search and not to bound
    # anything.
    sv = Float64[abs(x) for x in σ]
    kneeded(R) = begin
        j = findfirst(j -> sv[j] < R, eachindex(sv))
        j === nothing ? typemax(Int) : max(cld(j - 1, 2), 1)
    end
    rfloor = isempty(sv) ? 0.0 : sv[min(2 * maximum(kmax) + 1, length(sv))]
    radii = Rs === nothing ? candidate_radii(λ, nR; rmin = rfloor) :
            Float64.(collect(Rs))
    isempty(radii) && (radii = [sqrt(abs(λ[1]) * abs(λ[end]))])
    t_gkl = time() - t1

    # --- the column pool and its images, computed once
    t2 = time()
    Wall = V[:, 1:Lh] * F.vectors
    TT <: Real && (Wall = hcat(real.(Wall), imag.(Wall)))
    pool = hcat(Matrix{TT}(Wall), Matrix{TT}(Vr[:, 1:ksmax]), Matrix{TT}(Ul[:, 1:ksmax]))
    neig = size(Wall, 2)
    t_pool = time() - t2

    # which pool columns belong to the eigendirections kept at radius R
    eigcols(R) = begin
        sel = findall(θ -> abs(θ) > R, F.values)
        TT <: Real ? vcat(sel, sel .+ Lh) : sel
    end

    t3 = time()
    rows = NamedTuple[]
    for R in radii, ks in kmax
        # the ladder starts where this radius first becomes reachable, so no grid point is spent
        # on a frame that cannot hold the far block below R
        ks = min(max(ks, kneeded(R)), ksmax)
        idx = vcat(eigcols(R), neig .+ (1:ks), neig + ksmax .+ (1:ks))
        idx = filter(i -> 1 <= i <= size(pool, 2), unique(idx))
        isempty(idx) && continue
        Q = rrqr_frame(pool[:, idx]; droptol)
        k = size(Q, 2)
        k == 0 && continue
        # A and A* are applied to the frame honestly. Reusing images of the shared pool would be
        # wrong here: a frame built from the Arnoldi basis is invariant under A up to the Arnoldi
        # residual, and that residual is exactly what γ1 and γ2 measure, so the shortcut reports
        # couplings at the level of the residual (measured 9e-13 against a true 1.5e+02 on nos3).
        X = _apply_block(op, Amul, Q, false)
        Xt = _apply_block(op, Aadj, Q, true)
        nmv[] += 2k
        G = Q' * X
        Mq = Q' * Q
        C = X - Q * G
        D = Xt - Q * (Q' * Xt)
        γ2 = opnorm(C)
        γ1 = opnorm(D)
        s11 = minimum(minimum(svdvals(G - R * cis(2π * t / nθ) * Mq)) for t in 0:(nθ - 1))
        g = estimate_orders(op, Q, Int(mmax); nsamples = Int(nsamples))
        nmv[] += Int(mmax) * Int(nsamples)
        for m in 1:Int(mmax)
            s22 = _sigma22_est(R, g, m)
            gate = γ1 * γ2 < R * s22
            marg = γ1 * γ2 > 0 ? s11 * s22 / (γ1 * γ2) : Inf
            # What the certificate will cost, predicted from what is already here. The floor of
            # thm:assembly sets the cover, since an arc is accepted when the bound exceeds its
            # length, so the number of arcs is about 2πR/(η·floor); each point costs one verified
            # SVD of the leading block, which grows like k_EN³ and is bounded here by dim Q³, the
            # cut not being known until the compression is in Schur form. The value is the number
            # of eigenvalues put outside, so the currency is work per eigenvalue.
            fl = s11 * s22 > γ1 * γ2 ?
                 (s11 * s22 - γ1 * γ2) / (s11 + s22 + γ1 + γ2) : 0.0
            nz = count(θ -> abs(θ) > R, λ)
            arcs = fl > 0 ? 2π * R / (Float64(η) * fl) : Inf
            work = arcs * float(k)^3
            push!(rows, (; R, ks, m, dimQ = k, nout = nz,
                gbar = g[1], gbar_m = _gbar_est(g, m),
                s22, γ1, γ2, s11, gate, margin = marg, floor = fl, arcs, work,
                cost = nz > 0 ? work / nz : Inf,
                assembly = gate && marg > Float64(amargin)))
        end
    end
    t_grid = time() - t3

    feas = [r for r in rows if r.assembly]
    # `objective = :cost` takes the point of least predicted work per eigenvalue certified, which
    # prices the cover as well as the frame; `:count` takes the most eigenvalues outside, then the
    # smallest frame, which is what a user who wants depth at any price would choose. Neither is
    # certified: both read estimates.
    # A per-eigenvalue cost is minimised by the smallest job, since the work is cubic in the frame,
    # so the bare ratio drifts to cheap and shallow. `nmin` is the constraint that fixes it: the
    # cheapest point that certifies at least that many eigenvalues. The front is reported whatever
    # the choice, and choosing from it is the caller's.
    elig = [r for r in feas if r.nout >= Int(nmin)]
    isempty(elig) && (elig = feas)
    best = if isempty(feas)
        nothing
    elseif objective === :count
        sort(elig; by = r -> (-r.nout, r.dimQ, -r.margin))[1]
    else
        sort(elig; by = r -> (r.cost, r.dimQ))[1]
    end

    # for each (R, m) with nothing feasible, where the grid says dim Q would have to be
    pred = NamedTuple[]
    if best === nothing
        for R in radii, m in 1:Int(mmax)
            sub = [r for r in rows if r.R == R && r.m == m && r.margin > 0 &&
                       isfinite(r.margin)]
            length(sub) >= 2 || continue
            sort!(sub; by = r -> r.dimQ)
            x = log.(Float64[r.dimQ for r in sub])
            y = log.(Float64[r.margin for r in sub])
            sx, sy = sum(x) / length(x), sum(y) / length(y)
            den = sum((x .- sx) .^ 2)
            den > 0 || continue
            p = sum((x .- sx) .* (y .- sy)) / den
            p > 0 || continue
            kneed = exp(sx + (log(Float64(amargin)) - sy) / p)
            push!(pred, (; R, m, slope = p, margin_at_max = last(sub).margin,
                dimQ_at_max = last(sub).dimQ, dimQ_needed = kneed,
                gate = any(r -> r.gate, sub)))
        end
        sort!(pred; by = r -> r.dimQ_needed)
    end
    return SplitPlan(best !== nothing, rows, feas, best, pred, ComplexF64.(λ), radii,
        (; Leig = min(Leig, n - 1), Lsv = L, droptol, nθ, amargin, nsamples, mmax, η,
            objective, nmin, budget, Lgrew = grew), nmv[],
        (; arnoldi = t_arn, gkl = t_gkl, pool = t_pool, grid = t_grid, total = time() - t0))
end

function plan_split(A::AbstractMatrix; kwargs...)
    op = LinOp(A)
    return plan_split(op, x -> A * x, x -> A' * x; kwargs...)
end

# A or A* applied to a block, through the stored matrix when there is one.
function _apply_block(op::LinOp, f, Xb::AbstractMatrix, adj::Bool)
    if op.stored !== nothing
        return Matrix(adj ? op.stored' * Xb : op.stored * Xb)
    end
    Y = similar(Xb)
    for j in axes(Xb, 2)
        Y[:, j] = f(Xb[:, j])
    end
    return Y
end

"""
    plan_table(p::SplitPlan; io = stdout, all = false)

Print the grid, the feasible region and, when there is none, the frame the grid predicts would be
needed. Every number is a floating-point estimate.
"""
function plan_table(p::SplitPlan; io = stdout, all::Bool = false)
    println(io, "plan: ", length(p.rows), " grid points, ", length(p.feasible), " feasible, ",
        p.nmatvec, " matvecs, ", round(p.time.total; digits = 2), " s")
    println(io, rpad("R", 12), rpad("ks", 5), rpad("m", 4), rpad("dim Q", 7), rpad("ḡ_1", 12),
        rpad("ḡ_m", 12), rpad("σ22(R)", 12), rpad("γ1γ2", 12), rpad("σ11(R)", 12),
        rpad("margin", 12),
        rpad("gate", 6), "assembly")
    shown = all ? p.rows : [r for r in p.rows if r.assembly || r.m == p.rows[1].m ||
                                r.m == maximum(x -> x.m, p.rows)]
    for r in shown
        println(io, rpad(round(r.R; sigdigits = 6), 12), rpad(r.ks, 5), rpad(r.m, 4),
            rpad(r.dimQ, 7), rpad(round(r.gbar; sigdigits = 4), 12),
            rpad(round(r.gbar_m; sigdigits = 4), 12), rpad(round(r.s22; sigdigits = 4), 12),
            rpad(round(r.γ1 * r.γ2; sigdigits = 4), 12),
            rpad(round(r.s11; sigdigits = 4), 12),
            rpad(round(r.margin; sigdigits = 4), 12), rpad(r.gate, 6), r.assembly)
    end
    # the trade-off itself: for each number of eigenvalues that can be put outside, the smallest
    # frame that does it. One chosen point hides this curve, and the curve is what a user chooses
    # from.
    if !isempty(p.feasible)
        println(io, "eigenvalues outside, against the smallest frame that certifies them:")
        for nz in sort(unique([r.nout for r in p.feasible]); rev = true)
            sub = [r for r in p.feasible if r.nout == nz]
            b = sort(sub; by = r -> (r.dimQ, -r.margin))[1]
            println(io, "   ", lpad(nz, 4), " outside   dim Q ", lpad(b.dimQ, 6),
                "   R = ", round(b.R; sigdigits = 6), "   m = ", b.m,
                "   arcs ", lpad(isfinite(b.arcs) ? string(round(Int, b.arcs)) : "-", 8),
                "   work/eig ", isfinite(b.cost) ? round(b.cost; sigdigits = 3) : Inf)
        end
    end
    if p.best !== nothing
        b = p.best
        println(io, "best feasible: R = ", b.R, ", m = ", b.m, ", ks = ", b.ks, ", dim Q = ",
            b.dimQ, ", ", b.nout, " outside, margin = ", round(b.margin; sigdigits = 4))
    elseif !isempty(p.predicted)
        println(io, "no feasible point on the grid; the grid extrapolates to")
        for q in p.predicted[1:min(6, end)]
            println(io, "  R = ", round(q.R; sigdigits = 6), ", m = ", q.m, ": margin ",
                round(q.margin_at_max; sigdigits = 4), " at dim Q = ", q.dimQ_at_max,
                ", growing like dim Q^", round(q.slope; digits = 2),
                ", so dim Q ≈ ", round(Int, q.dimQ_needed), " would be needed",
                q.gate ? "" : " (and the gate fails on every frame tried)")
        end
    else
        println(io, "no feasible point on the grid and no growing trend to extrapolate")
    end
    return nothing
end
