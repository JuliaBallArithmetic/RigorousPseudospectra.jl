
# Code cut from RigorousPseudospectra, 2026-09-26

Kept for review. Reachability was established twice, by hand and by a fixpoint over
mention-edges from the bench scripts and the test suite.

## planner.jl:392-453  reach_table, reach_report

Never called: reach_figure.jl builds the reach figure by deserialising decay_sweep output and calls no package function. reach_table was rewritten on 2026-09-25 to read the planner grid; nothing ever consumed it.

```julia
# ------------------------------------------------------------------------------------------------
# What is reachable, before anything is certified
# ------------------------------------------------------------------------------------------------

"""
    reach_table(pl::SplitPlan) -> Vector{NamedTuple}
    reach_table(op, Amul, Aadj; kwargs...) -> Vector{NamedTuple}

Which radii a frame can reach, and at what size, read before any certificate is attempted.

This is a reading of [`plan_split`](@ref)'s own grid and computes nothing new: that grid already
carries, for every `(R, ks)`, the frame it assembles from the shared pool, its `dim Q`, and the
estimate of `‖G22‖` from the subspace iteration of [`estimate_orders`](@ref). The certificate has
to put that quantity below the radius, so the table reports the least frame on the grid whose
estimate clears each radius.

Nothing here is a bound. The estimate is a subspace iteration, measured a few per cent low; the
certificate afterwards is what decides. A radius no frame on the grid clears is reported with
`ks = -1`, and `plan_split`'s `predicted` field extrapolates the `dim Q` it would take.
"""
function reach_table(pl::SplitPlan)
    out = NamedTuple[]
    for R in pl.Rs
        sub = sort([r for r in pl.rows if r.R == R]; by = r -> r.dimQ)
        isempty(sub) && continue
        i = findfirst(r -> r.gbar < R, sub)
        nz = sub[1].nout
        push!(out, (; R, nout = nz,
            ks = i === nothing ? -1 : sub[i].ks,
            dimQ = i === nothing ? -1 : sub[i].dimQ,
            ghat = i === nothing ? minimum(r.gbar for r in sub) : sub[i].gbar,
            gate = i === nothing ? false : sub[i].gate,
            feasible = any(r -> r.assembly, sub),
            nmatvec = pl.nmatvec))
    end
    return sort(out; by = r -> -r.R)
end

function reach_table(op::LinOp, Amul, Aadj; kwargs...)
    return reach_table(plan_split(op, Amul, Aadj; kwargs...))
end

reach_table(A::AbstractMatrix; kwargs...) =
    reach_table(LinOp(A), x -> A * x, x -> A' * x; kwargs...)

"""
    reach_report(rows; io = stdout)

Print [`reach_table`](@ref): radius, eigenvalues outside it, the least frame that can reach it.
"""
function reach_report(rows; io = stdout)
    isempty(rows) && (println(io, "no radius: the Ritz moduli show no gap"); return nothing)
    println(io, "  radius        outside     ks    dim Q      ĝ     gate   feasible")
    for r in rows
        println(io, rpad(round(r.R; sigdigits = 6), 14), lpad(r.nout, 7), lpad(r.ks, 7),
            lpad(r.dimQ, 9), lpad(round(r.ghat; sigdigits = 4), 9),
            lpad(r.gate ? "yes" : "no", 7), lpad(r.feasible ? "yes" : "no", 11))
    end
    println(io, "  (", rows[1].nmatvec,
        " matvecs; ĝ is a subspace-iteration estimate of ‖G22‖, not a bound)")
    return nothing
end
```

## radius_sweep.jl:44-110  radius_sweep, radius_table, plan_radii

Never called. The planner supplanted the radius sweep; sigma11_estimate, the one live routine in the file, is called from triangular.jl and moves to the oracle.

```julia
"""
    radius_sweep(G, M, gbar, γ1, γ2; Rs, nθ = 64, η = 0.5, loss = 1.0) -> Vector{NamedTuple}

The sweep over the radii `Rs` at a fixed frame, with `gbar ≥ ‖G₂₂‖` and the two couplings of that
frame. `loss` scales the estimated floor to the route that will certify it (1 for the exact SVD of
the pencil, about 1/3 for the triangular route on the compressions measured so far). Each row
carries the count outside, the least predicted floor, the number of angles at which the hypothesis
is predicted to fail, and `arcs`, the predicted size of a cover.
"""
function radius_sweep(G, M, gbar::Real, γ1::Real, γ2::Real; Rs::AbstractVector,
        nθ::Integer = 64, η::Real = 0.5, loss::Real = 1.0, iters::Integer = 12)
    Gm = ComplexF64.(Matrix(mid(_ball(G))))
    Mm = ComplexF64.(Matrix(mid(_ball(M))))
    T = Matrix(schur(Mm \ Gm).T)
    λ = diag(T)
    out = NamedTuple[]
    for R in Rs
        R = Float64(R)
        s22 = R - Float64(gbar)
        floors = Float64[]
        nfail = 0
        for t in 0:(nθ - 1)
            z = R * cis(2π * t / nθ)
            s11 = sigma11_estimate(T, z; iters)
            f = (s22 > 0 && s11 * s22 > γ1 * γ2) ?
                Float64(loss) * (s11 * s22 - γ1 * γ2) / (s11 + s22 + γ1 + γ2) : 0.0
            f <= 0 && (nfail += 1)
            push!(floors, f)
        end
        arcs = nfail > 0 ? Inf :
               sum(2π * R / (Float64(η) * nθ * f) for f in floors)
        push!(out, (; R, count = count(z -> abs(z) > R, λ), minfloor = minimum(floors),
            nfail, arcs, s22))
    end
    return out
end

"""
    radius_table(rows; io = stdout)

Print the sweep: radius, count outside, least predicted floor, failing angles, predicted arcs.
"""
function radius_table(rows; io = stdout)
    println(io, "        R     count    min floor   fail   predicted arcs")
    for r in rows
        println(io, rpad(string(round(r.R; sigdigits = 5)), 11), lpad(string(r.count), 6),
            lpad(string(round(r.minfloor; sigdigits = 4)), 14), lpad(string(r.nfail), 7),
            lpad(isfinite(r.arcs) ? string(round(Int, r.arcs)) : "-", 10))
    end
    return nothing
end

"""
    plan_radii(sc; Rs = nothing, nθ = 64, η = 0.5, loss = 1/3, nR = 24) -> Vector{NamedTuple}

The sweep of [`radius_sweep`](@ref) straight from a certified split: the frame is fixed, so `ḡ`,
`γ₁` and `γ₂` are those of `sc` and no application of the operator is needed. With `Rs` omitted the
radii are geometric from just above `sc.gbar` to `‖A‖`. Nothing here is certified; it is what a
caller reads before choosing the radius to certify at.
"""
function plan_radii(sc; Rs = nothing, nθ::Integer = 64, η::Real = 0.5, loss::Real = 1 / 3,
        nR::Integer = 24, iters::Integer = 12)
    lo = 1.001 * sc.gbar
    hi = isfinite(sc.abar) && sc.abar > lo ? sc.abar : 1.0
    rs = Rs === nothing ? exp.(range(log(lo), log(hi); length = nR)) : Rs
    return radius_sweep(sc.G, sc.M, sc.gbar, sc.γ1, sc.γ2; Rs = rs, nθ, η, loss, iters)
end
```

## radius_sweep.jl:1-16  file header

Describes only the dead sweep.

```julia
# How far in can the circle be pushed, and what does the cover cost there?
#
# With the frame fixed, the certified numbers ḡ, γ₁, γ₂ are fixed with it: the radius enters the
# floor of Theorem thm:assembly only through σ₂₂(z) ≥ |z| − ḡ and through σ₁₁ on the circle. So a
# sweep in R costs no application of the operator: after one Schur factorisation of the
# compression, σ₁₁ at any z is estimated by inverse iteration on the triangular factor, two
# triangular solves per step, O(k²) each.
#
# For each radius the sweep reports
#
#   * the count outside, #{|λ̂| > R} from the Schur diagonal;
#   * the predicted floor at nθ angles, (σ̂₁₁σ̂₂₂ − γ₁γ₂)/(σ̂₁₁ + σ̂₂₂ + γ₁ + γ₂);
#   * the angles where the hypothesis is predicted to fail, which no refinement can repair;
#   * the predicted number of arcs a cover needs, (R/η)·Σ Δθ/floor(θ), which is the cost.
#
# Nothing here is certified: it chooses the radius, and the cover afterwards is the certificate.
```
