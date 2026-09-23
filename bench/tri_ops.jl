# The sixteen operators of the numerics section, end to end on the triangular route.
#
# One row per operator, at the radius and frame the planner chooses. The method sees the operator
# only through matvecs: plan_split grids (R, ks, m), reports where the gate and thm:assembly can
# hold, and the certificate is then run at its feasible point. An operator for which the planner
# finds nothing feasible is reported as such, which is the method declining rather than failing.
# The frame and the far block are as in pseudospectra_run.jl,
#
#     choose_frame  ->  certify_split(op, Q; compression = :none)  ->  α, ḡ, ā, γ1, γ2,
#
# and inside the frame the compression is put in Schur form and cut once, `triangular_split`,
# with `certify_exterior_tri` doing the circle, the count and the isolation discs.
#
# Every certified floor is compared against dense `svdvals(A − zI)` where the size allows, at 24
# angles on each of three circles; the number of points at which the certified bound exceeds the
# dense value is reported and must be zero.
#
#   OPENBLAS_NUM_THREADS=8 julia --project=. -t 32 bench/tri_ops.jl [op ...]
#
# With no arguments the whole catalogue runs. Output:
#   ~/ResearchData/KrylovPseudospectra_Data/pseudospectra/tri_ops.<host>.{csv,jls}

using RigorousPseudospectra
using LinearAlgebra, SparseArrays, Random, Printf, Dates
import Serialization
const RP = RigorousPseudospectra
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))

BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "8")))

const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
mkpath(OUT)
const HOST = gethostname()
const NDENSE_EIG = 6000       # dense eigvals up to this n, Arnoldi Ritz values above it
const NDENSE_SVD = 2200       # dense svdvals for the comparison up to this n
const NTHETA = 8          # start coarse; check_circle bisects where the Lipschitz test fails
const MAXDEPTH = 9        # 8·2^9 = 4096 arcs, the reach the old 64 × 2^6 had

const ORDER = ["rdb200", "rdb800l", "rdb2048", "rdb5000", "grcar500", "convdiff_cheb200",
    "davies200", "kahan500", "airy200", "orrsom200", "landau400", "hatano2000", "qh1484",
    "nos3", "olm1000"]

"The leading eigenvalues, dense where the size allows and Ritz values above it."
function spectrum(o)
    if o.A !== nothing && o.n <= NDENSE_EIG
        return eigvals(Matrix(o.A)), "dense eigvals"
    end
    return RP._ritz_leading(o.matvec, o.n, Float64; L = 200), "Arnoldi Ritz values"
end

"""
The radius and the frame, from the planner and from nothing else.

`plan_split` sees the operator only through matvecs, and chooses `(R, ks, m)` from floating-point
estimates of the two conditions over a grid. That is what the method prescribes and what
`rem:predictor` describes, and it is the only radius a user could actually place: a radius read
off a dense eigenvalue decomposition would be an oracle no user has, and a certificate that fails
at one says nothing about the method.
"""
function plan(o, op; kw...)
    # no frame ladder is supplied: the planner grows the Lanczos from its budget and derives the
    # ladder from the length it reaches
    t = @elapsed (pl = RP.plan_split(op, o.matvec, o.adjoint; mmax = 12, nR = 6,
        amargin = 1.5, Leig = 200, kw...))
    return pl, t
end

"""
The points of the planner's trade-off front worth certifying: for each number of eigenvalues it
can put outside, the cheapest frame that does it, capped at `nmax` of them so that a wide front
does not turn into a day of runs.
"""
function front(pl; nmax::Integer = 6)
    isempty(pl.feasible) && return NamedTuple[]
    pts = NamedTuple[]
    for nz in sort(unique([r.nout for r in pl.feasible]); rev = true)
        sub = [r for r in pl.feasible if r.nout == nz]
        push!(pts, sort(sub; by = r -> (r.cost, r.dimQ))[1])
    end
    length(pts) <= nmax && return pts
    # keep the extremes and spread the rest over the front
    idx = unique(round.(Int, range(1, length(pts); length = nmax)))
    return pts[idx]
end

"One point of the front, certified: the frame, the split, the circle, and what it cost."
function certify_at(o, op, λ, pr, label, rows)
    R = pr.R
    @printf("\n--- %s  n = %d  %s: R = %.10g, ks = %d, %d outside predicted, %s arcs predicted\n",
        o.name, o.n, label, R, pr.ks, pr.nout,
        isfinite(pr.arcs) ? string(round(Int, pr.arcs)) : "-")
    flush(stdout)

    t_frame = @elapsed ((Q, finfo) = RP.two_sided_frame(o.matvec, o.adjoint, o.n, R;
        ks = pr.ks, Leig = 200, Lsv = min(o.n, 2pr.ks + 60), T = op.T))
    t_split = @elapsed (sc = RP.certify_split(op, Q; target = R, compression = :none))
    @printf("    frame: ks = %d, dim Q = %d, α = %.3e, ḡ = %.6g, ā = %.6g, γ1 = %.3e, γ2 = %.3e (%.1f s + %.1f s)\n",
        finfo.ks, finfo.dimQ, sc.α, sc.gbar, sc.abar, sc.γ1, sc.γ2, t_frame, t_split)
    gate = sc.γ1 * sc.γ2 < R * (R - sc.gbar)
    @printf("    the outer gate γ1γ2 < R(R − ḡ): %.6g against %.6g, %s\n",
        sc.γ1 * sc.γ2, R * (R - sc.gbar), gate)
    flush(stdout)

    t_tri = @elapsed ts = RP.triangular_split(sc.G, sc.M, R; α = sc.α, factor = 2.0)
    if ts === nothing
        println("    triangular_split declined: no admissible cut brings the trailing block below R")
        push!(rows, (; op = o.name, o.n, R, label, dimQ = finfo.dimQ,
            ks = pr.ks, nout_pred = pr.nout, arcs_pred = pr.arcs, cost_pred = pr.cost,
            arcs = -1, sc.α, sc.gbar, sc.abar, sc.γ1, sc.γ2, gate, declined = true,
            cut = -1, tri_γ1 = NaN, tri_γ2 = NaN, tri_gbar = NaN, nR2 = NaN,
            hyp_ok = false, region_ok = false, floor = 0.0, count = -1, ndiscs = 0,
            dense_count = count(z -> abs(z) > R, λ), nsig = 0, nbad = 0, worst = NaN,
            t_frame, t_split, t_tri = NaN, t_ext = NaN, host = HOST))
        return
    end
    @printf("    triangular split: cut k_EN = %d of %d, γ1 = %.3e, γ2 = %.3e, ḡ₂₂ = %.3e, ‖R2‖ = %.2e (%.1f s)\n",
        ts.c, ts.k, ts.γ1, ts.γ2, ts.gbar, ts.nR2, t_tri)
    flush(stdout)

    σ22lo(z) = max(RP.farshort_lo(z, sc.gbar), RP.farpower_lo(z, sc.gbars_power))
    t_ext = @elapsed ext = RP.certify_exterior_tri(ts, sc.gbar, sc.γ1, sc.γ2; R, σ22lo,
        nθ = NTHETA, maxdepth = MAXDEPTH)
    arcs_actual = ext.exterior.npoints
    dense_count = count(z -> abs(z) > R, λ)
    @printf("    cor:agree: hypothesis %s, region %s, floor %.4e, count %s (dense %d), %d discs, all closed %s (%.1f s)\n",
        ext.exterior.ok, ext.region_ok, ext.exterior.floor,
        ext.exterior.ok ? string(ext.count) : "none", dense_count, length(ext.discs), ext.ok,
        t_ext)
    @printf("    the cover: %d evaluations against %s predicted by the planner\n",
        arcs_actual, isfinite(pr.arcs) ? string(round(Int, pr.arcs)) : "-")
    flush(stdout)

    # the certified floor against dense truth, three circles, 24 angles each
    nsig = 0; nbad = 0; worst = Inf
    if o.A !== nothing && o.n <= NDENSE_SVD
        Ad = Matrix(o.A)
        floor_at(z) = begin
            s11 = RP.sigma_tri_lo(ts, z)
            s11 > 0 || return 0.0
            s22 = σ22lo(z)
            s22 > 0 || return 0.0
            f1, _ = RP.assembly_floor(s11, s22, sc.γ1, sc.γ2)
            f2, _ = RP.blockdom_floor(s11, s22, sc.γ1, sc.γ2)
            return max(f1, f2)
        end
        for ρ in (1.0, 1.15, 1.6), t in 0:23
            z = ρ * R * cis(2π * t / 24)
            b = floor_at(z)
            tru = minimum(svdvals(Ad - z * I))
            nsig += 1
            b <= tru * (1 + 1e-10) + 1e-300 || (nbad += 1;
                @warn "above the truth" o.name R z b tru)
            b > 0 && (worst = min(worst, b / tru))
        end
        @printf("    dense comparison: %d points, %d above the truth, smallest ratio %.4e\n",
            nsig, nbad, isfinite(worst) ? worst : NaN)
    end

    push!(rows, (; op = o.name, o.n, R, label, dimQ = finfo.dimQ,
        ks = pr.ks, nout_pred = pr.nout, arcs_pred = pr.arcs, cost_pred = pr.cost,
        arcs = arcs_actual, sc.α, sc.gbar, sc.abar, sc.γ1, sc.γ2, gate, declined = false,
        cut = ts.c, tri_γ1 = ts.γ1, tri_γ2 = ts.γ2, tri_gbar = ts.gbar, nR2 = ts.nR2,
        hyp_ok = ext.exterior.ok, region_ok = ext.exterior.region_ok,
        floor = ext.exterior.floor, count = ext.exterior.ok ? ext.count : -1,
        ndiscs = count(d -> isfinite(d.radius), ext.discs), dense_count, nsig, nbad,
        worst = isfinite(worst) ? worst : NaN, t_frame, t_split, t_tri, t_ext, host = HOST))
    return
end

function main(names)
    rows = NamedTuple[]
    for nm in names
        o = try
            nm == "free" ? ps_free_operator(100, 400; kmax = 150) : PS_CATALOGUE[nm]()
        catch err
            @warn "operator unavailable" nm err
            continue
        end
        println("\n===== ", o.name, "  n = ", o.n, "  ", Dates.format(now(), "HH:MM:SS"))
        flush(stdout)
        λ, src = spectrum(o)
        println("    spectrum from ", src)
        op = o.A === nothing ? RP.LinOp(o.matvec, o.adjoint, o.n; epsA = o.epsA) :
             RP.LinOp(o.A)
        pl, t = plan(o, op)
        @printf("planner: %d grid points, %d feasible, %d matvecs, %.1f s (Lanczos stopped: %s)\n",
            length(pl.rows), length(pl.feasible), pl.nmatvec, t, pl.params.Lgrew)
        flush(stdout)
        pts = front(pl)
        if isempty(pts)
            isempty(pl.predicted) ||
                @printf("  none feasible; the grid says dim Q ≈ %.0f would be needed at R = %.6g\n",
                    pl.predicted[1].dimQ_needed, pl.predicted[1].R)
        end
        # the front: one certificate per achievable number of eigenvalues outside
        for pr in pts
            try
                certify_at(o, op, λ, pr, "front, $(pr.nout) outside", rows)
            catch err
                @warn "certificate failed" o.name pr.nout err
            end
        end
        # windows: ask for a circle in an annulus rather than at a point, at a few widths
        a = sort(abs.(λ); rev = true)
        for (j, w) in ((2, 4), (5, 4), (10, 4))
            j + 1 <= length(a) || continue
            r = sqrt(a[j] * a[j + 1])
            plw, tw = plan(o, op; Rwindow = (r / w, r * w))
            plw.best === nothing && continue
            b = plw.best
            try
                certify_at(o, op, λ, b,
                    @sprintf("window [%.4g, %.4g], %d matvecs", r / w, r * w, plw.nmatvec), rows)
            catch err
                @warn "window certificate failed" o.name j err
            end
        end
    end
    isempty(rows) && return
    Serialization.serialize(joinpath(OUT, "tri_ops.$(HOST).jls"), rows)
    open(joinpath(OUT, "tri_ops.$(HOST).csv"), "w") do io
        ks = collect(keys(rows[1]))
        println(io, join(ks, ","))
        for r in rows
            println(io, join([string(getfield(r, k)) for k in ks], ","))
        end
    end
    @printf("\n%d rows; %d declined; %d dense comparisons, %d above the truth\n", length(rows),
        count(r -> r.declined, rows), sum(r -> r.nsig, rows), sum(r -> r.nbad, rows))
    println("wrote ", joinpath(OUT, "tri_ops.$(HOST).csv"))
end

main(isempty(ARGS) ? ORDER : ARGS)
