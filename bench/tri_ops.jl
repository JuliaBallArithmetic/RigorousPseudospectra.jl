# The sixteen operators of the numerics section, end to end on the triangular route.
#
# The same operators, the same two target counts and the same placement of R as
# `bench/pseudospectra_run.jl`, so the rows sit beside the earlier ones; what changes is the
# treatment of the compression. The frame and the far block are as before,
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
const TARGETS = (10, 5)
const NDENSE_EIG = 6000       # dense eigvals up to this n, Arnoldi Ritz values above it
const NDENSE_SVD = 2200       # dense svdvals for the comparison up to this n
const NTHETA = 64
const MAXDEPTH = 6

const ORDER = ["rdb200", "rdb800l", "rdb2048", "rdb5000", "grcar500", "convdiff_cheb200",
    "davies200", "kahan500", "airy200", "orrsom200", "landau400", "hatano2000", "qh1484",
    "nos3", "olm1000"]

"The count raised until the two moduli that bracket R separate, so that R is not placed inside a
conjugate pair."
function admissible_count(a::AbstractVector, target::Integer; tol = 1e-9)
    c = target
    while c < length(a) - 1 && abs(a[c] - a[c + 1]) <= tol * max(1.0, a[c])
        c += 1
    end
    return c
end

"The leading eigenvalues, dense where the size allows and Ritz values above it."
function spectrum(o)
    if o.A !== nothing && o.n <= NDENSE_EIG
        return eigvals(Matrix(o.A)), "dense eigvals"
    end
    return RP._ritz_leading(o.matvec, o.n, Float64; L = 200), "Arnoldi Ritz values"
end

function run_target(o, λ, target::Integer, rows)
    a = sort(abs.(λ); rev = true)
    length(a) > target + 1 || return
    c = admissible_count(a, target)
    R = sqrt(a[c] * a[c + 1])
    @printf("\n--- %s  n = %d  target %d -> count %d,  R = %.10g\n", o.name, o.n, target, c, R)
    flush(stdout)
    op = o.A === nothing ? RP.LinOp(o.matvec, o.adjoint, o.n; epsA = o.epsA) : RP.LinOp(o.A)

    t_frame = @elapsed ((Q, finfo) = RP.choose_frame(op, o.matvec, o.adjoint, R; kmax = o.kmax))
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
        push!(rows, (; op = o.name, o.n, target, count_used = c, R, dimQ = finfo.dimQ,
            ks = finfo.ks, sc.α, sc.gbar, sc.abar, sc.γ1, sc.γ2, gate, declined = true,
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
    dense_count = count(z -> abs(z) > R, λ)
    @printf("    cor:agree: hypothesis %s, region %s, floor %.4e, count %s (dense %d), %d discs, all closed %s (%.1f s)\n",
        ext.exterior.ok, ext.exterior.region_ok, ext.exterior.floor,
        ext.exterior.ok ? string(ext.count) : "none", dense_count, length(ext.discs), ext.ok,
        t_ext)
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

    push!(rows, (; op = o.name, o.n, target, count_used = c, R, dimQ = finfo.dimQ,
        ks = finfo.ks, sc.α, sc.gbar, sc.abar, sc.γ1, sc.γ2, gate, declined = false,
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
        for target in TARGETS
            try
                run_target(o, λ, target, rows)
            catch err
                @warn "run failed" o.name target err
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
