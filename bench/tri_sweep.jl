# What the triangular route's three parameters do, on the operators of the numerics section.
#
# `bench/tri_ops.jl` ran every row at `ncuts = 10`, `mmax = 8`, `factor = 2.0`: ten candidate cuts
# on the scan, at most eight powers for the trailing block, and the shallowest cut whose predicted
# floor is within a factor of two of the best. This sweeps the three.
#
# The frame is the expensive half and does not depend on any of them, so it is built once per
# (operator, target) and every parameter combination is run against the same `sc`; what each
# combination costs is one `triangular_split` and one `certify_exterior_tri`.
#
#   OPENBLAS_NUM_THREADS=4 julia --project=. -t 16 bench/tri_sweep.jl [op ...]
#
# Output: ~/ResearchData/KrylovPseudospectra_Data/pseudospectra/tri_sweep.<host>.{csv,jls}

using RigorousPseudospectra
using LinearAlgebra, SparseArrays, Random, Printf, Dates
import Serialization
const RP = RigorousPseudospectra
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))

BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "4")))

const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
mkpath(OUT)
const HOST = gethostname()
const TARGETS = (10, 5)
const NDENSE_EIG = 6000
const NTHETA = 64
const MAXDEPTH = 6

const NSHORT = (8, 24, 64)
const MMAX = (8, 16)
const FACTOR = (1.0, 2.0)

const ORDER = ["rdb200", "rdb800l", "rdb2048", "rdb5000", "grcar500", "convdiff_cheb200",
    "davies200", "kahan500", "airy200", "orrsom200", "landau400", "hatano2000", "qh1484",
    "nos3", "olm1000"]

function admissible_count(a::AbstractVector, target::Integer; tol = 1e-9)
    c = target
    while c < length(a) - 1 && abs(a[c] - a[c + 1]) <= tol * max(1.0, a[c])
        c += 1
    end
    return c
end

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
    dense_count = count(z -> abs(z) > R, λ)
    op = o.A === nothing ? RP.LinOp(o.matvec, o.adjoint, o.n; epsA = o.epsA) : RP.LinOp(o.A)
    t_frame = @elapsed ((Q, finfo) = RP.choose_frame(op, o.matvec, o.adjoint, R; kmax = o.kmax))
    t_split = @elapsed (sc = RP.certify_split(op, Q; target = R, compression = :none))
    @printf("\n--- %s  n = %d  target %d -> count %d,  R = %.8g,  dim Q = %d, ḡ = %.6g, γ1 = %.3e, γ2 = %.3e (%.1f s)\n",
        o.name, o.n, target, c, R, finfo.dimQ, sc.gbar, sc.γ1, sc.γ2, t_frame + t_split)
    flush(stdout)
    σ22lo(z) = max(RP.farshort_lo(z, sc.gbar), RP.farpower_lo(z, sc.gbars_power))
    for nshort in NSHORT, mmax in MMAX, factor in FACTOR
        t = @elapsed ts = RP.triangular_split(sc.G, sc.M, R; α = sc.α, factor, nshort, mmax)
        if ts === nothing
            @printf("    nshort %3d mmax %2d factor %.1f : declined (%.1f s)\n", nshort, mmax,
                factor, t)
            push!(rows, (; op = o.name, o.n, target, R, dimQ = finfo.dimQ, sc.gbar, sc.γ1,
                sc.γ2, nshort, mmax, factor, declined = true, cut = -1, tri_γ1 = NaN,
                tri_gbar = NaN, hyp_ok = false, floor = 0.0, count = -1, ndiscs = 0,
                dense_count, t, host = HOST))
            continue
        end
        te = @elapsed ext = RP.certify_exterior_tri(ts, sc.gbar, sc.γ1, sc.γ2; R, σ22lo,
            nθ = NTHETA, maxdepth = MAXDEPTH)
        @printf("    nshort %3d mmax %2d factor %.1f : cut %5d of %d, γ1 %.3e, ḡ₂₂ %.3e, hyp %s, count %s (dense %d), %d discs (%.1f + %.1f s)\n",
            nshort, mmax, factor, ts.c, ts.k, ts.γ1, ts.gbar, ext.exterior.ok,
            ext.exterior.ok ? string(ext.count) : "-", dense_count, length(ext.discs), t, te)
        flush(stdout)
        push!(rows, (; op = o.name, o.n, target, R, dimQ = finfo.dimQ, sc.gbar, sc.γ1, sc.γ2,
            nshort, mmax, factor, declined = false, cut = ts.c, tri_γ1 = ts.γ1,
            tri_gbar = ts.gbar, hyp_ok = ext.exterior.ok, floor = ext.exterior.floor,
            count = ext.exterior.ok ? ext.count : -1,
            ndiscs = count(d -> isfinite(d.radius), ext.discs), dense_count, t = t + te,
            host = HOST))
    end
    return
end

function main(names)
    rows = NamedTuple[]
    for nm in names
        o = try
            PS_CATALOGUE[nm]()
        catch err
            @warn "operator unavailable" nm err
            continue
        end
        println("\n===== ", o.name, "  n = ", o.n, "  ", Dates.format(now(), "HH:MM:SS"))
        flush(stdout)
        λ, _ = spectrum(o)
        for target in TARGETS
            try
                run_target(o, λ, target, rows)
            catch err
                @warn "run failed" o.name target err
            end
        end
    end
    isempty(rows) && return
    Serialization.serialize(joinpath(OUT, "tri_sweep.$(HOST).jls"), rows)
    open(joinpath(OUT, "tri_sweep.$(HOST).csv"), "w") do io
        ks = collect(keys(rows[1]))
        println(io, join(ks, ","))
        for r in rows
            println(io, join([string(getfield(r, k)) for k in ks], ","))
        end
    end
    # how many (operator, target) pairs close for at least one parameter choice
    pairs = unique([(r.op, r.target) for r in rows])
    closed = [p for p in pairs if any(r -> (r.op, r.target) == p && r.hyp_ok, rows)]
    @printf("\n%d rows over %d (operator, target) pairs; %d pairs close for at least one setting\n",
        length(rows), length(pairs), length(closed))
    for p in closed
        best = [r for r in rows if (r.op, r.target) == p && r.hyp_ok]
        b = sort(best; by = r -> -r.floor)[1]
        @printf("   %-18s target %2d : best floor %.4e at nshort %d, mmax %d, factor %.1f, cut %d\n",
            p[1], p[2], b.floor, b.nshort, b.mmax, b.factor, b.cut)
    end
    println("wrote ", joinpath(OUT, "tri_sweep.$(HOST).csv"))
end

main(isempty(ARGS) ? ORDER : ARGS)
