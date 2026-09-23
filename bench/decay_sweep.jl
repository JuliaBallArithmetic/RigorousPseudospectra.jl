# How fast the far block falls, against the frame size: the figure the applicability claim rests on.
#
# Proposition prop:frame gives ḡ ≥ σ_{2k+1}(A), so the certified bound on the far block is, up to
# the frame's own slack, the singular value decay of the operator. The method needs ḡ < R, and R
# sits between |λ_c| and |λ_{c+1}| for a target of c eigenvalues outside, so the frame size an
# operator needs for a given c is where its curve crosses the line |λ_c|.
#
# For each operator and each frame size on a ladder: one two-sided frame at the radius of the
# ten-eigenvalue target, one certified split with no peeling, and the four numbers it returns.
# Nothing here uses the triangular route; ḡ, ā, γ1 and γ2 are properties of the frame alone.
#
#   OPENBLAS_NUM_THREADS=4 julia --project=. -t 16 bench/decay_sweep.jl [op ...]
#
# Output: ~/ResearchData/KrylovPseudospectra_Data/pseudospectra/decay_sweep.<host>.{csv,jls}

using RigorousPseudospectra
using LinearAlgebra, SparseArrays, Printf, Dates
import Serialization
const RP = RigorousPseudospectra
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))

BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "4")))

const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
mkpath(OUT)
const HOST = gethostname()
const KSS = (10, 20, 40, 80, 160, 320)
const NDENSE_EIG = 6000

const ORDER = ["rdb200", "rdb800l", "rdb2048", "rdb5000", "grcar500", "convdiff_cheb200",
    "davies200", "kahan500", "airy200", "orrsom200", "landau400", "hatano2000", "qh1484",
    "nos3", "olm1000"]

function spectrum(o)
    if o.A !== nothing && o.n <= NDENSE_EIG
        return eigvals(Matrix(o.A))
    end
    return RP._ritz_leading(o.matvec, o.n, Float64; L = 200)
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
        λ = spectrum(o)
        a = sort(abs.(λ); rev = true)
        length(a) > 11 || continue
        R = sqrt(a[10] * a[11])
        op = o.A === nothing ? RP.LinOp(o.matvec, o.adjoint, o.n; epsA = o.epsA) : RP.LinOp(o.A)
        println("\n===== ", o.name, "  n = ", o.n, "  R = ", R, "  ",
            Dates.format(now(), "HH:MM:SS"))
        flush(stdout)
        for ks in KSS
            2ks + 60 < o.n || continue
            t = @elapsed begin
                Q, fi = RP.two_sided_frame(o.matvec, o.adjoint, o.n, R; ks, Leig = 200,
                    Lsv = min(o.n, 2ks + 60), T = op.T)
                sc = RP.certify_split(op, Q; target = R, compression = :none)
            end
            @printf("  ks %4d  dim Q %5d  ḡ = %.6e  ā = %.6e  ḡ/ā = %.4e  γ1 = %.3e  γ2 = %.3e  (%.1f s)\n",
                ks, sc.k, sc.gbar, sc.abar, sc.gbar / sc.abar, sc.γ1, sc.γ2, t)
            flush(stdout)
            push!(rows, (; op = o.name, o.n, ks, dimQ = sc.k, R, gbar = sc.gbar, abar = sc.abar,
                ratio = sc.gbar / sc.abar, γ1 = sc.γ1, γ2 = sc.γ2, α = sc.α,
                lam5 = a[5], lam10 = a[10], lam1 = a[1], time = t, host = HOST))
        end
    end
    isempty(rows) && return
    Serialization.serialize(joinpath(OUT, "decay_sweep.$(HOST).jls"), rows)
    open(joinpath(OUT, "decay_sweep.$(HOST).csv"), "w") do io
        ks = collect(keys(rows[1]))
        println(io, join(ks, ","))
        for r in rows
            println(io, join([string(getfield(r, k)) for k in ks], ","))
        end
    end
    @printf("\n%d rows over %d operators\n", length(rows),
        length(unique([r.op for r in rows])))
    println("wrote ", joinpath(OUT, "decay_sweep.$(HOST).csv"))
end

main(isempty(ARGS) ? ORDER : ARGS)
