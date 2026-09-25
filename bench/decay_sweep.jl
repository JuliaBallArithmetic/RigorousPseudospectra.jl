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
# Beside each certified ḡ the run records the oracle floor σ_{2k+1}(A), from a dense SVD where the
# size allows. That floor is what no frame of k columns can go below whatever the sweep does, so
# the pair says two different things: the floor is whether a radius is reachable at all, and the
# gap between ḡ and the floor is what the certificate gives away. On olm1000 the floor first
# crosses the radius at index 394, so k ≥ 197 is needed and no order helps below that; at ks = 400
# the certificate goes through with ḡ = 8798 against a true far block of 8659.
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
const KSS = (10, 20, 40, 80, 160, 320, 480)
const NDENSE_EIG = 6000
const NDENSE_SVD = 3000

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
        # the oracle floor, σ_j(A), dense where the size allows
        sv = (o.A !== nothing && o.n <= NDENSE_SVD) ? svdvals(Matrix(o.A)) : Float64[]
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
            jf0 = 2 * sc.k + 1
            fo = (!isempty(sv) && jf0 <= length(sv)) ? sv[jf0] : NaN
            @printf("  ks %4d  dim Q %5d  ḡ = %.6e  floor σ_{2k+1} = %.6e  ḡ/floor = %.4g  ā = %.6e  order %3d  (%.1f s)\n",
                ks, sc.k, sc.gbar, fo, sc.gbar / fo, sc.abar,
                sc.far === nothing ? -1 : sc.far.m, t)
            flush(stdout)
            jf = 2 * sc.k + 1
            floor_oracle = (!isempty(sv) && jf <= length(sv)) ? sv[jf] : NaN
            push!(rows, (; op = o.name, o.n, ks, dimQ = sc.k, R, gbar = sc.gbar, abar = sc.abar,
                ratio = sc.gbar / sc.abar, floor_oracle, γ1 = sc.γ1, γ2 = sc.γ2, α = sc.α,
                lam1 = a[1], lam2 = a[2], lam5 = a[5], lam10 = a[10],
                lam20 = length(a) >= 20 ? a[20] : NaN,
                order = sc.far === nothing ? -1 : sc.far.m, time = t, host = HOST))
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
