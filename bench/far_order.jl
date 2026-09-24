# Does a higher order improve the far-block bound, and what does it cost?
#
# `certify_far_block` runs one sweep of order `m` over the canonical basis and then reads a bound
# out of it for every `t ≤ m`. The sweep costs `m` applications of `B = P⊥AP⊥` per column, so the
# cost is linear in `m`; what is not obvious is how much the bound improves, since the powers of a
# far block whose spectral radius is already near its norm decay not at all, and the improvement
# comes from the sweep's own rounding bound rather than from the mathematics of orders.
#
# For each operator: one frame at the radius of the fifth gap, then `certify_far_block` at a ladder
# of orders, recording ḡ, ā and the time, against a dense `opnorm` of the block where the size
# allows. The dense value is a floating-point reference, accurate to rounding and not certified;
# the certified ḡ must exceed it, and the ratio is what the sweep gives away.
#
#   OPENBLAS_NUM_THREADS=4 julia --project=. -t 16 bench/far_order.jl [op ...]
#
# Output: ~/ResearchData/KrylovPseudospectra_Data/pseudospectra/far_order.<host>.{csv,jls}

using RigorousPseudospectra
using LinearAlgebra, SparseArrays, Printf, Dates
import Serialization
const RP = RigorousPseudospectra
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))

BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "4")))

const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
mkpath(OUT)
const HOST = gethostname()
const ORDERS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
const NDENSE = 2200          # dense block and its opnorm up to this n
const KS = 60

const ORDER = ["rdb200", "rdb800l", "rdb2048", "grcar500", "convdiff_cheb200", "davies200",
    "kahan500", "airy200", "orrsom200", "landau400", "hatano2000", "nos3", "olm1000"]

function main(names)
    rows = NamedTuple[]
    for nm in names
        o = try
            PS_CATALOGUE[nm]()
        catch err
            @warn "operator unavailable" nm err
            continue
        end
        op = RP.LinOp(o.A)
        λ = o.n <= 6000 ? eigvals(Matrix(o.A)) : RP._ritz_leading(o.matvec, o.n, Float64; L = 200)
        a = sort(abs.(λ); rev = true)
        length(a) > 6 || continue
        R = sqrt(a[5] * a[6])
        Q, _ = RP.two_sided_frame(o.matvec, o.adjoint, o.n, R; ks = KS, Leig = 200,
            Lsv = min(o.n, 2KS + 60), T = op.T)
        P = Matrix(Q)
        fd = RP.frame_data(op, P; coupling = :auto)
        trueg = NaN
        if o.A !== nothing && o.n <= NDENSE
            A = Matrix(o.A)
            G22 = A - P * (P' * A) - (A * P) * P' + P * (P' * A * P) * P'
            trueg = opnorm(G22)
        end
        RP.certify_far_block(op, P, fd; order = 1)      # warm the compiler, not timed
        @printf("\n===== %s  n = %d, dim Q = %d, R = %.6g, dense ‖G22‖ = %s  %s\n",
            o.name, o.n, size(P, 2), R, isnan(trueg) ? "-" : @sprintf("%.6g", trueg),
            Dates.format(now(), "HH:MM:SS"))
        @printf("  %5s %12s %12s %10s %10s\n", "order", "ḡ", "ā", "ḡ/dense", "seconds")
        flush(stdout)
        for m in ORDERS
            m < o.n || continue
            t = @elapsed far = RP.certify_far_block(op, P, fd; order = m)
            @printf("  %5d %12.6g %12.6g %10s %10.3f\n", m, far.gbar, far.abar,
                isnan(trueg) ? "-" : @sprintf("%.4f", far.gbar / trueg), t)
            flush(stdout)
            push!(rows, (; op = o.name, o.n, dimQ = size(P, 2), R, order = m, gbar = far.gbar,
                abar = far.abar, dense = trueg, ratio = far.gbar / trueg, time = t, host = HOST))
        end
    end
    isempty(rows) && return
    Serialization.serialize(joinpath(OUT, "far_order.$(HOST).jls"), rows)
    open(joinpath(OUT, "far_order.$(HOST).csv"), "w") do io
        ks = collect(keys(rows[1]))
        println(io, join(ks, ","))
        for r in rows
            println(io, join([string(getfield(r, k)) for k in ks], ","))
        end
    end
    # the summary: how much of the gap between order 1 and the best is bought, and at what cost
    println("\noperator            dim Q   ḡ(1)/dense   ḡ(best)/dense   best order   time(1)   time(best)")
    for nm in unique([r.op for r in rows])
        rs = [r for r in rows if r.op == nm]
        r1 = rs[1]
        rb = sort(rs; by = r -> r.gbar)[1]
        @printf("%-18s %6d %12s %15s %12d %9.3f %10.3f\n", nm, r1.dimQ,
            isnan(r1.dense) ? "-" : @sprintf("%.3f", r1.ratio),
            isnan(rb.dense) ? "-" : @sprintf("%.3f", rb.ratio), rb.order, r1.time, rb.time)
    end
    println("\nwrote ", joinpath(OUT, "far_order.$(HOST).csv"))
end

main(isempty(ARGS) ? ORDER : ARGS)
