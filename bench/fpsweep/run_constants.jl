# Measured rounding errors against the constants of eq:roundA, eq:epsP, lem:onestep and the column
# bound ‖H^p e_j − w_p‖ ≤ ε_B E_j(g). The exact quantities are evaluated in BigFloat (256 bits)
# from the same Float64 data (A, Q, y), so the reference is exact up to 256-bit rounding.
#
#   julia --project=bench/fpsweep -t 16 bench/fpsweep/run_constants.jl OUT.csv NAME:ks ...

include(joinpath(@__DIR__, "FPSweep.jl"))
using .FPSweep
include(joinpath(@__DIR__, "operators.jl"))
using Printf, LinearAlgebra, SparseArrays, Random

setprecision(BigFloat, 256)
const RES = joinpath(DATA, "results"); mkpath(RES)
out = joinpath(RES, ARGS[1])
specs = ARGS[2:end]
NY = parse(Int, get(ENV, "NY", "20"))       # random test vectors
PMAX = parse(Int, get(ENV, "PMAX", "4"))

io = open(out, "w")
println(io, "name,n,ks,dimQ,quantity,max_ratio,mean_ratio,count,bound_value")

function record(io, name, n, ks, k, q, ratios, bnd)
    @printf("  %-22s max ratio %.3e  mean %.3e  (%d samples, bound %.3e)\n", q, maximum(ratios),
        sum(ratios) / length(ratios), length(ratios), bnd)
    println(io, join([name, n, ks, k, q, maximum(ratios), sum(ratios) / length(ratios),
        length(ratios), bnd], ","))
    flush(io)
end

for spec in specs
    name, kss = split(spec, ":")
    ks = parse(Int, kss)
    op0, source, sym = CATALOGUE[name]()
    n = FPSweep.nsize(op0)
    e = exponent(FPSweep.norm_estimate(op0)) + 1
    op = scaled(op0, 2.0^-e)
    Q = ks == 0 ? zeros(n, 0) : (sym ? lanczos_frame(op, min(n - 1, 300); kmax = ks)[1] :
                                  gkl_frame(op, ks)[1])
    k = size(Q, 2)
    println("==== $name n=$n ks=$ks dimQ=$k")
    fr = certify_frame(op, Q)
    A = to_sparse(op)
    Ab = SparseMatrixCSC{BigFloat, Int}(A); Abt = sparse(transpose(Ab))
    Qb = BigFloat.(Q)
    Gb = k > 0 ? inv(Qb' * Qb) : zeros(BigFloat, 0, 0)
    Pb(x) = k > 0 ? x - Qb * (Gb * (Qb' * x)) : x
    # true norms (Float64 dense, then used as a, g in the constants)
    base = dense_baseline(A, Q; symmetric = sym)
    a = max(base.normA, base.lbA); g = max(base.normG22, base.lbG)
    εB = FPSweep.epsB_up(fr, a, g)
    rng = MersenneTwister(99)
    Ys = [randn(rng, n) for _ in 1:NY]
    append!(Ys, [Float64.((1:n) .== j) for j in (1, n ÷ 2, n)])
    rA = Float64[]; rP = Float64[]; rB = Float64[]
    k1 = max(k, 1)
    Tk = zeros(1, k); Sb = zeros(1, n); buf = FPSweep.Buffers(1, n, k)
    for y in Ys
        yb = BigFloat.(y); ny = norm(yb)
        for adj in (false, true)
            fy = FPSweep.opvec(op, y, adj)
            ex = adj ? Abt * yb : Ab * yb
            push!(rA, Float64(norm(BigFloat.(fy) - ex) / (a * ny)) / fr.εA)
            yo = zeros(1, n)
            FPSweep.applyB!(yo, op, reshape(copy(y), 1, n), Q, buf, adj)
            exB = Pb(adj ? Abt * Pb(yb) : Ab * Pb(yb))
            push!(rB, Float64(norm(BigFloat.(vec(yo)) - exB) / ny) / εB)
        end
        if k > 0
            yo = zeros(1, n)
            FPSweep.proj!(yo, reshape(copy(y), 1, n), Q, Tk, Sb)
            push!(rP, Float64(norm(BigFloat.(vec(yo)) - Pb(yb)) / ny) / fr.εP)
        end
    end
    record(io, name, n, ks, k, "roundA/epsA", rA, fr.εA)
    k > 0 && record(io, name, n, ks, k, "proj/epsP", rP, fr.εP)
    record(io, name, n, ks, k, "onestep/epsB", rB, εB)
    # column chains: ‖H^p e_j − w_p‖ / (ε_B E_j(g)), p = 1..PMAX
    for p in 1:PMAX
        rc = Float64[]
        for j in unique([1, 2, n ÷ 3, n ÷ 2, n - 1, n])
            W = zeros(1, n); W[1, j] = 1.0; V = zeros(1, n)
            ex = BigFloat.(vec(W))
            E = 0.0
            for i in 0:(p - 1)
                wn = norm(W);
                FPSweep.applyB!(V, op, W, Q, buf, false)
                vn = norm(V)
                E = E * g^2 + (g * wn + vn)          # E_j(g) by the recursion of the paper
                FPSweep.applyB!(W, op, V, Q, buf, true)
                ex = Pb(Abt * Pb(Pb(Ab * Pb(ex))))
            end
            push!(rc, Float64(norm(BigFloat.(vec(W)) - ex)) / (εB * E))
        end
        record(io, name, n, ks, k, "column_p$(p)/epsB_Ej", rc, εB)
    end
end
close(io)
