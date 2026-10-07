# The same sweep in ball arithmetic (BallArithmetic.jl, MMul4 products with directed rounding):
# ball vectors through the projector I − QQᵀ and through the stored operator. Compared with the
# Float64 sweep on the same columns and the same number of threads. Records the ratio
# ‖rad‖/‖mid‖ after each application of B or Bᵀ for the first 4 columns of the subset.
#
#   julia --project=bench/fpsweep -t 128 bench/fpsweep/run_ballsweep.jl OUT.csv NAME:ks:ncols ...
# ncols = 0 means all n columns (then the ball value is a certificate of ‖B_Q‖ ≥ ‖G22‖).

include(joinpath(@__DIR__, "FPSweep.jl"))
using .FPSweep
include(joinpath(@__DIR__, "operators.jl"))
using Printf, LinearAlgebra, SparseArrays

const RES = joinpath(DATA, "results"); mkpath(RES)
out = joinpath(RES, ARGS[1])
specs = ARGS[2:end]
PMAX = parse(Int, get(ENV, "PMAX", "4"))

newfile = !isfile(out)
io = open(out, "a")
newfile && println(io, "name,n,ks,dimQ,ncols,allcols,m,fp_partial,ball_partial,ratio_ball_fp," *
                       "fp_value,ball_value,t_fp_book,t_ball,nthreads,normA,normG22,kappa")
io2 = open(replace(out, ".csv" => "_radii.csv"), newfile ? "w" : "a")
newfile && println(io2, "name,n,ks,dimQ,kappa,column,application,sub,substep,rad_over_mid")

for spec in specs
    name, kss, ncs = split(spec, ":")
    ks = parse(Int, kss); nc = parse(Int, ncs)
    op0, source, sym = CATALOGUE[name]()
    n = FPSweep.nsize(op0)
    e = exponent(FPSweep.norm_estimate(op0)) + 1
    op = scaled(op0, 2.0^-e); sc = 2.0^e
    Q = ks == 0 ? zeros(n, 0) : (sym ? lanczos_frame(op, min(n - 1, 300); kmax = ks)[1] :
                                  gkl_frame(op, ks)[1])
    k = size(Q, 2)
    cols = nc == 0 ? (1:n) : round.(Int, range(1, n; length = nc))
    Amat = op isa DenseOp ? op.A : to_sparse(op)
    # warm-up both on two columns
    sweep(op, Q, 1; cols = cols[1:2]); ball_sweep(Amat, Q, 1; cols = cols[1:2])
    S = sweep(op, Q, PMAX; cols = cols)
    B = ball_sweep(Amat, Q, PMAX; cols = cols)
    base = n <= 6000 ? dense_baseline(Amat isa SparseMatrixCSC ? Amat : sparse(Amat), Q; symmetric = sym) :
           (; normA = NaN, normG22 = NaN, lbA = NaN, lbG = NaN)
    nA = max(base.normA, base.lbA) * sc; nG = max(base.normG22, base.lbG) * sc
    println("==== $name n=$n ks=$ks dimQ=$k cols=$(length(cols)) fp $(round(S.time, digits=2))s ball $(round(B.time, digits=2))s  κ=$(nA/nG)")
    for pp in 1:PMAX
        fpv = S.fp_tr[pp]; bv = B.S[pp]
        fpval = fpv^(1 / (4pp)) * sc; bval = B.vals[pp] * sc
        @printf("  m=%d  fp partial %.6e  ball partial %.6e  ratio %.4e  | values %.6e %.6e\n",
            2pp, fpv, bv, bv / fpv, fpval, bval)
        println(io, join(Any[name, n, ks, k, length(cols), nc == 0, 2pp, fpv, bv, bv / fpv, fpval,
            bval, S.time, B.time, B.nthreads, nA, nG, nA / nG], ","))
    end
    subs = ["P", "A", "P", "P", "At", "P"]
    for (j, r) in sort(collect(B.track))
        println("  column $j rad/mid after each application: ",
            join((@sprintf("%.2e", r[3i]) for i in 1:(2PMAX)), " "))
        println("     growth per sub-step (P, A, P, P, Aᵀ, P) in application 3-4: ",
            join((@sprintf("%.2e", r[q] / r[q - 1]) for q in 13:18), " "))
        for (q, x) in enumerate(r)
            println(io2, join(Any[name, n, ks, k, nA / nG, j, cld(q, 3), mod1(q, 3), subs[mod1(q, 6)], x], ","))
        end
    end
    flush(io); flush(io2); flush(stdout)
end
close(io); close(io2)
