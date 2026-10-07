# Thread scaling of the sweep: columns are independent, so the sweep is run with T tasks pinned
# to threads 1..T (Threads.@threads :static) for T = 1, 2, 4, ..., in one process started with
# -t 128. BLAS is single-threaded inside the sweep.
#
#   julia --project=bench/fpsweep -t 128 bench/fpsweep/run_threads.jl OUT.csv NAME ks p [T1,T2,...]

include(joinpath(@__DIR__, "FPSweep.jl"))
using .FPSweep
include(joinpath(@__DIR__, "operators.jl"))
using Printf, LinearAlgebra

const RES = joinpath(DATA, "results"); mkpath(RES)
out = joinpath(RES, ARGS[1])
name = ARGS[2]; ks = parse(Int, ARGS[3]); p = parse(Int, ARGS[4])
Ts = length(ARGS) >= 5 ? parse.(Int, split(ARGS[5], ",")) : [1, 2, 4, 8, 16, 32, 64, 128]
BLOCK = parse(Int, get(ENV, "BLOCK", "0"))

op0, source, sym = CATALOGUE[name]()
n = FPSweep.nsize(op0)
e = exponent(FPSweep.norm_estimate(op0)) + 1
op = scaled(op0, 2.0^-e)
Q = ks == 0 ? zeros(n, 0) : (sym ? lanczos_frame(op, 300; kmax = ks)[1] : gkl_frame(op, ks)[1])
k = size(Q, 2)
bs = BLOCK > 0 ? BLOCK : max(1, min(32, 2^22 ÷ n))
sweep(op, Q, p; cols = 1:(2bs), blocksize = bs)                     # compile
newfile = !isfile(out)
io = open(out, "a")
newfile && println(io, "name,n,ks,dimQ,m,nthreads,blocksize,bookkeeping,time,trfp")
for T in Ts, book in (false, true)
    S = sweep(op, Q, p; nthreads = T, blocksize = bs, bookkeeping = book)
    trfp = book ? (S.fp_tr[p]^(1 / (4p))) * 2.0^e : NaN
    @printf("%s n=%d dimQ=%d m=%d T=%3d book=%s  %.3f s  %s\n", name, n, k, 2p, T, book, S.time,
        book ? @sprintf("trfp=%.10e", trfp) : "")
    println(io, join(Any[name, n, ks, k, 2p, S.nthreads, bs, book, S.time, trfp], ","))
    flush(io); flush(stdout)
end
close(io)
