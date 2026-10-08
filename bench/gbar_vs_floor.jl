# Experiment 2 of the numerics: the certified far-block bound against the floor of prop:frame.
#
# For each operator and every gap radius R = sqrt(|λ_N||λ_{N+1}|), K = #{σ_i(A) >= R}; frames of
# def:Q (two_sided_frame) with ks in ceil.(K .* (1/2, 3/4, 1, 3/2)) singular vectors per side,
# skipped when N + 2ks > n/2. One certify_split with compression = :none and a far-block sweep of
# order 8: ḡ the least over orders 1..8, and ḡ at order 1. Against it: the floor σ_{2 dimQ + 1}(A)
# of prop:frame (experiment 1), and the dense ‖G22‖ = ‖P⊥AP⊥‖ for n <= 2048. ḡ < R is the first
# index of feasibility.
#
#   JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=8 julia --project=bench bench/gbar_vs_floor.jl names
using RigorousPseudospectra, LinearAlgebra, Printf
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "8")))

const HOST = readchomp(`hostname -s`)
const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra", "reach")
mkpath(OUT)
names = split(ARGS[1], ",")
NMAX = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : typemax(Int)
DENSE_MAX = 2048

for nm in names
    name = String(nm)
    o = ps_operator(name)
    op = LinOp(o.A)
    A = Matrix(o.A)
    n = size(A, 1)
    lam = sort(abs.(eigvals(A)); rev = true)
    s = svdvals(A)
    io = open(joinpath(OUT, "gbar_vs_floor.$(name).$(HOST).csv"), "w")
    println(io, "operator,n,N,R,K,ks,dimQ,gbar,gbar1,floor,normG22,gamma1,gamma2,feasible,t_frame,t_cert,error")
    for N in 1:min(NMAX, n - 1)
        lam[N] > lam[N + 1] * (1 + 1e-9) || continue
        R = sqrt(lam[N] * lam[N + 1])
        K = count(>=(R), s)
        for ks in unique(max.(1, ceil.(Int, K .* (0.5, 0.75, 1.0, 1.5))))
            N + 2ks > n ÷ 2 && continue
            row = try
                tf = @elapsed ((Q, _) = two_sided_frame(o.matvec, o.adjoint, n, R; ks, Leig = 200,
                    Lsv = min(n, 2ks + 60), T = op.T))
                tc = @elapsed (sc = certify_split(op, Q; order = 8, target = R, compression = :none))
                d = sc.k
                fl = 2d + 1 <= n ? s[2d + 1] : 0.0
                g22 = NaN
                if n <= DENSE_MAX
                    Qo = Matrix(qr(Q).Q)[:, 1:d]
                    P = I - Qo * Qo'
                    g22 = opnorm(P * A * P)
                end
                g1 = sc.far === nothing ? NaN : sc.far.gbar_trace[1]
                @sprintf("%s,%d,%d,%.17g,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.6g,%.6g,%s,%.2f,%.2f,",
                    name, n, N, R, K, ks, d, sc.gbar, g1, fl, g22, sc.γ1, sc.γ2, sc.gbar < R, tf, tc)
            catch err
                @sprintf("%s,%d,%d,%.17g,%d,%d,,,,,,,,,,,%s", name, n, N, R, K, ks,
                    replace(sprint(showerror, err)[1:min(end, 80)], "," => ";", "\n" => " "))
            end
            println(io, row)
            flush(io)
        end
    end
    close(io)
    @printf("%s done\n", name)
end
