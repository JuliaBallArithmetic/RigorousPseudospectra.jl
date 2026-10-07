# Certified gbar, gamma1, gamma2 at every frame size k=1,...,256 (capped at n-2) for the
# gallery used in the KrylovPseudospectra paper's decay figure. gbar is from sweep+fp_checks
# (Section sec:fpsweep of the paper, order m=8); gamma1, gamma2 from certify_frame's Gram
# matrices (Section sec:coupling). Output: operator,k,gbar,g1,g2,abar,R CSV.
include(joinpath(homedir(), "Code", "RigorousPseudospectra.jl", "bench", "fpsweep", "FPSweep.jl"))
using .FPSweep
include(joinpath(homedir(), "Code", "RigorousPseudospectra.jl", "bench", "fpsweep", "operators.jl"))
using SparseArrays, LinearAlgebra, Printf

names14 = ["rdb200","rdb800l","rdb2048","nos3","olm1000","qh1484","grcar500","kahan500",
           "davies200","airy200","orrsom200","landau400","hatano2000","convdiff_cheb200"]
rfracs = Dict("rdb200"=>10,"rdb800l"=>10,"rdb2048"=>10,"nos3"=>8,"olm1000"=>10,"qh1484"=>10,
              "grcar500"=>10,"kahan500"=>10,"davies200"=>8,"airy200"=>8,"orrsom200"=>8,
              "landau400"=>10,"hatano2000"=>10,"convdiff_cheb200"=>8)

PMAX = 4   # m = 8
KMAX = 256

outpath = joinpath(homedir(), "scratch", "krylov_matrices", "decay_sweep_certified.csv")
csv = open(outpath, "w")
println(csv, "operator,k,gbar,g1,g2,abar,R")

for name in names14
    t0 = time()
    op0, source, sym = CATALOGUE[name]()
    n = FPSweep.nsize(op0)
    e = exponent(FPSweep.norm_estimate(op0)) + 1
    op = scaled(op0, 2.0^-e); sc = 2.0^e
    Asp = n <= 300_000 ? to_sparse(op) : nothing

    base0 = dense_baseline(Asp, zeros(n,0); symmetric=sym)
    abar = max(base0.normA, base0.lbA) * sc
    Amat = Matrix(Asp) .* sc
    eigsA = eigen(ComplexF64.(Amat)).values
    mags = sort(abs.(eigsA), rev=true)
    R = mags[min(rfracs[name], length(mags))]

    kmax_eff = min(KMAX, n - 2)
    for k in 1:kmax_eff
        local Q
        if sym
            Q, info = lanczos_frame(op, min(n - 1, max(k+20, 300)); kmax = k)
        else
            Q, info = gkl_frame(op, k)
        end
        size(Q,2) == 0 && continue
        fr = certify_frame(op, Q)
        S = sweep(op, Q, PMAX)
        c = FPSweep.fp_checks(S, fr, PMAX)
        gcert = (c.gF <= c.gI || isnan(c.gI) ? c.gF : c.gI) * sc
        g1cert = (fr.γ1[1] + fr.γ1[2]) * sc
        g2cert = (fr.γ2[1] + fr.γ2[2]) * sc
        println(csv, name, ",", k, ",", gcert, ",", g1cert, ",", g2cert, ",", abar, ",", R)
    end
    flush(csv)
    println(name, " done in ", round(time()-t0, digits=1), "s  (n=", n, ", R=", round(R,sigdigits=6), ")")
end
close(csv)
println("all done")
