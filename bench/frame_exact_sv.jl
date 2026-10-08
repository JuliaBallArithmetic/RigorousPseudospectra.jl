# Diagnostic for the numerics section: is ‖G22‖ far above σ_{2d+1}(A) because the Golub–Kahan–Lanczos singular
# vectors are inaccurate? Three frames at the same (R, ks):
#   gkl   : two_sided_frame as in experiment 2 (Arnoldi eigendirections, GKL with Lsv = 2ks+60)
#   gklfull: the same with a full-length GKL, Lsv = n
#   exactsv: the same Arnoldi eigendirections, exact dense singular vectors
#   dense : dense eigenvectors of modulus > R, exact dense singular vectors
# ‖G22‖ formed densely, against the floor σ_{2d+1}(A) and the bound σ_{ks+1}(A) of prop:frame.
using RigorousPseudospectra, LinearAlgebra, Printf
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))
BLAS.set_num_threads(16)
const HOST = readchomp(`hostname -s`)
out = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra", "reach",
    "frame_exact_sv.$(HOST).csv")
io = open(out, "w")
println(io, "operator,n,N,R,K,ks,frame,dimQ,normG22,floor,sigma_ks1")
g22(A, Q) = (Qo = Matrix(qr(Q).Q)[:, 1:size(Q, 2)]; P = I - Qo * Qo'; opnorm(P * A * P))
for name in (isempty(ARGS) ? ("davies200", "landau400", "olm1000", "qh1484") : Tuple(split(ARGS[1], ",")))
    o = ps_operator(name)
    A = Matrix(o.A)
    n = size(A, 1)
    T = eltype(A) <: Complex ? ComplexF64 : Float64
    E = eigen(A)
    ord = sortperm(abs.(E.values); rev = true)
    lam = abs.(E.values[ord])
    F = svd(A)
    s = F.S
    gaps = [N for N in 1:(n ÷ 4) if lam[N] > lam[N + 1] * (1 + 1e-9)]
    picks = unique(gaps[round.(Int, range(1, length(gaps); length = min(6, length(gaps))))])
    for N in picks
        R = sqrt(lam[N] * lam[N + 1])
        K = count(>=(R), s)
        for ks in unique(clamp.((cld(K, 2), K, 2K), 1, (n ÷ 2 - N) ÷ 2))
            ks >= 1 || continue
            Qg, _ = two_sided_frame(o.matvec, o.adjoint, n, R; ks, Leig = 200,
                Lsv = min(n, 2ks + 60), T)
            Qf, _ = two_sided_frame(o.matvec, o.adjoint, n, R; ks, Leig = 200, Lsv = n, T)
            Wl = first(eigendirections(o.matvec, n, R; L = 200, T))
            Qe = rrqr_frame(hcat(Wl, F.V[:, 1:ks], F.U[:, 1:ks]))
            Wd = E.vectors[:, ord[1:N]]
            Qd = rrqr_frame(hcat(Wd, F.V[:, 1:ks], F.U[:, 1:ks]))
            for (lab, Q) in (("gkl", Qg), ("gklfull", Qf), ("exactsv", Qe), ("dense", Qd))
                d = size(Q, 2)
                @printf(io, "%s,%d,%d,%.17g,%d,%d,%s,%d,%.6g,%.6g,%.6g\n", name, n, N, R, K, ks, lab, d,
                    g22(A, Q), 2d + 1 <= n ? s[2d + 1] : 0.0, s[ks + 1])
            end
            flush(io)
        end
    end
    println(name, " done")
end
close(io)
println("wrote ", out)
