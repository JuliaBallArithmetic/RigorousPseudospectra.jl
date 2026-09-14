# The two candidates of sec:candidate, compared on the gallery (bench/matrices.jl) and on a few
# random non-normal matrices: the Sylvester block diagonalisation (Bavely–Stewart, with merges when
# ‖X‖_F > xmax, for xmax = 1e4 and 1e8) and the iterated first-order Newton step. Both are certified by the same
# certify_compression. Reported: whether ‖R2‖ < 1 was certified, β_Λ, κ(W) (the upper bound
# ‖Y‖‖W‖/(1 − ‖R2‖)), number of blocks, largest component radius of the chosen family, Newton
# steps taken / step kept / stop reason / largest off-block Frobenius norm met along the way,
# Sylvester merges, and times (candidate, certification).
#
#   julia --project=. bench/candidates_compare.jl
using RigorousPseudospectra, LinearAlgebra, Random, Printf
include(joinpath(@__DIR__, "matrices.jl"))

function companion(mon)
    d = length(mon) - 1
    C = zeros(d, d)
    for i in 2:d
        C[i, i - 1] = 1
    end
    C[:, d] .= -mon[1:d]
    return C
end
function chebyshev_derivative_companion(n)
    Tm, T = [1.0], [0.0, 1.0]
    for _ in 2:n
        Tn = zeros(length(T) + 1)
        Tn[2:end] .+= 2 .* T
        Tn[1:length(Tm)] .-= Tm
        Tm, T = T, Tn
    end
    d = [k * T[k + 1] for k in 1:n]
    return companion(d ./ d[end])
end
graded_triu(n; seed = 1, decay = 0.5) = (rng = MersenneTwister(seed);
    triu(randn(rng, ComplexF64, n, n)) .* [2.0^(-decay * abs(i - j)) for i in 1:n, j in 1:n])

mats = [(e.name, e.A) for e in gallery()]
append!(mats, [
    ("jordan8+1e-10", jordan_block(8) + 1e-10 * randn(MersenneTwister(3), 8, 8)),
    ("chain40", graded_chain(40; s = 0.7)),
    ("chebyshev'12", chebyshev_derivative_companion(12)),
    ("chebyshev'20", chebyshev_derivative_companion(20)),
    ("triu20", graded_triu(20; seed = 1)),
    ("triu60", graded_triu(60; seed = 2, decay = 0.3)),
    ("randn50", randn(MersenneTwister(4), ComplexF64, 50, 50)),
    ("grcar80", grcar(80)),
])

# warm-up, so that compilation is not in the times
for method in (:sylvester, :newton)
    miyajima_rump(randn(MersenneTwister(0), ComplexF64, 6, 6); method)
    miyajima_rump(jordan_block(4); method)
end

@printf("%-15s %4s %-9s %3s %10s %10s %6s %10s %-28s %6s %8s %8s\n", "matrix", "k", "method", "ok",
    "β_Λ", "κ(W) ≤", "blocks", "max rad", "Newton it/best/stop/max off", "merges", "t_cand", "t_cert")
for (name, A) in mats
    for (method, xmax, label) in ((:sylvester, 1e4, "sylv 1e4"), (:sylvester, 1e8, "sylv 1e8"),
            (:newton, 1e8, "newton"))
        R = miyajima_rump(A; method, refine = 30, xmax)
        cc = R.cert
        info = cc.candidate.info
        nw = method === :newton ? @sprintf("%d/%d/%s/%.0e", info.iters, info.best, info.stop,
            info.maxoff) : "-"
        mg = method === :sylvester ? string(info.nmerge) : "-"
        maxr = R.ok ? maximum(C -> C.hull_radius, R.clusters) : Inf
        @printf("%-15s %4d %-9s %3s %10.2e %10.2e %6d %10.2e %-28s %6s %8.4f %8.4f\n", name,
            size(A, 1), label, R.ok ? "yes" : "no", cc.betaΛ, cc.kappa, length(cc.blocks), maxr,
            nw, mg, info.time, cc.time)
    end
end
