# Experiment 1 with the couplings: which frames could pass the hypothesis of the Main Theorem in exact
# (dense, floating-point, uncertified) arithmetic, for frames with right singular vectors only, left
# only, or both (prop:frame).
#
# For each operator and gap radius R = sqrt(|λ_N||λ_{N+1}|), the frame is the orthonormal factor of a
# rank-revealing QR of [W | V_k] (right), [W | U_k] (left) or [W | V_k | U_k] (both), W the dense
# eigenvectors of modulus > R and V_k, U_k the leading k dense right and left singular vectors. The
# hypothesis is tested with exact quantities: g = ‖G22‖ < R and
# min over 64 points of |z| = R of σ_min(G11 - z) * (R - g) > ‖G12‖ ‖G21‖. Reported, per side, the
# least k found by doubling from 1 up to kmax and bisecting (monotonicity in k is not guaranteed), and
# at that k: dim Q, ‖G22‖, ‖G12‖, ‖G21‖, min σ11, σ_1 σ_{k+1} and σ_{k+1}^2. kmax keeps dim Q <= n/2.
#
# Radii: every gap with N <= n/4 for n <= 1000; otherwise every gap with N <= 16 and NSAMPLE (default
# 48) more spread geometrically up to n/4.
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=8 julia --project=bench bench/ideal_sides.jl names
using LinearAlgebra, Printf
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "8")))

const HOST = readchomp(`hostname -s`)
const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra", "reach")
mkpath(OUT)
names = split(ARGS[1], ",")
NSAMPLE = parse(Int, get(ENV, "NSAMPLE", "48"))

function rrqr(M; droptol = 1e-8)
    F = qr(M, ColumnNorm())
    d = abs.(diag(F.R))
    r = count(>=(droptol * d[1]), d)
    return Matrix(F.Q)[:, 1:r]
end

function blocks(A, Q, R)
    n = size(A, 1)
    P = I - Q * Q'
    G11 = Q' * A * Q
    g22 = opnorm(P * A * P)
    g12 = opnorm(Q' * A * P)
    g21 = opnorm(P * A * Q)
    s11 = minimum(minimum(svdvals(G11 - R * cis(2π * t / 64) * I)) for t in 0:63)
    ok = g22 < R && s11 * (R - g22) > g12 * g21
    return (; d = size(Q, 2), g22, g12, g21, s11, ok)
end

for nm in names
    name = String(nm)
    A = Matrix(ps_operator(name).A)
    n = size(A, 1)
    E = eigen(A)
    ord = sortperm(abs.(E.values); rev = true)
    lam = abs.(E.values[ord])
    F = svd(A)
    s = F.S
    gapN = [N for N in 1:(n ÷ 4) if lam[N] > lam[N + 1] * (1 + 1e-9)]
    if n > 1000
        upper = [N for N in gapN if N > 16]
        pick = isempty(upper) ? Int[] : unique(upper[clamp.(round.(Int, exp.(range(0,
            log(length(upper)); length = NSAMPLE))), 1, length(upper))])
        gapN = sort(unique(vcat([N for N in gapN if N <= 16], pick)))
    end
    io = open(joinpath(OUT, "ideal_sides.$(name).$(HOST).csv"), "w")
    println(io, "operator,n,N,R,side,status,k,dimQ,normG22,g12,g21,min_sigma11,s1_sk1,sk1_sq,nevals")
    for N in gapN
        R = sqrt(lam[N] * lam[N + 1])
        W = E.vectors[:, ord[1:N]]
        for side in (:right, :left, :both)
            kmax = side === :both ? (n ÷ 2 - N) ÷ 2 : n ÷ 2 - N
            kmax >= 1 || continue
            frame(k) = side === :right ? hcat(W, F.V[:, 1:k]) :
                       side === :left ? hcat(W, F.U[:, 1:k]) : hcat(W, F.V[:, 1:k], F.U[:, 1:k])
            cache = Dict{Int, Any}()
            f(k) = get!(() -> blocks(A, rrqr(frame(k)), R), cache, k)
            lo, hi = 0, 1
            while !f(hi).ok && hi < kmax
                lo = hi
                hi = min(2hi, kmax)
            end
            status = f(hi).ok ? "found" : "none up to kmax"
            if f(hi).ok
                while hi - lo > 1
                    mid = (lo + hi) ÷ 2
                    f(mid).ok ? (hi = mid) : (lo = mid)
                end
            end
            b = f(hi)
            @printf(io, "%s,%d,%d,%.17g,%s,%s,%d,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n", name, n, N, R,
                side, status, hi, b.d, b.g22, b.g12, b.g21, b.s11, s[1] * s[hi + 1], s[hi + 1]^2,
                length(cache))
        end
        flush(io)
    end
    close(io)
    @printf("%s done\n", name)
end
