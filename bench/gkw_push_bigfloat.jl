# Push the BigFloat GKW "for free" count: sweep refinement depth (incrementally, reusing
# the refined basis) and watch ‖E‖, β₂ and the certified count. The free count is capped
# by max(grey radius ‖E‖+β₂, headroom ε·κ₂) on the isolating circle rⱼ=dⱼ/3; refinement
# drives ‖E‖ down toward the κ·u floor, so the count climbs until the FIXED headroom
# ε·κ₂ (~1e-30 here, independent of precision) binds — that ceiling is #{gaps > 3·ε·κ₂}.
#
# Run:  julia --project=bench bench/gkw_push_bigfloat.jl [prec_bits]   (default 256)

using RigorousPseudospectra
using LinearAlgebra
using BallArithmetic
using Serialization
using Printf

const DATA = "/home/isaia/Code/GKWExperiments.jl/data"
eps_K(::Type{T}; K = 256, N = 5000) where {T} =
    T(sum(sqrt(2n + 1) / (n - 0.5)^2 for n in 1:(N - 1))) * (T(2) / 3)^(K + 1)

function freecount(A, R, ε::T) where {T}
    fr = free_resolvent(A, R)
    λs = R.centres[sortperm(R.centres; by = abs, rev = true)]
    dnear(j) = minimum(abs(λs[j] - λs[k]) for k in eachindex(λs) if k != j)
    nsamp = 64
    free = 0
    for j in eachindex(λs)
        rj = dnear(j) / 3
        rj <= 0 && continue
        M = zero(T); ok = true
        for q in 0:(nsamp - 1)
            rb = resolvent_bound(fr, λs[j] + rj * cis(2 * T(π) * q / nsamp))
            isfinite(rb) || (ok = false; break)
            M = max(M, rb)
        end
        ok && ε * M < 1 && (free += 1)
    end
    return free, fr
end

function main(prec)
    setprecision(BigFloat, prec)
    A = Complex{BigFloat}.(BallArithmetic.mid(deserialize(joinpath(DATA, "ball_matrix_bf_K256.jls"))))
    n = size(A, 1)
    ε = eps_K(BigFloat)
    @printf("GKW K=256 BigFloat-%d : n=%d  ε=%.3e\n", prec, n, Float64(ε))
    @printf("%-8s | %-10s | %-10s | %-10s | %-8s | %-6s | %.0fs\n",
        "refine", "‖E‖₂", "β₂", "κ₂", "#single", "FREE", 0.0)
    W = Complex{BigFloat}.(Matrix(eigen(ComplexF64.(A)).vectors))   # Float64 candidate, promoted
    total = 0
    best = nothing
    for Δ in (4, 4, 4, 4, 4, 4, 4)                 # checkpoints 4,8,12,16,20,24,28
        t = @elapsed W = refine_basis(A, W; steps = Δ, mode = :entrywise)
        total += Δ
        R = miyajima_rump(A, W)                    # cheap κ₂ (default)
        free, fr = freecount(A, R, ε)
        nsingle = count(c -> length(c.idx) == 1, R.clusters)
        @printf("%-8d | %.3e | %.3e | %.3e | %-8d | %-6d | %.0fs\n",
            total, Float64(fr.coupling), Float64(fr.slack), Float64(fr.kappa),
            nsingle, free, t)
        best = (; R, fr)
    end
    serialize(joinpath(@__DIR__, "figures", "gkw_push_best.jls"),
        (; bits = prec, fr = best.fr,
            λs = best.R.centres[sortperm(best.R.centres; by = abs, rev = true)]))
    println("best fr cached → bench/figures/gkw_push_best.jls")
end

main(isempty(ARGS) ? 256 : parse(Int, ARGS[1]))
