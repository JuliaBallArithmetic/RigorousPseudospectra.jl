# Run the current VBD pipeline (Schur → Newton–Rump → block-diagonal → within-block
# orthogonalization → Miyajima certificate → block resolvent floor) on the gallery, and
# check the certified floor against the true σ_min(A−zI). Float64 (no BigFloat GC issue).
using RigorousPseudospectra, LinearAlgebra, BallArithmetic, Printf
include(joinpath(@__DIR__, "matrices.jl"))

truesig(A, z) = minimum(svdvals(A - z * I))   # Float64 reference σ_min(A−zI)

# β_Λ = ‖R₁‖₂/(1−‖R₂‖₂), the single block residual of the draft (eq. blockdev), in balls
function beta_lambda(A, R, clusters)
    CT = eltype(R.W)
    W = R.W
    Y = inv(W)
    Bw = BallMatrix(CT.(A))
    Wb = BallMatrix(W)
    Yb = BallMatrix(Y)
    Mt = Yb * Bw * Wb                                   # M̃ = YAW
    Λ = zeros(CT, size(W)...)
    for c in clusters, j in c, i in c
        Λ[i, j] = mid(Mt)[i, j]
    end
    R1 = Yb * (Bw * Wb - Wb * BallMatrix(Λ))            # R₁ = Y(AW − WΛ)
    R2 = Yb * Wb - BallMatrix(CT.(I(size(W, 1))))       # R₂ = YW − I
    nR1 = BallArithmetic.upper_bound_L2_opnorm(R1)
    nR2 = BallArithmetic.upper_bound_L2_opnorm(R2)
    return mid(nR1) + rad(nR1) < Inf ?
           (mid(nR1) + rad(nR1)) / (1 - (mid(nR2) + rad(nR2))) : Inf
end

function run_one(g)
    A = g.A
    n = size(A, 1)
    W, clusters = vbd_solve(A)
    R = miyajima_rump(A, W; kappa_mode = :svdbox)
    vr = vbd_resolvent(A, R; blocks = clusters)
    bL = beta_lambda(A, R, clusters)
    sizes = sort(length.(clusters); rev = true)
    @printf("%-13s n=%2d  #blocks=%2d  sizes=%-16s  κ₂=%.2e  ‖R₂‖∞=%.1e  β_Λ=%.1e\n",
        g.name, n, length(clusters), string(sizes[1:min(6, end)]),
        Float64(R.kappa), Float64(R.nrmR2), Float64(bL))

    # validate floor vs true σ_min on a grid in the box; rigor = floor ≤ true everywhere
    lo, hi = g.box
    xs = range(real(lo), real(hi); length = 25)
    ys = range(imag(lo), imag(hi); length = 25)
    for (lbl, near) in (("far ", false), ("blkSVD", true))
        worst = -Inf            # max(floor − true): MUST be ≤ 0 for rigor
        npos = 0; ratios = Float64[]
        for x in xs, y in ys
            z = x + im * y
            f = sigma_min_lower(vr, z; near = near)
            t = truesig(A, z)
            worst = max(worst, f - t)
            if f > 0
                npos += 1
                t > 0 && push!(ratios, f / t)
            end
        end
        med = isempty(ratios) ? NaN : sort(ratios)[cld(length(ratios), 2)]
        @printf("    %-6s rigor max(floor−true)=%+.2e %s  certified %4d/%d  median floor/true=%.2f\n",
            lbl, worst, worst <= 1e-10 ? "✓" : "VIOLATION", npos, length(xs) * length(ys), med)
        flush(stdout)
    end
end

println("VBD pipeline on the gallery (Float64):\n")
for g in gallery()
    run_one(g)
end
