# Block-Geršgorin (Feingold–Varga) sharpening of the OUTER Krylov resolvent floor (paper §4).
# The 2-block assembly σ₁₁σ₂₂/(σ₁₁+σ₂₂+γ) charges the single symmetric coupling γ=max(‖G₁₂‖,‖G₂₁‖)
# and never dies; the block-dominance floor min(σ₁₁,σ₂₂)(1−τ), τ²=(‖G₁₂‖/σ₁₁)²+(‖G₂₁‖/σ₂₂)², is
# κ-free (unitary outer basis) and ~2× tighter where the couplings are small vs the gaps (the FAR
# field), but dies once τ≥1 (near the spectrum). The engine returns their max (`_assemble2`); this
# script quantifies the gain on a transfer-like (decaying-σ) operator and plots the improved
# rigorous ε-pseudospectrum OUTSIDE the far circle.
#   julia --project=bench bench/blockdom_farfield.jl
using RigorousPseudospectra, LinearAlgebra, Printf, CairoMakie
const RP = RigorousPseudospectra

# transfer-like: Gaussian-smoothing convolution, row-stochastic → spectral radius 1, σ decays fast.
function smoothing(n; w = 0.06)
    xs = range(0, 1; length = n)
    A = [exp(-((xs[i] - xs[j]) / w)^2) for i in 1:n, j in 1:n]
    A ./= sum(A; dims = 2)
    ComplexF64.(A)
end

n = 300
A = smoothing(n)
s = svdvals(A)
@printf("smoothing n=%d: σ₁=%.3f σ₅₀/σ₁=%.2e (decaying ⇒ far block can be made deep)\n",
    n, s[1], s[50] / s[1])

# certify the leading eigenvalues outside a fixed circle; far block deep ⇒ ‖G₂₂‖ < Rcirc
Rcirc = 0.35
out = certify_leading_eigs_sing(
    A; Rcirc = Rcirc, nblock_eig = 40, margin = 0.8, tighten = false)
@printf("dim(Q)=%d  ‖G₂₂‖=%.4f  γ=max(‖G₁₂‖,‖G₂₁‖)=%.4f   ‖G₁₂‖=%.4f ‖G₂₁‖=%.4f\n",
    out.dimQ, out.G22norm, out.γ, out.g12, out.g21)

# pure-Schur floor (old) vs max(Schur, block-dominance) (new), on the same certified σ₁₁,σ₂₂.
schur_only(z) = begin
    r = abs(z)
    r <= out.G22norm && return 0.0
    σ11 = Float64(RP._sigma_min_cert(out.G11 - z * I))
    σ22 = r - out.G22norm
    (σ11 <= 0 || σ22 <= 0) ? 0.0 : σ11 * σ22 / (σ11 + σ22 + out.γ)
end
new_floor(z) = pseudospectrum_outside(out, z)   # = max(Schur, block-dominance)

# sweep a far-field grid and tally the improvement
gx = range(-1.2, 1.2; length = 141)
gy = range(-1.2, 1.2; length = 141)
Fnew = fill(NaN, length(gy), length(gx))
Fold = fill(NaN, length(gy), length(gx))
ratios = Float64[]
for (jx, x) in enumerate(gx), (jy, y) in enumerate(gy)

    z = x + im * y
    abs(z) <= out.G22norm && continue
    fo = schur_only(z)
    fn = new_floor(z)
    Fold[jy, jx] = fo
    Fnew[jy, jx] = fn
    fo > 0 && push!(ratios, fn / fo)
end
sr = sort(ratios);
med = sr[cld(length(sr), 2)]
@printf("far-field points certified: %d   improvement fn/fo: median=%.3f  max=%.3f  (>1.05 on %.0f%%)\n",
    length(ratios), med, maximum(ratios), 100 * count(>(1.05), ratios) / length(ratios))

# figure: rigorous pseudospectral contours (old dashed vs new solid) + the far-circle mask
fig = Figure(size = (760, 700))
ax = Axis(fig[1, 1]; xlabel = "Re z", ylabel = "Im z", aspect = DataAspect(),
    title = @sprintf("Rigorous σ_min(A−zI) floor outside |z|=‖G₂₂‖=%.2f:  Schur (dashed) vs +block-dominance (solid)",
        out.G22norm))
lv = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
contour!(ax, gx, gy, permutedims(Fold); levels = lv,
    color = :gray55, linestyle = :dash, linewidth = 1.4)
contour!(ax, gx, gy, permutedims(Fnew); levels = lv, color = :royalblue, linewidth = 2.0)
θ = range(0, 2π; length = 200)
lines!(
    ax, out.G22norm .* cos.(θ), out.G22norm .* sin.(θ); color = :firebrick, linewidth = 1.5)
poly!(ax, Point2f.(out.G22norm .* cos.(θ), out.G22norm .* sin.(θ)); color = (
    :firebrick, 0.08))
ev = eigvals(Matrix(A))
scatter!(ax, real.(ev), imag.(ev); color = :black, markersize = 5, marker = :xcross)
save(joinpath(@__DIR__, "figures", "blockdom_farfield.png"), fig)
println("saved bench/figures/blockdom_farfield.png")
