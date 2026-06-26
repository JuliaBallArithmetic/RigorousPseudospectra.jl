# Free resolvent bound on the legacy test-matrix gallery (implementation.md §8).
# For each matrix we build the certificate (miyajima_rump → free_resolvent) and plot the
# certified resolvent UPPER bound  ‖(A−zI)⁻¹‖₂ ≤ 1/σ_min_lower(z)  over a region:
#   * GREY where the free bound is undefined (σ_min_lower(z) = 0 — the near field, where
#     no positive σ_min is certifiable: the pole obstruction);
#   * a log₁₀ contour/heatmap of the bound where it IS defined.
# It also tests the intuition that the grey region is the UNION of discs of radius ‖Nⱼ‖
# (the per-block non-normality) around the cluster centres cⱼ. Analytically it is exactly
#       grey = { z : minⱼ(|z−cⱼ| − ‖Nⱼ‖) ≤ ‖E‖₂ + β₂ } = ⋃ⱼ disc(cⱼ, ‖Nⱼ‖ + ‖E‖₂ + β₂),
# i.e. balls of radius ‖Nⱼ‖ around each barycenter, UNIFORMLY inflated by the (global)
# coupling ‖E‖₂ and slack β₂ — for a well-refined basis those are tiny, so radius ≈ ‖Nⱼ‖
# (and a singleton/normal block has ‖Nⱼ‖ = 0, so its "ball" is just the eigenvalue).
#
# Run:  julia --project=bench bench/free_resolvent_plot.jl

using RigorousPseudospectra
using LinearAlgebra
using CairoMakie
using Printf

include("matrices.jl")

# certify B; fall back to the identity basis for a defective matrix (eigen ⇒ ‖R₂‖≥1)
function certify(B; refine = 15)
    Bc = ComplexF64.(Matrix(B))
    try
        # svdbox κ₂ here (small gallery): reports the TRUE conditioning for the table
        return miyajima_rump(Bc; refine = refine, mode = :auto, kappa_mode = :svdbox)
    catch
        return miyajima_rump(Bc, Matrix{ComplexF64}(I, size(Bc)...); kappa_mode = :svdbox)
    end
end

# square plot box containing the spectrum and all grey discs, with margin
function autobox(B, fr; margin = 0.30)
    ev = eigvals(ComplexF64.(Matrix(B)))
    r = max.(fr.rowsum, fr.colsum)                            # recentred-disc radii
    re = vcat(real.(ev), real.(fr.diag) .- r, real.(fr.diag) .+ r)
    im = vcat(imag.(ev), imag.(fr.diag) .- r, imag.(fr.diag) .+ r)
    cx = (minimum(re) + maximum(re)) / 2
    cy = (minimum(im) + maximum(im)) / 2
    hw = (1 + margin) * max(maximum(re) - minimum(re), maximum(im) - minimum(im)) / 2
    return cx, cy, hw
end

function plot_one!(fig, pos, name, B, fr; N = 350)
    cx, cy, hw = autobox(B, fr)
    xs = range(cx - hw, cx + hw; length = N)
    ys = range(cy - hw, cy + hw; length = N)
    # certified log₁₀ resolvent upper bound; NaN (grey) where the free bound is undefined
    Z = [(s = sigma_min_lower(fr, x + im * y); s > 0 ? -log10(s) : NaN) for x in xs, y in ys]
    greyfrac = count(isnan, Z) / length(Z)

    ax = Axis(fig[pos...]; title = name, xlabel = "Re z", ylabel = "Im z",
        aspect = DataAspect())
    fin = filter(isfinite, Z)
    crange = isempty(fin) ? (0.0, 1.0) : (minimum(fin), maximum(fin))
    hm = heatmap!(ax, xs, ys, Z; colormap = :viridis, nan_color = RGBAf(0.6, 0.6, 0.6, 1),
        colorrange = crange)
    contour!(ax, xs, ys, Z; levels = 8, color = (:white, 0.5), linewidth = 0.6)
    # predicted grey discs: centre Ãᵢᵢ, radius max(row-tail, col-tail) (dashed)
    θ = range(0, 2π; length = 160)
    for (c, ρ) in zip(fr.diag, max.(fr.rowsum, fr.colsum))
        ρ > 1e-9 && lines!(ax, real(c) .+ ρ .* cos.(θ), imag(c) .+ ρ .* sin.(θ);
            color = :cyan, linestyle = :dash, linewidth = 1.0)
    end
    ev = eigvals(ComplexF64.(Matrix(B)))
    scatter!(ax, real.(ev), imag.(ev); marker = :xcross, color = :red, markersize = 9)
    limits!(ax, cx - hw, cx + hw, cy - hw, cy + hw)
    return hm, greyfrac
end

function main()
    mkpath(joinpath(@__DIR__, "figures"))
    @printf("%-14s | %3s | %10s | %10s | %6s\n",
        "matrix", "n", "max-tail", "κ₂", "grey%")
    println("-"^52)
    for entry in gallery()
        B = entry.A
        R = certify(B)
        fr = free_resolvent(B, R)
        fig = Figure(; size = (640, 560))
        _, gf = plot_one!(fig, (1, 1), entry.name, B, fr)
        Colorbar(fig[1, 2], colormap = :viridis,
            colorrange = (0, 1), label = "log₁₀ ‖(A−zI)⁻¹‖ upper bound (grey = undefined)")
        save(joinpath(@__DIR__, "figures", "free_resolvent_$(entry.name).png"), fig)
        @printf("%-14s | %3d | %10.2e | %10.2e | %5.1f%%\n",
            entry.name, size(B, 1), maximum(max.(fr.rowsum, fr.colsum)), fr.kappa, 100 * gf)
    end
    println("\nFigures → bench/figures/free_resolvent_<name>.png")
    println("Cyan dashed circles = predicted grey discs ⋃ᵢ disc(Ãᵢᵢ, max(row-tail,col-tail));")
    println("they coincide with the grey region BY CONSTRUCTION — the free bound is")
    println("undefined exactly inside the union of non-normality balls about the clusters.")
end

main()
