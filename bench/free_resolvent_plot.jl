# Where eq:blockres gives no bound, and the discs that contain that set.
#
# The "free resolvent" routine this script used to plot (row/column Gershgorin in the eigenvector
# frame) is not part of the paper and was removed. What remains of the script's question, how
# large is the region where the resolvent bound of the block-diagonal certificate is undefined,
# and which discs contain it, is answered here for the certificate of sec:vbd: the bound
# eq:blockres is zero exactly on ∪_i Ω_i, Ω_i = {σ_min(P_i − zI) ≤ β_Λ}, and thm:enclosure is
# applied through outer discs D(c_i, ρ_i). Two radii are drawn: the m = 1 radius
# ‖P_i − c_iI‖ + β_Λ (dashed cyan), and the radius returned by `outer_radius`, the least over
# m ≤ 8 of the bound from Proposition prop:power applied to P_i − c_iI (solid white). The grey
# fraction of the box is reported.
#
#   julia --project=bench bench/free_resolvent_plot.jl

using RigorousPseudospectra
using LinearAlgebra
using CairoMakie
using Printf

include(joinpath(@__DIR__, "matrices.jl"))

function plot_one!(fig, pos, name, A, R; N = 300)
    cc = R.cert
    ev = eigvals(Matrix(A))
    ρ1 = [opnorm(cc.P[i] - cc.centres[i] * I) + cc.betaΛ for i in eachindex(cc.P)]
    re = vcat(real.(ev), real.(cc.centres) .- ρ1, real.(cc.centres) .+ ρ1)
    imv = vcat(imag.(ev), imag.(cc.centres) .- ρ1, imag.(cc.centres) .+ ρ1)
    cx = (minimum(re) + maximum(re)) / 2
    cy = (minimum(imv) + maximum(imv)) / 2
    hw = 0.65 * max(maximum(re) - minimum(re), maximum(imv) - minimum(imv), 0.5)
    xs = range(cx - hw, cx + hw; length = N)
    ys = range(cy - hw, cy + hw; length = N)
    Z = [(s = sigma_blocks_lo(cc, x + im * y); s > 0 ? -log10(s) : NaN) for x in xs, y in ys]
    greyfrac = count(isnan, Z) / length(Z)
    ax = Axis(fig[pos...]; title = name, xlabel = "Re z", ylabel = "Im z", aspect = DataAspect())
    fin = filter(isfinite, Z)
    crange = isempty(fin) ? (0.0, 1.0) : (minimum(fin), maximum(fin))
    hm = heatmap!(ax, xs, ys, Z; colormap = :viridis, nan_color = RGBAf(0.6, 0.6, 0.6, 1),
        colorrange = crange)
    θ = range(0, 2π; length = 160)
    for (j, c) in enumerate(cc.centres)
        lines!(ax, real(c) .+ ρ1[j] .* cos.(θ), imag(c) .+ ρ1[j] .* sin.(θ); color = :cyan,
            linestyle = :dash, linewidth = 1.0)
        ρ = cc.ρglobal[j]
        ρ > 1e-9 && lines!(ax, real(c) .+ ρ .* cos.(θ), imag(c) .+ ρ .* sin.(θ);
            color = :white, linewidth = 1.0)
    end
    scatter!(ax, real.(ev), imag.(ev); marker = :xcross, color = :red, markersize = 9)
    limits!(ax, cx - hw, cx + hw, cy - hw, cy + hw)
    return hm, greyfrac, maximum(ρ1), maximum(cc.ρglobal)
end

function main()
    mkpath(joinpath(@__DIR__, "figures"))
    @printf("%-14s | %3s | %6s | %10s | %10s | %10s | %6s\n", "matrix", "n", "blocks",
        "max ρ(m=1)", "max ρ", "κ", "grey%")
    println("-"^72)
    for entry in gallery()
        R = miyajima_rump(entry.A)
        fig = Figure(; size = (640, 560))
        hm, gf, r1, rr = plot_one!(fig, (1, 1), entry.name, entry.A, R)
        Colorbar(fig[1, 2], hm; label = "log₁₀ ‖(A−zI)⁻¹‖ upper bound (grey = none)")
        save(joinpath(@__DIR__, "figures", "blockres_region_$(entry.name).png"), fig)
        @printf("%-14s | %3d | %6d | %10.2e | %10.2e | %10.2e | %5.1f%%\n", entry.name,
            size(entry.A, 1), length(R.cert.blocks), r1, rr, R.kappa, 100 * gf)
    end
    println("\nFigures → bench/figures/blockres_region_<name>.png")
end

main()
