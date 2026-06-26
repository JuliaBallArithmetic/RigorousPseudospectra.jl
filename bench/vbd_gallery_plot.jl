# Plot the VBD block resolvent bound on the gallery: for each matrix, vbd_solve →
# miyajima_rump → vbd_resolvent, then the certified log₁₀‖(A−zI)⁻¹‖₂ from the per-z block
# SVD floor (sigma_min_lower …; near = true). Dashed cyan = the block discs of radius
# ‖N_j‖ (the kept blocks); red × = eigenvalues. Run: julia --project=bench bench/vbd_gallery_plot.jl
using RigorousPseudospectra, LinearAlgebra, CairoMakie, Printf
include("matrices.jl")

function build(B)
    Bc = ComplexF64.(Matrix(B))
    W, clusters = vbd_solve(Bc)
    R = miyajima_rump(Bc, W; kappa_mode = :svdbox)
    return vbd_resolvent(Bc, R; blocks = clusters), clusters, R
end

function plot_one!(fig, pos, name, B, vr; N = 170)
    ev = eigvals(ComplexF64.(Matrix(B)))
    res = real.(ev); ims = imag.(ev)
    cx = (minimum(res) + maximum(res)) / 2
    cy = (minimum(ims) + maximum(ims)) / 2
    hw = 0.6 * max(maximum(res) - minimum(res), maximum(ims) - minimum(ims), 1.0) + 0.3
    xs = range(cx - hw, cx + hw; length = N)
    ys = range(cy - hw, cy + hw; length = N)
    Z = [(s = sigma_min_lower(vr, x + im * y; near = true); s > 0 ? -log10(s) : NaN)
         for x in xs, y in ys]
    ax = Axis(fig[pos...]; title = name, xlabel = "Re z", ylabel = "Im z",
        aspect = DataAspect())
    fin = filter(isfinite, Z)
    crange = isempty(fin) ? (0.0, 1.0) : (minimum(fin), maximum(fin))
    hm = heatmap!(ax, xs, ys, Z; colormap = :viridis,
        nan_color = RGBAf(0.6, 0.6, 0.6, 1), colorrange = crange)
    contour!(ax, xs, ys, Z; levels = 8, color = (:white, 0.5), linewidth = 0.6)
    θ = range(0, 2π; length = 160)
    for j in eachindex(vr.centres)
        c = ComplexF64(vr.centres[j]); ρ = Float64(vr.blocknn[j])
        ρ > 1e-9 && lines!(ax, real(c) .+ ρ .* cos.(θ), imag(c) .+ ρ .* sin.(θ);
            color = :cyan, linestyle = :dash, linewidth = 1.2)
    end
    scatter!(ax, res, ims; marker = :xcross, color = :red, markersize = 9)
    limits!(ax, cx - hw, cx + hw, cy - hw, cy + hw)
    return hm
end

function main()
    mkpath(joinpath(@__DIR__, "figures"))
    for e in gallery()
        vr, clusters, R = build(e.A)
        fig = Figure(; size = (640, 560))
        hm = plot_one!(fig, (1, 1), e.name, e.A, vr)
        Colorbar(fig[1, 2], hm; label = "log₁₀ ‖(A−zI)⁻¹‖₂ upper bound")
        save(joinpath(@__DIR__, "figures", "vbd_$(e.name).png"), fig)
        sizes = sort(length.(clusters); rev = true)
        @printf("%-13s #blocks=%2d sizes=%-12s κ₂=%.2e -> vbd_%s.png\n",
            e.name, length(clusters), string(sizes[1:min(4, end)]),
            Float64(R.kappa), e.name)
    end
    println("\nFigures → bench/figures/vbd_<name>.png")
end
main()
