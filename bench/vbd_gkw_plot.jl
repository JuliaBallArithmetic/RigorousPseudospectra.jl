# Plot the VBD block resolvent bound for GKW (K=256) at 256- and 512-bit, from the cached
# decompositions. Uses the cheap far-field floor (near=false) so the heatmap is feasible:
# the near-0 geometric tail shows as ONE block region (disc of radius ‖N_block‖ about its
# barycentre), the leading eigenvalues as resolved singletons. The block fields are
# converted to Float64 once, so the grid scan is pure-Float64 (no BigFloat per-z).
# Run:  julia --project=bench bench/vbd_gkw_plot.jl
using RigorousPseudospectra, LinearAlgebra, BallArithmetic, CairoMakie, Serialization, Printf

cachepath(p) = joinpath(@__DIR__, "gkw_vbd_bf$(p).jls")

# Float64 view of a VBDResolvent for fast far-field evaluation
function f64view(prec)
    setprecision(BigFloat, prec)
    c = deserialize(cachepath(prec))
    vr = vbd_resolvent(c.A, c.R; blocks = c.clusters)         # one BigFloat inv + ball product
    centres = ComplexF64.(vr.centres)
    blocknn = Float64.(vr.blocknn)
    bigblk = argmax(length.(c.clusters))                       # index of the near-0 block
    return centres, blocknn, Float64(vr.coupling), Float64(vr.kappa),
    ComplexF64.(c.R.centres), centres[bigblk], blocknn[bigblk]
end

farfield(centres, blocknn, coup, κ, z) =
    (g = minimum(abs(z - centres[j]) - blocknn[j] for j in eachindex(centres));
    s = (g - coup) / κ; s > 0 ? -log10(s) : NaN)

function panel!(fig, pos, prec, N = 320)
    centres, blocknn, coup, κ, evs, cblk, ρblk = f64view(prec)
    # region around the leading real spectrum and the near-0 block
    xs = range(-0.45, 1.12; length = N)
    ys = range(-0.42, 0.42; length = round(Int, N * 0.84 / 1.57))
    Z = [farfield(centres, blocknn, coup, κ, x + im * y) for x in xs, y in ys]
    ax = Axis(fig[pos...]; title = "GKW K=256, $(prec)-bit  (κ₂=$(round(κ,digits=0)))",
        xlabel = "Re z", ylabel = "Im z", aspect = DataAspect())
    fin = filter(isfinite, Z)
    hm = heatmap!(ax, xs, ys, Z; colormap = :viridis, nan_color = RGBAf(0.6, 0.6, 0.6, 1),
        colorrange = isempty(fin) ? (0.0, 1.0) : (minimum(fin), maximum(fin)))
    contour!(ax, xs, ys, Z; levels = 8, color = (:white, 0.4), linewidth = 0.5)
    θ = range(0, 2π; length = 200)                              # the near-0 block disc
    lines!(ax, real(cblk) .+ ρblk .* cos.(θ), imag(cblk) .+ ρblk .* sin.(θ);
        color = :cyan, linestyle = :dash, linewidth = 1.5)
    scatter!(ax, real.(evs), imag.(evs); marker = :xcross, color = :red, markersize = 6)
    limits!(ax, -0.45, 1.12, -0.42, 0.42)
    return hm
end

function main()
    mkpath(joinpath(@__DIR__, "figures"))
    fig = Figure(; size = (1180, 460))
    panel!(fig, (1, 1), 256)
    hm = panel!(fig, (1, 2), 512)
    Colorbar(fig[1, 3], hm; label = "log₁₀ ‖(A−zI)⁻¹‖₂ upper bound (VBD far-field)")
    save(joinpath(@__DIR__, "figures", "vbd_gkw.png"), fig)
    println("Figure → bench/figures/vbd_gkw.png")
end
main()
