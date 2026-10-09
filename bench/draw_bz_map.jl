# The figures for the Belousov-Zhabotinsky maps of experiment 5, drawn from the results serialised
# by `bench/bz/paper_map_bz.jl` (paper_map_bz.<tag>.<host>.jls, tag bz_k<KE>_xi<ξ>_N<N>): one figure
# with one panel per tag given, each panel on its own colour scale with its own colour bar. The two
# scripts are separate because the computation needs the `bench/bz` project
# (RigorousInvariantMeasures) and the drawing needs `bench` (CairoMakie).
#
#   julia --project=bench bench/draw_bz_map.jl name bz_k12_xi0.1_N4 bz_k16_xi0.1_N4

using CairoMakie
using Printf
import Serialization

const MAPDIR = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra",
    "maps")
const FIGDIR = joinpath(@__DIR__, "figures")
const PAPERDIR = joinpath(homedir(), "Write", "KrylovPseudospectra", "figures")
foreach(mkpath, (FIGDIR, PAPERDIR))

const INK = RGBf(0.15, 0.15, 0.14)
const MUTED = RGBf(0.42, 0.42, 0.40)
const NOBOUND = RGBAf(0.72, 0.72, 0.71, 1)
const DISCCOL = RGBf(1.0, 0.85, 0.0)
const EVCOL = RGBf(0.85, 0.15, 0.1)
const COMPCOL = RGBf(1.0, 1.0, 1.0)

const NAME = ARGS[1]
const TAGS = ARGS[2:end]

"The serialised map for this tag, the most recent when several machines have written one."
function load(tag)
    pre = "paper_map_bz.$(tag)."
    cand = [joinpath(MAPDIR, f) for f in readdir(MAPDIR) if startswith(f, pre) && endswith(f, ".jls")]
    isempty(cand) && error("no $(pre)*.jls in $MAPDIR")
    return Serialization.deserialize(sort(cand; by = mtime)[end])
end

logs(F) = [f > 0 ? log10(f) : NaN for f in F]

newaxis(fig, pos) = Axis(fig[pos...]; xlabel = "Re z", ylabel = "Im z", aspect = DataAspect(),
    xlabelcolor = INK, ylabelcolor = INK, xticklabelcolor = MUTED, yticklabelcolor = MUTED,
    spinewidth = 0.8, leftspinecolor = MUTED, bottomspinecolor = MUTED,
    rightspinecolor = MUTED, topspinecolor = MUTED, xgridvisible = false, ygridvisible = false)

"""
One panel: the certified floor as a heat map, grey where there is none, the two circles, the
eigenvalues of `A`, the centres of the certified enclosure of `G11`, and the isolation discs.
`rings = false` drops the circles, for a zoom panel that does not contain them.
"""
function panel!(ax, xs, ys, F, r, cr; rings = true, evsize = 6, compsize = 7, discs = true)
    hm = heatmap!(ax, xs, ys, logs(F); colormap = :viridis, colorrange = cr,
        nan_color = NOBOUND)
    θ = range(0, 2π; length = 600)
    if rings
        lines!(ax, r.R .* cos.(θ), r.R .* sin.(θ); color = :white, linewidth = 2.0)
        lines!(ax, r.gbar .* cos.(θ), r.gbar .* sin.(θ); color = :white, linewidth = 1.6,
            linestyle = :dash)
    end
    if haskey(r, :comps) && !isempty(r.comps)
        c = [x.center for x in r.comps]
        scatter!(ax, real.(c), imag.(c); marker = :circle, color = (COMPCOL, 0.0),
            strokecolor = COMPCOL, strokewidth = 1.0, markersize = compsize)
    end
    scatter!(ax, real.(r.ev), imag.(r.ev); marker = :xcross, color = EVCOL,
        markersize = evsize, strokewidth = 0)
    if discs
        for d in r.discs
            lines!(ax, real(d.center) .+ d.radius .* cos.(θ),
                imag(d.center) .+ d.radius .* sin.(θ); color = DISCCOL, linewidth = 1.8)
        end
    end
    return hm
end

"The size, R and ḡ of a panel, written inside it at the top left."
function stamp!(ax, r, half)
    txt = @sprintf("k = 2^%d,  ξ = %.0e,  R = %.4g,  ḡ = %.4g", round(Int, log2(r.n)), r.ξ, r.R, r.gbar)
    any(>(0), r.F) || (txt *= "\nno bound anywhere in the box")
    text!(ax, -0.94half, 0.90half; text = txt, align = (:left, :top), color = :white,
        fontsize = 11, strokecolor = RGBAf(0, 0, 0, 0.6), strokewidth = 1.6)
end

have = [load(t) for t in TAGS]
fig = Figure(size = (560 * length(have), 500), fontsize = 13, backgroundcolor = :white)
for (i, r) in enumerate(have)
    fin = filter(isfinite, logs(r.F))
    cr = isempty(fin) ? (0.0, 1.0) : (minimum(fin), maximum(fin))
    ax = newaxis(fig, (1, 2i - 1))
    hm = panel!(ax, r.xs, r.ys, r.F, r, cr)
    xlims!(ax, -r.half, r.half)
    ylims!(ax, -r.half, r.half)
    stamp!(ax, r, r.half)
    length(have) > 1 && Label(fig[1, 2i - 1, TopLeft()], string('a' + i - 1), font = :bold,
        color = INK, padding = (0, 6, 4, 0))
    Colorbar(fig[1, 2i], hm; label = "log₁₀ of the certified lower bound on σ_min(A − zI)",
        labelcolor = INK, ticklabelcolor = MUTED, labelsize = 12, ticklabelsize = 11)
    bb = r.Fc .> 0
    ra = sort([r.Fc[j] / r.Dc[j] for j in eachindex(r.Fc) if bb[j]])
    @printf("%s: dim Q = %d, R = %.6g, ḡ = %.6g, half-width %.4g, bounded %d of %d, count %s, %d discs",
        r.name, r.k, r.R, r.gbar, r.half, count(>(0), r.F), length(r.F),
        r.count < 0 ? "none" : string(r.count), length(r.discs))
    isempty(ra) || @printf("; dense check %d points, violations %d, ratio min %.4f median %.4f",
        length(r.Fc), r.strict, ra[1], ra[cld(end, 2)])
    println()
end
for ext in ("pdf", "png")
    p = joinpath(FIGDIR, "$(NAME).$ext")
    save(p, fig; px_per_unit = ext == "png" ? 2 : 1)
    cp(p, joinpath(PAPERDIR, "$(NAME).$ext"); force = true)
    println("wrote ", p)
end
