# Figure: what a certified point costs, against the dense alternative.
#
# The claim the method makes in practice is that a point of a contour costs one verified SVD of the
# leading block, of size k_EN, and not one of the whole matrix, of size n. The gallery records both
# at every case: `t_z` for the triangular route and `t_sv` for `sigmin_lo` on A − zI.
#
# One panel, log-log against the cut size, with the two times as separate series: the cases where
# they meet are those whose cut takes almost the whole matrix, and there the route has nothing left
# to save.
#
#   julia --project=bench bench/cost_figure.jl

using CairoMakie
using Printf
import Serialization

const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
const FIGDIR = joinpath(@__DIR__, "figures")
const PAPERDIR = joinpath(homedir(), "Write", "KrylovPseudospectra", "figures")
foreach(mkpath, (FIGDIR, PAPERDIR))

const INK = RGBf(0.15, 0.15, 0.14)
const MUTED = RGBf(0.42, 0.42, 0.40)
const TRI = RGBf(0.15, 0.35, 0.75)
const SVD = RGBf(0.80, 0.30, 0.10)

rows = let files = filter(f -> startswith(f, "tri_gallery.") && endswith(f, ".jls"),
        readdir(OUT))
    isempty(files) && error("no tri_gallery.<host>.jls in $OUT")
    newest = sort(files; by = f -> mtime(joinpath(OUT, f)))[end]
    println("reading ", newest)
    Serialization.deserialize(joinpath(OUT, newest))
end
good = [r for r in rows if r.ok && isfinite(r.t_z) && isfinite(r.t_sv) && r.t_z > 0]
@printf("%d cases with both timings\n", length(good))

# the gallery does not record the matrix size, only the cut; the cut is the right abscissa
# anyway, since it is what the per-point cost is cubic in
sizes = Float64[r.c for r in good]

fig = Figure(size = (640, 470), fontsize = 13, backgroundcolor = :white)
ax = Axis(fig[1, 1]; xlabel = "cut size k_EN", ylabel = "seconds per point of the contour",
    xscale = log10, yscale = log10, xlabelcolor = INK, ylabelcolor = INK,
    xticklabelcolor = MUTED, yticklabelcolor = MUTED, xgridvisible = false,
    ygridvisible = false, spinewidth = 0.8, leftspinecolor = MUTED, bottomspinecolor = MUTED,
    rightspinecolor = MUTED, topspinecolor = MUTED)

scatter!(ax, sizes, Float64[r.t_z for r in good]; color = TRI, marker = :circle,
    markersize = 9, label = "triangular route, one SVD of size k_EN")
scatter!(ax, sizes, Float64[r.t_sv for r in good]; color = SVD, marker = :utriangle,
    markersize = 9, label = "one verified SVD of A − zI")
axislegend(ax; position = :lt, framevisible = false, labelsize = 11)

for ext in ("pdf", "png")
    p = joinpath(FIGDIR, "cost.$ext")
    save(p, fig; px_per_unit = ext == "png" ? 2 : 1)
    cp(p, joinpath(PAPERDIR, "cost.$ext"); force = true)
    println("wrote ", joinpath(PAPERDIR, "cost.$ext"))
end

sp = sort([r.t_sv / r.t_z for r in good])
@printf("\nspeedup over the dense route: min %.3g, median %.3g, max %.3g, over %d cases\n",
    sp[1], sp[cld(end, 2)], sp[end], length(sp))
