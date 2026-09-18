# The figures for the Belousov-Zhabotinsky maps, drawn from the results serialised by
# `bench/bz/paper_map_bz.jl`: one figure per noise size, and one four-panel figure comparing them.
# The two scripts are separate because the computation needs the `bench/bz` project
# (RigorousInvariantMeasures) and the drawing needs `bench` (CairoMakie), and neither
# Project.toml was modified.
#
#   julia --project=bench bench/draw_bz_map.jl

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

const XIS = [1e-1, 1e-2, 1e-3, 1e-4]
const KE = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
tag(ξ) = "bz_k$(KE)_xi$(ξ)"      # the TAG of bench/bz/paper_map_bz.jl

"""
The serialised map for this noise size, taking the most recent when several machines have written
one; the file is `map_<tag>.<host>.jls`, and the older runs, written before the hostname was part
of the name, are `map_<tag>.jls`.
"""
function load(ξ)
    pre = "map_$(tag(ξ))"
    cand = [joinpath(MAPDIR, f) for f in readdir(MAPDIR)
            if startswith(f, pre * ".") && endswith(f, ".jls")]
    isempty(cand) && return nothing
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

"The ξ, R and ḡ of a panel, written inside it at the top left."
function stamp!(ax, r, half)
    txt = @sprintf("ξ = %.0e,  R = %.4g,  ḡ = %.4g", r.ξ, r.R, r.gbar)
    any(>(0), r.F) || (txt *= "\nno bound anywhere in the box")
    text!(ax, -0.94half, 0.90half; text = txt, align = (:left, :top), color = :white,
        fontsize = 11, strokecolor = RGBAf(0, 0, 0, 0.6), strokewidth = 1.6)
end

results = [(ξ, load(ξ)) for ξ in XIS]
have = [(ξ, r) for (ξ, r) in results if r !== nothing]
isempty(have) && error("no serialised Belousov-Zhabotinsky map found in $MAPDIR")

# one colour range shared by every panel, so that the four are comparable
allfin = Float64[]
for (_, r) in have
    append!(allfin, filter(isfinite, logs(r.F)))
end
CR = isempty(allfin) ? (0.0, 1.0) : (minimum(allfin), maximum(allfin))
@printf("shared colour range log10 floor in [%.3f, %.3f]\n", CR...)

# --- one figure per noise size ---------------------------------------------------------------------
for (ξ, r) in have
    zoom = haskey(r, :Fz) && any(>(0), r.Fz)
    fig = Figure(size = zoom ? (1060, 470) : (620, 540), fontsize = 13,
        backgroundcolor = :white)
    ax = newaxis(fig, (1, 1))
    hm = panel!(ax, r.xs, r.ys, r.F, r, CR)
    xlims!(ax, -r.half, r.half)
    ylims!(ax, -r.half, r.half)
    stamp!(ax, r, r.half)
    if zoom
        Label(fig[1, 1, TopLeft()], "a", font = :bold, color = INK, padding = (0, 6, 4, 0))
        ax2 = newaxis(fig, (1, 2))
        panel!(ax2, r.zxs, r.zys, r.Fz, r, CR; rings = false, evsize = 11, compsize = 12)
        xlims!(ax2, r.zxs[1], r.zxs[end])
        ylims!(ax2, r.zys[1], r.zys[end])
        Label(fig[1, 2, TopLeft()], "b", font = :bold, color = INK, padding = (0, 6, 4, 0))
        Colorbar(fig[1, 3], hm; label = "log₁₀ of the certified lower bound on σ_min(A − zI)",
            labelcolor = INK, ticklabelcolor = MUTED, labelsize = 12, ticklabelsize = 11)
    else
        Colorbar(fig[1, 2], hm; label = "log₁₀ of the certified lower bound on σ_min(A − zI)",
            labelcolor = INK, ticklabelcolor = MUTED, labelsize = 12, ticklabelsize = 11)
    end
    for ext in ("pdf", "png")
        p = joinpath(FIGDIR, "map_$(tag(ξ)).$ext")
        save(p, fig; px_per_unit = ext == "png" ? 2 : 1)
        cp(p, joinpath(PAPERDIR, "map_$(tag(ξ)).$ext"); force = true)
    end
    # the ξ = 1e-1 figure keeps the name the earlier version had, so that a draft already
    # including it does not break
    if ξ == 1e-1
        for ext in ("pdf", "png")
            cp(joinpath(FIGDIR, "map_$(tag(ξ)).$ext"), joinpath(FIGDIR, "map_bz_k12.$ext");
                force = true)
            cp(joinpath(FIGDIR, "map_$(tag(ξ)).$ext"),
                joinpath(PAPERDIR, "map_bz_k12.$ext"); force = true)
        end
    end
    bb = r.Fc .> 0
    ra = sort([r.Fc[i] / r.Dc[i] for i in eachindex(r.Fc) if bb[i]])
    @printf("ξ = %.0e: dim Q = %d, R = %.6g, ḡ = %.6g, box half-width %.4g, bounded %d of %d, ",
        ξ, r.k, r.R, r.gbar, r.half, count(>(0), r.F), length(r.F))
    @printf("count %s, %d discs; dense check %d points (%d bounded), violations %d",
        r.count < 0 ? "none" : string(r.count), length(r.discs), length(r.Fc), count(bb),
        r.strict)
    isempty(ra) || @printf(", ratio min %.4f median %.4f max %.4f", ra[1], ra[cld(end, 2)],
        ra[end])
    println()
end

# --- the four panels side by side -------------------------------------------------------------------
if length(have) > 1
    fig = Figure(size = (1080, 980), fontsize = 13, backgroundcolor = :white)
    local hm
    for (i, (ξ, r)) in enumerate(have)
        row, col = fld(i - 1, 2) + 1, mod(i - 1, 2) + 1
        ax = newaxis(fig, (row, col))
        hm = panel!(ax, r.xs, r.ys, r.F, r, CR)
        xlims!(ax, -r.half, r.half)
        ylims!(ax, -r.half, r.half)
        stamp!(ax, r, r.half)
        Label(fig[row, col, TopLeft()], string('a' + i - 1), font = :bold, color = INK,
            padding = (0, 6, 4, 0))
    end
    Colorbar(fig[:, 3], hm; label = "log₁₀ of the certified lower bound on σ_min(A − zI)",
        labelcolor = INK, ticklabelcolor = MUTED, labelsize = 12, ticklabelsize = 11)
    for ext in ("pdf", "png")
        p = joinpath(FIGDIR, "map_bz_family.$ext")
        save(p, fig; px_per_unit = ext == "png" ? 2 : 1)
        cp(p, joinpath(PAPERDIR, "map_bz_family.$ext"); force = true)
        println("wrote ", p)
    end
end
