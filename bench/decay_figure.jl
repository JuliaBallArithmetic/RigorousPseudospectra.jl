# Figure: what decides whether the method reaches an operator.
#
# Two panels, sharing the frame size on the horizontal axis and one curve per operator.
#
#   (a) the far block against the operator, ḡ/ā, on a log scale, with the radius R/ā drawn as a
#       marker on each curve: the method needs ḡ < R, so the frame size an operator requires is
#       where its curve passes below its marker, and a curve that never does is out of reach.
#   (b) the gate of Proposition prop:absorb, R(R − ḡ) against γ₁γ₂, as the ratio γ₁γ₂/(R(R − ḡ)):
#       below one the gate holds, above it the couplings are what fails rather than the decay.
#
# Read from decay_sweep.<host>.jls.
#
#   julia --project=bench bench/decay_figure.jl

using CairoMakie
using Printf
import Serialization

const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
const FIGDIR = joinpath(@__DIR__, "figures")
const PAPERDIR = joinpath(homedir(), "Write", "KrylovPseudospectra", "figures")
foreach(mkpath, (FIGDIR, PAPERDIR))

const INK = RGBf(0.15, 0.15, 0.14)
const MUTED = RGBf(0.42, 0.42, 0.40)

rows = let files = filter(f -> startswith(f, "decay_sweep.") && endswith(f, ".jls"),
        readdir(OUT))
    isempty(files) && error("no decay_sweep.<host>.jls in $OUT")
    vcat([Serialization.deserialize(joinpath(OUT, f)) for f in files]...)
end
ops = unique([r.op for r in rows])
@printf("%d rows over %d operators\n", length(rows), length(ops))

# a colour per operator, and a marker cycle so the curves are separable in print
const COLS = cgrad(:tab20, max(length(ops), 3); categorical = true)
const MARKS = [:circle, :rect, :utriangle, :diamond, :cross, :xcross, :star5, :pentagon]

fig = Figure(size = (1020, 430), fontsize = 13, backgroundcolor = :white)
ax1 = Axis(fig[1, 1]; xlabel = "frame size dim Q", ylabel = "ḡ / ā", yscale = log10,
    xscale = log10, xlabelcolor = INK, ylabelcolor = INK, xticklabelcolor = MUTED,
    yticklabelcolor = MUTED, xgridvisible = false, ygridvisible = false,
    spinewidth = 0.8, leftspinecolor = MUTED, bottomspinecolor = MUTED,
    rightspinecolor = MUTED, topspinecolor = MUTED)
ax2 = Axis(fig[1, 2]; xlabel = "frame size dim Q", ylabel = "γ₁γ₂ / R(R − ḡ)", yscale = log10,
    xscale = log10, xlabelcolor = INK, ylabelcolor = INK, xticklabelcolor = MUTED,
    yticklabelcolor = MUTED, xgridvisible = false, ygridvisible = false,
    spinewidth = 0.8, leftspinecolor = MUTED, bottomspinecolor = MUTED,
    rightspinecolor = MUTED, topspinecolor = MUTED)

nzero = Tuple{String, Int}[]     # points at which the coupling product is exactly zero
for (i, op) in enumerate(ops)
    rs = sort([r for r in rows if r.op == op]; by = r -> r.dimQ)
    x = Float64[r.dimQ for r in rs]
    y = Float64[r.gbar / r.abar for r in rs]
    c = COLS[i]
    m = MARKS[1 + (i - 1) % length(MARKS)]
    lines!(ax1, x, y; color = c, linewidth = 1.6, label = op)
    scatter!(ax1, x, y; color = c, marker = m, markersize = 7)
    # the radius this operator is aiming at, as a horizontal segment over its own range
    rr = rs[1].R / rs[1].abar
    lines!(ax1, [x[1], x[end]], [rr, rr]; color = (c, 0.45), linewidth = 1.0,
        linestyle = :dash)

    g = Float64[]
    xg = Float64[]
    for r in rs
        d = r.R * (r.R - r.gbar)
        d > 0 || continue
        # the product reaches 4e-25 on rdb200, and a floor at 1e-20 drew that as a flat line at
        # the floor; only an exact zero, which no log axis can carry, is dropped
        v = Float64(r.γ1 * r.γ2 / d)
        v > 0 || (push!(nzero, (op, r.dimQ)); continue)
        push!(xg, Float64(r.dimQ))
        push!(g, v)
    end
    if !isempty(g)
        lines!(ax2, xg, g; color = c, linewidth = 1.6)
        scatter!(ax2, xg, g; color = c, marker = m, markersize = 7)
    end
end
isempty(nzero) ||
    println("coupling product exactly zero, omitted from panel b: ", join(string.(nzero), ", "))
hlines!(ax2, [1.0]; color = INK, linewidth = 1.4)
text!(ax2, 0.02, 0.96; text = "the gate fails above this line", space = :relative,
    align = (:left, :top), color = INK, fontsize = 11)

Legend(fig[1, 3], ax1; framevisible = false, labelsize = 10, patchsize = (16, 2),
    rowgap = 0, padding = (2, 2, 2, 2))
Label(fig[1, 1, TopLeft()], "a", font = :bold, color = INK, padding = (0, 6, 4, 0))
Label(fig[1, 2, TopLeft()], "b", font = :bold, color = INK, padding = (0, 6, 4, 0))

for ext in ("pdf", "png")
    p = joinpath(FIGDIR, "decay_frame.$ext")
    save(p, fig; px_per_unit = ext == "png" ? 2 : 1)
    cp(p, joinpath(PAPERDIR, "decay_frame.$ext"); force = true)
    println("wrote ", joinpath(PAPERDIR, "decay_frame.$ext"))
end

# the same content as a table, for the text
println("\noperator            dim Q      ḡ/ā      R/ā   γ₁γ₂/R(R−ḡ)")
for op in ops
    rs = sort([r for r in rows if r.op == op]; by = r -> r.dimQ)
    r = rs[end]
    d = r.R * (r.R - r.gbar)
    @printf("%-18s %6d %8.4g %8.4g %14s\n", op, r.dimQ, r.gbar / r.abar, r.R / r.abar,
        d > 0 ? @sprintf("%.3g", r.γ1 * r.γ2 / d) : "ḡ ≥ R")
end
