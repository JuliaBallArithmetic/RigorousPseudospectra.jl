# Figure: which radii a frame of k columns can reach, and what the certificate gives away.
#
# One panel per operator, the frame size across and the radius up, both on log scales and the
# radius in units of ‖A‖. Two curves:
#
#   * the oracle floor σ_{2k+1}(A), which by Proposition prop:frame no frame of k columns can go
#     below, whatever the sweep does. A radius above this curve is reachable in principle;
#   * the certified ḡ(k), which is what the sweep actually delivers. The gap between them is the
#     price of certifying, and it has been measured at a few per cent once the order is chosen
#     from the estimate.
#
# Over them, the moduli of the first, second, fifth, tenth and twentieth eigenvalues as horizontal
# lines. Where a line crosses the oracle floor is the frame size that radius needs; where it
# crosses ḡ is the frame size this certificate needs. An operator whose lines never cross the floor
# below k = n/2 is one the method cannot reach at any frame, and that is a statement about the
# operator rather than about the implementation.
#
#   julia --project=bench bench/reach_figure.jl

using CairoMakie
using Printf
import Serialization

const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
const FIGDIR = joinpath(@__DIR__, "figures")
const PAPERDIR = joinpath(homedir(), "Write", "KrylovPseudospectra", "figures")
foreach(mkpath, (FIGDIR, PAPERDIR))

const INK = RGBf(0.15, 0.15, 0.14)
const MUTED = RGBf(0.42, 0.42, 0.40)
const FLOOR = RGBf(0.15, 0.35, 0.75)
const CERT = RGBf(0.80, 0.30, 0.10)

rows = let files = filter(f -> startswith(f, "decay_sweep.") && endswith(f, ".jls"),
        readdir(OUT))
    isempty(files) && error("no decay_sweep.<host>.jls in $OUT")
    vcat([Serialization.deserialize(joinpath(OUT, f)) for f in files]...)
end
ops = unique([r.op for r in rows])
@printf("%d rows over %d operators\n", length(rows), length(ops))

ncol = 4
nrow = cld(length(ops), ncol)
fig = Figure(size = (280ncol, 240nrow), fontsize = 12, backgroundcolor = :white)

for (i, op) in enumerate(ops)
    rs = sort([r for r in rows if r.op == op]; by = r -> r.dimQ)
    isempty(rs) && continue
    nA = rs[1].abar
    row, col = fld(i - 1, ncol) + 1, mod(i - 1, ncol) + 1
    ax = Axis(fig[row, col]; title = "$(op)  n = $(rs[1].n)", titlesize = 11,
        xlabel = row == nrow ? "dim Q" : "", ylabel = col == 1 ? "radius / ‖A‖" : "",
        xscale = log10, yscale = log10, xlabelcolor = INK, ylabelcolor = INK,
        titlecolor = INK, xticklabelcolor = MUTED, yticklabelcolor = MUTED,
        xgridvisible = false, ygridvisible = false, spinewidth = 0.8,
        leftspinecolor = MUTED, bottomspinecolor = MUTED, rightspinecolor = MUTED,
        topspinecolor = MUTED, xticklabelsize = 9, yticklabelsize = 9)
    x = Float64[r.dimQ for r in rs]
    g = Float64[max(r.gbar / nA, 1e-18) for r in rs]
    lines!(ax, x, g; color = CERT, linewidth = 1.8)
    scatter!(ax, x, g; color = CERT, markersize = 6)
    fo = [(r.dimQ, r.floor_oracle / nA) for r in rs if isfinite(r.floor_oracle)]
    if !isempty(fo)
        lines!(ax, first.(fo), max.(last.(fo), 1e-18); color = FLOOR, linewidth = 1.8)
        scatter!(ax, first.(fo), max.(last.(fo), 1e-18); color = FLOOR, marker = :rect,
            markersize = 6)
    end
    for (lam, lab) in ((rs[1].lam2, "2"), (rs[1].lam5, "5"), (rs[1].lam10, "10"),
        (rs[1].lam20, "20"))
        isfinite(lam) || continue
        hlines!(ax, [lam / nA]; color = (INK, 0.35), linewidth = 0.8, linestyle = :dash)
        text!(ax, x[end], lam / nA; text = lab, align = (:right, :bottom), fontsize = 8,
            color = (INK, 0.55))
    end
end

Legend(fig[nrow + 1, 1:ncol],
    [LineElement(color = FLOOR, linewidth = 1.8), LineElement(color = CERT, linewidth = 1.8),
        LineElement(color = (INK, 0.35), linewidth = 0.8, linestyle = :dash)],
    ["σ_{2k+1}(A), the floor no frame of k columns goes below",
        "ḡ(k), what the certificate delivers",
        "|λ_c| for c = 2, 5, 10, 20"];
    orientation = :horizontal, framevisible = false, labelsize = 10)

for ext in ("pdf", "png")
    p = joinpath(FIGDIR, "reach.$ext")
    save(p, fig; px_per_unit = ext == "png" ? 2 : 1)
    cp(p, joinpath(PAPERDIR, "reach.$ext"); force = true)
    println("wrote ", joinpath(PAPERDIR, "reach.$ext"))
end

println("\noperator            n    dim Q at which ḡ first falls below |λ_10|    (oracle floor)")
for op in ops
    rs = sort([r for r in rows if r.op == op]; by = r -> r.dimQ)
    tgt = rs[1].lam10
    ic = findfirst(r -> r.gbar < tgt, rs)
    io = findfirst(r -> isfinite(r.floor_oracle) && r.floor_oracle < tgt, rs)
    @printf("%-18s %6d %20s %20s\n", op, rs[1].n,
        ic === nothing ? "none up to $(rs[end].dimQ)" : string(rs[ic].dimQ),
        io === nothing ? "none" : string(rs[io].dimQ))
end
