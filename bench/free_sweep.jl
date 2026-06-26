# Removal-fraction sweep: how localized is each matrix's non-normality?
#
# For each matrix, remove the worst eigenvalues (highest off-diagonal tail) one
# fraction at a time, and measure the fraction of the box where the free triangular
# gap is still positive (certifies something). The SHAPE of the curve measures
# localization: a sharp early rise = one removable bad block (localized); a slow
# rise = uniform non-normality (no single culprit — must remove nearly everything).
#
#   julia --project=bench bench/free_sweep.jl

using RigorousPseudospectra
using CairoMakie

include(joinpath(@__DIR__, "matrices.jl"))
const FIGDIR = joinpath(@__DIR__, "figures")
mkpath(FIGDIR)

function tri_gap(z, eigs, trow, tcol, keep)
    ginf = minimum(abs(z - eigs[i]) - trow[i] for i in keep)
    g1 = minimum(abs(z - eigs[i]) - tcol[i] for i in keep)
    return max(ginf, g1)
end

const FRACS = collect(0.0:0.05:0.95)

function works_curve(entry; ng = 110)
    A, box = entry.A, entry.box
    S = certified_setup(A)
    τ = max.(S.tailrow, S.tailcol)
    order = sortperm(τ; rev = true)           # worst (highest tail) first
    n = length(τ)
    lo, hi = box
    xs = range(real(lo), real(hi); length = ng)
    ys = range(imag(lo), imag(hi); length = ng)
    pts = [complex(x, y) for x in xs, y in ys]
    curve = Float64[]
    for f in FRACS
        k = round(Int, f * n)
        keep = k >= n ? [order[end]] : sort(order[(k + 1):end])
        frac = count(z -> tri_gap(z, S.eigs, S.tailrow, S.tailcol, keep) > 0, pts) / length(pts)
        push!(curve, 100 * frac)
    end
    return curve
end

fig = Figure(size = (820, 620))
ax = Axis(fig[1, 1];
    xlabel = "fraction of eigenvalues removed (worst off-diagonal tail first)",
    ylabel = "% of box where the free estimate works (g > 0)",
    title = "Free-estimate recovery vs worst-block removal\n(sharp early rise = localized non-normality; slow rise = uniform)")
for entry in gallery()
    c = works_curve(entry)
    lines!(ax, FRACS, c; linewidth = 2.5, label = entry.name)
    scatter!(ax, FRACS, c; markersize = 6)
    println(rpad(entry.name, 14), " works%: f=0 -> $(round(c[1],digits=0))%, ",
        "f=0.25 -> $(round(c[findfirst(>=(0.25),FRACS)],digits=0))%, ",
        "f=0.5 -> $(round(c[findfirst(>=(0.5),FRACS)],digits=0))%")
end
ylims!(ax, 0, 102)
axislegend(ax; position = :rb)
save(joinpath(FIGDIR, "free_removal_sweep.png"), fig)
println("wrote bench/figures/free_removal_sweep.png")
