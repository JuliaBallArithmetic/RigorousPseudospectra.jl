# Free-estimate field report: for each gallery matrix, the cheap certified σ_min
# lower bound g(z) = max(gap_tri, gap_svd, gap_recentred) as a field over the box.
# At each z, g(z) is the *best ε we can certify exclusion for from the formulas
# alone* (z ∉ σ_ε for every ε < g(z)). The g = 0 contour is the failure boundary:
# inside it the free formulas certify nothing; moving outward, g grows.
#
#   julia --project=bench bench/free_estimates.jl

using RigorousPseudospectra
using CairoMakie
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "matrices.jl"))

const FIGDIR = joinpath(@__DIR__, "figures")
mkpath(FIGDIR)

# robust upper colour limit (≈97th percentile) without a Statistics dependency
function _hi(v)
    s = sort(vec(v))
    return s[max(1, ceil(Int, 0.97 * length(s)))]
end

function free_fields(A, box; ng = 160)
    lo, hi = box
    xs = range(real(lo), real(hi); length = ng)
    ys = range(imag(lo), imag(hi); length = ng)
    S = certified_setup(A)
    cf = cheap_floor(A)
    M = ComplexF64.(Matrix(A))
    G = [max(cf(complex(x, y)), gap_recentred(complex(x, y), S)) for x in xs, y in ys]
    Tg = [svdvals(M - complex(x, y) * I)[end] for x in xs, y in ys]
    return collect(xs), collect(ys), G, Tg, S.eigs
end

function free_figure(entry)
    xs, ys, G, Tg, eigs = free_fields(entry.A, entry.box)
    vmax = _hi(Tg)
    fig = Figure(size = (1200, 560))
    panels = ((max.(G, 0.0), "free certified σ_min lower bound  g(z)\n(= best ε certifiable per point)"),
        (Tg, "true σ_min  (dense-SVD ground truth)"))
    for (col, (Z, ttl)) in enumerate(panels)
        ax = Axis(fig[1, col]; xlabel = "Re z", ylabel = "Im z",
            aspect = DataAspect(), title = ttl)
        heatmap!(ax, xs, ys, Z; colormap = :viridis, colorrange = (0, vmax))
        if col == 1
            contour!(ax, xs, ys, G; levels = [0.0], color = :red, linewidth = 2.5)
        end
        scatter!(ax, real.(eigs), imag.(eigs); color = :white, marker = :xcross,
            markersize = 9, strokecolor = :black, strokewidth = 0.4)
    end
    Colorbar(fig[1, 3]; colorrange = (0, vmax), colormap = :viridis, label = "σ_min")
    save(joinpath(FIGDIR, "free_$(entry.name).png"), fig)
    return (; name = entry.name, frac_useful = count(>(0), G) / length(G),
        gmax = maximum(G), tmax = maximum(Tg))
end

results = [free_figure(e) for e in gallery()]

open(joinpath(@__DIR__, "REPORT_FREE.md"), "w") do io
    println(io, "# Free-estimate field report\n")
    println(io, "**Left panel** — the cheap certified lower bound ",
        "`g(z) = max(gap_tri, gap_svd, gap_recentred)`. At each `z`, `g(z)` is the ",
        "**best ε we can certify exclusion for from the formulas alone**: `z` is ",
        "provably outside `σ_ε` for every `ε < g(z)`. The **red curve `g(z)=0`** is ",
        "the failure boundary — inside it the free formulas certify nothing; moving ",
        "outward, `g` grows toward the far-field value. **Right panel** — the true ",
        "`σ_min` for comparison (same colour scale). × = eigenvalues.\n")
    println(io, "| matrix | box-fraction where free works (`g>0`) | max free `ε` | max true `σ_min` |")
    println(io, "|---|---|---|---|")
    for r in results
        @printf(io, "| %s | %.0f%% | %.2f | %.2f |\n",
            r.name, 100 * r.frac_useful, r.gmax, r.tmax)
    end
    println(io)
    for r in results
        println(io, "## $(r.name)\n\n![free_$(r.name)](figures/free_$(r.name).png)\n")
    end
end

println("wrote bench/REPORT_FREE.md")
for r in results
    @printf("  %-13s free works on %3.0f%% of box,  max free ε = %.2f  (true max %.2f)\n",
        r.name, 100 * r.frac_useful, r.gmax, r.tmax)
end
