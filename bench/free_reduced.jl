# Free-estimate field with the WORST block removed (diagnostic).
#
# The free triangular gap g(z) = max_p min_i (|z − t̃ᵢ| − τᵢ^p) is dragged negative
# by the eigenvalues with the largest off-diagonal tails τᵢ — the "worst block".
# Those are exactly the ones the cluster path would hand to a block SVD. Here we
# DROP them (tail above a threshold) and recompute the gap over the kept eigenvalues
# to see how much of the plane the free formula recovers once the worst block is set
# aside. NOTE: the reduced field is a DIAGNOSTIC, not a certified bound for A (it
# ignores some eigenvalues); it illustrates the far-set the assembly keeps cheap.
#
#   julia --project=bench bench/free_reduced.jl

using RigorousPseudospectra
using CairoMakie
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "matrices.jl"))

const FIGDIR = joinpath(@__DIR__, "figures")
mkpath(FIGDIR)

_hi(v) = (s = sort(vec(v)); s[max(1, ceil(Int, 0.97 * length(s)))])

# triangular free gap (σ_min(T̃−zI) lower bound) over a subset `keep` of eigenvalues
function tri_gap(z, eigs, trow, tcol, keep)
    ginf = minimum(abs(z - eigs[i]) - trow[i] for i in keep)
    g1 = minimum(abs(z - eigs[i]) - tcol[i] for i in keep)
    return max(ginf, g1)
end

function reduced_figure(entry; keep_quantile = 0.75)
    A, box = entry.A, entry.box
    S = certified_setup(A)
    τ = max.(S.tailrow, S.tailcol)                 # per-eigenvalue off-diagonal mass
    thr = sort(τ)[clamp(ceil(Int, keep_quantile * length(τ)), 1, length(τ))]
    keep = findall(<=(thr), τ)                     # drop the worst (high-tail) block
    drop = findall(>(thr), τ)
    isempty(keep) && (keep = collect(eachindex(τ)))

    lo, hi = box
    ng = 160
    xs = range(real(lo), real(hi); length = ng)
    ys = range(imag(lo), imag(hi); length = ng)
    M = ComplexF64.(Matrix(A))
    allidx = collect(eachindex(τ))
    Gfull = [tri_gap(complex(x, y), S.eigs, S.tailrow, S.tailcol, allidx) for x in xs, y in ys]
    Gred = [tri_gap(complex(x, y), S.eigs, S.tailrow, S.tailcol, keep) for x in xs, y in ys]
    Tg = [svdvals(M - complex(x, y) * I)[end] for x in xs, y in ys]
    vmax = _hi(Tg)

    fig = Figure(size = (1500, 520))
    panels = ((Gfull, "free gap — all eigenvalues"),
        (Gred, "free gap — worst block removed ($(length(drop)) of $(length(τ)) dropped)"),
        (Tg, "true σ_min"))
    for (col, (Z, ttl)) in enumerate(panels)
        ax = Axis(fig[1, col]; xlabel = "Re z", ylabel = "Im z",
            aspect = DataAspect(), title = ttl)
        heatmap!(ax, xs, ys, max.(Z, 0.0); colormap = :viridis, colorrange = (0, vmax))
        col <= 2 && contour!(ax, xs, ys, Z; levels = [0.0], color = :red, linewidth = 2)
        if !isempty(keep)
            scatter!(ax, real.(S.eigs[keep]), imag.(S.eigs[keep]);
                color = :white, marker = :xcross, markersize = 9,
                strokecolor = :black, strokewidth = 0.4)
        end
        if !isempty(drop)                          # the removed worst block, in red
            scatter!(ax, real.(S.eigs[drop]), imag.(S.eigs[drop]);
                color = :red, marker = :diamond, markersize = 11,
                strokecolor = :black, strokewidth = 0.5)
        end
    end
    Colorbar(fig[1, 4]; colorrange = (0, vmax), colormap = :viridis, label = "σ_min")
    save(joinpath(FIGDIR, "free_reduced_$(entry.name).png"), fig)
    return (; name = entry.name, ndrop = length(drop), n = length(τ),
        thr, full_useful = count(>(0), Gfull) / length(Gfull),
        red_useful = count(>(0), Gred) / length(Gred))
end

results = [reduced_figure(e) for e in gallery()]

open(joinpath(@__DIR__, "REPORT_FREE_REDUCED.md"), "w") do io
    println(io, "# Free estimate with the worst block removed\n")
    println(io, "Left: the free triangular gap over **all** eigenvalues — dragged ",
        "negative (failing, dark; red = `g=0`) by the high-tail eigenvalues. ",
        "Middle: the same gap with the **worst block removed** (eigenvalues whose ",
        "off-diagonal tail exceeds the threshold — red diamonds), recomputed over ",
        "the kept eigenvalues (white ×). Right: true `σ_min`. The middle panel is a ",
        "diagnostic (it ignores the dropped eigenvalues, so it is not a certified ",
        "bound for `A`) — it shows how much of the plane the free formula recovers ",
        "once the worst block is set aside for the block SVD.\n")
    println(io, "| matrix | eigenvalues dropped | tail threshold | free works (all) | free works (worst removed) |")
    println(io, "|---|---|---|---|---|")
    for r in results
        @printf(io, "| %s | %d / %d | %.2f | %.0f%% | %.0f%% |\n",
            r.name, r.ndrop, r.n, r.thr, 100 * r.full_useful, 100 * r.red_useful)
    end
    println(io)
    for r in results
        println(io, "## $(r.name)\n\n![free_reduced_$(r.name)](figures/free_reduced_$(r.name).png)\n")
    end
end

println("wrote bench/REPORT_FREE_REDUCED.md")
for r in results
    @printf("  %-13s dropped %d/%d (τ>%.2f):  free works %.0f%% -> %.0f%%\n",
        r.name, r.ndrop, r.n, r.thr, 100 * r.full_useful, 100 * r.red_useful)
end
