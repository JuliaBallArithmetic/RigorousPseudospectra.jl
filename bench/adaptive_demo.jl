# z-adaptive vs global block escalation on the weakly-coupled chain.
#
# The chain is ONE global coupling cluster (m = n), so global escalation does an
# n×n verified SVD on every unresolved cell. The z-adaptive active set is a small
# local window, so it does a tiny m×m SVD instead. This script reports the cluster
# sizes and the sweep timings for both modes, and renders the (identical, rigorous)
# enclosure.
#
#   julia --project=bench bench/adaptive_demo.jl

using RigorousPseudospectra
using CairoMakie
using Printf

include(joinpath(@__DIR__, "matrices.jl"))
const FIGDIR = joinpath(@__DIR__, "figures")
mkpath(FIGDIR)

entry = first(e for e in gallery() if e.name == "chain")
A, box, ε, h = entry.A, entry.box, entry.ε, entry.h
n = size(A, 1)
S = certified_setup(A)

# global clustering: one big component of size n
gsize = maximum(length, cluster_components(S; θ = 0.25, ε_target = ε))

# z-adaptive: largest active window along the spectrum
margin = max(maximum(S.tailrow), maximum(S.tailcol))
awin = maximum(length(RigorousPseudospectra._active_set(complex(x, 0.0), S, margin))
for x in 0.0:0.2:Float64(n + 1))

# time both modes (warm up first)
bisect(cheap_floor(A); box, ε, min_halfdiag = 0.4, cluster = true)
tg = @elapsed encg = bisect(cheap_floor(A); box, ε, min_halfdiag = h, adaptive = false)
ta = @elapsed enca = bisect(cheap_floor(A); box, ε, min_halfdiag = h, adaptive = true)

save(joinpath(FIGDIR, "chain_adaptive.png"), plot_enclosure(enca))

@printf("n = %d\n", n)
@printf("global  cluster size (max component) = %d   (one m=n block)\n", gsize)
@printf("adaptive max active window           = %d   (local, per z)\n", awin)
@printf("global   sweep: %.2f s   (ex=%d in=%d bd=%d)\n", tg,
    length(encg.excluded), length(encg.included), length(encg.boundary))
@printf("adaptive sweep: %.2f s   (ex=%d in=%d bd=%d)\n", ta,
    length(enca.excluded), length(enca.included), length(enca.boundary))
@printf("speedup x%.1f\n", tg / ta)
