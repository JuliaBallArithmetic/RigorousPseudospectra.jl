# Exterior pseudospectrum maps on the real operators, for the numerics section.
#
# For one operator, at the radius the planner chooses: the frame of
# def:Q at the radius and frame size `plan_split` chooses, one `certify_split` with no peeling,
# one `triangular_split` of the compression, and then on a grid
# over a box containing |z| = R the certified lower bound on σ_min(A − zI) of thm:assembly with
# eq:farshort, grey where there is none. Over it: the circle |z| = R, the circle |z| = ḡ, the dense
# σ_min = ε contours for a few ε, the eigenvalues, and the isolation discs of cor:agree where the
# exterior count closed.
#
# Every grid point is compared against the dense σ_min(A − zI) and the number of points at which
# the certified bound exceeds it is reported; that number must be zero.
#
#   JULIA_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1 julia --project=bench bench/paper_maps.jl [names]
#
# with `names` a comma-separated subset of the keys of CONFIGS; the default is the three the
# planner finds a feasible point for, rdb200, airy200 and davies200. A second argument `redraw`
# draws again from the serialised grid, which is what a change to the contour levels needs.

using RigorousPseudospectra
using CairoMakie
using LinearAlgebra
using Printf
import Serialization

BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "1")))

include(joinpath(@__DIR__, "pseudospectra_ops.jl"))

const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
const MAPDIR = joinpath(OUT, "maps")
const HOST = readchomp(`hostname -s`)
const FIGDIR = joinpath(@__DIR__, "figures")
const PAPERDIR = joinpath(homedir(), "Write", "KrylovPseudospectra", "figures")
foreach(mkpath, (MAPDIR, FIGDIR, PAPERDIR))

const INK = RGBf(0.15, 0.15, 0.14)
const MUTED = RGBf(0.42, 0.42, 0.40)
const NOBOUND = RGBAf(0.72, 0.72, 0.71, 1)   # grey: no certified bound at this point

function readcsv(path)
    lines = readlines(path)
    hdr = split(lines[1], ',')
    return [Dict(zip(hdr, split(l, ','))) for l in lines[2:end] if !isempty(strip(l))]
end
num(r, k) = parse(Float64, r[k])
pick(rows, name) = only(filter(r -> r["op"] == name && r["target"] == "10", rows))


# The configurations. `ng` is the number of grid points per side; `levels` the ε of the dense
# σ_min = ε contours, as fractions of ‖A‖.
#
# `zoom` is a box (xlo, xhi, ylo, yhi) for a second panel at the isolation discs, which on rdb200
# are of radius 2e-11 at their certified minimum and 0.26 to 0.47 at the largest radius the count
# still passes, against a box of half-width 36: invisible in the whole-plane panel.
const CONFIGS = Dict(
    "rdb200" => (; ng = 340, levels = [1e-3, 1e-2, 1e-1], discs = true,
        zoom = (-34.4, -28.0, -3.2, 3.2), ngz = 320),
    "airy200" => (; ng = 300, levels = [1e-3, 1e-2, 1e-1], discs = true, zoom = nothing,
        ngz = 0),
    "davies200" => (; ng = 300, levels = [1e-3, 1e-2, 1e-1], discs = true, zoom = nothing,
        ngz = 0),
    "grcar500" => (; ng = 280, levels = [1e-3, 1e-2, 1e-1], discs = false, zoom = nothing,
        ngz = 0),
    "kahan500" => (; ng = 240, levels = [1e-4, 1e-3, 1e-2], discs = false, zoom = nothing,
        ngz = 0),
    "convdiff_cheb200" => (; ng = 300, levels = [1e-3, 1e-2, 1e-1], discs = true, zoom = nothing,
        ngz = 0),
    "orrsom200" => (; ng = 300, levels = [1e-3, 1e-2, 1e-1], discs = true, zoom = nothing,
        ngz = 0),
    "landau400" => (; ng = 300, levels = [1e-3, 1e-2, 1e-1], discs = true, zoom = nothing,
        ngz = 0),
    "rdb800l" => (; ng = 200, levels = [1e-3, 1e-2, 1e-1], discs = true, zoom = nothing,
        ngz = 0),
    "nos3" => (; ng = 200, levels = [1e-3, 1e-2, 1e-1], discs = true, zoom = nothing,
        ngz = 0),
    "olm1000" => (; ng = 200, levels = [1e-3, 1e-2, 1e-1], discs = true, zoom = nothing,
        ngz = 0),
    "qh1484" => (; ng = 160, levels = [1e-3, 1e-2, 1e-1], discs = true, zoom = nothing,
        ngz = 0),
    "hatano2000" => (; ng = 160, levels = [1e-3, 1e-2, 1e-1], discs = true, zoom = nothing,
        ngz = 0),
    "rdb2048" => (; ng = 160, levels = [1e-3, 1e-2, 1e-1], discs = true, zoom = nothing,
        ngz = 0))

function build(name)
    o = ps_operator(name)
    op = LinOp(o.A)
    # R=<radius> KS=<ks> in the environment fix the radius and the frame (a row certified by
    # certify_feasible.jl) and bypass the planner; the far-block sweep is then of order 8 as there
    if haskey(ENV, "R")
        R = parse(Float64, ENV["R"])
        ks = parse(Int, ENV["KS"])
        t_frame = @elapsed ((Q, _) = two_sided_frame(o.matvec, o.adjoint, o.n, R; ks, Leig = 200,
            Lsv = min(o.n, 2ks + 60), T = op.T))
        t_split = @elapsed (sc = certify_split(op, Q; order = 8, target = R, compression = :none))
        @printf("%s: R = %.6g, ks = %d (given), dim Q = %d, gbar = %.6g\n", name, R, ks, sc.k, sc.gbar)
        floor_given(z) = resolvent_floor(sc, z; method = :svd).floor
        return (; o, op, sc, floor_at = floor_given, R, t_frame, t_split, plan = nothing,
            absl1 = maximum(abs, eigvals(Matrix(o.A))))
    end
    # the radius and the frame are the planner's, not a radius read off a dense eigendecomposition
    t_plan = @elapsed (pl = plan_split(op, o.matvec, o.adjoint; mmax = 12, nR = 6,
        amargin = 1.5, Leig = 200))
    if pl.best === nothing
        @printf("%s: the planner found no feasible point over %d grid points, %d matvecs, %.1f s",
            name, length(pl.rows), pl.nmatvec, t_plan)
        isempty(pl.predicted) ? println() :
            @printf("; the grid says dim Q ≈ %.0f would be needed at R = %.6g\n",
                pl.predicted[1].dimQ_needed, pl.predicted[1].R)
        return nothing
    end
    b = pl.best
    R = b.R
    @printf("%s: planner %d grid points, %d feasible, %d matvecs, %.1f s -> R = %.6g, ks = %d, dim Q = %d, %d outside\n",
        name, length(pl.rows), length(pl.feasible), pl.nmatvec, t_plan, R, b.ks, b.dimQ, b.nout)
    plan_table(pl)
    t_frame = @elapsed ((Q, finfo) = two_sided_frame(o.matvec, o.adjoint, o.n, R; ks = b.ks,
        Leig = 200, Lsv = min(o.n, 2b.ks + 60), T = op.T))
    # sigma11 by one verified SVD of the compression pencil per z (sec:vbd), sigma22 = |z| - gbar;
    # the compression is certified by Miyajima-Rump for the counts
    t_split = @elapsed (sc = certify_split(op, Q; target = R, compression = :full))
    floor_at(z) = resolvent_floor(sc, z; method = :svd).floor
    return (; o, op, sc, floor_at, R, t_frame, t_split, plan = pl.best,
        absl1 = maximum(abs, pl.lambda))
end

"The certified floor and the dense σ_min on the grid `xs × ys`."
function evaluate(floor_at, A, xs, ys)
    F = zeros(length(xs), length(ys))
    D = zeros(length(xs), length(ys))
    t_cert = @elapsed Threads.@threads for j in eachindex(ys)
        for i in eachindex(xs)
            F[i, j] = floor_at(xs[i] + im * ys[j])
        end
    end
    # `svdvals!` overwrites its argument, so the whole of `Aj` is refilled at every point and not
    # only the diagonal
    t_dense = @elapsed Threads.@threads for j in eachindex(ys)
        Aj = Matrix{ComplexF64}(undef, size(A))
        for i in eachindex(xs)
            z = xs[i] + im * ys[j]
            copyto!(Aj, A)
            @inbounds for d in 1:size(A, 1)
                Aj[d, d] -= z
            end
            D[i, j] = svdvals!(Aj)[end]
        end
    end
    return (; F, D, t_cert, t_dense)
end

function gridmap(b; ng)
    A = Matrix(b.o.A)
    ev = eigvals(A)
    half = max(1.15 * b.R, 1.08 * maximum(abs, ev))
    xs = collect(range(-half, half; length = ng))
    ys = copy(xs)
    e = evaluate(b.floor_at, A, xs, ys)
    return (; xs, ys, e.F, e.D, ev, half, e.t_cert, e.t_dense, A)
end

"""
The largest radius in a ten-point geometric sequence up to the limit prop:count allows (inside
`{|z| > R}` and half the distance to the nearest other disc centre) at which the count still
passes, which is what the "max" rows of `discs.csv` of 2026-09-12 hold; the disc returned by
`certify_exterior` is the smallest of the sequence and is too small to draw.
"""
function widest_disc(b, R, centres, i; nradii = 10, nθ = 64, maxdepth = 6)
    c = centres[i].center
    rmax = abs(c) - R
    for (j, d) in enumerate(centres)
        j == i && continue
        rmax = min(rmax, abs(c - d.center) / 2)
    end
    rmin = centres[i].radius * (1 + 1e-6) + 1e-300
    best = centres[i].radius
    rmin < rmax || return best
    for r in reverse(exp.(range(log(rmin), log(rmax * (1 - 1e-6)); length = nradii)))
        c2, okc, ck = count_disc(b.sc, c, r; nθ, maxdepth, method = :svd)
        if okc && c2 == centres[i].count
            return r
        end
    end
    return best
end

function run_one(name)
    cfg = CONFIGS[name]
    b = build(name)
    b === nothing && return nothing
    g = gridmap(b; cfg.ng)
    sc = b.sc
    @printf("%s: n = %d, dim Q = %d, R = %.6g, ḡ = %.6g at order %d, γ1γ2 = %.4g, frame %.1f s, split %.1f s\n",
        name, sc.n, sc.k, b.R, sc.gbar, sc.far === nothing ? -1 : sc.far.m,
        sc.γ1 * sc.γ2, b.t_frame, b.t_split)
    @printf("   grid %d × %d over [%.4g, %.4g]², certified %.1f s, dense %.1f s\n",
        cfg.ng, cfg.ng, -g.half, g.half, g.t_cert, g.t_dense)

    # soundness: the certified bound must never exceed the dense σ_min
    bounded = g.F .> 0
    nb = count(bounded)
    strict = count(i -> bounded[i] && g.F[i] > g.D[i], eachindex(g.F))
    tol = 1e-10 * sc.abar
    loose = count(i -> bounded[i] && g.F[i] > g.D[i] + tol, eachindex(g.F))
    ratio = [g.F[i] / g.D[i] for i in eachindex(g.F) if bounded[i]]
    @printf("   bounded at %d of %d points (%.3f); bound > dense σ_min at %d points strictly, ",
        nb, length(g.F), nb / length(g.F), strict)
    @printf("%d beyond %.3g; bound/σ_min min %.4f median %.4f max %.4f\n", loose, tol,
        isempty(ratio) ? NaN : minimum(ratio),
        isempty(ratio) ? NaN : sort(ratio)[cld(length(ratio), 2)],
        isempty(ratio) ? NaN : maximum(ratio))

    discs = NamedTuple[]
    cnt = -1
    if cfg.discs && !haskey(ENV, "R")
        t = @elapsed (ext = certify_exterior(sc, b.R; nθ = 256, maxdepth = 8, nradii = 10,
            method = :svd))
        cnt = ext.count
        good = [d for d in ext.discs if isfinite(d.radius)]
        tw = @elapsed (wide = [widest_disc(b, b.R, good, i) for i in eachindex(good)])
        discs = [(; d.center, d.radius, rmax = w, d.count) for (d, w) in zip(good, wide)]
        @printf("   cor:agree: count %d, %d discs, all closed %s (%.1f s + %.1f s for the widest)\n",
            ext.count, length(discs), ext.ok, t, tw)
        e = ext.exterior
        @printf("      circle cover %s over %d evaluations, floor %.4e, min σ11 %.4e, min σ22 %.4e; region (R > ḡ = %.6g) %s\n",
            e.ok, e.nevals, e.floor, e.minσ11, e.minσ22, Float64(sc.gbar), e.region_ok)
        for d in discs
            @printf("      centre %+.10f %+.10fi  r_min %.4e  r_max %.4e  count %d\n",
                real(d.center), imag(d.center), d.radius, d.rmax, d.count)
        end
        # a centre at which no radius on the ladder closed: `discs` keeps only the ones that did,
        # so without this the log shows an empty list and not the centres it failed at
        for d in ext.discs
            isfinite(d.radius) && continue
            c = d.center
            rmax = min(abs(c) - b.R,
                minimum([abs(c - e.center) / 2 for e in ext.discs if e.center != c]; init = Inf))
            @printf("      centre %+.10f %+.10fi at |z| = %.6g: no radius closed, ladder up to r_max = %.4e (|c| − R = %.4e, ḡ = %.6g)\n",
                real(c), imag(c), abs(c), rmax, abs(c) - b.R, Float64(sc.gbar))
        end
    end

    # the zoom panel at the discs
    z = nothing
    if cfg.zoom !== nothing
        xlo, xhi, ylo, yhi = cfg.zoom
        nzy = max(40, round(Int, cfg.ngz * (yhi - ylo) / (xhi - xlo)))
        zxs = collect(range(xlo, xhi; length = cfg.ngz))
        zys = collect(range(ylo, yhi; length = nzy))
        e = evaluate(b.floor_at, g.A, zxs, zys)
        zb = e.F .> 0
        zstrict = count(i -> zb[i] && e.F[i] > e.D[i], eachindex(e.F))
        zloose = count(i -> zb[i] && e.F[i] > e.D[i] + tol, eachindex(e.F))
        z = (; xs = zxs, ys = zys, e.F, e.D, nb = count(zb), strict = zstrict, loose = zloose,
            e.t_cert, e.t_dense)
        @printf("   zoom %d × %d over [%.4g, %.4g] × [%.4g, %.4g]: bounded at %d of %d, ",
            cfg.ngz, nzy, xlo, xhi, ylo, yhi, z.nb, length(e.F))
        @printf("bound > dense σ_min at %d points (%.1f s + %.1f s)\n", zstrict, e.t_cert,
            e.t_dense)
    end

    res = (; name, n = sc.n, k = sc.k, R = b.R, gbar = sc.gbar, abar = sc.abar, γ1 = sc.γ1,
        γ2 = sc.γ2, ng = cfg.ng, half = g.half, xs = g.xs, ys = g.ys, F = g.F, D = g.D,
        ev = g.ev, nb, strict, loose, tol, t_cert = g.t_cert, t_dense = g.t_dense,
        t_frame = b.t_frame, t_split = b.t_split, discs, count = cnt, levels = cfg.levels, z)
    Serialization.serialize(joinpath(MAPDIR, "map_$(name).$(HOST).jls"), res)
    return res
end

const DISCCOL = RGBf(1.0, 0.85, 0.0)
const EVCOL = RGBf(0.85, 0.15, 0.1)

newaxis(fig, pos) = Axis(fig[pos...]; xlabel = "Re z", ylabel = "Im z", aspect = DataAspect(),
    xlabelcolor = INK, ylabelcolor = INK, xticklabelcolor = MUTED, yticklabelcolor = MUTED,
    spinewidth = 0.8, leftspinecolor = MUTED, bottomspinecolor = MUTED,
    rightspinecolor = MUTED, topspinecolor = MUTED, xgridvisible = false, ygridvisible = false)

function panel!(ax, xs, ys, F, D, r, cr, levs; rings = true, discs = NamedTuple[])
    L = [f > 0 ? log10(f) : NaN for f in F]
    hm = heatmap!(ax, xs, ys, L; colormap = :viridis, colorrange = cr, nan_color = NOBOUND)
    contour!(ax, xs, ys, D; levels = levs, color = (:black, 0.85), linewidth = 1.2)
    θ = range(0, 2π; length = 600)
    if rings
        lines!(ax, r.R .* cos.(θ), r.R .* sin.(θ); color = :white, linewidth = 2.0)
        lines!(ax, r.gbar .* cos.(θ), r.gbar .* sin.(θ); color = :white, linewidth = 1.6,
            linestyle = :dash)
    else
        lines!(ax, r.R .* cos.(θ), r.R .* sin.(θ); color = :white, linewidth = 2.0)
    end
    scatter!(ax, real.(r.ev), imag.(r.ev); marker = :xcross, color = EVCOL, markersize = 7,
        strokewidth = 0)
    for d in discs
        lines!(ax, real(d.center) .+ d.rmax .* cos.(θ), imag(d.center) .+ d.rmax .* sin.(θ);
            color = DISCCOL, linewidth = 1.8)
    end
    return hm
end

function draw(r)
    fin = filter(isfinite, [f > 0 ? log10(f) : NaN for f in r.F])
    cr = isempty(fin) ? (0.0, 1.0) : (minimum(fin), maximum(fin))
    levs = sort(r.levels .* r.abar)
    haszoom = r.z !== nothing
    fig = Figure(size = haszoom ? (1060, 470) : (620, 540), fontsize = 13,
        backgroundcolor = :white)
    ax = newaxis(fig, (1, 1))
    hm = panel!(ax, r.xs, r.ys, r.F, r.D, r, cr, levs)
    xlims!(ax, -r.half, r.half)
    ylims!(ax, -r.half, r.half)
    if haszoom
        Label(fig[1, 1, TopLeft()], "a", font = :bold, color = INK, padding = (0, 6, 4, 0))
        ax2 = newaxis(fig, (1, 2))
        panel!(ax2, r.z.xs, r.z.ys, r.z.F, r.z.D, r, cr, levs; discs = r.discs)
        xlims!(ax2, r.z.xs[1], r.z.xs[end])
        ylims!(ax2, r.z.ys[1], r.z.ys[end])
        Label(fig[1, 2, TopLeft()], "b", font = :bold, color = INK, padding = (0, 6, 4, 0))
        Colorbar(fig[1, 3], hm; label = "log₁₀ of the certified lower bound on σ_min(A − zI)",
            labelcolor = INK, ticklabelcolor = MUTED, labelsize = 12, ticklabelsize = 11)
    else
        Colorbar(fig[1, 2], hm; label = "log₁₀ of the certified lower bound on σ_min(A − zI)",
            labelcolor = INK, ticklabelcolor = MUTED, labelsize = 12, ticklabelsize = 11)
    end
    for ext in ("pdf", "png")
        p = joinpath(FIGDIR, "map_$(r.name).$ext")
        save(p, fig; px_per_unit = ext == "png" ? 2 : 1)
        cp(p, joinpath(PAPERDIR, "map_$(r.name).$ext"); force = true)
    end
    println("   wrote ", joinpath(FIGDIR, "map_$(r.name).pdf"), " (levels ",
        join((@sprintf("%.4g", l) for l in levs), ", "), ")")
end

# the operators the planner finds a point for; grcar500 and kahan500 are in CONFIGS but declined,
# and a map of a radius the method refuses to choose would say nothing
names = isempty(ARGS) ? ["rdb200", "rdb800l", "rdb2048", "nos3", "olm1000", "qh1484", "grcar500",
    "kahan500", "davies200", "airy200", "orrsom200", "landau400", "hatano2000",
    "convdiff_cheb200"] : split(ARGS[1], ',')
redraw = length(ARGS) >= 2 && ARGS[2] == "redraw"
@info "threads" julia = Threads.nthreads() blas = BLAS.get_num_threads()
for nm in names
    name = String(nm)
    # `redraw` draws again from the serialised grid, which is what a change to the contour levels
    # needs; nothing is recomputed and the levels come from CONFIGS.
    r = redraw ?
        merge(Serialization.deserialize(joinpath(MAPDIR, "map_$(name).$(HOST).jls")),
            (; levels = CONFIGS[name].levels)) :
        run_one(name)
    # the planner declines an operator no frame of the sizes it will try can reach; there is no
    # radius to draw a map at, and the reason has already been printed by `build`
    r === nothing && continue
    draw(r)
    flush(stdout)
end
