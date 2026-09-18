# The exterior pseudospectrum map of the Belousov-Zhabotinsky operator A = N∘L at 2^12, at one
# noise size ξ, on the best frame the planner can find for that ξ.
#
# The frame is not fixed by hand. `plan_split` (src/planner.jl, the driver `bench/bz/plancert.jl`
# is the same call) is run over a grid of (R, m, dim Q) in floating point; if it finds a feasible
# point, that point is certified, and if it does not, the deepest frame on the grid is certified at
# the radius and order with the largest margin, and the run is labelled as such. This matters at
# the small noise sizes: a frame of dim Q = 101 gives ḡ = 1.44 at ξ = 1e-3 and 2.19 at ξ = 1e-4,
# so the map drawn on it would be a picture of a frame too shallow for the operator.
#
# The box is the larger of the unit disc with a margin and a disc of radius 1.3 ḡ, chosen per
# panel, so that the region {|z| > ḡ} where the certificate can say anything is inside the panel.
#
# The certified lower bound on σ_min(A − zI) is evaluated on a full grid whose resolution is
# chosen from a measurement of the cost of one point, so that the grid takes about `budget`
# seconds; each point is one verified SVD of a dim Q × dim Q ball pencil and the cost grows like
# dim Q³. The dense σ_min is evaluated on a coarse subgrid only, one `svdvals` of a 4096 × 4096
# complex matrix per point, and only for Im z ≥ 0, since A is real and
# σ_min(A − z̄I) = σ_min(A − zI).
#
# This script only computes and serialises; the figures are drawn by `bench/draw_bz_map.jl`, which
# runs under `bench/Project.toml` (the one with CairoMakie).
#
#   OPENBLAS_NUM_THREADS=24 julia -t 24 --project=. bench/bz/paper_map_bz.jl [xi] [budget_s] [ncheck]

include(joinpath(@__DIR__, "bzcommon.jl"))
using Serialization

const Ξ = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1e-1
const KE = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
const BUDGET = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 900.0
const NCHECK = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 15
const NGMIN, NGMAX = 120, 400

# The dense comparison holds one k × k complex matrix and takes its singular values at every check
# point, so it is available at 2^12 and not above it; past that the eigenvalues drawn under the map
# are the leading Ritz values instead.
const DENSEMAX = 4096

const MAPDIR = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra",
    "maps")
mkpath(MAPDIR)

k = 2^KE
# a frame of ks vectors gives dim Q of about 2ks + 1, so the grid stops well below the space
const KSS = filter(ks -> 2ks + 60 < k, [20, 50, 100, 150, 300, 600, 900, 1200])
const TAG = "bz_k$(KE)_xi$(Ξ)"

L, _ = build_L(k)
# `build_L` reads a cache but never writes one, and 2^12 was not cached; write it once so that the
# four noise sizes do not each re-assemble the Ulam matrix
let cache = joinpath(OUT, "L_$(KE).jls")
    isfile(cache) || Serialization.serialize(cache, L)
end
l = noise_halfwidth(Ξ, k)
Amul, Aadj, η, oi = make_operator(L, l)
op = RP.LinOp(Amul, Aadj, k; epsA = 0.0, etaA = η)

# --- the planning phase ----------------------------------------------------------------------------
t_plan = @elapsed (pl = RP.plan_split(op, Amul, Aadj; kss = KSS, mmax = 16, nR = 7,
    amargin = 1.5, Leig = 200, T = Float64))
@printf("ξ = %.0e, l = %d: planner %d grid points, %d feasible, %d matvecs, %.1f s\n",
    Ξ, l, length(pl.rows), length(pl.feasible), pl.nmatvec, pl.time.total)
RP.plan_table(pl)

local R, ksuse, muse, how
if pl.best !== nothing
    b = pl.best
    R, ksuse, muse = b.R, b.ks, b.m
    how = "the planner's feasible point"
else
    # the deepest frame on the grid, at the radius and order with the largest margin there
    kmaxrow = maximum(r -> r.ks, pl.rows)
    deep = [r for r in pl.rows if r.ks == kmaxrow]
    b = sort(deep; by = r -> -r.margin)[1]
    R, ksuse, muse = b.R, b.ks, b.m
    how = "the deepest frame on the planner's grid (no feasible point)"
end
@printf("   frame: %s; R = %.10g, ks = %d, m = %d, planner dim Q = %d, margin %.4g\n",
    how, R, ksuse, muse, b.dimQ, b.margin)

# --- the certified split at that point ---------------------------------------------------------------
t_frame = @elapsed ((Q, finfo) = RP.two_sided_frame(Amul, Aadj, k, R; ks = ksuse,
    Leig = pl.params.Leig, Lsv = pl.params.Lsv, droptol = pl.params.droptol, T = Float64))
t_split = @elapsed (sc = RP.certify_split(op, Q; target = R, order = muse,
    compression = :split))
@printf("   certified: dim Q = %d, α ≤ %.3e, ḡ ≤ %.6g, ā ≤ %.6g, γ1 ≤ %.6g, γ2 ≤ %.6g, γ1γ2 = %.6g\n",
    sc.k, sc.α, sc.gbar, sc.abar, sc.γ1, sc.γ2, sc.γ1 * sc.γ2)
@printf("   R − ḡ = %.6g, the gate needs γ1γ2 < R(R − ḡ) = %.6g: %s (frame %.1f s, split %.1f s)\n",
    R - sc.gbar, R * (R - sc.gbar), sc.γ1 * sc.γ2 < R * (R - sc.gbar), t_frame, t_split)

t_ext = @elapsed (ext = RP.certify_exterior(sc, R; nθ = 64, maxdepth = 2, method = :svd))
discs = [(; d.center, d.radius, d.count) for d in ext.discs if isfinite(d.radius)]
@printf("   cor:agree: hypothesis %s, region %s, count %s, %d discs, all closed %s (%.1f s)\n",
    ext.exterior.ok, ext.exterior.region_ok,
    ext.exterior.ok ? string(ext.count) : "none", length(discs), ext.ok, t_ext)
for d in discs
    @printf("      disc at %+.10f %+.10fi, radius %.4e, count %d\n", real(d.center),
        imag(d.center), d.radius, d.count)
end

comps = [(; C.center, C.radius, C.count) for C in RP._comps(sc)]
@printf("   the certified enclosure of G11 has %d components, %d of them outside R\n",
    length(comps), count(c -> abs(c.center) - c.radius > R, comps))

# --- the certified grid, at a resolution measured rather than guessed -------------------------------
half = max(1.15, 1.3 * sc.gbar)
probe = [half * 0.7 * cis(2π * t / 12) for t in 0:11]
RP.resolvent_floor(sc, probe[1])
tprobe = @elapsed for z in probe
    RP.resolvent_floor(sc, z)
end
percost = tprobe / length(probe)
ng = clamp(round(Int, sqrt(BUDGET * Threads.nthreads() / percost)), NGMIN, NGMAX)
@printf("   one certified point costs %.4f s at %d threads; grid %d × %d, estimated %.0f s\n",
    percost, Threads.nthreads(), ng, ng, ng^2 * percost / Threads.nthreads())

xs = collect(range(-half, half; length = ng))
ys = copy(xs)
F = zeros(ng, ng)
t_cert = @elapsed Threads.@threads for j in 1:ng
    for i in 1:ng
        F[i, j] = RP.resolvent_floor(sc, xs[i] + im * ys[j]).floor
    end
end
@printf("   certified grid %d × %d over [%.4g, %.4g]² in %.1f s, bounded at %d of %d points (%.4f)\n",
    ng, ng, -half, half, t_cert, count(>(0), F), length(F), count(>(0), F) / length(F))

# a zoom panel at the first isolation disc away from the real axis, where there is one
zc = nothing
zxs = Float64[]; zys = Float64[]; Fz = zeros(0, 0); zhalf = 0.0; t_zoom = 0.0
let off = [d for d in discs if abs(imag(d.center)) > d.radius]
    if !isempty(off)
        zc = off[1].center
        zhalf = max(5 * off[1].radius, 0.02)
        nz = 240
        zxs = collect(range(real(zc) - zhalf, real(zc) + zhalf; length = nz))
        zys = collect(range(imag(zc) - zhalf, imag(zc) + zhalf; length = nz))
        Fz = zeros(nz, nz)
        t_zoom = @elapsed Threads.@threads for j in 1:nz
            for i in 1:nz
                Fz[i, j] = RP.resolvent_floor(sc, zxs[i] + im * zys[j]).floor
            end
        end
        @printf("   zoom grid %d × %d about %+.7f %+.7fi, half-width %.4g, in %.1f s\n",
            nz, nz, real(zc), imag(zc), zhalf, t_zoom)
    end
end

# --- the dense check on a coarse subgrid, upper half plane ------------------------------------------
const DENSE = NCHECK > 0 && k <= DENSEMAX
xc = collect(range(-half, half; length = DENSE ? NCHECK : 0))
yc = collect(range(0.0, half; length = DENSE ? cld(NCHECK, 2) : 0))
Dc = zeros(length(xc), length(yc))
Fc = zeros(length(xc), length(yc))
local ev, t_dense
if DENSE
    A = dense_A(L, l)
    ev = eigvals(A)
    Az = Matrix{ComplexF64}(undef, k, k)
    t_dense = @elapsed for j in eachindex(yc), i in eachindex(xc)
        z = xc[i] + im * yc[j]
        copyto!(Az, A)
        @inbounds for d in 1:k
            Az[d, d] -= z
        end
        Dc[i, j] = svdvals!(Az)[end]
        Fc[i, j] = RP.resolvent_floor(sc, z).floor
    end
else
    ev = RP._ritz_leading(Amul, k, Float64; L = 200)
    t_dense = 0.0
    @printf("   no dense check at k = %d (above %d); the points drawn are the leading Ritz values\n",
        k, DENSEMAX)
end
bnd = Fc .> 0
strict = count(i -> bnd[i] && Fc[i] > Dc[i], eachindex(Fc))
ratio = [Fc[i] / Dc[i] for i in eachindex(Fc) if bnd[i]]
@printf("   dense check on %d points (%d with a bound): bound > dense σ_min at %d; ",
    length(Fc), count(bnd), strict)
@printf("bound/σ_min min %.4f median %.4f max %.4f  (%.1f s)\n",
    isempty(ratio) ? NaN : minimum(ratio),
    isempty(ratio) ? NaN : sort(ratio)[cld(length(ratio), 2)],
    isempty(ratio) ? NaN : maximum(ratio), t_dense)

res = (; name = TAG, n = k, k = sc.k, ξ = Ξ, l, R, ks = ksuse, m = muse, how,
    plan_best = pl.best, plan_rows = pl.rows, plan_feasible = length(pl.feasible),
    plan_predicted = pl.predicted, nmatvec_plan = pl.nmatvec, t_plan,
    gbar = sc.gbar, abar = sc.abar, α = sc.α, γ1 = sc.γ1, γ2 = sc.γ2,
    ng, half, xs, ys, F, ev, discs, comps, percost,
    count = ext.exterior.ok ? ext.count : -1, ext_ok = ext.ok, hyp_ok = ext.exterior.ok,
    region_ok = ext.exterior.region_ok, xc, yc, Dc, Fc, strict, t_cert, t_dense, t_frame,
    t_split, t_ext, zxs, zys, Fz, zc, zhalf, t_zoom)
let out = joinpath(MAPDIR, "map_$(TAG).$(gethostname()).jls")
    Serialization.serialize(out, res)
    println("wrote ", out)
end
