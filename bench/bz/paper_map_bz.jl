# The exterior pseudospectrum map of the Belousov-Zhabotinsky operator A = N∘L at 2^KE cells and
# noise ξ, on the frame that experiment 5 (bench/bz/exp5_bz.jl) certified at the radius with N
# eigenvalues outside: R and ks are read from exp5_bz.k<KE>.xi<ξ>.<host>.csv, the frame is rebuilt
# as there, the split certified as there (order 8, compression :full), and certify_exterior run with
# the same cover (64 initial arcs, maxdepth 64) and the same ladder of 12 disc radii, so that the
# discs drawn are those of the experiment.
#
# The box is the larger of the unit disc with a margin and a disc of radius 1.3 ḡ, so that the
# region {|z| > ḡ} where the certificate can say anything is inside it.
#
# The certified lower bound on σ_min(A − zI) is evaluated on a full grid whose resolution is
# chosen from a measurement of the cost of one point, so that the grid takes about `budget`
# seconds; each point is one verified SVD of a dim Q × dim Q ball pencil. The dense σ_min is
# evaluated on a coarse subgrid only, for 2^KE <= 4096 and Im z ≥ 0, since A is real and
# σ_min(A − z̄I) = σ_min(A − zI).
#
# This script only computes and serialises, to paper_map_bz.bz_k<KE>_xi<ξ>_N<N>.<host>.jls; the
# figures are drawn by `bench/draw_bz_map.jl`, which runs under `bench/Project.toml`.
#
#   OPENBLAS_NUM_THREADS=4 julia -t 16 --project=bench/bz bench/bz/paper_map_bz.jl xi KE N [budget_s] [ncheck]

include(joinpath(@__DIR__, "bzcommon.jl"))
include(joinpath(@__DIR__, "..", "datadir.jl"))
using Serialization

const Ξ = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1e-1
const KE = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
const NOUT = parse(Int, ARGS[3])
const BUDGET = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 900.0
const NCHECK = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 15
const HOST = readchomp(`hostname -s`)
const NGMIN, NGMAX = 120, 400

# The dense comparison holds one k × k complex matrix and takes its singular values at every check
# point, so it is available at 2^12 and not above it; past that the eigenvalues drawn under the map
# are the leading Ritz values instead.
const DENSEMAX = 4096

const MAPDIR = joinpath(PSOUT, "maps")
mkpath(MAPDIR)

k = 2^KE
const TAG = "bz_k$(KE)_xi$(Ξ)_N$(NOUT)"

L, _ = build_L(k)
# `build_L` reads a cache but never writes one, and 2^12 was not cached; write it once so that the
# four noise sizes do not each re-assemble the Ulam matrix
let cache = joinpath(OUT, "L_$(KE).jls")
    isfile(cache) || Serialization.serialize(cache, L)
end
l = noise_halfwidth(Ξ, k)
Amul, Aadj, op, oi = bz_linop(L, l)
η = oi.eta_rel

# --- the radius and the frame of experiment 5 ----------------------------------------------------------
exp5 = joinpath(PSOUT, "bz_exp5",
    "exp5_bz.k$(KE).xi$(Ξ).$(HOST).csv")
lines = readlines(exp5)
hdr = split(lines[1], ',')
rows = [Dict(zip(hdr, split(x, ','))) for x in lines[2:end]]
row = only(r for r in rows if parse(Int, r["N"]) == NOUT)
row["cover_ok"] == "true" || error("experiment 5 did not certify N = $NOUT at 2^$KE, ξ = $Ξ")
R = parse(Float64, row["R"])
ksuse = parse(Int, row["ks"])
@printf("ξ = %.0e, l = %d, N = %d: R = %.10g, ks = %d from %s\n", Ξ, l, NOUT, R, ksuse, basename(exp5))

# --- the certified split at that point ---------------------------------------------------------------
t_frame = @elapsed ((Q, finfo) = RP.two_sided_frame(Amul, Aadj, k, R; ks = ksuse, Leig = 200,
    Lsv = min(k, 2max(ksuse, 1) + 60), T = Float64))
t_split = @elapsed (sc = RP.certify_split(op, Q; target = R, order = 8, compression = :full))
@printf("   certified: dim Q = %d, α ≤ %.3e, ḡ ≤ %.6g, γ1 ≤ %.6g, γ2 ≤ %.6g, γ1γ2 = %.6g (frame %.1f s, split %.1f s)\n",
    sc.k, sc.α, sc.gbar, sc.γ1, sc.γ2, sc.γ1 * sc.γ2, t_frame, t_split)

t_ext = @elapsed (ext = RP.certify_exterior(sc, R; nθ = 64, nradii = 12, maxdepth = 64,
    method = :svd))
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
RP.resolvent_floor(sc, probe[1]; method = :svd)
tprobe = @elapsed for z in probe
    RP.resolvent_floor(sc, z; method = :svd)
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
        F[i, j] = RP.resolvent_floor(sc, xs[i] + im * ys[j]; method = :svd).floor
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
                Fz[i, j] = RP.resolvent_floor(sc, zxs[i] + im * zys[j]; method = :svd).floor
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
        Fc[i, j] = RP.resolvent_floor(sc, z; method = :svd).floor
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

res = (; name = TAG, n = k, k = sc.k, ξ = Ξ, l, R, ks = ksuse, m = 8, N = NOUT,
    gbar = sc.gbar, abar = sc.abar, α = sc.α, γ1 = sc.γ1, γ2 = sc.γ2,
    ng, half, xs, ys, F, ev, discs, comps, percost,
    count = ext.exterior.ok ? ext.count : -1, ext_ok = ext.ok, hyp_ok = ext.exterior.ok,
    region_ok = ext.exterior.region_ok, xc, yc, Dc, Fc, strict, t_cert, t_dense, t_frame,
    t_split, t_ext, zxs, zys, Fz, zc, zhalf, t_zoom)
let out = joinpath(MAPDIR, "paper_map_bz.$(TAG).$(HOST).jls")
    Serialization.serialize(out, res)
    println("wrote ", out)
end
