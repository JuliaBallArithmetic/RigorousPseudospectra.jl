# The exterior pseudospectrum map of the Belousov-Zhabotinsky operator A = N∘L, with the frame and
# the far block as before and the triangular route of sec:triangular inside the compression.
#
# The two halves of the certificate are separated on purpose. Outside the frame nothing changes:
# `plan_split` chooses (R, ks, m), `two_sided_frame` builds Q, and `certify_split` with
# `compression = :none` returns ḡ, ā, γ1, γ2 and α without peeling anything, so the estimates on
# the far block are those the earlier runs already paid for. Inside the frame the compression is
# put in Schur form and cut once, `triangular_split`, and σ11 comes from one verified singular value
# of the leading block rather than from the peel.
#
# The floor at z is then the two-block assembly of thm:assembly with
#
#     σ11(z) = sigma_tri_lo(ts, z),   σ22(z) ≥ max(|z| − ḡ, the power bound),
#
# and the circle |z| = R is covered by `cover_tri`, which also returns the count.
#
# Because the frame is the expensive half, it is memoised: the ball compression, its Gram and the
# frame quantities are written to gt_k<KE>_xi<XI>_ks<ks>_m<m>.<host>.jls and read back on a rerun.
#
#   OPENBLAS_NUM_THREADS=1 julia -t 32 --project=. bench/bz/tri_map_bz.jl XI KE [BUDGET] [NCHECK]

using RigorousPseudospectra
using LinearAlgebra, Printf
import Serialization
const RP = RigorousPseudospectra
include(joinpath(@__DIR__, "bzcommon.jl"))

const Ξ = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1e-1
const KE = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
const BUDGET = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 900.0
const NCHECK = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 15
const NGMIN, NGMAX = 120, 400
const DENSEMAX = 4096

const MAPDIR = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra",
    "maps")
mkpath(MAPDIR)
const HOST = gethostname()
const TAG = "tribz_k$(KE)_xi$(Ξ)"

k = 2^KE
L, _ = build_L(k)
let cache = joinpath(OUT, "L_$(KE).jls")
    isfile(cache) || Serialization.serialize(cache, L)
end
l = noise_halfwidth(Ξ, k)
Amul, Aadj, η, oi = make_operator(L, l)
op = RP.LinOp(Amul, Aadj, k; epsA = 0.0, etaA = η)
@printf("=== %s on %s: k = %d, ξ = %.0e, l = %d, %d threads\n", TAG, HOST, k, Ξ, l,
    Threads.nthreads())
flush(stdout)

# --- the planning phase, unchanged ------------------------------------------------------------------
const KSS = filter(ks -> 2ks + 60 < k, [20, 50, 100, 150, 300, 600, 900, 1200])
t_plan = @elapsed (pl = RP.plan_split(op, Amul, Aadj; kss = KSS, mmax = 16, nR = 7,
    amargin = 1.5, Leig = 200, T = Float64))
@printf("planner: %d grid points, %d feasible, %d matvecs, %.1f s\n", length(pl.rows),
    length(pl.feasible), pl.nmatvec, pl.time.total)
RP.plan_table(pl)
flush(stdout)

local R, ksuse, muse, how
if pl.best !== nothing
    b = pl.best
    R, ksuse, muse, how = b.R, b.ks, b.m, "the planner's feasible point"
else
    kmaxrow = maximum(r -> r.ks, pl.rows)
    b = sort([r for r in pl.rows if r.ks == kmaxrow]; by = r -> -r.margin)[1]
    R, ksuse, muse, how = b.R, b.ks, b.m, "the deepest frame on the grid (no feasible point)"
end
@printf("frame: %s; R = %.10g, ks = %d, m = %d, planner dim Q = %d, margin %.4g\n",
    how, R, ksuse, muse, b.dimQ, b.margin)
flush(stdout)

# --- the frame and the far block, memoised -----------------------------------------------------------
memo = joinpath(OUT, "gt_k$(KE)_xi$(Ξ)_ks$(ksuse)_m$(muse).$(HOST).jls")
local sc, t_frame, t_split
if isfile(memo)
    d = Serialization.deserialize(memo)
    sc = d.sc
    t_frame, t_split = d.t_frame, d.t_split
    println("frame read back from ", memo)
else
    t_frame = @elapsed ((Q, finfo) = RP.two_sided_frame(Amul, Aadj, k, R; ks = ksuse,
        Leig = 200, Lsv = min(k, 2ksuse + 60), T = Float64))
    t_split = @elapsed (sc = RP.certify_split(op, Q; target = R, order = muse,
        compression = :none))
    Serialization.serialize(memo, (; KE, Ξ, ks = ksuse, m = muse, host = HOST, R, sc,
        t_frame, t_split))
    println("frame written to ", memo)
end
@printf("dim Q = %d, α = %.4e, ḡ = %.6g, ā = %.6g, γ1 = %.4e, γ2 = %.4e (frame %.1f s, split %.1f s)\n",
    sc.k, sc.α, sc.gbar, sc.abar, sc.γ1, sc.γ2, t_frame, t_split)
@printf("the outer gate γ1γ2 < R(R − ḡ): %.6g against %.6g, %s\n", sc.γ1 * sc.γ2,
    R * (R - sc.gbar), sc.γ1 * sc.γ2 < R * (R - sc.gbar))
flush(stdout)

# --- the triangular route inside the compression -----------------------------------------------------
t_tri = @elapsed ts = RP.triangular_split(sc.G, sc.M, R; α = sc.α, factor = 2.0)
if ts === nothing
    println("triangular_split declined: no admissible cut brings the trailing block below R")
    exit(0)
end
@printf("triangular split: cut k_EN = %d of %d, γ1 = %.4e, γ2 = %.4e, ḡ₂₂ = %.4e, ‖R2‖ = %.2e (%.1f s)\n",
    ts.c, ts.k, ts.γ1, ts.γ2, ts.gbar, ts.nR2, t_tri)
flush(stdout)

# σ11 from the triangular route, σ22 from the far block of the frame, assembled as in thm:assembly
σ22lo(z) = max(RP.farshort_lo(z, sc.gbar), RP.farpower_lo(z, sc.gbars_power))
function floor_at(z)
    s11 = RP.sigma_tri_lo(ts, z)
    s11 > 0 || return 0.0
    s22 = σ22lo(z)
    s22 > 0 || return 0.0
    f1, _ = RP.assembly_floor(s11, s22, sc.γ1, sc.γ2)
    f2, _ = RP.blockdom_floor(s11, s22, sc.γ1, sc.γ2)
    return max(f1, f2)
end

t_cov = @elapsed cv = RP.cover_tri(ts, R; nθ = 64, η = 0.5, maxpoints = 20000)
@printf("cover of |z| = R on the compression: ok = %s, floor %.4e, count %s, %d points, %d bisections (%.1f s)\n",
    cv.ok, cv.floor, cv.ok ? string(cv.count) : "none", cv.npoints, cv.nbisect, t_cov)
flush(stdout)

# --- the certified grid --------------------------------------------------------------------------
half = max(1.15, 1.3 * sc.gbar)
probe = [half * 0.7 * cis(2π * t / 12) for t in 0:11]
floor_at(probe[1])
tprobe = @elapsed for z in probe
    floor_at(z)
end
percost = tprobe / length(probe)
ng = clamp(round(Int, sqrt(BUDGET * Threads.nthreads() / percost)), NGMIN, NGMAX)
@printf("one certified point costs %.4f s; grid %d × %d\n", percost, ng, ng)
flush(stdout)

xs = collect(range(-half, half; length = ng))
ys = copy(xs)
F = zeros(ng, ng)
t_cert = @elapsed Threads.@threads for j in 1:ng
    for i in 1:ng
        F[i, j] = floor_at(xs[i] + im * ys[j])
    end
end
@printf("grid %d × %d over [%.4g, %.4g]² in %.1f s, bounded at %d of %d points (%.4f)\n",
    ng, ng, -half, half, t_cert, count(>(0), F), length(F), count(>(0), F) / length(F))
flush(stdout)

# --- the dense comparison, where the matrix still fits ------------------------------------------------
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
        Fc[i, j] = floor_at(z)
    end
else
    ev = RP._ritz_leading(Amul, k, Float64; L = 200)
    t_dense = 0.0
    @printf("no dense check at k = %d (above %d); the points drawn are the leading Ritz values\n",
        k, DENSEMAX)
end
bnd = Fc .> 0
strict = count(i -> bnd[i] && Fc[i] > Dc[i], eachindex(Fc))
ratio = [Fc[i] / Dc[i] for i in eachindex(Fc) if bnd[i]]
if DENSE
    @printf("dense check on %d points (%d with a bound): bound above the truth at %d; ",
        length(Fc), count(bnd), strict)
    @printf("ratio min %.4f median %.4f max %.4f (%.1f s)\n",
        isempty(ratio) ? NaN : minimum(ratio),
        isempty(ratio) ? NaN : sort(ratio)[cld(length(ratio), 2)],
        isempty(ratio) ? NaN : maximum(ratio), t_dense)
end

res = (; name = TAG, host = HOST, n = k, k = sc.k, ξ = Ξ, l, R, ks = ksuse, m = muse, how,
    plan_best = pl.best, plan_rows = pl.rows, plan_feasible = length(pl.feasible),
    nmatvec_plan = pl.nmatvec, t_plan,
    gbar = sc.gbar, abar = sc.abar, α = sc.α, γ1 = sc.γ1, γ2 = sc.γ2,
    cut = ts.c, dimQ = ts.k, tri_γ1 = ts.γ1, tri_γ2 = ts.γ2, tri_gbar = ts.gbar, nR2 = ts.nR2,
    cover_ok = cv.ok, cover_floor = cv.floor, npoints = cv.npoints, nbisect = cv.nbisect,
    ng, half, xs, ys, F, ev, discs = NamedTuple[], comps = NamedTuple[], percost,
    count = cv.ok ? cv.count : -1, Fc, Dc, xc, yc, strict,
    t_cert, t_dense, t_frame, t_split, t_tri, t_cov,
    zxs = Float64[], zys = Float64[], Fz = zeros(0, 0), zc = nothing, zhalf = 0.0, t_zoom = 0.0)
let out = joinpath(MAPDIR, "map_$(TAG).$(HOST).jls")
    Serialization.serialize(out, res)
    println("wrote ", out)
end
