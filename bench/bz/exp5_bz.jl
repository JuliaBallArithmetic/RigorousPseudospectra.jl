# Experiment 5 of the numerics: the certificate and the isolation discs on the Belousov-Zhabotinsky
# operator A = N∘L, applied as two maps (bzcommon.jl), at a partition of 2^KE cells and noise ξ.
#
# What is certified is the floating-point operator A = N∘L with L the midpoint of the Ulam matrix,
# not the transfer operator (bzcommon.jl); the lift to the transfer operator is not done here.
#
# Radii: R = sqrt(|λ_N||λ_{N+1}|) in each gap with N <= NMAX, the λ from a dense eigendecomposition
# when 2^KE <= DENSEMAX and otherwise the Ritz values of an Arnoldi of length LRITZ, in floating
# point; they place the radius and are not part of the certificate. At each radius, the frames of
# def:Q with ks singular vectors per side, as in certify_feasible.jl: ks = 0, then 8, doubled up to
# KSMAX until the certified ḡ (order 8) is below R and the circle cover (count_exterior, method
# :svd, 64 initial arcs, maxdepth 64) completes; then bisection back to the first frame that passed.
# At that frame, certify_exterior: the count outside R and, for every component of the verified
# enclosure of G11 outside R, the smallest isolation disc on a geometric ladder of NRADII radii
# whose boundary passes the hypothesis of prop:count, with the same cover. When 2^KE <= DENSEMAX the
# dense eigenvalues outside R are compared with the discs and the count.
#
# The frame reported at each radius is saved with ../save_certificate.jl to
# frames/exp5_bz.k<KE>.xi<ξ>.N<N>.<host>.jld2, with the discs.
#
#   JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=4 julia --project=bench/bz bench/bz/exp5_bz.jl KE xi
include(joinpath(@__DIR__, "bzcommon.jl"))
include(joinpath(@__DIR__, "..", "save_certificate.jl"))
include(joinpath(@__DIR__, "..", "datadir.jl"))

const HOST = readchomp(`hostname -s`)
const DIR = joinpath(PSOUT, "bz_exp5")
mkpath(DIR)
const KE = parse(Int, ARGS[1])
const Ξ = parse(Float64, ARGS[2])
const NMAX = parse(Int, get(ENV, "NMAX", "16"))
const KSMAX = parse(Int, get(ENV, "KSMAX", "512"))
const NRADII = parse(Int, get(ENV, "NRADII", "12"))
const LRITZ = parse(Int, get(ENV, "LRITZ", "300"))
const DENSEMAX = 4096

k = 2^KE
L, _ = build_L(k)
l = noise_halfwidth(Ξ, k)
Amul, Aadj, op, oi = bz_linop(L, l)
η = oi.eta_rel
dense = k <= DENSEMAX
lam = if dense
    λ = eigvals(dense_A(L, l))
    λ[sortperm(abs.(λ); rev = true)]
else
    RP._ritz_leading(Amul, k, Float64; L = LRITZ)
end
@printf("bz 2^%d, ξ = %g, l = %d, η = %.3e, %s eigenvalues\n", KE, Ξ, l, η, dense ? "dense" : "Ritz")

function evaluate(R, ks)
    t = @elapsed begin
        Q, _ = RP.two_sided_frame(Amul, Aadj, k, R; ks, Leig = 200, Lsv = min(k, 2max(ks, 1) + 60),
            T = Float64)
        sc = RP.certify_split(op, Q; order = 8, target = R, compression = :full)
        if sc.gbar < R
            cnt, cok, e = RP.count_exterior(sc, R; nθ = 64, maxdepth = 64, method = :svd)
            ok = e.ok && e.region_ok
        else
            cnt, cok, e, ok = -1, false, nothing, false
        end
    end
    @printf("   R = %.6g ks = %d dim Q = %d ḡ = %.6g γ1γ2 = %.3e cover %s%s (%.1f s)\n", R, ks, sc.k,
        sc.gbar, sc.γ1 * sc.γ2, ok, e === nothing ? "" : " " * string(e.reason), t)
    flush(stdout)
    return (; ks, sc, ok, cnt, e, t)
end

gapN = [N for N in 1:min(NMAX, length(lam) - 1) if abs(lam[N]) > abs(lam[N + 1]) * (1 + 1e-9)]
base = "$(splitext(basename(@__FILE__))[1]).k$(KE).xi$(Ξ).$(HOST)"
io = open(joinpath(DIR, base * ".csv"), "w")
println(io, "KE,xi,N,R,ks,dimQ,gbar,gamma1,gamma2,cover_ok,reason,floor,count,ndiscs,discs_ok,",
    "max_disc_radius,dense_in_discs,nframes,t")
discio = open(joinpath(DIR, base * ".discs.csv"), "w")
println(discio, "KE,xi,N,R,re_center,im_center,radius,count,dense_inside")
for N in gapN
    R = sqrt(abs(lam[N]) * abs(lam[N + 1]))
    @printf("N = %d, R = %.6g\n", N, R)
    cache = Dict{Int, Any}()
    f(ks) = get!(() -> evaluate(R, ks), cache, ks)
    lo, hi = -1, 0
    while !f(hi).ok && hi < KSMAX
        lo = hi
        hi = hi == 0 ? 8 : min(2hi, KSMAX)
    end
    if f(hi).ok && lo >= 0
        while hi - lo > 1
            mid = (lo + hi) ÷ 2
            f(mid).ok ? (hi = mid) : (lo = mid)
        end
    end
    b = f(hi)
    tt = sum(v.t for v in values(cache))
    framefile = joinpath(DIR, "frames", "exp5_bz.k$(KE).xi$(Ξ).N$(N).$(HOST).jld2")
    if !b.ok
        save_certificate(framefile, b.sc, op; R, chk = b.e, script = "bench/bz/exp5_bz.jl",
            operator = "bz", KE, xi = Ξ, N, ks = b.ks)
        @printf(io, "%d,%g,%d,%.17g,%d,%d,%.17g,%.6g,%.6g,false,%s,,,,,,,%d,%.1f\n", KE, Ξ, N, R, b.ks,
            b.sc.k, b.sc.gbar, b.sc.γ1, b.sc.γ2, b.e === nothing ? "gbar" : string(b.e.reason),
            length(cache), tt)
        flush(io)
        continue
    end
    td = @elapsed (ext = RP.certify_exterior(b.sc, R; nθ = 64, nradii = NRADII, maxdepth = 64,
        method = :svd))
    good = [d for d in ext.discs if isfinite(d.radius)]
    save_certificate(framefile, b.sc, op; R, chk = ext.exterior, discs = good,
        script = "bench/bz/exp5_bz.jl", operator = "bz", KE, xi = Ξ, N, ks = b.ks, count = ext.count)
    out = dense ? [z for z in lam if abs(z) > R] : ComplexF64[]
    inside(d) = count(z -> abs(z - d.center) < d.radius, out)
    din = dense ? (all(z -> any(d -> abs(z - d.center) < d.radius, good), out) &&
                   all(d -> inside(d) == d.count, good)) : missing
    for d in ext.discs
        @printf(discio, "%d,%g,%d,%.17g,%.17g,%.17g,%.6g,%d,%s\n", KE, Ξ, N, R, real(d.center),
            imag(d.center), d.radius, d.count, dense && isfinite(d.radius) ? string(inside(d)) : "")
    end
    @printf(io, "%d,%g,%d,%.17g,%d,%d,%.17g,%.6g,%.6g,true,%s,%.6g,%d,%d,%s,%.6g,%s,%d,%.1f\n", KE, Ξ, N, R,
        b.ks, b.sc.k, b.sc.gbar, b.sc.γ1, b.sc.γ2, string(b.e.reason), b.e.floor, ext.count,
        length(ext.discs), ext.ok, maximum((d.radius for d in good); init = 0.0), string(din),
        length(cache), tt + td)
    @printf("   count %d (N = %d), %d discs, all closed %s, dense inside %s (%.1f s)\n", ext.count, N,
        length(ext.discs), ext.ok, string(din), td)
    flush(io); flush(discio); flush(stdout)
end
close(io); close(discio)
println("done")
