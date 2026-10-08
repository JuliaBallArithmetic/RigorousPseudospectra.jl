# Experiment 3 of the numerics: the certificate, run in full where experiment 2 found ḡ < R.
#
# For every radius at which experiment 2 (effective_frame.<op>.<method>*.<host>.csv, all files)
# found a frame with ḡ < R, frames of def:Q at that R are certified (certify_split, compression =
# :full, far-block sweep of order 8) and the hypothesis of prop:count checked on the whole circle
# |z| = R. The search over ks starts at experiment 2's ks0, grows by factors of 2 up to
# ksmax = (n÷2 - N)÷2 until the cover completes, and bisects between the last failure and the first
# success; as in experiment 2, nothing makes the outcome monotone in ks, so what is reported is a
# transition the search found, with the outcome at ks0 alongside. The cover is count_exterior: σ11 by one verified SVD per point
# (method = :svd), σ22 = |z| - ḡ, 64 initial arcs, bisected with no practical cap (maxdepth = 64,
# past the floating-point resolution of the angle), so that a failed cover is one at which the
# hypothesis fails at a sample point (reason :point_fails) or an arc cannot be split further
# (:not_refinable), never a spent budget (rem:cover). No isolation discs. Recorded: the cover's outcome and floor, the certified count against N, and for
# n <= 2048 the least dense σ_min(A - zI) over 64 points of the circle.
#
# On the larger operators the radii can be sampled and sharded as in experiment 2: NSAMPLE=m keeps
# every feasible radius with N <= 16 and m more spread geometrically up to the largest; SHARD=i/S
# keeps every S-th of those starting at the i-th. The output file is tagged accordingly.
#
#   JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=8 julia --project=bench bench/certify_feasible.jl names [method]
using RigorousPseudospectra, LinearAlgebra, Printf
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "8")))

const HOST = readchomp(`hostname -s`)
const DIR = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra", "reach")
names = split(ARGS[1], ",")
method = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :qr
DENSE_MAX = 2048
NSAMPLE = parse(Int, get(ENV, "NSAMPLE", "0"))
SHARD = get(ENV, "SHARD", "")
shard_i, shard_S = isempty(SHARD) ? (0, 1) : parse.(Int, split(SHARD, "/"))
tag = (NSAMPLE > 0 ? ".sampled" : "") * (shard_S > 1 ? ".s$(shard_i)of$(shard_S)" : "")

function evaluate(o, op, A, n, R, ks, method)
    tf = @elapsed ((Q, _) = two_sided_frame(o.matvec, o.adjoint, n, R; ks, Leig = 200,
        Lsv = min(n, 2max(ks, 1) + 60), T = op.T, method))
    tc = @elapsed (sc = certify_split(op, Q; order = 8, target = R, compression = :full))
    tv = @elapsed ((cnt, cok, e) = count_exterior(sc, R; nθ = 64, maxdepth = 64, method = :svd))
    return (; ks, d = sc.k, gbar = sc.gbar, γ1 = sc.γ1, γ2 = sc.γ2, ok = e.ok && e.region_ok,
        e, cnt, cok, t = tf + tc + tv)
end

function readrows(path)
    lines = readlines(path)
    hdr = split(lines[1], ',')
    return [Dict(zip(hdr, split(l, ','))) for l in lines[2:end] if !isempty(strip(l))]
end

for nm in names
    name = String(nm)
    files = filter(f -> startswith(f, "effective_frame.$(name).$(method).") &&
                         endswith(f, ".$(HOST).csv"), readdir(DIR))
    rows = [r for f in files for r in readrows(joinpath(DIR, f)) if r["status"] == "found"]
    sort!(unique!(r -> r["N"], rows); by = r -> parse(Int, r["N"]))
    if NSAMPLE > 0
        low = [r for r in rows if parse(Int, r["N"]) <= 16]
        high = [r for r in rows if parse(Int, r["N"]) > 16]
        pick = isempty(high) ? eltype(rows)[] : unique(high[clamp.(round.(Int, exp.(range(0,
            log(length(high)); length = NSAMPLE))), 1, length(high))])
        rows = vcat(low, pick)
    end
    rows = rows[(shard_i + 1):shard_S:end]
    o = ps_operator(name)
    op = LinOp(o.A)
    A = Matrix(o.A)
    n = size(A, 1)
    io = open(joinpath(DIR, "certify_feasible.$(name).$(method)$(tag).$(HOST).csv"), "w")
    println(io, "operator,n,method,N,R,ks0,reason0,ks,dimQ,gbar,gamma1,gamma2,cover_ok,reason,floor,",
        "min_sigma11,min_sigma22,nevals,count,count_ok,count_valid,dense_min_sigma,nframes,t,error")
    for r in rows
        N = parse(Int, r["N"])
        R = parse(Float64, r["R"])
        ks0 = parse(Int, r["ks"])
        ksmax = (n ÷ 2 - N) ÷ 2
        line = try
            cache = Dict{Int, Any}()
            f(ks) = get!(() -> evaluate(o, op, A, n, R, ks, method), cache, ks)
            r0 = f(ks0)
            best = nothing
            if r0.ok
                best = r0
            else
                lo, hi = ks0, ks0
                while !f(hi).ok && hi < ksmax
                    lo = hi
                    hi = min(max(2hi, hi + 1), ksmax)
                end
                if f(hi).ok
                    while hi - lo > 1
                        mid = (lo + hi) ÷ 2
                        f(mid).ok ? (hi = mid) : (lo = mid)
                    end
                    best = f(hi)
                else
                    best = f(hi)
                end
            end
            b = best
            dmin = NaN
            if n <= DENSE_MAX
                dmin = minimum(minimum(svdvals(A - R * cis(2π * t / 64) * I)) for t in 0:63)
            end
            @sprintf("%s,%d,%s,%d,%.17g,%d,%s,%d,%d,%.17g,%.6g,%.6g,%s,%s,%.6g,%.6g,%.6g,%d,%d,%s,%s,%.6g,%d,%.2f,",
                name, n, method, N, R, ks0, r0.e.reason, b.ks, b.d, b.gbar, b.γ1, b.γ2, b.ok,
                b.e.reason, b.e.floor, b.e.minσ11, b.e.minσ22, b.e.nevals, b.cnt, b.cnt == N,
                b.cok, dmin, length(cache), sum(v.t for v in values(cache)))
        catch err
            @sprintf("%s,%d,%s,%d,%.17g,%d,,,,,,,,,,,,,,,,,,,%s", name, n, method, N, R, ks0,
                replace(sprint(showerror, err)[1:min(end, 80)], "," => ";", "\n" => " "))
        end
        println(io, line)
        flush(io)
    end
    close(io)
    @printf("%s (%s): %d rows done\n", name, method, length(rows))
end
