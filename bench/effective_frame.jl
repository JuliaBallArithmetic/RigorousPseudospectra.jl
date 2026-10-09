# Experiment 2 of the numerics: the effective frame dimension at which the certified far-block bound
# falls below the radius, against the floor of prop:frame (experiment 1).
#
# For each operator and every gap radius R = sqrt(|λ_N||λ_{N+1}|), the frames of def:Q
# (two_sided_frame, rank-revealing factorisation `method`) with ks singular vectors per side, ks from
# 0 (eigendirections only) to ksmax = (n÷2 - N)÷2. Feasible means the certified ḡ (certify_split,
# compression = :none, far-block sweep of order 8, ḡ the least over orders 1..8) is below R. The
# search: ks = 0; then ks = K = #{σ_i(A) >= R}, doubled up to ksmax until feasible; then bisection
# between the last infeasible and the first feasible ks. It assumes feasibility is monotone in ks,
# which nothing guarantees, so what is reported is a transition the bisection found, with the
# number of evaluations. Recorded at that ks: dim Q after the factorisation, ḡ, ḡ at order 1, the
# floor σ_{2 dimQ + 1}(A), the dense ‖G22‖ for n <= 2048, γ1, γ2; and at ks = 0 the same frame
# dimension and ḡ, which is the frame carrying only the eigendirections.
#
# On the larger operators the radii can be sampled and sharded: NSAMPLE=m keeps every gap with
# N <= 16 and m more spread geometrically up to n/2; SHARD=i/S keeps every S-th of those starting
# at the i-th (0-based). The output file is tagged accordingly.
#
# The frame reported at each radius is saved with save_certificate.jl to
# frames/effective_frame.<op>.<method>.N<N>.<host>.jld2. With REPLAY=1 the search is skipped: the
# radii and ks are read back from this script's own earlier output (same tag) and only the frame
# reported there is rebuilt, certified and saved, the numbers going to a file tagged .replay.
#
#   JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=8 julia --project=bench bench/effective_frame.jl names [method]
#   NSAMPLE=48 SHARD=0/4 ... julia --project=bench bench/effective_frame.jl rdb5000
using RigorousPseudospectra, LinearAlgebra, Printf
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))
include(joinpath(@__DIR__, "save_certificate.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "8")))

const HOST = readchomp(`hostname -s`)
const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra", "reach")
mkpath(OUT)
names = split(ARGS[1], ",")
method = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :qr
DENSE_MAX = 2048
NSAMPLE = parse(Int, get(ENV, "NSAMPLE", "0"))
SHARD = get(ENV, "SHARD", "")
shard_i, shard_S = isempty(SHARD) ? (0, 1) : parse.(Int, split(SHARD, "/"))
tag = (NSAMPLE > 0 ? ".sampled" : "") * (shard_S > 1 ? ".s$(shard_i)of$(shard_S)" : "")
REPLAY = get(ENV, "REPLAY", "0") == "1"
const FRAMES = joinpath(OUT, "frames")

function evaluate(o, op, A, s, n, R, ks)
    tf = @elapsed ((Q, _) = two_sided_frame(o.matvec, o.adjoint, n, R; ks, Leig = 200,
        Lsv = min(n, 2max(ks, 1) + 60), T = op.T, method))
    d = size(Q, 2)
    d == 0 && return (; ks, d, gbar = Inf, gbar1 = Inf, floor = s[1], g22 = NaN, γ1 = NaN,
        γ2 = NaN, ok = false, sc = nothing, t = tf)
    tc = @elapsed (sc = certify_split(op, Q; order = 8, target = R, compression = :none))
    g22 = NaN
    if n <= DENSE_MAX
        Qo = Matrix(qr(Q).Q)[:, 1:d]
        P = I - Qo * Qo'
        g22 = opnorm(P * A * P)
    end
    g1 = sc.far === nothing ? NaN : sc.far.gbar_trace[1]
    return (; ks, d, gbar = sc.gbar, gbar1 = g1, floor = 2d + 1 <= n ? s[2d + 1] : 0.0, g22,
        γ1 = sc.γ1, γ2 = sc.γ2, ok = sc.gbar < R, sc, t = tf + tc)
end

for nm in names
    name = String(nm)
    o = ps_operator(name)
    op = LinOp(o.A)
    A = Matrix(o.A)
    n = size(A, 1)
    lam = sort(abs.(eigvals(A)); rev = true)
    s = svdvals(A)
    prevrows = REPLAY ? let ls = readlines(joinpath(OUT, "effective_frame.$(name).$(method)$(tag).$(HOST).csv"))
        h = split(ls[1], ',')
        Dict(parse(Int, d["N"]) => d for d in (Dict(zip(h, split(x, ','))) for x in ls[2:end])
             if d["status"] != "error")
    end : nothing
    io = open(joinpath(OUT, "effective_frame.$(name).$(method)$(tag)$(REPLAY ? ".replay" : "").$(HOST).csv"), "w")
    println(io, "operator,n,method,N,R,K,dmin,status,nevals,ks,dimQ,gbar,gbar1,floor,normG22,",
        "gamma1,gamma2,dimQ0,gbar0,t,error")
    gapN = [N for N in 1:(n - 1) if lam[N] > lam[N + 1] * (1 + 1e-9)]
    if NSAMPLE > 0
        upper = [N for N in gapN if 16 < N < n ÷ 2]
        pick = isempty(upper) ? Int[] : unique(upper[clamp.(round.(Int, exp.(range(0,
            log(length(upper)); length = NSAMPLE))), 1, length(upper))])
        gapN = sort(unique(vcat([N for N in gapN if N <= 16], pick)))
    end
    gapN = REPLAY ? sort(collect(keys(prevrows))) : gapN[(shard_i + 1):shard_S:end]
    for N in gapN
        R = sqrt(lam[N] * lam[N + 1])
        K = count(>=(R), s)
        j = findfirst(<(R), s)
        dmin = j === nothing ? -1 : max(cld(j - 1, 2), 0)
        ksmax = (n ÷ 2 - N) ÷ 2
        ksmax >= 0 || continue
        line = try
            cache = Dict{Int, Any}()
            f(ks) = get!(() -> evaluate(o, op, A, s, n, R, ks), cache, ks)
            r0 = f(0)
            status = "found"
            best = nothing
            if REPLAY
                status = prevrows[N]["status"]
                best = f(parse(Int, prevrows[N]["ks"]))
            elseif r0.ok
                best = r0
            else
                lo = 0
                hi = min(max(K, 1), ksmax)
                while hi >= 1 && !f(hi).ok && hi < ksmax
                    lo = hi
                    hi = min(2hi, ksmax)
                end
                if hi < 1 || !f(hi).ok
                    status = "infeasible to ksmax"
                    best = f(max(hi, 0))
                else
                    while hi - lo > 1
                        mid = (lo + hi) ÷ 2
                        f(mid).ok ? (hi = mid) : (lo = mid)
                    end
                    best = f(hi)
                end
            end
            b = best
            b.sc === nothing || save_certificate(joinpath(FRAMES,
                    "effective_frame.$(name).$(method).N$(N).$(HOST).jld2"), b.sc, op; R,
                script = "bench/effective_frame.jl", operator = name, N, ks = b.ks, method,
                status, normG22_dense = b.g22)
            @sprintf("%s,%d,%s,%d,%.17g,%d,%d,%s,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.6g,%.6g,%d,%.17g,%.2f,",
                name, n, method, N, R, K, dmin, status, length(cache), b.ks, b.d, b.gbar, b.gbar1,
                b.floor, b.g22, b.γ1, b.γ2, r0.d, r0.gbar, sum(v.t for v in values(cache)))
        catch err
            @sprintf("%s,%d,%s,%d,%.17g,%d,%d,error,,,,,,,,,,,,,%s", name, n, method, N, R, K, dmin,
                replace(sprint(showerror, err)[1:min(end, 80)], "," => ";", "\n" => " "))
        end
        println(io, line)
        flush(io)
    end
    close(io)
    @printf("%s (%s) done\n", name, method)
end
