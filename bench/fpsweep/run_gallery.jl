# Driver: for each operator, frame size and order m = 2, 4, 6, 8, run the floating-point sweep,
# certify it by Proposition prop:fpchecks, and compare with dense baselines.
#
#   julia --project=bench/fpsweep -t 128 bench/fpsweep/run_gallery.jl OUT.csv NAME1 NAME2 ...
#
# Environment: KS="0,5,10,20,40" (singular triplets, or converged Ritz vectors for symmetric A),
# PMAX=4, NDENSE=6000 (largest n for the dense baselines), LANCZOS=300 (Lanczos steps),
# ITERS=30 (Double64 power iterations), BLOCK (columns per block; default min(32, 2^22 ÷ n)).
# Output: one CSV row per (operator, k, m) in ~/ResearchData/KrylovPseudospectra_Data/fpsweep/results.

include(joinpath(@__DIR__, "FPSweep.jl"))
using .FPSweep
include(joinpath(@__DIR__, "operators.jl"))
using Printf, LinearAlgebra, SparseArrays

const RES = joinpath(DATA, "results")
mkpath(RES)

out = joinpath(RES, ARGS[1])
names = ARGS[2:end]
KS = parse.(Int, split(get(ENV, "KS", "0,5,10,20,40"), ","))
PMAX = parse(Int, get(ENV, "PMAX", "4"))
NDENSE = parse(Int, get(ENV, "NDENSE", "6000"))
LANCZOS = parse(Int, get(ENV, "LANCZOS", "300"))
BLOCK = parse(Int, get(ENV, "BLOCK", "0"))
ITERS = parse(Int, get(ENV, "ITERS", "30"))      # Double64 power iterations for the lower bounds

const COLS = ["name", "n", "w", "sym", "frame", "ks", "dimQ", "nconv", "scale_exp", "epsA", "epsP",
    "alpha", "s0", "g1_0", "g2_0", "m", "trfp", "ipfp", "gF", "aF", "gapF", "gI", "aI", "gapI",
    "epsB_over_g", "g_cert", "a_cert", "normA_f64", "normG22_f64", "lbA_dd", "lbG_dd", "kappa",
    "scale_est", "sound", "pass_1pct", "j_pm_float", "t_frame", "t_certframe", "t_sweep_bare",
    "t_sweep_book", "t_checks", "t_baseline", "nthreads", "blocksize", "minnorm", "maxnorm",
    "source"]

newfile = !isfile(out)
io = open(out, "a")
newfile && println(io, join(COLS, ","))

fmt(x::AbstractFloat) = isnan(x) ? "NaN" : @sprintf("%.17g", x)
fmt(x) = string(x)

for name in names
    println("\n==== $name"); flush(stdout)
    op0, source, sym = CATALOGUE[name]()
    n = FPSweep.nsize(op0)
    est = FPSweep.norm_estimate(op0)
    e = exponent(est) + 1
    op = scaled(op0, 2.0^-e)
    sc = 2.0^e
    wv = op isa SparseOp ? op.w : (op isa DenseOp ? n : (op isa Stencil2D ? 5 : 7))
    Asp = n <= 300_000 ? to_sparse(op) : nothing
    dense = n <= NDENSE
    blocksize = BLOCK > 0 ? BLOCK : max(1, min(32, 2^22 ÷ n))
    # warm-up (compilation) on a few columns
    sweep(op, zeros(n, 0), 1; cols = 1:min(n, 2blocksize), blocksize = blocksize)
    @printf("n=%d w=%d ‖A‖est=%.4e scale=2^%d sym=%s\n", n, wv, est, e, sym)
    Apart = nothing
    for ks in KS
        ks >= n ÷ 3 && continue
        t0 = time()
        local Q, nconv, frame, θ
        if ks == 0
            Q = zeros(n, 0); nconv = 0; frame = "none"; θ = Float64[]
        elseif sym
            Q, info = lanczos_frame(op, min(n - 1, LANCZOS); kmax = ks)
            nconv = info.nconv; frame = "lanczos_converged_ritz"; θ = info.theta
        else
            Q, info = gkl_frame(op, ks)
            nconv = 0; frame = "gkl_two_sided"; θ = Float64[]
        end
        tframe = time() - t0
        fr = certify_frame(op, Q)
        Sb = sweep(op, Q, PMAX; bookkeeping = false, blocksize = blocksize)
        S = sweep(op, Q, PMAX; blocksize = blocksize)
        base = Asp === nothing ? (; normA = NaN, normG22 = NaN, lbA = NaN, lbG = NaN, time = NaN) :
               dense_baseline(Asp, Q; symmetric = sym, dense = dense, Apart = Apart, iters = ITERS)
        Asp !== nothing && Apart === nothing && (Apart = (; normA = base.normA, xA = base.xA, lbA = base.lbA))
        nanmax(x, y) = isnan(x) ? y : (isnan(y) ? x : max(x, y))
        trueA = nanmax(base.normA, base.lbA) * sc
        trueG = nanmax(base.normG22, base.lbG) * sc
        @printf("  ks=%d dimQ=%d α=%.2e εP=%.2e εA=%.2e  ‖A‖=%.6e ‖G22‖=%.6e  (frame %.1fs, sweep %.2f/%.2fs, base %.1fs)\n",
            ks, size(Q, 2), fr.α, fr.εP, fr.εA, trueA, trueG, tframe, Sb.time, S.time, base.time)
        for pp in 1:PMAX
            c = FPSweep.fp_checks(S, fr, pp)
            gcert, acert = c.gF <= c.gI || isnan(c.gI) ? (c.gF, c.aF) : (c.gI, c.aI)
            κ = trueA / trueG
            scale_est = sqrt(n) * pp * fr.εA * κ
            okg(x) = isnan(x) || x * sc >= trueG
            oka(x) = isnan(x) || x * sc >= trueA
            sound = isnan(trueG) ? "n/a" :
                    string(okg(c.gF) && okg(c.gI) && oka(c.aF) && oka(c.aI))
            # cor:weyl: j_± = #{i : ±θ_i > ḡ}; θ are Float64 Ritz values (not certified)
            next = sym ? count(t -> abs(t) > gcert, θ) : -1
            row = Any[name, n, wv, sym, frame, ks, size(Q, 2), nconv, e, fr.εA, fr.εP, fr.α,
                fr.s[1] * sc, fr.γ1[1] * sc, fr.γ2[1] * sc, 2pp, c.trfp * sc, c.ipfp * sc,
                c.gF * sc, c.aF * sc, c.gapF, c.gI * sc, c.aI * sc, c.gapI, c.εBF / c.gF,
                gcert * sc, acert * sc, base.normA * sc, base.normG22 * sc, base.lbA * sc,
                base.lbG * sc, κ, scale_est, sound, c.passc, next, tframe, fr.time, Sb.time,
                S.time, c.time, base.time, S.nthreads, S.blocksize, S.minnorm, S.maxnorm, "\"" * source * "\""]
            println(io, join(fmt.(row), ",")); flush(io)
            @printf("    m=%d  trfp=%.6e gF=%.6e gap=%.2e | ipfp=%.4e gapI=%.2e | ā=%.6e εB/ḡ=%.2e κ=%.2e est=%.2e sound=%s pass1%%=%s\n",
                2pp, c.trfp * sc, c.gF * sc, c.gapF, c.ipfp * sc, c.gapI, acert * sc,
                c.εBF / c.gF, κ, scale_est, sound, c.passc)
            if sound == "false"
                println("    !!!! SOUNDNESS FAILURE: certified below the baseline")
            end
        end
        flush(stdout)
    end
end
close(io)
