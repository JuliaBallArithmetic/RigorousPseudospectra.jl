# Which ordering of the Schur factor should the cut of sec:triangular use?
#
# `triangular_split` sorts the compression by decreasing |λ| once and cuts it once, so the cut is
# the same at every z of the contour. The alternative is to order at z, by increasing |z − T_ii|,
# so that the leading block carries the directions in which zI − T is nearest to singular, which is
# where the verified SVD earns its cost. This script runs both at every admissible depth c, on the
# same frame, the same far block and the same 64 points of the circle, and reports the certified
# floor of each together with ‖T22‖ at that depth.
#
# One constraint decides the shape of the answer and is enforced in `sigma_zorder_lo`: the leading
# block has to contain every eigenvalue of modulus ≥ R, or the trailing block carries them,
# ‖T22‖ > R, and eq:farpower gives nothing.
#
#   OPENBLAS_NUM_THREADS=4 julia --project=. -t 16 bench/cut_order.jl [op:target ...]
#
# Output: ~/ResearchData/KrylovPseudospectra_Data/pseudospectra/cut_order.<host>.{csv,jls}

using RigorousPseudospectra
using LinearAlgebra, Printf, Dates
import Serialization
const RP = RigorousPseudospectra
include(joinpath(@__DIR__, "pseudospectra_ops.jl"))

BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "4")))
const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
mkpath(OUT)
const HOST = gethostname()
const NTHETA = 64
const ROWS = ["rdb200:6", "rdb200:10", "airy200:10", "davies200:10", "convdiff_cheb200:10",
    "orrsom200:10", "kahan500:10", "nos3:10"]

"""
The floor of the cut ordered at `z`, kept here rather than in the package: the leading block is
every eigenvalue of modulus at least `R`, which prop:count is counting and without which
`‖T22‖ > R`, together with the nearest of the rest to `z`.
"""
function zorder_lo(ts, z, c)
    k = ts.k
    zc = ComplexF64(z)
    out = [abs(ts.T[i, i]) >= ts.R for i in 1:k]
    nout = count(out)
    (nout < k && max(c, nout) < k) || return 0.0
    c = max(c, nout)
    inside = findall(!, out)
    d = [abs(zc - ts.T[i, i]) for i in inside]
    sel = copy(out)
    sel[inside[partialsortperm(d, 1:(c - nout))]] .= true
    Tf = copy(ts.T)
    U = Matrix{ComplexF64}(I, k, k)
    RP._ordschur!(Tf, U, sel)
    Tz = RP._bmul(RP._bmul(RP.BallMatrix(Matrix(U')), ts.That), RP.BallMatrix(U))
    cur, rest = 1:c, (c + 1):k
    s11 = RP.subd(RP.sigmin_lo(RP._bsub(RP._tri_blockball(Tz, cur, cur),
        RP._zm_ball_id(c, zc))), ts.corr)
    s11 > 0 || return 0.0
    γ1 = RP.addu(RP.opnorm_up(RP._tri_blockball(Tz, cur, rest)), ts.corr)
    γ2 = RP.addu(RP.opnorm_up(RP._tri_blockball(Tz, rest, cur)), ts.corr)
    g1 = RP.addu(RP.opnorm_up(RP._tri_blockball(Tz, rest, rest); svd = false), ts.corr)
    s22 = RP.farshort_lo(zc, g1)
    s22 > 0 || return 0.0
    num = RP.subd(RP.muld(s11, s22), RP.mulu(γ1, γ2))
    num > 0 || return 0.0
    return RP.divd(RP.divd(num, RP.sumu((s11, s22, γ1, γ2))), ts.divisor)
end

"The floor of the |λ|-sorted cut at depth c, from the same ball compression."
function fixed_lo(ts, z, c)
    P11 = RP._tri_blockball(ts.That, 1:c, 1:c)
    s11 = RP.subd(RP.sigmin_lo(RP._bsub(P11, RP._zm_ball_id(c, z))), ts.corr)
    s11 > 0 || return 0.0
    Pd = RP._tri_blockball(ts.That, (c + 1):ts.k, (c + 1):ts.k)
    g1 = RP.addu(RP.opnorm_up(Pd; svd = false), ts.corr)
    s22 = RP.farshort_lo(z, g1)
    s22 > 0 || return 0.0
    γ1 = RP.addu(RP.opnorm_up(RP._tri_blockball(ts.That, 1:c, (c + 1):ts.k)), ts.corr)
    γ2 = RP.addu(RP.opnorm_up(RP._tri_blockball(ts.That, (c + 1):ts.k, 1:c)), ts.corr)
    num = RP.subd(RP.muld(s11, s22), RP.mulu(γ1, γ2))
    num > 0 || return 0.0
    return RP.divd(RP.divd(num, RP.sumu((s11, s22, γ1, γ2))), ts.divisor)
end

function main(rows)
    out = NamedTuple[]
    for spec in rows
        name, tgt = split(String(spec), ':')
        j = parse(Int, tgt)
        o = try
            ps_operator(name)
        catch err
            @warn "unavailable" name err
            continue
        end
        λ = eigvals(Matrix(o.A))
        a = sort(abs.(λ); rev = true)
        length(a) > j + 1 || continue
        R = sqrt(a[j] * a[j + 1])
        op = RP.LinOp(o.A)
        ks = 8
        Q, _ = RP.two_sided_frame(o.matvec, o.adjoint, o.n, R; ks, Leig = 200,
            Lsv = min(o.n, 2ks + 60), T = op.T)
        sc = RP.certify_split(op, Q; target = R, compression = :none)
        ts = RP.triangular_split(sc.G, sc.M, R; α = sc.α)
        ts === nothing && (@printf("%s:%d  no cut\n", name, j); continue)
        zs = ComplexF64[R * cis(2π * t / NTHETA) for t in 0:(NTHETA - 1)]
        t_svd = @elapsed s_svd = minimum(RP.sigma11_lo(sc, z; method = :svd) for z in zs)
        nout = count(i -> abs(ts.T[i, i]) >= R, 1:ts.k)
        @printf("\n%s:%d  n = %d, dim Q = %d, R = %.6g, scan chose c = %d, #{|λ| ≥ R} = %d, svd %.4e at %.3e s/z  %s\n",
            name, j, o.n, ts.k, R, ts.c, nout, s_svd, t_svd / NTHETA,
            Dates.format(now(), "HH:MM:SS"))
        @printf("  %4s %14s %14s %14s %11s %11s\n", "c", "|λ|-sorted", "z-ordered", "‖T22‖",
            "s/z fixed", "s/z zord")
        for c in nout:(ts.k - 1)
            tf = @elapsed sf = minimum(fixed_lo(ts, z, c) for z in zs)
            tz = @elapsed sz = minimum(zorder_lo(ts, z, c) for z in zs)
            g1 = RP.opnorm_up(RP._tri_blockball(ts.That, (c + 1):ts.k, (c + 1):ts.k); svd = false)
            @printf("  %4d %14.4e %14.4e %14.4e %11.3e %11.3e\n", c, sf, sz, g1,
                tf / NTHETA, tz / NTHETA)
            push!(out, (; op = name, target = j, o.n, dimQ = ts.k, R, cscan = ts.c, nout, c,
                fixed = sf, zorder = sz, nT22 = g1, svd = s_svd, t_fixed = tf / NTHETA,
                t_zorder = tz / NTHETA, t_svd = t_svd / NTHETA, host = HOST))
            flush(stdout)
        end
    end
    isempty(out) && return
    Serialization.serialize(joinpath(OUT, "cut_order.$(HOST).jls"), out)
    open(joinpath(OUT, "cut_order.$(HOST).csv"), "w") do io
        ks = collect(keys(out[1]))
        println(io, join(ks, ","))
        for r in out
            println(io, join([string(getfield(r, k)) for k in ks], ","))
        end
    end
    println("\nsummary: the best depth of each ordering, against one verified SVD")
    @printf("%-20s %6s %8s %12s %8s %12s %12s\n", "operator", "dim Q", "c*fixed", "best fixed",
        "c*zord", "best zord", "svd")
    for nm in unique([(r.op, r.target) for r in out])
        rs = [r for r in out if (r.op, r.target) == nm]
        bf = sort(rs; by = r -> -r.fixed)[1]
        bz = sort(rs; by = r -> -r.zorder)[1]
        @printf("%-20s %6d %8d %12.4e %8d %12.4e %12.4e\n", string(nm[1], ":", nm[2]), bf.dimQ,
            bf.c, bf.fixed, bz.c, bz.zorder, bf.svd)
    end
    println("\nwrote ", joinpath(OUT, "cut_order.$(HOST).csv"))
end

main(isempty(ARGS) ? ROWS : ARGS)
