# Experiment 3 of the numerics: the certificate, run in full where experiment 2 found ḡ < R.
#
# For every row of effective_frame.<op>.<method>.<host>.csv with status "found", the same frame is
# rebuilt (two_sided_frame at that R and ks; the Arnoldi and Golub-Kahan-Lanczos start vectors are
# the package's, so ḡ is recomputed and recorded rather than read back), certified with
# certify_split (compression = :full, far-block sweep of order 8), and the hypothesis of prop:count
# checked on the whole circle |z| = R by count_exterior: σ11 by one verified SVD per point
# (method = :svd), σ22 = |z| - ḡ, 256 initial arcs, up to 8 bisections each (rem:cover). No
# isolation discs. Recorded: the cover's outcome and floor, the certified count against N, and for
# n <= 2048 the least dense σ_min(A - zI) over 64 points of the circle.
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

function readrows(path)
    lines = readlines(path)
    hdr = split(lines[1], ',')
    return [Dict(zip(hdr, split(l, ','))) for l in lines[2:end] if !isempty(strip(l))]
end

for nm in names
    name = String(nm)
    rows = [r for r in readrows(joinpath(DIR, "effective_frame.$(name).$(method).$(HOST).csv"))
            if r["status"] == "found"]
    o = ps_operator(name)
    op = LinOp(o.A)
    A = Matrix(o.A)
    n = size(A, 1)
    io = open(joinpath(DIR, "certify_feasible.$(name).$(method).$(HOST).csv"), "w")
    println(io, "operator,n,method,N,R,ks,dimQ,gbar,gamma1,gamma2,cover_ok,region_ok,floor,",
        "min_sigma11,min_sigma22,nevals,count,count_ok,count_valid,dense_min_sigma,t_frame,t_cert,t_cover,error")
    for r in rows
        N = parse(Int, r["N"])
        R = parse(Float64, r["R"])
        ks = parse(Int, r["ks"])
        line = try
            tf = @elapsed ((Q, _) = two_sided_frame(o.matvec, o.adjoint, n, R; ks, Leig = 200,
                Lsv = min(n, 2max(ks, 1) + 60), T = op.T, method))
            tc = @elapsed (sc = certify_split(op, Q; order = 8, target = R, compression = :full))
            tv = @elapsed ((cnt, cok, e) = count_exterior(sc, R; nθ = 256, maxdepth = 8,
                method = :svd))
            dmin = NaN
            if n <= DENSE_MAX
                dmin = minimum(minimum(svdvals(A - R * cis(2π * t / 64) * I)) for t in 0:63)
            end
            @sprintf("%s,%d,%s,%d,%.17g,%d,%d,%.17g,%.6g,%.6g,%s,%s,%.6g,%.6g,%.6g,%d,%d,%s,%s,%.6g,%.2f,%.2f,%.2f,",
                name, n, method, N, R, ks, sc.k, sc.gbar, sc.γ1, sc.γ2, e.ok, e.region_ok, e.floor,
                e.minσ11, e.minσ22, e.nevals, cnt, cnt == N, cok, dmin, tf, tc, tv)
        catch err
            @sprintf("%s,%d,%s,%d,%.17g,%d,,,,,,,,,,,,,,,,,,%s", name, n, method, N, R, ks,
                replace(sprint(showerror, err)[1:min(end, 80)], "," => ";", "\n" => " "))
        end
        println(io, line)
        flush(io)
    end
    close(io)
    @printf("%s (%s): %d rows done\n", name, method, length(rows))
end
