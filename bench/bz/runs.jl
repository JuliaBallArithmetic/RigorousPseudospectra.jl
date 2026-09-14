# Driver: a list of certified runs in one process, one line of summary per run appended to
# OUT/summary.tsv and the full tables to stdout.
#
# Usage: julia -t 24 --project=. runs.jl "12:1e-1:1:50:full 12:1e-2:1:50:full ..."
# Each item is KE:XI:NEIG:KS:MODE:ORDER[:dense] with MODE one of `full` (arc covers of
# prop:count) and `light` (pointwise floors only), and ORDER the order m of the sweep (0 = auto).

include("certify.jl")

items = split(ARGS[1])
sumfile = joinpath(OUT, "summary.tsv")
if !isfile(sumfile)
    open(sumfile, "w") do io
        println(io, join(["tag", "ke", "n", "xi", "l", "neig", "dimQ", "alpha", "R", "Rlo", "Rhi",
            "gbar", "abar", "gamma1", "gamma2", "m", "sigma11", "gamma1gamma2", "sigma22",
            "assembly_ok", "count", "ext_ok", "Rmin", "t_frame", "t_sweep", "t_split", "mem_MB",
            "eta", "epsP", "ncover", "tcover", "full"], '\t'))
    end
end

for it in items
    f = split(it, ':')
    ke = parse(Int, f[1]); ξ = parse(Float64, f[2]); ne = parse(Int, f[3]); ks = parse(Int, f[4])
    full = f[5] == "full"
    ord = length(f) >= 6 && parse(Int, f[6]) > 0 ? parse(Int, f[6]) : :auto
    dense = length(f) >= 7 && f[7] == "dense"
    r = try
        opt = length(f) >= 8 ? f[8] : ""
        base = (; full, dense, order = ord, nθ = full ? 64 : 8, maxdepth = full ? 2 : 0)
        kw = opt == "R1" ?
                merge(base, (; radii_factors = [1.0], nθ = 128, maxdepth = 3,
                    tagsuffix = "_R1")) :
             opt == "newton" ? merge(base, (; vbd_method = :newton, tagsuffix = "_newton")) :
             opt == "xmax12" ? merge(base, (; xmax = 1e12, tagsuffix = "_xmax12")) :
             startswith(opt, "bs") ? merge(base, (; blocksize = parse(Int, opt[3:end]))) : base
        run_one(ke, ξ, ne, ks; kw...)
    catch e
        @error "run failed" it exception = (e, catch_backtrace())
        continue
    end
    c1 = r.circles[1]
    open(sumfile, "a") do io
        println(io, join(string.([("k$(r.ke)_xi$(r.ξ)_n$(r.neig)_ks$(r.ksarg)"), r.ke, r.k, r.ξ,
            r.l, r.neig, r.dimQ, r.α, r.R, r.Rwin[1], r.Rwin[2], r.gbar, r.abar, r.γ1, r.γ2, r.m,
            r.s11_svd, r.γ1 * r.γ2, r.R - r.gbar, c1.ok, r.count, r.ext_ok, r.Rmin, r.t_frame,
            r.t_sweep, r.t_split, r.mem_MB, r.η, r.epsP, r.ncover, r.tcover, r.full]), '\t'))
    end
    flush(stdout)
end
