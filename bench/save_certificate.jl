# The record of one certified frame, for the data deposit: everything needed to recheck a reported
# certificate without rebuilding the frame. Written as JLD2, which is HDF5, and only as plain arrays,
# numbers and strings, so that the file reads from h5py or MATLAB without Julia types:
#
#   meta/          script, host, the commits of RigorousPseudospectra and BallArithmetic (with
#                  "+dirty" when their trees carry uncommitted changes), date, and the caller's
#                  configuration (operator, N, R, ...)
#   operator/      n, epsA, etaA (the rounding constants of eq:roundA, or the absolute η)
#   frame/         Q (n × k), alpha (certified ‖I − QᵀQ‖)
#   G11/           mid and rad of the ball compression Qᵀ A Q, and of the ball Gram QᵀQ
#   far/           gbar, abar, gamma1, gamma2, the bound at every order of the sweep (gbar_trace)
#   enclosure/     the verified eigenvalue enclosure of G11: component centres, radii, counts
#   cover/         the circle check: ok, region_ok, reason, floor, nevals, min σ11, min σ22
#   discs/         isolation discs, when given: centres, radii, counts
#
#   include("save_certificate.jl"); save_certificate(path, sc, op; R, chk, discs, config...)
using JLD2, BallArithmetic

# the short commit of the repository holding `dir`, with "+dirty" when its tree has uncommitted
# changes; empty when it is not a git checkout
function _git_state(dir)
    try
        c = readchomp(`git -C $dir rev-parse --short HEAD`)
        dirty = !isempty(readchomp(`git -C $dir status --porcelain --untracked-files=no`))
        return dirty ? c * "+dirty" : c
    catch
        return ""
    end
end

function save_certificate(path::AbstractString, sc, op; R = NaN, chk = nothing, discs = nothing,
        script = "", config...)
    mkpath(dirname(path))
    commit = _git_state(@__DIR__)
    ba_commit = _git_state(pkgdir(BallArithmetic))
    comps = RigorousPseudospectra._has_counts(sc) ? RigorousPseudospectra._comps(sc) : []
    far = sc.far
    jldopen(path, "w") do f
        f["meta/script"] = String(script)
        f["meta/host"] = readchomp(`hostname -s`)
        f["meta/commit"] = commit
        f["meta/ballarithmetic_commit"] = ba_commit
        f["meta/date"] = string(Libc.strftime("%Y-%m-%dT%H:%M:%S", time()))
        for (key, v) in pairs(config)
            f["meta/config/$(key)"] = v isa Symbol ? String(v) : v
        end
        f["meta/R"] = Float64(R)
        f["operator/n"] = sc.n
        f["operator/epsA"] = op.epsA
        f["operator/etaA"] = op.etaA
        f["frame/Q"] = Matrix(sc.Q)
        f["frame/alpha"] = sc.α
        f["G11/mid"] = Matrix(BallArithmetic.mid(sc.G))
        f["G11/rad"] = Matrix(BallArithmetic.rad(sc.G))
        f["G11/gram_mid"] = Matrix(BallArithmetic.mid(sc.M))
        f["G11/gram_rad"] = Matrix(BallArithmetic.rad(sc.M))
        f["far/gbar"] = sc.gbar
        f["far/abar"] = sc.abar
        f["far/gamma1"] = sc.γ1
        f["far/gamma2"] = sc.γ2
        f["far/gbar_trace"] = far === nothing ? Float64[] : far.gbar_trace
        f["far/gbar_interp"] = far === nothing ? NaN : far.gbar_interp
        f["enclosure/ok"] = RigorousPseudospectra._has_counts(sc)
        f["enclosure/centres"] = ComplexF64[C.center for C in comps]
        f["enclosure/radii"] = Float64[C.radius for C in comps]
        f["enclosure/counts"] = Int[C.count for C in comps]
        if chk !== nothing
            f["cover/ok"] = chk.ok
            f["cover/region_ok"] = chk.region_ok
            f["cover/reason"] = String(chk.reason)
            f["cover/floor"] = chk.floor
            f["cover/nevals"] = chk.nevals
            f["cover/min_sigma11"] = chk.minσ11
            f["cover/min_sigma22"] = chk.minσ22
        end
        if discs !== nothing
            f["discs/centres"] = ComplexF64[d.center for d in discs]
            f["discs/radii"] = Float64[d.radius for d in discs]
            f["discs/counts"] = Int[d.count for d in discs]
        end
    end
    return path
end
