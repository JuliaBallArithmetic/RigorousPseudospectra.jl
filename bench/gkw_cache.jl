# Build a reusable cache of the expensive GKW BigFloat VBD pipeline (Schur + Rump–Newton
# separation + one rigorous certification) so resolvent experiments load instantly.
# Run once:  julia --project=bench bench/gkw_cache.jl [prec]
using RigorousPseudospectra, LinearAlgebra, GenericSchur, BallArithmetic, Serialization, Printf

const SRC = "/home/isaia/Code/GKWExperiments.jl/data/ball_matrix_bf_K256.jls"
cachepath(prec) = joinpath(@__DIR__, "gkw_vbd_bf$(prec).jls")

function build(prec::Int = 256)
    setprecision(BigFloat, prec)
    A = Complex{BigFloat}.(BallArithmetic.mid(deserialize(SRC)))
    @printf("loaded GKW n=%d at %d-bit\n", size(A, 1), prec)
    t0 = time()
    W, clusters = vbd_solve(A; maxsteps = 6)
    @printf("vbd_solve %.0fs  #clusters=%d  sizes(top)=%s\n",
        time() - t0, length(clusters), string(sort(length.(clusters); rev = true)[1:min(6, end)]))
    t0 = time()
    R = miyajima_rump(A, W; kappa_mode = :cheap)
    @printf("certify   %.0fs  κ₂=%.3e  ‖R₂‖∞=%.2e\n",
        time() - t0, Float64(R.kappa), Float64(R.nrmR2))
    out = cachepath(prec)
    serialize(out, (; A, W, clusters, R, prec))
    @printf("saved -> %s\n", out)
end

build(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 256)
