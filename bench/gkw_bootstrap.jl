# Precision bootstrap: take the 256-bit VBD frame, cast to 512-bit, and refine with
# vbd_refine (Newton + re-grounding S correction) — NO cold 512-bit Schur. Compare the
# resulting decomposition to the cold 512-bit cache (158 clusters / 100-block / κ₂≈141).
using RigorousPseudospectra, LinearAlgebra, GenericSchur, BallArithmetic, Serialization, Printf

const SRC = "/home/isaia/Code/GKWExperiments.jl/data/ball_matrix_bf_K256.jls"
cachepath(p) = joinpath(@__DIR__, "gkw_vbd_bf$(p).jls")

function go()
    # warm start: 256-bit decomposition
    setprecision(BigFloat, 256)
    W256 = deserialize(cachepath(256)).W
    # original matrix at 512-bit
    setprecision(BigFloat, 512)
    A = Complex{BigFloat}.(BallArithmetic.mid(deserialize(SRC)))
    @printf("warm-start W from 256-bit; refining at 512-bit (no cold Schur)\n")
    t0 = time()
    W, clusters = vbd_refine(A, W256; outer = 5, inner = 3)
    @printf("vbd_refine %.0fs  #clusters=%d  sizes(top)=%s\n",
        time() - t0, length(clusters), string(sort(length.(clusters); rev = true)[1:min(6, end)]))
    t0 = time()
    R = miyajima_rump(A, W; kappa_mode = :cheap)
    @printf("certify    %.0fs  κ₂=%.3e  ‖R₂‖∞=%.2e\n",
        time() - t0, Float64(R.kappa), Float64(R.nrmR2))
    @printf("  cold 512-bit baseline was: #clusters=158  block=100  κ₂=1.412e+02  ‖R₂‖∞=1.24e-151\n")
end
go()
