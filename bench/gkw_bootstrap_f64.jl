# Aggressive bootstrap: ONE cheap Float64 Schur for the frame, then ALL residual + Newton +
# recentering in BigFloat against the full-precision matrix (vbd_refine). Tests whether
# Newton–Kantorovich converges from a 1e-16 start down to the 512-bit floor — i.e. whether
# the expensive cold BigFloat Schur can be skipped entirely.
using RigorousPseudospectra, LinearAlgebra, GenericSchur, BallArithmetic, Serialization, Printf

const SRC = "/home/isaia/Code/GKWExperiments.jl/data/ball_matrix_bf_K256.jls"

function go(prec = 512, outer = 8, inner = 3)
    setprecision(BigFloat, prec)
    A = Complex{BigFloat}.(BallArithmetic.mid(deserialize(SRC)))
    # cheap Float64 Schur frame (the only Float64 step)
    t0 = time()
    Qf = schur(ComplexF64.(A)).vectors          # orthonormal Float64 frame
    @printf("Float64 Schur frame %.2fs\n", time() - t0)
    # refine entirely in BigFloat against the full-precision A
    t0 = time()
    W, clusters = vbd_refine(A, Qf; outer = outer, inner = inner)
    @printf("vbd_refine(%d-bit, outer=%d) %.0fs  #clusters=%d  sizes(top)=%s\n",
        prec, outer, time() - t0, length(clusters),
        string(sort(length.(clusters); rev = true)[1:min(6, end)]))
    t0 = time()
    R = miyajima_rump(A, W; kappa_mode = :cheap)
    @printf("certify %.0fs  κ₂=%.3e  ‖R₂‖∞=%.2e\n",
        time() - t0, Float64(R.kappa), Float64(R.nrmR2))
    @printf("  cold 512-bit baseline: #clusters=158 block=100 κ₂=1.412e+02 ‖R₂‖∞=1.24e-151\n")
end
go()
