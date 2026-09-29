# §3.1 (sec:dense), Theorem thm:svdbox: an economy-SVD enclosure of all the singular values.
#
# This calls BallArithmetic's `svdbox(B; method = :miyajima2014_thm10_weyl)`: Miyajima's
# Theorem 10 applied to mid(B) alone, each interval then widened by one bound on ‖rad(B)‖₂. The
# widening is the spectral-norm case of Weyl's singular-value perturbation inequality,
# |σᵢ(X) − σᵢ(B̃)| ≤ ‖X − B̃‖₂ ≤ ‖rad(B)‖₂ for every X in the ball, index by index in decreasing
# order, which is the order Theorem 10 anchors its intervals to.
#
# Why not Theorem 7, which is what thm:svdbox states. Both theorems use the same economy frames: a
# Float64 (LAPACK) economy SVD of the midpoint, F̂ = V̂*V̂ − I and Ĝ = Û*Û − I, infinite radii when
# ‖F̂‖ ≥ 1 or ‖Ĝ‖ ≥ 1. They differ in the residual. Theorem 7 measures ÛΣ̂V̂* − A against the whole
# ball, so rad(B) enters additively and once; Theorem 10 measures AV̂ − ÛΣ̂, so rad(B) reaches the
# bound as a ball product and is bounded entrywise by rad(B)|V̂|, losing the cancellation that makes
# ‖rad(B)V̂‖₂ ≤ ‖rad(B)‖₂ at a cost growing like √n. Keeping rad(B) out of the residual entirely
# avoids that without giving up Theorem 10's tighter exact part. Maximum radii on the shape this
# package hands over, a square 40×40 compression shifted off its spectrum:
#
#   rad(B)   Thm 10 on B   Thm 7 on B   Thm 10 + Weyl
#   1e-13    2.084e-11     4.562e-12    4.494e-12
#   1e-16    5.187e-13     5.720e-13    5.009e-13
#
# so this route is the narrower of the three at both radii, by 1 to 12 per cent against Theorem 7.
# The theorem is named rather than left to `svdbox`'s `:auto`, which resolves to exactly this, so
# that a change of default in the kernel cannot silently change what the paper's numbers mean.
#
# Before BallArithmetic 0.4 this was `svdbox(B; apply_vbd = false)`, then the Theorem 7 route, with
# `apply_vbd = true` additionally running a block diagonalisation of Σ*Σ whose result was discarded.

"""
    svd_enclosure(B::BallMatrix) -> (lo, hi)

Certified bounds `lo[i] ≤ s_i(X) ≤ hi[i]` for every `X` in the ball `B` and every
`i = 1:min(size(B)...)`, singular values in decreasing order. A failed verification gives
`lo = 0`, `hi = Inf`.

This is Miyajima's Theorem 10 on `mid(B)` widened by `‖rad(B)‖₂` through Weyl's inequality, which
is narrower than the Theorem 7 route of thm:svdbox on every matrix measured; the comparison is at
the top of this file.
"""
function svd_enclosure(B::BallMatrix)
    q = minimum(size(B))
    q == 0 && return (Float64[], Float64[])
    Bm = BallMatrix(Matrix(mid(B)), Matrix(rad(B)))
    all(isfinite, mid(Bm)) && all(isfinite, rad(Bm)) || return (zeros(q), fill(Inf, q))
    s = svdbox(Bm; method = :miyajima2014_thm10_weyl)
    lo = Float64[max(0.0, subd(Float64(mid(x)), Float64(rad(x)))) for x in s]
    hi = Float64[addu(Float64(mid(x)), Float64(rad(x))) for x in s]
    for i in 1:q
        (isfinite(lo[i]) && isfinite(hi[i])) || (lo[i] = 0.0; hi[i] = Inf)
    end
    return lo, hi
end
svd_enclosure(A::AbstractMatrix) = svd_enclosure(BallMatrix(Matrix(A)))

"Certified lower bound on σ_min of every matrix in the ball (the q-th singular value, q = min size)."
function sigmin_lo(B)
    lo, _ = svd_enclosure(B)
    return isempty(lo) ? 0.0 : lo[end]
end

"Certified upper bound on σ_max = ‖·‖₂ of every matrix in the ball."
function sigmax_up(B)
    _, hi = svd_enclosure(B)
    return isempty(hi) ? 0.0 : hi[1]
end
