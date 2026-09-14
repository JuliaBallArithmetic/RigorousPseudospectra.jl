# §3.1 (sec:dense), Theorem thm:svdbox: the economy-SVD enclosure of Miyajima (2014, Thm 7).
#
# BallArithmetic's `svdbox` with its default method `MiyajimaM1()` is that theorem in economy form:
# a Float64 (LAPACK) economy SVD of the midpoint, the three residuals Ê = ÛΣ̂V̂* − A, F̂ = V̂*V̂ − I,
# Ĝ = Û*Û − I formed in ball arithmetic (so the radius of A is included), their norms bounded by
# BallArithmetic's norm bounds, and the two lines of thm:svdbox with directed rounding. It returns
# infinite radii when ‖F̂‖ ≥ 1 or ‖Ĝ‖ ≥ 1. We call it with `apply_vbd = false`: the default
# `apply_vbd = true` additionally runs a block diagonalisation of Σ*Σ whose result `svdbox` discards
# (checked by reading `BallArithmetic/src/decompositions/svd/svd.jl` at v0.3.0).

"""
    svd_enclosure(B::BallMatrix) -> (lo, hi)

Certified bounds `lo[i] ≤ s_i(X) ≤ hi[i]` for every `X` in the ball `B` and every
`i = 1:min(size(B)...)`, singular values in decreasing order (Theorem thm:svdbox). A failed
verification gives `lo = 0`, `hi = Inf`.
"""
function svd_enclosure(B::BallMatrix)
    q = minimum(size(B))
    q == 0 && return (Float64[], Float64[])
    Bm = BallMatrix(Matrix(mid(B)), Matrix(rad(B)))
    all(isfinite, mid(Bm)) && all(isfinite, rad(Bm)) || return (zeros(q), fill(Inf, q))
    s = svdbox(Bm; apply_vbd = false)
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
