# Certified upper bounds on the spectral norm of a ball matrix, valid for every matrix in the ball.
#
# For X in the ball with midpoint C and radius R, |X| ≤ |C| + R entrywise, and the spectral norm
# is monotone in the moduli of the entries (Higham, Lemma 6.6), so any upper bound on the norm of
# the nonnegative matrix |C| + R bounds ‖X‖. We use the Frobenius norm and the interpolation
# bound √(‖·‖₁‖·‖∞) of that matrix, both evaluated with upward rounding, and optionally the upper
# end of the economy-SVD enclosure (Theorem thm:svdbox, through BallArithmetic's `svdbox`), which
# is tight but costs an O(q³) factorisation. This replaces BallArithmetic's own
# `upper_bound_L2_opnorm`, which this package does not call.

"Entrywise upper bound of `|C| + R` for a ball matrix (or of `|A|` for a plain matrix)."
function absmat_up(B::BallMatrix)
    C, R = mid(B), rad(B)
    out = Matrix{Float64}(undef, size(C))
    @inbounds for j in axes(C, 2), i in axes(C, 1)
        out[i, j] = addu(absu(C[i, j]), Float64(R[i, j]))
    end
    return out
end
absmat_up(A::AbstractMatrix) = Float64[absu(x) for x in A]

"Upper bound on the Frobenius norm of every matrix in the ball."
function frobenius_up(B)
    M = absmat_up(B)
    s = 0.0
    @inbounds for x in M
        s = addu(s, mulu(x, x))
    end
    return sqrtu(s)
end

"Upper bounds on ‖·‖₁ (max column sum) and ‖·‖∞ (max row sum) of every matrix in the ball."
function onenorm_infnorm_up(B)
    M = absmat_up(B)
    m, n = size(M)
    c1 = 0.0
    rs = zeros(m)
    @inbounds for j in 1:n
        s = 0.0
        for i in 1:m
            s = addu(s, M[i, j])
            rs[i] = addu(rs[i], M[i, j])
        end
        c1 = max(c1, s)
    end
    return c1, (m == 0 ? 0.0 : maximum(rs))
end

"Interpolation bound √(‖·‖₁‖·‖∞), rounded upward."
function interp_up(B)
    a, b = onenorm_infnorm_up(B)
    return sqrtu(mulu(a, b))
end

"""
    opnorm_up(B; svd = true) -> Float64

Certified upper bound on ‖X‖₂ for every `X` in the ball matrix `B` (a plain matrix is a ball of
radius zero): the least of the Frobenius bound, the interpolation bound and, when `svd = true`,
the upper end of `svdbox(B)[1]`.
"""
function opnorm_up(B::BallMatrix; svd::Bool = true)
    isempty(B) && return 0.0
    v = min(frobenius_up(B), interp_up(B))
    if svd
        v = min(v, sigmax_up(B))
    end
    return v
end
opnorm_up(A::AbstractMatrix; kwargs...) = opnorm_up(BallMatrix(Matrix(A)); kwargs...)
