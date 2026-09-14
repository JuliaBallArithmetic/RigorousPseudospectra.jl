# §3.3 (sec:coupling) and the frame quantities of §5.5 (sec:almostortho): the enclosures that live
# on objects of size k, computed once per operator and frame.
#
#   α ≥ ‖I − Q*Q‖ < 1                        eq:alpha, one ball Gram and svdbox
#   G̃ ⊇ Q*AQ, M ⊇ Q*Q                        ball enclosures of the pencil of prop:almostortho
#   γ2_Q ≥ ‖P⊥AQ‖,  γ1_Q ≥ ‖Q*AP⊥‖            eq:gram: γ² = ‖C*C‖, C = P⊥AQ, D = P⊥A*Q
#   s_Q ≥ ‖Q*AQ‖
#
# P⊥ is the orthogonal projector onto (ran Q)^⊥. Since P⊥Q = 0, P⊥X = P⊥(X − QQ*X), and ‖P⊥‖ ≤ 1,
# so ‖P⊥AQ‖ ≤ ‖(I − QQ*)AQ‖, which is formed in ball arithmetic without the inverse of Q*Q.
#
# For a stored matrix every product is a ball product (BallArithmetic, outward rounded). For a
# matrix-free operator only X = fl(AQ) and X' = fl(A*Q) are available; each column of X − AQ has
# norm at most ε_A‖A‖‖q_i‖ + η_A (eq:roundA), so ‖X − AQ‖ ≤ ε_A a ‖Q‖_F + √k η_A whenever ‖A‖ ≤ a,
# and each quantity is returned as an affine function c0 + c1·a with nonnegative coefficients, as
# prop:fpchecks needs. The values for the isometry Z of the polar decomposition Q = ZS carry the
# factors of prop:almostortho: ‖G12‖ ≤ γ1_Q/√(1−α), ‖G21‖ ≤ γ2_Q/√(1−α), ‖G11‖ ≤ s_Q/(1−α).

"""
    FrameData

Certified frame quantities. `α ≥ ‖I − Q*Q‖`; `Gt` a ball matrix and `rhoG = (c0, c1)` such that
every entry of `Q*AQ` lies within `rad(Gt) + c0 + c1·a` of `mid(Gt)` whenever `‖A‖ ≤ a` (for a
stored matrix `rhoG = (0, 0)`); `Mq ⊇ Q*Q`; `s`, `γ1Q`, `γ2Q` the affine bounds on `‖Q*AQ‖`,
`‖Q*AP⊥‖`, `‖P⊥AQ‖` for the frame `Q` itself (not yet divided by the factors in α).
"""
struct FrameData
    n::Int
    k::Int
    α::Float64
    Gt::BallMatrix
    Mq::BallMatrix
    rhoG::NTuple{2, Float64}
    s::NTuple{2, Float64}
    γ1Q::NTuple{2, Float64}
    γ2Q::NTuple{2, Float64}
    qF::Float64
    stored::Bool
    coupling::Symbol
    time::Float64
end

aff(c::NTuple{2, Float64}, a::Real) = addu(c[1], mulu(c[2], Float64(a)))

"Upper bound on ‖C‖ for a tall ball matrix C by eq:gram (`:gram`) or by svdbox of C (`:direct`)."
function _tall_norm_up(C::BallMatrix, coupling::Symbol)
    size(C, 2) == 0 && return 0.0
    if coupling === :direct
        return min(sigmax_up(C), frobenius_up(C))
    end
    Gm = _bmul(BallMatrix(Matrix(mid(C)'), Matrix(rad(C)')), C)
    return min(sqrtu(opnorm_up(Gm)), frobenius_up(C))
end

"α ≥ ‖I − Q*Q‖ and the ball Q*Q."
function frame_alpha(Q::AbstractMatrix)
    k = size(Q, 2)
    Qb = BallMatrix(Matrix(Q))
    Mq = _bmul(BallMatrix(Matrix(Q')), Qb)
    E = _bsub(Mq, Matrix{ComplexF64}(I, k, k))
    α = k == 0 ? 0.0 : opnorm_up(E)
    return α, Mq
end

"""
    frame_data(op::LinOp, Q; coupling = :gram) -> FrameData

The quantities above for the frame `Q` (n×k). Ball products with the stored matrix when
`op.stored !== nothing`, otherwise from `fl(AQ)` and `fl(A*Q)` with the affine rounding terms.
Throws if `α < 1` is not certified.
"""
function frame_data(op::LinOp, Q::AbstractMatrix; coupling::Symbol = :gram)
    t0 = time()
    n, k = size(Q)
    n == op.n || error("frame has $n rows, operator has size $(op.n)")
    all(isfinite, Q) || error("frame has non-finite entries")
    α, Mq = frame_alpha(Q)
    α < 1 || error("the frame is not certified to have full rank: α = $α ≥ 1")
    qF = frobenius_up(Q)
    Qb = BallMatrix(Matrix(Q))
    Qh = BallMatrix(Matrix(Q'))
    if op.stored !== nothing
        A = op.stored
        X = _bmul_op(A, Qb)
        Xt = _bmul_op(adjoint(A), Qb)
        rho = (0.0, 0.0)
        e0, e1 = 0.0, 0.0
    else
        X = BallMatrix(apply_cols(op, Q, false))
        Xt = BallMatrix(apply_cols(op, Q, true))
        # ‖X − AQ‖ ≤ ε_A a ‖Q‖_F + √k η_A
        e0 = mulu(sqrtu(Float64(k)), op.etaA)
        e1 = mulu(op.epsA, qF)
        sq = sqrtu(addu(1.0, α))                 # ‖Q‖ ≤ √(1 + α)
        rho = (mulu(sq, e0), mulu(sq, e1))       # every entry of Q*(AQ − X) is at most its norm
    end
    Gt = _bmul(Qh, X)
    C = _bsub(X, _bmul(Qb, Gt))                  # (I − QQ*) A Q
    D = _bsub(Xt, _bmul(Qb, _bmul(Qh, Xt)))      # (I − QQ*) A* Q
    γ2 = (addu(_tall_norm_up(C, coupling), e0), e1)
    γ1 = (addu(_tall_norm_up(D, coupling), e0), e1)
    s = (addu(opnorm_up(Gt), rho[1]), rho[2])
    return FrameData(n, k, α, Gt, Mq, rho, s, γ1, γ2, qF, op.stored !== nothing, coupling,
        time() - t0)
end

# Ball product of a stored (dense or sparse, possibly adjoint) matrix with a ball matrix.
function _bmul_op(A, B::BallMatrix)
    As = A isa Adjoint ? (parent(A) isa SparseMatrixCSC ? sparse(A) : Matrix(A)) : A
    TA = eltype(As)
    TB = eltype(mid(B))
    if TA <: Real && TB <: Real
        return As * B
    elseif TA <: Real
        # real matrix times complex ball: real and imaginary parts separately
        Re = As * BallMatrix(real.(mid(B)), rad(B))
        Im = As * BallMatrix(imag.(mid(B)), rad(B))
        return BallMatrix(complex.(mid(Re), mid(Im)), _radsum(rad(Re), rad(Im)))
    else
        Ar = real.(As)
        Ai = imag.(As)
        Bc = _promote_ball(B)
        Br = BallMatrix(real.(mid(Bc)), rad(Bc))
        Bi = BallMatrix(imag.(mid(Bc)), rad(Bc))
        Re = _bsub_real(Ar * Br, Ai * Bi)
        Im = Ar * Bi + Ai * Br
        return BallMatrix(complex.(mid(Re), mid(Im)), _radsum(rad(Re), rad(Im)))
    end
end
_radsum(a, b) = [addu(Float64(x), Float64(y)) for (x, y) in zip(a, b)] |> v -> reshape(v, size(a))
_bsub_real(A::BallMatrix, B::BallMatrix) = A - B

"""
    isometry_bounds(fd::FrameData, a) -> (s, γ1, γ2)

Bounds for the isometry Z of the polar decomposition, valid whenever ‖A‖ ≤ a:
`‖G11‖ ≤ s_Q(a)/(1 − α)`, `‖G12‖ ≤ γ1_Q(a)/√(1 − α)`, `‖G21‖ ≤ γ2_Q(a)/√(1 − α)`
(proof of prop:almostortho).
"""
function isometry_bounds(fd::FrameData, a::Real)
    oma = subd(1.0, fd.α)
    soma = sqrtd(oma)
    return (divu(aff(fd.s, a), oma), divu(aff(fd.γ1Q, a), soma), divu(aff(fd.γ2Q, a), soma))
end

"Affine coefficients of the isometry bounds, each coefficient divided by the α factor."
function isometry_affine(fd::FrameData)
    oma = subd(1.0, fd.α)
    soma = sqrtd(oma)
    d(c, x) = (divu(c[1], x), divu(c[2], x))
    return (d(fd.s, oma), d(fd.γ1Q, soma), d(fd.γ2Q, soma))
end

"""
    compression_ball(fd, a) -> BallMatrix

A ball matrix containing `Q*AQ` when ‖A‖ ≤ a: `Gt` with every radius increased by `rhoG(a)`.
"""
function compression_ball(fd::FrameData, a::Real)
    ρ = aff(fd.rhoG, a)
    ρ == 0 && return fd.Gt
    R = rad(fd.Gt)
    return BallMatrix(Matrix(mid(fd.Gt)), [addu(Float64(x), ρ) for x in R])
end
