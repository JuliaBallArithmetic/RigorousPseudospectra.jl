# The operator and its rounding model, eq:roundA.
#
# The one thing assumed of a matrix-free operator is a constant ε_A with
#     ‖fl(Ay) − Ay‖ ≤ ε_A ‖A‖ ‖y‖,   ‖fl(A*y) − A*y‖ ≤ ε_A ‖A‖ ‖y‖        (eq:roundA)
# for every y, where ‖A‖ is not assumed and is certified together with ‖G22‖ (prop:fpchecks).
# We carry in addition an absolute term η_A,
#     ‖fl(Ay) − Ay‖ ≤ ε_A ‖A‖ ‖y‖ + η_A,
# which the paper does not have: Higham's bounds (3.11)-(3.12) that eq:roundA rests on assume no
# underflow, and gradual underflow adds an absolute error of at most half the smallest subnormal
# η per product. For a stored matrix we compute both constants; for a matrix-free operator the
# user supplies ε_A (and η_A, default 0, which is the paper's model).
#
# For a stored matrix with at most w nonzeros in each row and in each column, every entry of fl(Ay)
# is an inner product of length ≤ w, so |fl(Ay) − Ay| ≤ γ_w |A||y| + wη componentwise, and
# ‖|A|‖ ≤ √(‖A‖₁‖A‖∞) ≤ √w ‖A‖, whence ε_A = √w γ_w (the paper's √w u_w) and η_A = √n wη. For
# complex data γ_{w+2} (see `gammau`); the absolute term is taken as 4wη per entry in all cases.

"""
    LinOp

A linear operator on `ℂⁿ` (or `ℝⁿ`) with its rounding model: `mul!(Y, X)` and `adj!(Y, X)` write
`fl(A X)` and `fl(A* X)` column by column; `epsA`, `etaA` are the constants of eq:roundA (with the
absolute term); `stored` is the matrix when one is stored (used for ball products), else
`nothing`; `T` is the element type of the vectors (`Float64` or `ComplexF64`).
"""
struct LinOp{FM, FA}
    n::Int
    T::DataType
    mul!::FM
    adj!::FA
    epsA::Float64
    etaA::Float64
    stored::Any
    w::Int
end

function _eta_stored(n, w, T)
    η = eta_min(Float64)
    c = 4.0                          # covers complex data and a real matrix applied to complex vectors
    return mulu(sqrtu(Float64(n)), mulu(c * w, η))
end

"Maximum number of nonzeros in a row or a column."
function _stencil_width(A::SparseMatrixCSC)
    wc = maximum(diff(A.colptr); init = 0)
    rows = zeros(Int, size(A, 1))
    for r in rowvals(A)
        rows[r] += 1
    end
    return max(wc, maximum(rows; init = 0))
end
_stencil_width(A::AbstractMatrix) = size(A, 1)

"""
    LinOp(A::AbstractMatrix; T = eltype)

A stored matrix (dense or `SparseMatrixCSC`): `ε_A = √w γ_w`, `η_A = √n wη` with `w` the largest
number of nonzeros in a row or a column (`w = n` for a dense matrix, whose products are BLAS inner
products of length `n`).
"""
function LinOp(A::AbstractMatrix; T::DataType = (eltype(A) <: Complex ? ComplexF64 : Float64))
    n = size(A, 1)
    size(A, 2) == n || error("square operator expected")
    As = A isa SparseMatrixCSC ? dropzeros(A) : Matrix(A)
    w = max(1, _stencil_width(As))
    epsA = mulu(sqrtu(Float64(w)), gammau(w, T))
    etaA = _eta_stored(n, w, T)
    Ast = As isa SparseMatrixCSC ? sparse(adjoint(As)) : Matrix(adjoint(As))
    m!(Y, X) = mul!(Y, As, X)
    a!(Y, X) = mul!(Y, Ast, X)
    return LinOp(n, T, m!, a!, epsA, etaA, As, w)
end

"""
    LinOp(Amul, Aadj, n; epsA, etaA = 0, T = nothing)

A matrix-free operator given by `x ↦ Amul(x)` and `x ↦ Aadj(x)` (vectors in, vectors out, applied
column by column to blocks; they must be safe to call from several threads when the sweep is
threaded). `epsA` is the constant of eq:roundA and must be supplied: it is an assumption on the
implementation of the operator, not something the package can check. `T` defaults to the element
type of `Amul` applied to a real vector.
"""
function LinOp(Amul, Aadj, n::Integer; epsA::Real, etaA::Real = 0.0, T = nothing)
    if T === nothing
        y = Amul(zeros(n))
        T = eltype(y) <: Complex ? ComplexF64 : Float64
    end
    function m!(Y, X)
        for c in axes(X, 2)
            Y[:, c] .= Amul(X[:, c])
        end
        return Y
    end
    function a!(Y, X)
        for c in axes(X, 2)
            Y[:, c] .= Aadj(X[:, c])
        end
        return Y
    end
    return LinOp(Int(n), T, m!, a!, Float64(epsA), Float64(etaA), nothing, 0)
end

"fl(A Q) (adj = false) or fl(A* Q) (adj = true) for the columns of Q, in floating point."
function apply_cols(op::LinOp, Q::AbstractMatrix, adj::Bool)
    T = promote_type(op.T, eltype(Q)) <: Complex ? ComplexF64 : Float64
    X = Matrix{T}(Q)
    Y = similar(X)
    adj ? op.adj!(Y, X) : op.mul!(Y, X)
    return Y
end
