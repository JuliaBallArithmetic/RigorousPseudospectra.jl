# FPSweep: the floating-point trace sweep of draft_lean.tex §3.4 (label sec:fpsweep), with the a
# posteriori certification of Proposition prop:fpchecks.
#
#   B = G22 = P⊥ A P⊥,  H = BᵀB,  order m = 2p.
#   eq:roundA   ‖fl(Ay) − Ay‖ ≤ ε_A ‖A‖ ‖y‖ (and for Aᵀ)
#   eq:epsP     ‖fl(y − Q fl(Qᵀy)) − P⊥y‖ ≤ ε_P ‖y‖,
#               ε_P = α + u + (1+u)(1+α)√k (u_n + u_k(1 + √k u_n))
#   lem:onestep ‖fl(By) − By‖ ≤ ε_B ‖y‖,
#               ε_B = ε_P √(γ2²+g²) + ε_A a (1+ε_P) + ε_P(√(γ1²+g²) + a(ε_P + ε_A(1+ε_P)))
#   eq:FI       F_a(g) = Σ_j (c_j + ε_B E_j(g))²,  I_a(g) = (N1 + √n ε_B)(N∞ + n ε_B)
#               E_j(g) = Σ_{i=0}^{p-1} g^{2(p-1-i)} (g‖w_i‖ + ‖v_i‖)
#   prop:fpchecks (i) F_ā(ḡ) < ḡ^{4p} or I_ā(ḡ) < ḡ²;  (ii) ‖[s̄(ā) γ̄1(ā); γ̄2(ā) ḡ]‖ < ā
#
# Everything is real Float64, round-to-nearest, for the sweep. Upper bounds on running sums and
# the scalar checks are evaluated with RoundingEmulator's directed operations (add_up, mul_up, ...),
# which are exact emulations of directed rounding and do not touch the hardware rounding mode.
#
# Vectors inside the sweep are stored as ROWS of a b×n block ("row block"), b columns of the
# canonical basis processed together, so that the projector is two BLAS-3 calls.

module FPSweep

using LinearAlgebra, SparseArrays, Random, Printf
using BallArithmetic
import RoundingEmulator: add_up, add_down, sub_up, sub_down, mul_up, mul_down, div_up,
                         div_down, sqrt_up, sqrt_down

export u64, gamma_up, epsA_sparse, epsP_up
export SparseOp, DenseOp, Stencil2D, Stencil3D, nsize, apply!, to_sparse, scaled
export gkl_frame, lanczos_frame
export certify_frame, sweep, fp_checks, ball_sweep, dense_baseline

const u64 = 2.0^-53                 # unit roundoff of binary64
const ETA = 2.0^-1074               # smallest positive subnormal

# ----------------------------------------------------------------------------------------------
# Rounding constants
# ----------------------------------------------------------------------------------------------

"γ_n = n u/(1 − n u), rounded upward (Higham (3.8)); the paper's u_n."
function gamma_up(n::Integer)
    nu = mul_up(Float64(n), u64)          # exact for n < 2^53
    nu < 1 || error("n u ≥ 1")
    return div_up(nu, sub_down(1.0, nu))
end

"ε_A = √w u_w for a stored matrix (or a stencil) with at most w terms per row and per column."
epsA_sparse(w::Integer) = mul_up(sqrt_up(Float64(w)), gamma_up(w))

"ε_P of eq:epsP, rounded upward."
function epsP_up(α::Float64, n::Integer, k::Integer)
    un = gamma_up(n)
    uk = gamma_up(k)
    sk = sqrt_up(Float64(k))
    inner = add_up(un, mul_up(uk, add_up(1.0, mul_up(sk, un))))
    t = mul_up(mul_up(mul_up(add_up(1.0, u64), add_up(1.0, α)), sk), inner)
    return add_up(add_up(α, u64), t)
end

# ----------------------------------------------------------------------------------------------
# Operators. apply!(Y, op, X, adj) writes the rows of Y as op (adj=false) or opᵀ (adj=true)
# applied to the rows of X, X and Y of size b×n. Every output entry is formed as a sum of
# products taken one after another, so Higham's inner-product bound (3.11)-(3.12) applies with
# the number of terms w, and ε_A = √w u_w (eq:roundA and the text after it).
# ----------------------------------------------------------------------------------------------

abstract type Op end

"Stored sparse matrix. `A` and `At = Aᵀ` are both kept in CSC so that each output entry is an
inner product over the nonzeros of one row, accumulated in the stored order."
struct SparseOp <: Op
    n::Int
    A::SparseMatrixCSC{Float64, Int}
    At::SparseMatrixCSC{Float64, Int}
    w::Int
    name::String
end
function SparseOp(A::SparseMatrixCSC{Float64, Int}, name::AbstractString)
    n = size(A, 1)
    size(A, 2) == n || error("square matrix expected")
    A = dropzeros(A)
    At = sparse(transpose(A))
    wr = maximum(diff(At.colptr))          # nonzeros per row of A
    wc = maximum(diff(A.colptr))           # nonzeros per column of A
    return SparseOp(n, A, At, max(wr, wc), String(name))
end
nsize(op::Op) = op.n
epsA(op::SparseOp) = epsA_sparse(op.w)

function apply!(Y::AbstractMatrix{Float64}, op::SparseOp, X::AbstractMatrix{Float64}, adj::Bool)
    M = adj ? op.A : op.At                 # column i of M = row i of (adj ? Aᵀ : A)
    b = size(X, 1)
    rv = rowvals(M)
    nz = nonzeros(M)
    @inbounds for i in 1:(op.n)
        for c in 1:b
            Y[c, i] = 0.0
        end
        for kk in nzrange(M, i)
            r = rv[kk]
            v = nz[kk]
            @simd for c in 1:b
                Y[c, i] = Y[c, i] + v * X[c, r]
            end
        end
    end
    return Y
end
to_sparse(op::SparseOp) = op.A

"Stored dense matrix, applied by BLAS: each entry an inner product of length n, so w = n."
struct DenseOp <: Op
    n::Int
    A::Matrix{Float64}
    name::String
end
DenseOp(A::Matrix{Float64}, name::AbstractString) = DenseOp(size(A, 1), A, String(name))
epsA(op::DenseOp) = epsA_sparse(op.n)
function apply!(Y::AbstractMatrix{Float64}, op::DenseOp, X::AbstractMatrix{Float64}, adj::Bool)
    # rows: y = A x  ⇔  Yrow = Xrow Aᵀ;  y = Aᵀ x ⇔ Yrow = Xrow A
    adj ? mul!(Y, X, op.A) : mul!(Y, X, transpose(op.A))
    return Y
end
to_sparse(op::DenseOp) = sparse(op.A)

"""
Matrix-free 2D five-point stencil on an n1×n2 grid, column-major index i1 + (i2-1) n1, zero
Dirichlet boundary:
  (Au)_{i1,i2} = c0 u + cW u_{i1-1} + cE u_{i1+1} + cS u_{i2-1} + cN u_{i2+1}.
The transpose swaps W↔E and S↔N. Each output entry is formed as the sum of its (at most 5)
products in this fixed order, so w = 5.
"""
struct Stencil2D <: Op
    n::Int
    n1::Int
    n2::Int
    c0::Float64
    cW::Float64
    cE::Float64
    cS::Float64
    cN::Float64
    name::String
end
Stencil2D(n1, n2, c0, cW, cE, cS, cN, name) = Stencil2D(n1 * n2, n1, n2, c0, cW, cE, cS, cN, name)
epsA(op::Stencil2D) = epsA_sparse(5)

function apply!(Y::AbstractMatrix{Float64}, op::Stencil2D, X::AbstractMatrix{Float64}, adj::Bool)
    n1, n2 = op.n1, op.n2
    c0 = op.c0
    cW, cE, cS, cN = adj ? (op.cE, op.cW, op.cN, op.cS) : (op.cW, op.cE, op.cS, op.cN)
    b = size(X, 1)
    @inbounds for i2 in 1:n2, i1 in 1:n1
        idx = i1 + (i2 - 1) * n1
        hasW = i1 > 1
        hasE = i1 < n1
        hasS = i2 > 1
        hasN = i2 < n2
        @simd for c in 1:b
            s = c0 * X[c, idx]
            s = hasW ? s + cW * X[c, idx - 1] : s
            s = hasE ? s + cE * X[c, idx + 1] : s
            s = hasS ? s + cS * X[c, idx - n1] : s
            s = hasN ? s + cN * X[c, idx + n1] : s
            Y[c, idx] = s
        end
    end
    return Y
end
function to_sparse(op::Stencil2D)
    n1, n2 = op.n1, op.n2
    I = Int[]; J = Int[]; V = Float64[]
    for i2 in 1:n2, i1 in 1:n1
        idx = i1 + (i2 - 1) * n1
        push!(I, idx); push!(J, idx); push!(V, op.c0)
        i1 > 1 && (push!(I, idx); push!(J, idx - 1); push!(V, op.cW))
        i1 < n1 && (push!(I, idx); push!(J, idx + 1); push!(V, op.cE))
        i2 > 1 && (push!(I, idx); push!(J, idx - n1); push!(V, op.cS))
        i2 < n2 && (push!(I, idx); push!(J, idx + n1); push!(V, op.cN))
    end
    return sparse(I, J, V, op.n, op.n)
end

"""
Matrix-free 3D seven-point stencil on an N1×N2×N3 grid with zero Dirichlet boundary and an
optional diagonal vector `d` (for the Anderson model): (Au)_i = (c0 + d_i) u_i + Σ_dir c_dir u_nb.
The diagonal coefficient c0 + d_i is formed once and stored (it is the stored matrix entry), and
each output entry is the sum of at most 7 products in a fixed order, so w = 7.
"""
struct Stencil3D <: Op
    n::Int
    N1::Int
    N2::Int
    N3::Int
    diag::Vector{Float64}          # the diagonal entries of A, as stored
    cm1::Float64; cp1::Float64     # coefficient of u_{i1-1}, u_{i1+1}
    cm2::Float64; cp2::Float64
    cm3::Float64; cp3::Float64
    name::String
end
epsA(op::Stencil3D) = epsA_sparse(7)
function apply!(Y::AbstractMatrix{Float64}, op::Stencil3D, X::AbstractMatrix{Float64}, adj::Bool)
    N1, N2, N3 = op.N1, op.N2, op.N3
    s1 = 1; s2 = N1; s3 = N1 * N2
    if adj
        cm1, cp1, cm2, cp2, cm3, cp3 = op.cp1, op.cm1, op.cp2, op.cm2, op.cp3, op.cm3
    else
        cm1, cp1, cm2, cp2, cm3, cp3 = op.cm1, op.cp1, op.cm2, op.cp2, op.cm3, op.cp3
    end
    b = size(X, 1)
    d = op.diag
    @inbounds for i3 in 1:N3, i2 in 1:N2, i1 in 1:N1
        idx = i1 + (i2 - 1) * N1 + (i3 - 1) * s3
        a1 = i1 > 1; b1 = i1 < N1; a2 = i2 > 1; b2 = i2 < N2; a3 = i3 > 1; b3 = i3 < N3
        dd = d[idx]
        @simd for c in 1:b
            s = dd * X[c, idx]
            s = a1 ? s + cm1 * X[c, idx - s1] : s
            s = b1 ? s + cp1 * X[c, idx + s1] : s
            s = a2 ? s + cm2 * X[c, idx - s2] : s
            s = b2 ? s + cp2 * X[c, idx + s2] : s
            s = a3 ? s + cm3 * X[c, idx - s3] : s
            s = b3 ? s + cp3 * X[c, idx + s3] : s
            Y[c, idx] = s
        end
    end
    return Y
end
function to_sparse(op::Stencil3D)
    N1, N2, N3 = op.N1, op.N2, op.N3
    s3 = N1 * N2
    I = Int[]; J = Int[]; V = Float64[]
    for i3 in 1:N3, i2 in 1:N2, i1 in 1:N1
        idx = i1 + (i2 - 1) * N1 + (i3 - 1) * s3
        push!(I, idx); push!(J, idx); push!(V, op.diag[idx])
        i1 > 1 && (push!(I, idx); push!(J, idx - 1); push!(V, op.cm1))
        i1 < N1 && (push!(I, idx); push!(J, idx + 1); push!(V, op.cp1))
        i2 > 1 && (push!(I, idx); push!(J, idx - N1); push!(V, op.cm2))
        i2 < N2 && (push!(I, idx); push!(J, idx + N1); push!(V, op.cp2))
        i3 > 1 && (push!(I, idx); push!(J, idx - s3); push!(V, op.cm3))
        i3 < N3 && (push!(I, idx); push!(J, idx + s3); push!(V, op.cp3))
    end
    return sparse(I, J, V, op.n, op.n)
end

"""
    scaled(op, s) -> op with every stored coefficient multiplied by s = 2^e.

Multiplying by a power of two is exact (no coefficient may become subnormal: checked), and
fl(sA y) = s fl(Ay) exactly in the absence of underflow and overflow, so the sweep on the scaled
operator is the sweep on A rescaled, as in Remark rem:impl, done once on the operator.
"""
function scaled(op::SparseOp, s::Float64)
    _chkpow2(s)
    B = copy(op.A)
    nonzeros(B) .*= s
    _chknormal(nonzeros(B))
    return SparseOp(op.n, B, sparse(transpose(B)), op.w, op.name)
end
function scaled(op::DenseOp, s::Float64)
    _chkpow2(s)
    B = op.A .* s
    _chknormal(B)
    DenseOp(op.n, B, op.name)
end
function scaled(op::Stencil2D, s::Float64)
    _chkpow2(s)
    c = (op.c0, op.cW, op.cE, op.cS, op.cN) .* s
    _chknormal(collect(c))
    Stencil2D(op.n, op.n1, op.n2, c..., op.name)
end
function scaled(op::Stencil3D, s::Float64)
    _chkpow2(s)
    d = op.diag .* s
    c = (op.cm1, op.cp1, op.cm2, op.cp2, op.cm3, op.cp3) .* s
    _chknormal(d); _chknormal(collect(c))
    Stencil3D(op.n, op.N1, op.N2, op.N3, d, c..., op.name)
end
_chkpow2(s) = (s > 0 && significand(s) == 1.0) || error("scale must be a power of two")
_chknormal(v) = all(x -> x == 0 || abs(x) >= floatmin(Float64), v) ||
                error("scaling produced a subnormal coefficient")

# single-vector convenience
function opvec(op::Op, x::AbstractVector{Float64}, adj::Bool)
    X = reshape(copy(x), 1, :)
    Y = similar(X)
    apply!(Y, op, X, adj)
    return vec(Y)
end

"Power-iteration estimate of ‖A‖ (floating point, used only to choose the scale)."
function norm_estimate(op::Op; iters = 30, rng = MersenneTwister(7))
    x = randn(rng, op.n); x ./= norm(x)
    s = 0.0
    for _ in 1:iters
        y = opvec(op, x, false)
        z = opvec(op, y, true)
        s = sqrt(norm(z))
        x = z ./ norm(z)
    end
    return s
end

# ----------------------------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------------------------

"""
    gkl_frame(op, ks; L, rng) -> (Q, info)

Two-sided singular frame (Definition def:Q without eigendirections): L steps of Golub–Kahan–
Lanczos bidiagonalisation in Float64 with full reorthogonalisation (twice), the top `ks` Ritz
singular triplets, Q = orthonormal factor of a column-pivoted QR of [V_ks U_ks] with columns
whose |R_ii| < 1e-8 |R_11| dropped, then one Householder QR.
"""
function gkl_frame(op::Op, ks::Int; L::Int = min(op.n - 1, max(2ks + 40, 3ks)),
        rng = MersenneTwister(11), droptol = 1e-8)
    n = op.n
    V = zeros(n, L + 1); U = zeros(n, L)
    α = zeros(L); β = zeros(L)
    v = randn(rng, n); v ./= norm(v); V[:, 1] = v
    big = 0.0
    Lreq = L
    for j in 1:L
        uu = opvec(op, V[:, j], false)
        j > 1 && (uu .-= β[j - 1] .* U[:, j - 1])
        for _ in 1:2
            j > 1 && (Uj = view(U, :, 1:(j - 1)); uu .-= Uj * (Uj' * uu))
        end
        α[j] = norm(uu)
        big = max(big, α[j])
        if α[j] <= 1e-13 * big             # breakdown: the Krylov space is exhausted
            L = j - 1; break
        end
        U[:, j] = uu ./ α[j]
        vv = opvec(op, U[:, j], true)
        vv .-= α[j] .* V[:, j]
        for _ in 1:2
            Vj = view(V, :, 1:j); vv .-= Vj * (Vj' * vv)
        end
        β[j] = norm(vv)
        if β[j] <= 1e-13 * big || j == L
            L = j; break
        end
        V[:, j + 1] = vv ./ β[j]
    end
    α = α[1:L]; β = β[1:L]
    ks = min(ks, L)
    Bd = Bidiagonal(α, β[1:(L - 1)], :U)       # A V_L = U_L B (+ β_L v_{L+1} e_Lᵀ on the Aᵀ side)
    F = svd(Matrix(Bd))
    Vk = V[:, 1:L] * F.V[:, 1:ks]
    Uk = U[:, 1:L] * F.U[:, 1:ks]
    M = hcat(Vk, Uk)
    P = qr(M, ColumnNorm())
    d = abs.(diag(P.R))
    r = count(>=(droptol * d[1]), d)
    Q0 = Matrix(P.Q)[:, 1:r]
    Q = Matrix(qr(Q0).Q)[:, 1:r]
    return Q, (; L, Lreq, ks, dimQ = r, ritz_sv = F.S[1:min(ks + 1, L)])
end

"""
    lanczos_frame(op, L; kmax, tol) -> (Q, info)

Frame of converged Ritz vectors (§4, label sec:herm) for symmetric A: L Lanczos steps in Float64
with full reorthogonalisation (twice), Ritz pairs (θ_i, s_i) of T_L, residuals r_i = β_L |(s_i)_L|,
converged set C = {i : r_i ≤ tol·max|θ|}, sorted by |θ| decreasing and truncated to `kmax`;
Q = Q_L S_C. No re-orthonormalisation: the defect α is certified.
"""
function lanczos_frame(op::Op, L::Int; kmax::Int = L, tol = 1e-10, rng = MersenneTwister(13))
    n = op.n
    Qk = zeros(n, L + 1)
    a = zeros(L); bb = zeros(L)
    q = randn(rng, n); q ./= norm(q); Qk[:, 1] = q
    for j in 1:L
        w = opvec(op, Qk[:, j], false)
        a[j] = dot(Qk[:, j], w)
        w .-= a[j] .* Qk[:, j]
        j > 1 && (w .-= bb[j - 1] .* Qk[:, j - 1])
        for _ in 1:2
            Qj = view(Qk, :, 1:j); w .-= Qj * (Qj' * w)
        end
        bb[j] = norm(w); Qk[:, j + 1] = w ./ bb[j]
    end
    T = SymTridiagonal(a, bb[1:(L - 1)])
    E = eigen(T)
    θ = E.values; S = E.vectors
    r = bb[L] .* abs.(S[L, :])
    scale = maximum(abs, θ)
    conv = findall(i -> r[i] <= tol * scale, eachindex(θ))
    conv = conv[sortperm(abs.(θ[conv]); rev = true)]
    C = conv[1:min(kmax, length(conv))]
    Q = Qk[:, 1:L] * S[:, C]
    return Q, (; L, nconv = length(conv), k = length(C), theta = θ[C], resid = r[C],
        theta_all_conv = θ[conv], betaL = bb[L])
end

# ----------------------------------------------------------------------------------------------
# Certified frame data: α, and s̄(a), γ̄1(a), γ̄2(a) as affine functions c0 + c1 a (§6.2, §6.5)
# ----------------------------------------------------------------------------------------------

"Apply op to the k columns of Q; returns fl(AQ) (adj=false) or fl(AᵀQ) (adj=true), n×k."
function op_cols(op::Op, Q::Matrix{Float64}, adj::Bool)
    Qt = Matrix(transpose(Q))            # k×n row block
    Y = similar(Qt)
    apply!(Y, op, Qt, adj)
    return Matrix(transpose(Y))
end

"Certified upper bound on the Frobenius norm of a Float64 matrix."
function frob_up(Q::Matrix{Float64})
    s = 0.0
    for x in Q
        s = add_up(s, mul_up(x, x))
    end
    return sqrt_up(s)
end

"Certified upper bound on ‖C‖₂ for a ball matrix C (n×k, k small) via its k×k Gram (eq:gram)."
function norm_up_gram(C::BallMatrix)
    G = C' * C
    return sqrt_up(svd_bound_L2_opnorm(G))
end

"""
    certify_frame(op, Q) -> NamedTuple

α ≥ ‖I − QᵀQ‖ (ball Gram, upper_bound_L2_opnorm); with X = fl(AQ), X' = fl(AᵀQ):
  ‖P⊥AQ‖  ≤ ‖X − Q(QᵀX)‖ + ε_A a ‖Q‖_F            (P⊥X = P⊥(X − QQᵀX), ‖P⊥‖ ≤ 1)
  ‖QᵀAQ‖  ≤ ‖QᵀX‖ + ‖Q‖ ε_A a ‖Q‖_F,  ‖Q‖ ≤ √(1+α)
and the isometry factors of prop:almostortho: γ̄2 = ‖P⊥AQ‖/√(1−α), γ̄1 = ‖P⊥AᵀQ‖/√(1−α),
s̄ = ‖QᵀAQ‖/(1−α). Each returned as (c0, c1) with bound = c0 + c1·a.
"""
function certify_frame(op::Op, Q::Matrix{Float64})
    n, k = size(Q)
    all(isfinite, Q) || error("frame has non-finite entries")
    t0 = time()
    if k == 0      # no deflation: P⊥ = I is applied as a copy; the formula of eq:epsP gives u
        return (; α = 0.0, εA = epsA(op), εP = epsP_up(0.0, n, 0), s = (0.0, 0.0),
            γ1 = (0.0, 0.0), γ2 = (0.0, 0.0), k, n, qF = 0.0, time = time() - t0)
    end
    Qb = BallMatrix(Q)
    G = Qb' * Qb
    E = BallMatrix(Matrix{Float64}(I, k, k)) - G
    α = upper_bound_L2_opnorm(E)
    α < 1 || error("α ≥ 1")
    εA = epsA(op)
    εP = epsP_up(α, n, k)
    X = op_cols(op, Q, false)
    Xt = op_cols(op, Q, true)
    Xb = BallMatrix(X); Xtb = BallMatrix(Xt)
    QX = Qb' * Xb
    nQX = svd_bound_L2_opnorm(QX)
    C2 = Xb - Qb * QX
    C1 = Xtb - Qb * (Qb' * Xtb)
    g2raw = norm_up_gram(C2)
    g1raw = norm_up_gram(C1)
    qF = frob_up(Q)
    sq1ma = sqrt_down(sub_down(1.0, α))
    ome = sub_down(1.0, α)
    sq1pa = sqrt_up(add_up(1.0, α))
    γ2 = (div_up(g2raw, sq1ma), div_up(mul_up(εA, qF), sq1ma))
    γ1 = (div_up(g1raw, sq1ma), div_up(mul_up(εA, qF), sq1ma))
    s = (div_up(nQX, ome), div_up(mul_up(mul_up(sq1pa, εA), qF), ome))
    return (; α, εA, εP, s, γ1, γ2, k, n, qF, time = time() - t0)
end

# ----------------------------------------------------------------------------------------------
# The sweep (Algorithm alg:fpsweep)
# ----------------------------------------------------------------------------------------------

struct Buffers{M <: AbstractMatrix{Float64}}
    W::M
    V::M
    T1::M
    T2::M
    S::M
    Tk::M
end
Buffers(b, n, k) = Buffers(zeros(b, n), zeros(b, n), zeros(b, n), zeros(b, n), zeros(b, n),
    zeros(b, k))

"out = fl(X − fl(fl(X Q) Qᵀ)): rows of X projected as y − Q fl(Qᵀ y) (eq:epsP)."
function proj!(out, X, Q, Tk, S)
    if size(Q, 2) == 0
        out .= X
        return out
    end
    mul!(Tk, X, Q)                 # inner products of length n
    mul!(S, Tk, transpose(Q))      # inner products of length k
    @. out = X - S
    return out
end

"out = fl(B X) (adj=false) or fl(Bᵀ X) (adj=true), B = P⊥AP⊥, as in lem:onestep."
function applyB!(out, op, X, Q, buf::Buffers, adj::Bool)
    proj!(buf.T1, X, Q, buf.Tk, buf.S)
    apply!(buf.T2, op, buf.T1, adj)
    proj!(out, buf.T2, Q, buf.Tk, buf.S)
    return out
end

"""
Certified ‖x‖₂ for each row of a b×n block: s = fl(Σ x_r²) summed in order, and
Σ x_r² ≤ s/(1 − γ_n) + nη (Higham (3.5), products included; nη covers gradual underflow).
Also the ℓ1 norms, Σ|x_r| ≤ s1/(1 − γ_n).
"""
function rownorms!(ss, s1, X, n)
    b = size(X, 1)
    fill!(ss, 0.0); fill!(s1, 0.0)
    @inbounds for r in 1:n
        @simd for c in 1:b
            x = X[c, r]
            ss[c] = ss[c] + x * x
            s1[c] = s1[c] + abs(x)
        end
    end
    return ss, s1
end

struct SweepResult
    p::Int
    n::Int
    # certified running sums for each order p' = 1..p (upper bounds)
    Scc::Vector{Float64}
    Scz::Vector{Vector{Float64}}
    Szz::Vector{Matrix{Float64}}
    # floating-point values
    fp_tr::Vector{Float64}        # Σ_j ‖w_p'‖² in plain floating point, p' = 1..p
    N1::Float64                   # certified ≥ max_j ‖v0‖₁
    Ninf::Float64                 # certified ≥ ‖Σ_j |v0|‖_∞
    N1_fp::Float64
    Ninf_fp::Float64
    time::Float64
    ncols::Int
    bookkeeping::Bool
    nthreads::Int
    blocksize::Int
    minnorm::Float64              # smallest computed ‖w_i‖,‖v_i‖ (under/overflow watch)
    maxnorm::Float64
end

"""
    sweep(op, Q, p; cols=1:n, blocksize, bookkeeping=true, nthreads)

The floating-point sweep of order m = 2p over the canonical basis vectors in `cols`. With
`bookkeeping` it accumulates, for every order p' ≤ p, the O(p²) running sums from which F of
eq:FI is evaluated (products of two of ‖w_{p'}‖, ‖w_i‖, ‖v_i‖, rounded upward), and N1, N∞.
Without it, it runs the bare Float64 sweep (for timing).
"""
function sweep(op::Op, Q::Matrix{Float64}, p::Int; cols = 1:(op.n),
        blocksize::Int = max(1, min(32, 2^22 ÷ op.n)), bookkeeping::Bool = true,
        nthreads::Int = Threads.nthreads())
    n = op.n
    k = size(Q, 2)
    cols = collect(cols)
    nb = cld(length(cols), blocksize)
    γn = gamma_up(n)
    onemγ = sub_down(1.0, γn)
    nη = mul_up(Float64(n), ETA)
    np = p
    # per-task accumulators
    nt = min(nthreads, nb)
    accs = Vector{Any}(undef, nt)
    counter = Threads.Atomic{Int}(0)
    oldblas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    t0 = time()
    Threads.@threads :static for t in 1:nt
        # `local`: these names are also assigned after the loop, and the loop body is a closure
        local Scc, Scz, Szz, fptr, acc, N1, N1fp, mn, mx, buf, xs, ys, xfp, ss, s1
        buf = Buffers(blocksize, n, k)
        Scc = zeros(np); Scz = [zeros(2pp) for pp in 1:np]; Szz = [zeros(2pp, 2pp) for pp in 1:np]
        fptr = zeros(np)
        acc = bookkeeping ? zeros(n) : zeros(0)
        N1 = 0.0; N1fp = 0.0
        xs = zeros(blocksize, np + 1)       # certified ‖w_i‖, i = 0..p
        ys = zeros(blocksize, np)           # certified ‖v_i‖, i = 0..p-1
        xfp = zeros(blocksize, np + 1)
        ss = zeros(blocksize); s1 = zeros(blocksize)
        mn = Inf; mx = 0.0
        while true
            ib = Threads.atomic_add!(counter, 1) + 1
            ib > nb && break
            J = cols[((ib - 1) * blocksize + 1):min(ib * blocksize, length(cols))]
            bJ = length(J)
            W = view(buf.W, 1:bJ, :); V = view(buf.V, 1:bJ, :)
            b2 = Buffers(W, V, view(buf.T1, 1:bJ, :), view(buf.T2, 1:bJ, :),
                view(buf.S, 1:bJ, :), view(buf.Tk, 1:bJ, :))
            fill!(W, 0.0)
            for c in 1:bJ
                W[c, J[c]] = 1.0
            end
            if bookkeeping
                xs[1:bJ, 1] .= 1.0          # w_0 = e_j exactly
                xfp[1:bJ, 1] .= 1.0
            end
            for i in 0:(np - 1)
                applyB!(V, op, W, Q, b2, false)            # v_i = fl(B w_i)
                if bookkeeping
                    rownorms!(ss, s1, V, n)
                    for c in 1:bJ
                        ys[c, i + 1] = sqrt_up(add_up(div_up(ss[c], onemγ), nη))
                        mn = min(mn, sqrt(ss[c])); mx = max(mx, sqrt(ss[c]))
                    end
                    if i == 0
                        for c in 1:bJ
                            N1 = max(N1, div_up(s1[c], onemγ))
                            N1fp = max(N1fp, s1[c])
                        end
                        @inbounds for r in 1:n
                            @simd for c in 1:bJ
                                acc[r] = acc[r] + abs(V[c, r])
                            end
                        end
                    end
                end
                applyB!(W, op, V, Q, b2, true)             # w_{i+1} = fl(Bᵀ v_i)
                if bookkeeping
                    rownorms!(ss, s1, W, n)
                    for c in 1:bJ
                        xs[c, i + 2] = sqrt_up(add_up(div_up(ss[c], onemγ), nη))
                        xfp[c, i + 2] = ss[c]
                        mn = min(mn, sqrt(ss[c])); mx = max(mx, sqrt(ss[c]))
                    end
                end
            end
            if bookkeeping
                for c in 1:bJ, pp in 1:np
                    cj = xs[c, pp + 1]
                    # z_d, d = 0..2pp-1: d = 2pp-1-2i ↦ ‖w_i‖, d = 2pp-2-2i ↦ ‖v_i‖
                    z = zeros(2pp)
                    for i in 0:(pp - 1)
                        z[2pp - 1 - 2i + 1] = xs[c, i + 1]
                        z[2pp - 2 - 2i + 1] = ys[c, i + 1]
                    end
                    Scc[pp] = add_up(Scc[pp], mul_up(cj, cj))
                    fptr[pp] += xfp[c, pp + 1]
                    for d in 1:(2pp)
                        Scz[pp][d] = add_up(Scz[pp][d], mul_up(cj, z[d]))
                        for d2 in 1:(2pp)
                            Szz[pp][d, d2] = add_up(Szz[pp][d, d2], mul_up(z[d], z[d2]))
                        end
                    end
                end
            end
        end
        accs[t] = (; Scc, Scz, Szz, fptr, acc, N1, N1fp, mn, mx)
    end
    el = time() - t0
    BLAS.set_num_threads(oldblas)
    if !bookkeeping
        return SweepResult(p, n, Float64[], Vector{Float64}[], Matrix{Float64}[], Float64[],
            NaN, NaN, NaN, NaN, el, length(cols), false, nt, blocksize, NaN, NaN)
    end
    Scc = zeros(np); Scz = [zeros(2pp) for pp in 1:np]; Szz = [zeros(2pp, 2pp) for pp in 1:np]
    fptr = zeros(np); acc = zeros(n); N1 = 0.0; N1fp = 0.0; mn = Inf; mx = 0.0
    for a in accs
        for pp in 1:np
            Scc[pp] = add_up(Scc[pp], a.Scc[pp])
            fptr[pp] += a.fptr[pp]
            Scz[pp] .= add_up.(Scz[pp], a.Scz[pp])
            Szz[pp] .= add_up.(Szz[pp], a.Szz[pp])
        end
        acc .+= a.acc                      # still a sum of nonnegative terms, ≤ n roundings each
        N1 = max(N1, a.N1); N1fp = max(N1fp, a.N1fp)
        mn = min(mn, a.mn); mx = max(mx, a.mx)
    end
    Ninf_fp = maximum(acc)
    Ninf = div_up(Ninf_fp, onemγ)
    (isfinite(mx) && mn > 1e-250) || @warn "norms near underflow/overflow" mn mx
    return SweepResult(p, n, Scc, Scz, Szz, fptr, N1, Ninf, N1fp, Ninf_fp, el, length(cols),
        true, nt, blocksize, mn, mx)
end

# ----------------------------------------------------------------------------------------------
# The checks of Proposition prop:fpchecks, evaluated with upward rounding
# ----------------------------------------------------------------------------------------------

aff_up(c, a) = add_up(c[1], mul_up(c[2], a))
hyp_up(x, y) = sqrt_up(add_up(mul_up(x, x), mul_up(y, y)))

"ε_B(a, g) of lem:onestep (the two constants for B and Bᵀ coincide), rounded upward."
function epsB_up(fr, a, g)
    εA, εP = fr.εA, fr.εP
    γ1 = aff_up(fr.γ1, a); γ2 = aff_up(fr.γ2, a)
    opP = add_up(1.0, εP)
    t1 = mul_up(εP, hyp_up(γ2, g))
    t2 = mul_up(mul_up(εA, a), opP)
    t3 = mul_up(εP, add_up(hyp_up(γ1, g), mul_up(a, add_up(εP, mul_up(εA, opP)))))
    return add_up(add_up(t1, t2), t3)
end

"F_a(g) of eq:FI, upper bound, from the running sums of order pp."
function F_up(S::SweepResult, pp::Int, εB, g)
    gp = ones(2 * (2pp))                       # g^0 .. g^{4pp-1}, rounded up
    for d in 2:length(gp)
        gp[d] = mul_up(gp[d - 1], g)
    end
    s1 = 0.0
    for d in 1:(2pp)
        s1 = add_up(s1, mul_up(gp[d], S.Scz[pp][d]))
    end
    s2 = 0.0
    for d in 1:(2pp), d2 in 1:(2pp)
        s2 = add_up(s2, mul_up(gp[d + d2 - 1], S.Szz[pp][d, d2]))
    end
    return add_up(add_up(S.Scc[pp], mul_up(mul_up(2.0, εB), s1)), mul_up(mul_up(εB, εB), s2))
end

"I_a(g) of eq:FI, upper bound."
I_up(S::SweepResult, εB) = mul_up(add_up(S.N1, mul_up(sqrt_up(Float64(S.n)), εB)),
    add_up(S.Ninf, mul_up(Float64(S.n), εB)))

pow_down(g, e) = (r = 1.0; for _ in 1:e; r = mul_down(r, g); end; r)

"Upper bound on the 2-norm of the nonnegative matrix [s γ1; γ2 g] of check (ii)."
function Mnorm_up(fr, a, g)
    s = aff_up(fr.s, a); γ1 = aff_up(fr.γ1, a); γ2 = aff_up(fr.γ2, a)
    # MᵀM = [P Qo; Qo R], λmax = (P+R)/2 + √(((P−R)/2)² + Qo²)
    Pu = add_up(mul_up(s, s), mul_up(γ2, γ2)); Pd = add_down(mul_down(s, s), mul_down(γ2, γ2))
    Ru = add_up(mul_up(γ1, γ1), mul_up(g, g)); Rd = add_down(mul_down(γ1, γ1), mul_down(g, g))
    Qo = add_up(mul_up(s, γ1), mul_up(γ2, g))
    dmax = max(sub_up(Pu, Rd), sub_up(Ru, Pd), 0.0)
    h = div_up(dmax, 2.0)
    λ = add_up(div_up(add_up(Pu, Ru), 2.0), sqrt_up(add_up(mul_up(h, h), mul_up(Qo, Qo))))
    return sqrt_up(λ)
end

check_ii(fr, a, g) = Mnorm_up(fr, a, g) < a

"Least ā (by bisection) with ‖M(ā, ḡ)‖ < ā; returns a value that passes (ii)."
function least_a(fr, g; iters = 200)
    lo = Mnorm_up(fr, 0.0, g)                  # ‖M(a)‖ ≥ ‖M(0)‖, so a ≤ lo fails
    hi = max(lo, 1e-300) * 2
    k = 0
    while !check_ii(fr, hi, g)
        hi *= 2; k += 1
        k > 60 && return NaN
    end
    for _ in 1:iters
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        check_ii(fr, mid, g) ? (hi = mid) : (lo = mid)
    end
    return hi
end

passF(S, pp, fr, a, g) = F_up(S, pp, epsB_up(fr, a, g), g) < pow_down(g, 4pp)
passI(S, fr, a, g) = I_up(S, epsB_up(fr, a, g)) < mul_down(g, g)

"""
Smallest ḡ ≥ g0 at which `test(ā(ḡ), ḡ)` passes, ā(ḡ) the least number passing (ii): bracket by
raising ḡ = g0(1+δ) with δ = 1e-15·2^j, then bisect. Returns (ḡ, ā) or (NaN, NaN) if nothing
passes below 2^20·g0.
"""
function least_g(test, fr, g0; iters = 200)
    lo = g0
    δ = 1e-15
    hi = g0 * (1 + δ)
    while true
        a = least_a(fr, hi)
        isfinite(a) && test(a, hi) && break
        lo = hi
        δ *= 2
        δ > 2.0^20 && return (NaN, NaN)
        hi = g0 * (1 + δ)
    end
    for _ in 1:iters
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        a = least_a(fr, mid)
        (isfinite(a) && test(a, mid)) ? (hi = mid) : (lo = mid)
    end
    a = least_a(fr, hi)
    @assert test(a, hi) && check_ii(fr, a, hi)
    return (hi, a)
end

"""
    fp_checks(S, fr, pp) -> NamedTuple

For the sweep of order m = 2pp: the floating-point trace value (Σ_j‖w_pp‖²)^{1/4pp}, the
floating-point interpolation value √(N1 N∞), and, for each half of check (i), the smallest ḡ
passing it with ā the least number passing (ii) at that ḡ.
"""
function fp_checks(S::SweepResult, fr, pp::Int; margin = 1e-2)
    t0 = time()
    trfp = S.fp_tr[pp]^(1 / (4pp))
    ipfp = sqrt(S.N1_fp * S.Ninf_fp)
    gF, aF = least_g((a, g) -> passF(S, pp, fr, a, g), fr, trfp)
    gI, aI = least_g((a, g) -> passI(S, fr, a, g), fr, ipfp)
    # at a fixed candidate ḡ = (1+margin) × floating-point trace value
    gc = trfp * (1 + margin)
    ac = least_a(fr, gc)
    passc = passF(S, pp, fr, ac, gc)
    εBF = isfinite(gF) ? epsB_up(fr, aF, gF) : NaN
    return (; m = 2pp, trfp, ipfp, gF, aF, gI, aI, εBF,
        gapF = gF / trfp - 1, gapI = gI / ipfp - 1, gc, passc, time = time() - t0)
end

# ----------------------------------------------------------------------------------------------
# The same sweep in ball arithmetic (BallArithmetic.jl): ball vectors through the projector
# (I − QQᵀ, point Q) and through the operator (a stored sparse or dense matrix, point entries).
# Certifies ‖B_Q‖ with B_Q = (I−QQᵀ)A(I−QQᵀ); since P⊥(I−QQᵀ) = P⊥, B = P⊥ B_Q P⊥ and
# ‖B‖ ≤ ‖B_Q‖, so the value also bounds ‖G22‖.
# ----------------------------------------------------------------------------------------------

function ball_proj(Y::BallMatrix, Q::Matrix{Float64})
    size(Q, 2) == 0 && return Y
    T = transpose(Q) * Y
    return Y - Q * T
end

"Certified upper bound of the 2-norm of each column of a ball matrix, with upward rounding."
function ball_colnorms_up(Y::BallMatrix)
    c = mid(Y); r = rad(Y)
    nc = size(c, 2)
    out = zeros(nc); mids = zeros(nc); rads = zeros(nc)
    for j in 1:nc
        sm = 0.0; sr = 0.0
        for i in axes(c, 1)
            sm = add_up(sm, mul_up(c[i, j], c[i, j]))
            sr = add_up(sr, mul_up(r[i, j], r[i, j]))
        end
        mids[j] = sqrt_up(sm); rads[j] = sqrt_up(sr)
        out[j] = add_up(mids[j], rads[j])
    end
    return out, mids, rads
end

"""
    ball_sweep(A, Q, p; cols, blocksize, nthreads) -> NamedTuple

A is the stored matrix (SparseMatrixCSC or Matrix). Returns Σ_{j∈cols} (upper bound of ‖H_Q^p e_j‖)²
rounded upward, per-order values, the ratio ‖rad‖/‖mid‖ after each application for the first
`ntrack` columns, and the time. A certificate of ‖B‖ only when cols = 1:n.
"""
function ball_sweep(A::AbstractMatrix{Float64}, Q::Matrix{Float64}, p::Int; cols = 1:size(A, 1),
        blocksize::Int = 16, nthreads::Int = Threads.nthreads(), ntrack::Int = 4)
    n = size(A, 1)
    At = A isa SparseMatrixCSC ? sparse(transpose(A)) : Matrix(transpose(A))
    cols = collect(cols)
    nb = cld(length(cols), blocksize)
    nt = min(nthreads, nb)
    res = Vector{Any}(undef, nt)
    oldblas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    counter = Threads.Atomic{Int}(0)
    t0 = time()
    Threads.@threads :static for t in 1:nt
        local S, track
        S = zeros(p)
        track = Dict{Int, Vector{Float64}}()
        while true
            ib = Threads.atomic_add!(counter, 1) + 1
            ib > nb && break
            J = cols[((ib - 1) * blocksize + 1):min(ib * blocksize, length(cols))]
            E = zeros(n, length(J))
            for (c, j) in enumerate(J)
                E[j, c] = 1.0
            end
            W = BallMatrix(E)
            # ratios[c]: ‖rad‖/‖mid‖ after each sub-step P⊥, A, P⊥, P⊥, Aᵀ, P⊥ of each i
            ratios = [Float64[] for _ in J]
            function rec!(Y)
                _, mm, rr = ball_colnorms_up(Y)
                for c in eachindex(J)
                    push!(ratios[c], rr[c] / mm[c])
                end
            end
            for i in 1:p
                Y1 = ball_proj(W, Q); rec!(Y1)
                Y2 = A * Y1; rec!(Y2)
                V = ball_proj(Y2, Q); rec!(V)
                Y1 = ball_proj(V, Q); rec!(Y1)
                Y2 = At * Y1; rec!(Y2)
                W = ball_proj(Y2, Q); rec!(W)
                nw, _, _ = ball_colnorms_up(W)
                for c in eachindex(J)
                    S[i] = add_up(S[i], mul_up(nw[c], nw[c]))
                end
            end
            for (c, j) in enumerate(J)
                j in cols[1:min(ntrack, length(cols))] && (track[j] = ratios[c])
            end
        end
        res[t] = (; S, track)
    end
    el = time() - t0
    BLAS.set_num_threads(oldblas)
    S = zeros(p); track = Dict{Int, Vector{Float64}}()
    for r in res
        S .= add_up.(S, r.S)
        merge!(track, r.track)
    end
    vals = [S[i] <= 0 ? 0.0 : _root_up(S[i], 4i) for i in 1:p]
    return (; S, vals, track, time = el, ncols = length(cols), nthreads = nt)
end

"r ≥ x^(1/e): start from the floating-point root and step up until r^e, rounded down, is ≥ x."
function _root_up(x, e)
    r = x^(1 / e)
    while pow_down(r, e) < x
        r = nextfloat(r)
    end
    return r
end

# ----------------------------------------------------------------------------------------------
# Baselines. (1) Float64 LAPACK: ‖A‖ = σ1(A), ‖G22‖ = σ1(P A P) with P = I − Q(QᵀQ)⁻¹Qᵀ formed
# densely. (2) Lower bounds ‖Mx‖/‖x‖ ≤ ‖M‖ evaluated in Double64 (≈ 106-bit) for M = A and
# M = P⊥AP⊥ (P⊥ applied with a Double64 solve of the k×k Gram), x from power iteration on MᵀM
# started at the Float64 top singular vector (or a random vector when no dense SVD was done).
# Double64 is not rigorous; its rounding is of relative order 1e-32.
# ----------------------------------------------------------------------------------------------

using DoubleFloats

function dense_baseline(A::AbstractMatrix{Float64}, Q::Matrix{Float64}; symmetric::Bool = false,
        dense::Bool = true, iters::Int = 30, seed::Int = 17, Apart = nothing)
    n = size(A, 1)
    k = size(Q, 2)
    t0 = time()
    oldblas = BLAS.get_num_threads()
    BLAS.set_num_threads(min(64, Sys.CPU_THREADS))
    nA = NaN; nG = NaN; xA = nothing; xG = nothing
    if dense
        Ad = Matrix(A)
        if k > 0
            Gi = inv(Symmetric(Q' * Q))
            P = Matrix{Float64}(I, n, n) - Q * (Gi * Q')
            G22 = P * Ad * P
        else
            G22 = Ad
        end
        if Apart !== nothing
            nA = Apart.normA; xA = Apart.xA
        elseif symmetric
            EA = eigen(Symmetric(Ad)); iA = argmax(abs.(EA.values))
            nA = abs(EA.values[iA]); xA = EA.vectors[:, iA]
        else
            FA = svd(Ad); nA = FA.S[1]; xA = FA.V[:, 1]
        end
        if symmetric
            EG = eigen(Symmetric((G22 + G22') / 2)); iG = argmax(abs.(EG.values))
            nG = abs(EG.values[iG]); xG = EG.vectors[:, iG]
        else
            FG = svd(G22); nG = FG.S[1]; xG = FG.V[:, 1]
        end
    end
    tdense = time() - t0
    BLAS.set_num_threads(oldblas)
    rng = MersenneTwister(seed)
    Apart !== nothing && xA === nothing && (xA = Apart.xA)
    xA === nothing && (xA = randn(rng, n); iters = max(iters, 100))
    xG === nothing && (xG = randn(rng, n))
    T = Double64
    Ad2 = SparseMatrixCSC{T, Int}(A isa SparseMatrixCSC ? A : sparse(A))
    Adt = sparse(transpose(Ad2))
    Qd = T.(Q)
    Gd = k > 0 ? inv(Qd' * Qd) : zeros(T, 0, 0)
    Pd(x) = k > 0 ? x - Qd * (Gd * (Qd' * x)) : x
    if Apart !== nothing && hasproperty(Apart, :lbA)
        lbA = Apart.lbA; nA = Apart.normA
    else
        x = T.(xA)
        for _ in 1:iters
            x = Adt * (Ad2 * x); x ./= norm(x)
        end
        lbA = Float64(norm(Ad2 * x) / norm(x), RoundDown)
    end
    x = Pd(T.(xG))
    for _ in 1:iters
        x = Pd(Adt * Pd(Pd(Ad2 * Pd(x)))); x ./= norm(x)
    end
    lbG = Float64(norm(Pd(Ad2 * Pd(x))) / norm(x), RoundDown)
    return (; normA = nA, normG22 = nG, lbA, lbG, xA, tdense, time = time() - t0)
end

end # module
