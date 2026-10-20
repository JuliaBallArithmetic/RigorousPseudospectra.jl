# Common setup for the Belousov-Zhabotinsky runs: the stored matrix L (the midpoint of RIM's
# interval Ulam matrix), the periodic uniform noise kernel N in its :block summation scheme, the
# matrix-free operator A = N∘L, and its rounding constant.
#
# The object certified is the floating-point matrix, not the transfer operator: L is
# `mid.(RIM's interval Ulam matrix)` and nothing about the discretisation error is claimed here.
#
# The rounding constant.  The paper's eq:roundA is relative, ‖fl(Ay) − Ay‖ ≤ ε_A‖A‖‖y‖.  For the
# composite A = N∘L applied as y ↦ fl(N fl(L y)) the natural constant is absolute.  With
#   fl(Ly) = Ly + e₁,      ‖e₁‖ ≤ ε_L‖L‖‖y‖,          ε_L = √w γ_w, w = max nnz in a row or column,
#   fl(Nv) = Nv + e₂,      |e₂|ᵢ ≤ γ_{2l+2}(N|v|)ᵢ  hence ‖e₂‖ ≤ γ_{2l+2}‖N‖‖v‖  (:block scheme),
# we get fl(Ay) − Ay = N e₁ + e₂ with v = Ly + e₁, so
#   ‖fl(Ay) − Ay‖ ≤ ‖N‖‖L‖ (ε_L + γ_{2l+2}(1 + ε_L)) ‖y‖ =: η‖y‖.
# The adjoint y ↦ fl(Lᵀ fl(N y)) gives ‖N‖‖L‖(γ_{2l+2} + ε_L(1 + γ_{2l+2}))‖y‖, the same number.
# ‖N‖ = 1 (N is symmetric doubly stochastic), ‖L‖ ≤ √(‖L‖₁‖L‖_∞) is computed once from L.
# The bound is proportional to ‖y‖, and the package's absolute slot η_A is not (operator.jl:
# ‖fl(Ay) − Ay‖ ≤ ε_A‖A‖‖y‖ + η_A), so η goes into ε_A: with a certified lower bound a_lo ≤ ‖A‖
# (`norm_lower`), ε_A := η/a_lo gives ε_A·a ≥ η for every a ≥ ‖A‖ that prop:fpchecks tries, and
# ε_A‖A‖‖y‖ ≥ η‖y‖. The absolute slot keeps only gradual underflow, as for a stored matrix with
# w + 2l + 2 products per entry (`_eta_stored`). Until 2026-10-20 η was passed as η_A, which drops
# the factor ‖y‖ wherever the sweep's vectors have norm above one (bz_linop).

using RigorousInvariantMeasures
using RigorousPseudospectra
using LinearAlgebra, SparseArrays, Printf, Random
import Serialization
import BallArithmetic
using BallArithmetic: mul_up, add_up, sqrt_up, sum_up, gamma_bound, mul_down, add_down, sub_down,
    sqrt_down, div_up, div_down

const OUT = "/home/isaia/ResearchData/KrylovPseudospectra_Data/bz"
ispowerof2(k) = k > 0 && (k & (k - 1)) == 0
import IntervalArithmetic
using IntervalArithmetic: mid
const RP = RigorousPseudospectra
const RIM = RigorousInvariantMeasures

# ---------------------------------------------------------------------------------------------
# the stored matrix

# The map is RIM's `BZ()`. Its cube-root exponent is written `(…)^(1//3)`; with the interval
# exponent `interval(1.0)/3` it used to have, the Newton step for the preimages takes a derivative
# through `Taylor1{Interval{Float64}} ^ Interval{Float64}` and TaylorSeries throws
# `MethodError: no method matching Int64(::Interval{Float64})`. That fix is RigorousInvariantMeasures
# commit 1cfdca5, in `src/sample_dynamics.jl`.

"""
    build_L(k) -> (L::SparseMatrixCSC{Float64}, info)

Midpoint of RIM's interval Ulam matrix for the BZ map on a partition of `k` cells.
"""
function build_L(k::Integer)
    cache = joinpath(OUT, "L_$(round(Int, log2(k))).jls")
    # The cache lives in the annex, so on a machine whose annex has not fetched the content the
    # path holds a pointer file and not a matrix; assemble in place rather than fail.
    if ispowerof2(k) && isfile(cache)
        t0 = time()
        L = try
            Serialization.deserialize(cache)
        catch err
            @warn "the cached operator could not be read; assembling instead" cache err
            nothing
        end
        if L isa AbstractMatrix
            return L, (; k, t_assemble = time() - t0, nnz = nnz(L), bytes = 0,
                maxrad = NaN, integral_preserving = true, cached = true)
        end
    end
    t0 = time(); b0 = Base.gc_bytes()
    D = RIM.BZ()
    B = RIM.Ulam(k)
    Q = RIM.DiscretizedOperator(B, D)
    t_assemble = time() - t0
    Li = Q.L
    L = SparseMatrixCSC(Li.m, Li.n, copy(Li.colptr), copy(Li.rowval), Float64.(mid.(Li.nzval)))
    dropzeros!(L)
    bytes = Base.gc_bytes() - b0
    return L, (; k, t_assemble, nnz = nnz(L), bytes,
        maxrad = maximum(IntervalArithmetic.radius.(Li.nzval); init = 0.0),
        integral_preserving = Q isa RIM.IntegralPreservingDiscretizedOperator)
end

"Upper bound on the largest column sum of `|L|` for a sparse `L`, summed over the stored entries."
function _max_colsum_up(L::SparseMatrixCSC)
    vals = nonzeros(L)
    return maximum(j -> sum_up(abs(vals[p]) for p in nzrange(L, j)), 1:size(L, 2); init = 0.0)
end

# ---------------------------------------------------------------------------------------------
# the operator A = N∘L and its constant

"Half-width in cells for a noise of half-width ξ (a fraction of the interval)."
noise_halfwidth(ξ::Real, k::Integer) = max(1, round(Int, ξ * k))

"""
    make_operator(L, l) -> (Amul, Aadj, etaA, info)

`A = N∘L` applied matrix-free, with `N` the periodic uniform Ulam kernel of half-width `l` cells
in the `:block` summation scheme, and the absolute rounding constant η derived in the header.
Each call allocates its own kernel, so `Amul`/`Aadj` are safe to call from several threads.
"""
function make_operator(L::SparseMatrixCSC, l::Integer)
    k = size(L, 1)
    Lt = sparse(transpose(L))
    B = RIM.Ulam(k)
    nthr = Threads.maxthreadid()
    Ks = [RIM.UniformKernelUlamPeriodic(B, l; summation = :block) for _ in 1:nthr]
    N!(v) = RIM.mul!(Ks[Threads.threadid()], v)
    Amul(x) = N!(L * Vector{Float64}(x))
    Aadj(x) = Lt * N!(copy(Vector{Float64}(x)))
    # the constant of the header, every operation rounded upward
    w = RP._stencil_width(L)
    εL = mul_up(sqrt_up(Float64(w)), gamma_bound(w, Float64))
    normL = sqrt_up(mul_up(_max_colsum_up(L), _max_colsum_up(Lt)))
    γN = gamma_bound(2l + 2, Float64)
    η = mul_up(normL, add_up(εL, mul_up(γN, add_up(1.0, εL))))
    return Amul, Aadj, η, (; w, εL, normL, γN, l, etaA = η, nnzL = nnz(L))
end

"""
    norm_lower(Amul, k, η, ηA) -> a_lo

A certified lower bound on ‖A‖ from one application to the constant vector x = (1, …, 1):
‖A‖ ≥ ‖Ax‖/‖x‖ ≥ (‖fl(Ax)‖ − η‖x‖ − η_A)/‖x‖, with ‖fl(Ax)‖ summed rounding downward and
‖x‖ = √k rounded upward. Errors when the bound is not positive.
"""
function norm_lower(Amul, k::Integer, η::Float64, ηA::Float64)
    y = Amul(ones(k))
    s = 0.0
    for v in y
        s = add_down(s, mul_down(v, v))
    end
    nx = sqrt_up(Float64(k))
    a_lo = div_down(sub_down(sub_down(sqrt_down(s), mul_up(η, nx)), ηA), nx)
    a_lo > 0 || error("norm_lower: no positive lower bound on ‖A‖ ($(a_lo))")
    return a_lo
end

"""
    bz_linop(L, l) -> (Amul, Aadj, op, info)

The operator A = N∘L of `make_operator` as a `LinOp` with its rounding model: ε_A = η/a_lo with
a_lo = `norm_lower`, and η_A the underflow term of w + 2l + 2 products per entry (see the header).
"""
function bz_linop(L::SparseMatrixCSC, l::Integer)
    k = size(L, 1)
    Amul, Aadj, η, oi = make_operator(L, l)
    etaA = RP._eta_stored(k, oi.w + 2l + 2, Float64)
    a_lo = norm_lower(Amul, k, η, etaA)
    epsA = div_up(η, a_lo)
    op = RP.LinOp(Amul, Aadj, k; epsA, etaA)
    return Amul, Aadj, op, merge(oi, (; eta_rel = η, a_lo, epsA, etaA))
end

"Dense N for a small k, to check that the periodic uniform kernel is symmetric."
function dense_N(k::Integer, l::Integer)
    B = RIM.Ulam(k)
    K = RIM.UniformKernelUlamPeriodic(B, l; summation = :block)
    N = zeros(k, k)
    for j in 1:k
        e = zeros(k); e[j] = 1.0
        N[:, j] = RIM.mul!(K, e)
    end
    return N
end

"Dense A = N L, for the small-k comparisons against dense truth."
dense_A(L::SparseMatrixCSC, l::Integer) = dense_N(size(L, 1), l) * Matrix(L)


