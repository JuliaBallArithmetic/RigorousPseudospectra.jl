# §4 (sec:norm): the far block B = G22 = P⊥AP⊥, bounded by the certificate eq:mincert,
#     ‖B‖ ≤ min{ √(‖B‖₁‖B‖∞),  (tr H^m)^{1/2m} },   H = B*B,
# the interpolation bound eq:interp and the trace bound of prop:ladder, evaluated by the sweep over
# the canonical basis eq:sweep in floating point (sec:fpsweep, Algorithm alg:fpsweep) and certified
# afterwards by the two scalar checks of prop:fpchecks.
#
# One run of order m computes, for every column j, the chain x_0 = e_j, x_1 = fl(B x_0),
# x_2 = fl(B* x_1), …, x_m, alternating B and B*. For every order t ≤ m, Σ_j ‖x_t‖² is the
# floating-point value of tr H^t (for t = 2p, x_t = w_p ≈ H^p e_j; for t = 2p+1, x_t ≈ B H^p e_j and
# ‖BH^pe_j‖² summed over j is tr H^{2p+1}, eq:sweep), so the run returns every lower order with it.
# The paper writes the error analysis for even orders; for a general order t the recursion
# err_{s+1} ≤ g err_s + ε_B ‖x_s‖ + η_B (lem:onestep, with g ≥ ‖B‖ and the absolute term η_B
# described in operator.jl) gives
#     ‖x_t − exact‖ ≤ ε_B E_{j,t}(g) + η_B S_t(g),
#     E_{j,t}(g) = Σ_{s<t} g^{t−1−s} ‖x_s‖,   S_t(g) = Σ_{s<t} g^s,
# which for t = 2p and η_B = 0 is the paper's E_j(g). Hence, with ν_{j,s} ≥ ‖x_{j,s}‖ certified,
#     ‖B‖^{2t} ≤ tr H^t ≤ F_t(g) := Σ_j (ν_{j,t} + ε_B E_{j,t}(g) + η_B S_t(g))²       (eq:FI)
# and, with N1 ≥ max_j ‖x_{j,1}‖₁, N∞ ≥ ‖Σ_j |x_{j,1}|‖∞,
#     ‖B‖² ≤ I(g) := (N1 + √n(ε_B + η_B))(N∞ + n(ε_B + η_B)).
# F_t expands into the sums Γ[s,s'] = Σ_j ν_{j,s}ν_{j,s'} and Σ_j ν_{j,s}, s, s' ≤ m: O(m²) scalars,
# accumulated with upward rounding during the sweep; the checks are evaluated after it.

# ------------------------------------------------------------------------------------------------
# Rounding constants (eq:epsP, lem:onestep)
# ------------------------------------------------------------------------------------------------

"""
    epsP_up(α, n, k, T) -> ε_P

eq:epsP: `‖fl(y − Q fl(Q*y)) − P⊥y‖ ≤ ε_P ‖y‖` with
`ε_P = α + u + (1+u)(1+α)√k (u_n + u_k(1 + √k u_n))`, `u_ℓ = γ_ℓ` (complex: `γ_{ℓ+2}`).
"""
function epsP_up(α::Real, n::Integer, k::Integer, T::Type = Float64)
    u = unit_roundoff(Float64)
    k == 0 && return u
    un = gammau(n, T)
    uk = gammau(k, T)
    sk = sqrtu(Float64(k))
    inner = addu(un, mulu(uk, addu(1.0, mulu(sk, un))))
    t = mulu(mulu(mulu(addu(1.0, u), addu(1.0, Float64(α))), sk), inner)
    return addu(addu(Float64(α), u), t)
end

"""
    etaP_up(n, k, T) -> η_P

The absolute (underflow) term of the projector, not in the paper: the inner products of length n
of `fl(Q*y)` carry at most `nη` each (4nη complex), `k` of them, multiplied by `‖Q‖ ≤ √2`; the
inner products of length k of `fl(Qt)` at most `kη` each, `n` of them; the subtraction is exact
in the subnormal range.
"""
function etaP_up(n::Integer, k::Integer, T::Type = Float64)
    k == 0 && return 0.0
    η = eta_min(Float64)
    c = T <: Complex ? 4.0 : 1.0
    a = mulu(mulu(sqrtu(2.0), sqrtu(Float64(k))), mulu(Float64(n), c * η))
    b = mulu(sqrtu(Float64(n)), mulu(Float64(k), c * η))
    return addu(a, b)
end

"""
    onestep_up(a, g, γ1, γ2, εA, ηA, εP, ηP) -> (ε_B, η_B)

Lemma lem:onestep: `‖fl(By) − By‖ ≤ ε_B ‖y‖ + η_B` for `B = P⊥AP⊥` applied as
`fl(P⊥ fl(A fl(P⊥y)))`, whenever `‖A‖ ≤ a`, `‖G12‖ ≤ γ1`, `‖G21‖ ≤ γ2`, `‖G22‖ ≤ g`:
`ε_B = ε_P√(γ2² + g²) + ε_A a(1+ε_P) + ε_P(√(γ1² + g²) + a(ε_P + ε_A(1+ε_P)))`, the larger of the
constants for `B` and for `B*` (γ1 and γ2 exchanged), and
`η_B = √(γ2²+g²)η_P + ε_A a η_P + η_A + ε_P(aη_P + ε_A aη_P + η_A) + η_P`, the absolute term
(zero when η_A = η_P = 0, the paper's case).
"""
function onestep_up(a, g, γ1, γ2, εA, ηA, εP, ηP)
    hyp(x, y) = sqrtu(addu(mulu(x, x), mulu(y, y)))
    opP = addu(1.0, εP)
    function one(gl, gr)            # gl couples through P⊥A, gr through AP⊥
        t1 = mulu(εP, hyp(gl, g))
        t2 = mulu(mulu(εA, a), opP)
        t3 = mulu(εP, addu(hyp(gr, g), mulu(a, addu(εP, mulu(εA, opP)))))
        ε = addu(addu(t1, t2), t3)
        aηP = mulu(a, ηP)
        η = addu(addu(addu(mulu(hyp(gl, g), ηP), mulu(εA, aηP)), ηA),
            addu(mulu(εP, addu(addu(aηP, mulu(εA, aηP)), ηA)), ηP))
        return ε, η
    end
    e1, h1 = one(γ2, γ1)            # B
    e2, h2 = one(γ1, γ2)            # B*
    return max(e1, e2), max(h1, h2)
end

# ------------------------------------------------------------------------------------------------
# The sweep (Algorithm alg:fpsweep)
# ------------------------------------------------------------------------------------------------

"""
    SweepSums

The running sums of one sweep of order `m` (upper bounds, rounded upward): `Γ[s+1, s'+1] ≥
Σ_j ν_{j,s}ν_{j,s'}`, `Σν[s+1] ≥ Σ_j ν_{j,s}` for `s, s' = 0:m`, `N1`, `Ninf`; and the plain
floating-point values `fp[t] = Σ_j ‖x_{j,t}‖²` (t = 1:m), `N1fp`, `Ninffp`, used only to choose
where to start the checks.
"""
struct SweepSums
    m::Int
    n::Int
    Γ::Matrix{Float64}
    Σν::Vector{Float64}
    fp::Vector{Float64}
    N1::Float64
    Ninf::Float64
    N1fp::Float64
    Ninffp::Float64
    N1s::Vector{Float64}
    Ninfs::Vector{Float64}
    N1fps::Vector{Float64}
    Ninffps::Vector{Float64}
    finite::Bool
    scale::Float64
    nmatvec::Int
    nthreads::Int
    blocksize::Int
    time::Float64
end

# out = fl(X − Q fl(Q* X)), the projector of eq:epsP applied to the columns of X
function _proj!(out, X, Q, Tk, S)
    if size(Q, 2) == 0
        out .= X
        return out
    end
    mul!(Tk, Q', X)
    mul!(S, Q, Tk)
    out .= X .- S
    return out
end

# out = fl(B X) or fl(B* X), B = P⊥ (c·A) P⊥ with c = scale a power of two
function _applyB!(out, X, op::LinOp, Q, scale, T1, T2, Tk, S, adj::Bool)
    _proj!(T1, X, Q, Tk, S)
    adj ? op.adj!(T2, T1) : op.mul!(T2, T1)
    scale == 1 || (T2 .*= scale)
    _proj!(out, T2, Q, Tk, S)
    return out
end

"""
    trace_sweep(op, Q, m; scale = 1.0, blocksize, nthreads, cols = 1:n) -> SweepSums

The floating-point sweep of order `m` over the canonical basis (eq:sweep), for the operator
`c·A` with `c = scale` a power of two (Remark rem:impl: the scale is exact in binary floating
point, apart from underflow, which the absolute terms cover). Columns are processed in blocks of
`blocksize`, the projector as two matrix products per block; blocks are independent tasks on
`nthreads` threads, each keeping its own sums, combined with upward rounding at the end. Memory:
`O(n·blocksize)` per thread plus one length-`n` accumulator per thread for `Σ_j |x_{j,1}|`.
For each computed vector the norm bound is `ν ≥ ‖x‖` from the computed sum of squares `ŝ`,
`‖x‖² ≤ ŝ/(1 − γ_n) + 2nη`, and the ℓ¹ norm `‖x‖₁ ≤ ŝ₁/(1 − γ_{n+2}) + nη` (complex moduli are
computed by `hypot`, whose error is taken as at most 2u, hence the extra terms).
"""
function trace_sweep(op::LinOp, Q::AbstractMatrix, m::Integer; scale::Real = 1.0,
        blocksize::Integer = max(1, min(64, 2^22 ÷ max(op.n, 1))),
        nthreads::Integer = Threads.nthreads(), cols = 1:(op.n))
    t0 = time()
    n = op.n
    k = size(Q, 2)
    T = promote_type(op.T, eltype(Q)) <: Complex ? ComplexF64 : Float64
    Qm = Matrix{T}(Q)
    scale = Float64(scale)
    η = eta_min(Float64)
    γsq = gammau(n, T)
    γl1 = gammau(n + 2, T)
    onemsq = subd(1.0, γsq)
    oneml1 = subd(1.0, γl1)
    ηsq = mulu(2.0 * n, η)
    ηl1 = mulu(Float64(n), η)
    cols = collect(cols)
    b = Int(blocksize)
    nb = cld(length(cols), b)
    nt = max(1, min(Int(nthreads), nb))
    counter = Threads.Atomic{Int}(0)
    results = Vector{Any}(undef, nt)
    oldblas = BLAS.get_num_threads()
    nt > 1 && BLAS.set_num_threads(1)
    try
        Threads.@threads :static for tid in 1:nt
            # `local`: these names are assigned again after the loop, and the loop body is a
            # closure; without it every task would share (and race on) the same accumulators
            local X, Yv, T1, T2, S, Tk, Γ, Σν, fp, acc, N1, N1fp, fin, ν, nmv
            X = zeros(T, n, b)
            Yv = zeros(T, n, b)
            T1 = zeros(T, n, b)
            T2 = zeros(T, n, b)
            S = zeros(T, n, b)
            Tk = zeros(T, k, b)
            Γ = zeros(m + 1, m + 1)
            Σν = zeros(m + 1)
            fp = zeros(m)
            acc = zeros(n, m)
            N1 = zeros(m)
            N1fp = zeros(m)
            fin = true
            ν = zeros(b, m + 1)
            nmv = 0
            while true
                ib = Threads.atomic_add!(counter, 1) + 1
                ib > nb && break
                J = cols[((ib - 1) * b + 1):min(ib * b, length(cols))]
                bJ = length(J)
                Xv = view(X, :, 1:bJ)
                Yw = view(Yv, :, 1:bJ)
                T1v = view(T1, :, 1:bJ)
                T2v = view(T2, :, 1:bJ)
                Sv = view(S, :, 1:bJ)
                Tkv = view(Tk, :, 1:bJ)
                fill!(Xv, zero(T))
                for c in 1:bJ
                    Xv[J[c], c] = one(T)
                    ν[c, 1] = 1.0
                end
                cur, nxt = Xv, Yw
                for s in 1:m
                    _applyB!(nxt, cur, op, Qm, scale, T1v, T2v, Tkv, Sv, iseven(s))
                    nmv += bJ
                    for c in 1:bJ
                        ss = 0.0
                        s1 = 0.0
                        @inbounds for r in 1:n
                            x = nxt[r, c]
                            ss += abs2(x)
                            s1 += abs(x)
                        end
                        fin &= isfinite(ss)
                        ν[c, s + 1] = sqrtu(addu(divu(ss, onemsq), ηsq))
                        fp[s] += ss
                        # the ℓ¹ and ℓ∞ sums at every order, not only the first: the column sums
                        # give ‖B^s‖₁ and the accumulated row sums give ‖B^s‖_∞, whose geometric
                        # mean bounds ‖B^s‖₂ and is far below the Frobenius norm for a sparse or
                        # banded block
                        N1[s] = max(N1[s], addu(divu(s1, oneml1), ηl1))
                        N1fp[s] = max(N1fp[s], s1)
                        @inbounds for r in 1:n
                            acc[r, s] += abs(nxt[r, c])
                        end
                    end
                    cur, nxt = nxt, cur
                end
                for c in 1:bJ
                    for s in 1:(m + 1)
                        Σν[s] = addu(Σν[s], ν[c, s])
                        for s2 in 1:(m + 1)
                            Γ[s, s2] = addu(Γ[s, s2], mulu(ν[c, s], ν[c, s2]))
                        end
                    end
                end
            end
            results[tid] = (; Γ, Σν, fp, acc, N1, N1fp, fin, nmv)
        end
    finally
        nt > 1 && BLAS.set_num_threads(oldblas)
    end
    Γ = zeros(m + 1, m + 1)
    Σν = zeros(m + 1)
    fp = zeros(m)
    acc = zeros(n, m)
    N1s = zeros(m)
    N1fps = zeros(m)
    fin = true
    nmv = 0
    for r in results
        Γ .= addu.(Γ, r.Γ)
        Σν .= addu.(Σν, r.Σν)
        fp .+= r.fp
        acc .+= r.acc
        N1s .= max.(N1s, r.N1)
        N1fps .= max.(N1fps, r.N1fp)
        fin &= r.fin
        nmv += r.nmv
    end
    Ninffps = [n == 0 ? 0.0 : maximum(view(acc, :, s)) for s in 1:m]
    Ninfs = [addu(divu(x, oneml1), ηl1) for x in Ninffps]
    N1 = m == 0 ? 0.0 : N1s[1]
    Ninf = m == 0 ? 0.0 : Ninfs[1]
    N1fp = m == 0 ? 0.0 : N1fps[1]
    Ninffp = m == 0 ? 0.0 : Ninffps[1]
    fin &= all(isfinite, Γ) && all(isfinite, N1s) && all(isfinite, Ninfs)
    return SweepSums(m, n, Γ, Σν, fp, N1, Ninf, N1fp, Ninffp, N1s, Ninfs, N1fps, Ninffps,
        fin, scale, nmv, nt, b, time() - t0)
end

# ------------------------------------------------------------------------------------------------
# The checks of Proposition prop:fpchecks, evaluated with upward rounding after the sweep
# ------------------------------------------------------------------------------------------------

"Upper bound of F_t(g) of eq:FI from the running sums."
function F_up(S::SweepSums, t::Int, εB, ηB, g)
    # coefficients w_s: w_t = 1, w_s = ε_B g^{t−1−s} for s < t (index s+1 in Γ)
    w = zeros(t + 1)
    w[t + 1] = 1.0
    for s in 0:(t - 1)
        w[s + 1] = mulu(εB, powu(g, t - 1 - s))
    end
    q = 0.0
    for s in 1:(t + 1), s2 in 1:(t + 1)
        q = addu(q, mulu(mulu(w[s], w[s2]), S.Γ[s, s2]))
    end
    if ηB > 0
        St = 0.0
        for d in 0:(t - 1)
            St = addu(St, powu(g, d))
        end
        h = mulu(ηB, St)
        lin = 0.0
        for s in 1:(t + 1)
            lin = addu(lin, mulu(w[s], S.Σν[s]))
        end
        q = addu(q, addu(mulu(mulu(2.0, h), lin), mulu(Float64(S.n), mulu(h, h))))
    end
    return q
end

"""
Upper bound on `‖B^t‖₂²` from the ℓ¹ and ℓ∞ sums at order `t`, by interpolation:
`‖M‖₂ ≤ sqrt(‖M‖₁‖M‖_∞)`. The rounding allowance is the one `I_up` uses, applied at the order in
question. Returns `Inf` when the order was not accumulated.
"""
function I_up(S::SweepSums, t::Int, εB, ηB)
    (1 <= t <= length(S.N1s)) || return Inf
    e = addu(εB, ηB)
    return mulu(addu(S.N1s[t], mulu(sqrtu(Float64(S.n)), e)),
        addu(S.Ninfs[t], mulu(Float64(S.n), e)))
end

"Upper bound of I(g) of eq:FI."
function I_up(S::SweepSums, εB, ηB)
    e = addu(εB, ηB)
    return mulu(addu(S.N1, mulu(sqrtu(Float64(S.n)), e)), addu(S.Ninf, mulu(Float64(S.n), e)))
end

"Upper bound on the spectral norm of the nonnegative 2×2 matrix [s γ1; γ2 g]."
function norm2x2_up(s, γ1, γ2, g)
    Pu = addu(mulu(s, s), mulu(γ2, γ2))
    Pd = addd(muld(s, s), muld(γ2, γ2))
    Ru = addu(mulu(γ1, γ1), mulu(g, g))
    Rd = addd(muld(γ1, γ1), muld(g, g))
    Qo = addu(mulu(s, γ1), mulu(γ2, g))
    h = divu(max(subu(Pu, Rd), subu(Ru, Pd), 0.0), 2.0)
    λ = addu(divu(addu(Pu, Ru), 2.0), sqrtu(addu(mulu(h, h), mulu(Qo, Qo))))
    return sqrtu(λ)
end

"""
    FPCheckData

Everything the checks of prop:fpchecks need, in the units of the scaled operator `c·A`:
the affine isometry bounds `s(a)`, `γ1(a)`, `γ2(a)` (coefficients `(c0, c1)`), `εA`, `ηA`, `εP`,
`ηP`, and the sweep sums.
"""
struct FPCheckData
    S::SweepSums
    s::NTuple{2, Float64}
    γ1::NTuple{2, Float64}
    γ2::NTuple{2, Float64}
    εA::Float64
    ηA::Float64
    εP::Float64
    ηP::Float64
end

function _consts(D::FPCheckData, a, g)
    γ1 = aff(D.γ1, a)
    γ2 = aff(D.γ2, a)
    return onestep_up(a, g, γ1, γ2, D.εA, D.ηA, D.εP, D.ηP)
end

"(ii) of prop:fpchecks: ‖[s(a) γ1(a); γ2(a) g]‖ < a."
check_ii(D::FPCheckData, a, g) = norm2x2_up(aff(D.s, a), aff(D.γ1, a), aff(D.γ2, a), g) < a

"(i), trace half, order t: F_t(g) < g^{2t}."
function check_iF(D::FPCheckData, t::Int, a, g)
    εB, ηB = _consts(D, a, g)
    return F_up(D.S, t, εB, ηB, g) < powd(g, 2t)
end

"(i), interpolation half: I(g) < g²."
function check_iI(D::FPCheckData, a, g)
    εB, ηB = _consts(D, a, g)
    return I_up(D.S, εB, ηB) < muld(g, g)
end

"Least `a` (by bisection) passing (ii) at `g`; `NaN` if none is found."
function least_a(D::FPCheckData, g; iters::Int = 200)
    lo = norm2x2_up(aff(D.s, 0.0), aff(D.γ1, 0.0), aff(D.γ2, 0.0), g)
    hi = max(2lo, floatmin(Float64))
    j = 0
    while !check_ii(D, hi, g)
        hi *= 2
        j += 1
        (j > 1100 || !isfinite(hi)) && return NaN
    end
    for _ in 1:iters
        mid_ = (lo + hi) / 2
        (mid_ <= lo || mid_ >= hi) && break
        check_ii(D, mid_, g) ? (hi = mid_) : (lo = mid_)
    end
    return hi
end

"""
    least_g(D, test, g0) -> (ḡ, ā)

Smallest ḡ ≥ g0 found at which `test(ā(ḡ), ḡ)` passes, ā(ḡ) the least number passing (ii):
bracketed by `g0(1 + δ)`, δ doubling from 1e-15, then bisected. `(NaN, NaN)` if nothing passes.
"""
function least_g(D::FPCheckData, test, g0; iters::Int = 200)
    g0 = max(Float64(g0), floatmin(Float64))
    δ = 1e-15
    lo = g0
    hi = g0 * (1 + δ)
    while true
        a = least_a(D, hi)
        isfinite(a) && test(a, hi) && break
        lo = hi
        δ *= 2
        hi = g0 * (1 + δ)
        (isfinite(hi) && δ < 2.0^1100) || return (NaN, NaN)
    end
    for _ in 1:iters
        mid_ = (lo + hi) / 2
        (mid_ <= lo || mid_ >= hi) && break
        a = least_a(D, mid_)
        (isfinite(a) && test(a, mid_)) ? (hi = mid_) : (lo = mid_)
    end
    a = least_a(D, hi)
    (isfinite(a) && test(a, hi) && check_ii(D, a, hi)) || return (NaN, NaN)
    return (hi, a)
end

# ------------------------------------------------------------------------------------------------
# The order: a floating-point stochastic estimate (no certificate needed, eq:mincert)
# ------------------------------------------------------------------------------------------------

"""
    estimate_orders(op, Q, mmax; nsamples = 8, rng) -> Vector{Float64}

Floating-point estimates `v_t ≈ (tr H^t)^{1/2t}`, `t = 1:mmax`, from `nsamples` Gaussian vectors
`x`: `E‖x_t‖² = tr H^t` for the chain `x_t` of the sweep started at `x`. Costs `mmax·nsamples`
applications of B; each column is renormalised at every step and the logarithms accumulated.
Used to choose the order of the certified sweep and the power-of-two scale; it certifies nothing.
"""
function estimate_orders(op::LinOp, Q::AbstractMatrix, mmax::Integer; nsamples::Integer = 8,
        nwarm::Integer = 8, rng = Random.MersenneTwister(20260911))
    n = op.n
    k = size(Q, 2)
    T = promote_type(op.T, eltype(Q)) <: Complex ? ComplexF64 : Float64
    Qm = Matrix{T}(Q)
    s = Int(nsamples)
    X = T <: Complex ? (randn(rng, n, s) .+ im .* randn(rng, n, s)) ./ sqrt(2) :
        randn(rng, n, s)
    X = Matrix{T}(Matrix(qr(Matrix{T}(X)).Q))
    Y = similar(X)
    T1 = similar(X)
    T2 = similar(X)
    S = similar(X)
    Tk = zeros(T, k, s)
    # A few applications before anything is recorded, so that the block has turned towards the
    # dominant singular directions: from random starts the first estimate is otherwise well below
    # the norm, and the order and scale chosen from it are poor.
    for w in 1:nwarm
        _applyB!(Y, X, op, Qm, 1.0, T1, T2, Tk, S, iseven(w))
        σw = opnorm(Y)
        (isfinite(σw) && σw > 0) || break
        Y ./= σw
        X .= Matrix(qr(Y).Q)
    end
    v = zeros(mmax)
    logscale = 0.0
    for t in 1:mmax
        _applyB!(Y, X, op, Qm, 1.0, T1, T2, Tk, S, iseven(t))
        # ‖B^t‖ is estimated by ‖B^t X‖ for X orthonormal, which is the largest singular value of
        # an n × s block and a lower bound converging upward as the block aligns with the dominant
        # singular directions. The running scale keeps it away from overflow.
        σ = opnorm(Y)
        if !(isfinite(σ) && σ > 0)
            v[t] = 0.0
            break
        end
        logscale += log(σ)
        v[t] = exp(logscale / t)
        Y ./= σ
        X, Y = Y, X
    end
    return v
end

# ------------------------------------------------------------------------------------------------
# The certificate
# ------------------------------------------------------------------------------------------------

"""
    FarBlockCertificate

`ok` means that `‖G22‖ < gbar` and `‖A‖ < abar` are certified (prop:fpchecks), `gbar` the least
of the certified bounds over the orders `t = 1:m` of the trace half and the interpolation half
(eq:mincert). Also: `gbar_trace[t]` and `gbar_interp` (NaN when a check did not pass), the
floating-point values `trace_fp[t] = (Σ_j‖x_t‖²)^{1/2t}` and `interp_fp`, the estimates used to
choose the order, the scale, the one-step constants at the returned pair, and timings.
"""
struct FarBlockCertificate
    ok::Bool
    gbar::Float64
    abar::Float64
    m::Int
    passed::Symbol
    tpass::Int
    gbar_trace::Vector{Float64}
    abar_trace::Vector{Float64}
    gbar_interp::Float64
    abar_interp::Float64
    trace_fp::Vector{Float64}
    interp_fp::Float64
    estimates::Vector{Float64}
    scale::Float64
    epsB::Float64
    etaB::Float64
    epsP::Float64
    gbars_power::Vector{Float64}
    sums::SweepSums
    nmatvec::Int
    time_estimate::Float64
    time_sweep::Float64
    time_checks::Float64
end

"""
    certify_far_block(op, Q, fd; order = :auto, order_max = 12, order_tol = 0.05, target = nothing,
                      nsamples = 8, blocksize, nthreads) -> FarBlockCertificate

Algorithm alg:fpsweep. The order `m` is `order` if an integer, else predicted from
[`estimate_orders`](@ref): the sweep measures a Frobenius norm over a block of rank at most
`n − k` and tests it against `g^{2t}`, so the certified bound behaves like
`v_t·(n − k)^{1/2t}`, and `m` is the least order whose prediction clears `target`, or, with no
target, the least at which the prediction stops improving. The operator is scaled by the power of two
`c = 2^{-e}` that brings the least estimate into `[1, 2)` (rem:impl). One sweep of order `m`
([`trace_sweep`](@ref)); then, for every `t ≤ m` and for the interpolation half, the least `ḡ`
passing (i) with `ā(ḡ)` the least number passing (ii) (bisection, at no further matvec). The frame
data `fd` supplies α (for ε_P) and the affine isometry bounds.
"""
function certify_far_block(op::LinOp, Q::AbstractMatrix, fd::FrameData;
        order = :auto, order_max::Integer = 64, order_tol::Real = 0.05, target = nothing,
        nsamples::Integer = 8, blocksize = nothing, nthreads::Integer = Threads.nthreads(),
        rng = Random.MersenneTwister(20260911))
    n, k = size(Q)
    T = promote_type(op.T, eltype(Q)) <: Complex ? ComplexF64 : Float64
    te = time()
    mmax = order isa Integer ? Int(order) : Int(order_max)
    est = estimate_orders(op, Q, mmax; nsamples, rng)
    te = time() - te
    if order isa Integer
        m = Int(order)
    else
        # The order the target needs, predicted before the sweep is run.
        #
        # What the sweep measures is a Frobenius norm, over a block of rank at most n − k, and the
        # test compares it against g^{2t}, so the certified bound carries that factor to the power
        # 1/2t:
        #
        #     g(t) ≈ v_t · (n − k)^{1/2t},
        #
        # with v_t the estimate of ‖B^t‖^{1/t}. On grcar500, rank 270: predicted looseness 16.4,
        # 1.26, 1.12, 1.06 at t = 1, 12, 24, 48 against measured 7.4, 1.09, 1.02, 1.005.
        #
        # So the smallest order that can clear the target is the smallest t with g(t) < target, and
        # when no t up to `order_max` does, no sweep of any order on this ladder will certify the
        # far block below the target and there is no point running one. The order is otherwise
        # where the prediction stops improving.
        rk = max(n - k, 1)
        gpred = [est[t] * rk^(1 / (2t)) for t in 1:mmax]
        m = nothing
        if target !== nothing
            m = findfirst(g -> g * (1 + order_tol) < Float64(target), gpred)
        end
        if m === nothing
            vmin = minimum(gpred)
            m = findfirst(g -> g <= (1 + order_tol) * vmin, gpred)
            m === nothing && (m = mmax)
        end
    end
    vref = minimum(est[1:m])
    c = (isfinite(vref) && vref > 0) ? 2.0^(-exponent(vref)) : 1.0
    bs = blocksize === nothing ? max(1, min(64, 2^22 ÷ max(n, 1))) : Int(blocksize)
    S = trace_sweep(op, Q, m; scale = c, blocksize = bs, nthreads)
    tc = time()
    # constants in the units of the scaled operator c·A
    εP = epsP_up(fd.α, n, k, T)
    ηP = etaP_up(n, k, T)
    ηA = addu(mulu(c, op.etaA), mulu(sqrtu(Float64(n)), eta_min(Float64)))
    sa, g1a, g2a = isometry_affine(fd)
    sc(x) = (mulu(c, x[1]), x[2])
    D = FPCheckData(S, sc(sa), sc(g1a), sc(g2a), op.epsA, ηA, εP, ηP)
    gt = fill(NaN, m)
    at = fill(NaN, m)
    git = fill(NaN, m)
    ait = fill(NaN, m)
    trfp = [S.fp[t] > 0 ? S.fp[t]^(1 / (2t)) : 0.0 for t in 1:m]
    ipfp = sqrt(S.N1fp * S.Ninffp)
    if S.finite
        for t in 1:m
            gt[t], at[t] = least_g(D, (a, g) -> check_iF(D, t, a, g), trfp[t])
            # the interpolation half at the same order: ‖B^t‖₂ ≤ sqrt(‖B^t‖₁‖B^t‖_∞) < g^t
            git[t], ait[t] = least_g(D, (a, g) -> I_up(D.S, t, _consts(D, a, g)...) <
                                                  powd(g, 2t), sqrt(S.N1fps[t] * S.Ninffps[t]))
        end
        gi, ai = least_g(D, (a, g) -> check_iI(D, a, g), ipfp)
    else
        gi, ai = NaN, NaN
    end
    cands = [(gt[t], at[t], :trace, t) for t in 1:m if isfinite(gt[t])]
    append!(cands, [(git[t], ait[t], :interp, t) for t in 1:m if isfinite(git[t])])
    isfinite(gi) && push!(cands, (gi, ai, :interp, 0))
    tc = time() - tc
    if isempty(cands)
        return FarBlockCertificate(false, Inf, Inf, m, :none, 0, gt ./ c, at ./ c, NaN, NaN,
            trfp ./ c, ipfp / c, est, c, NaN, NaN, εP, Float64[], S,
            S.nmatvec + m * nsamples, te, S.time, tc)
    end
    g, a, which, t = cands[argmin(first.(cands))]
    εB, ηB = _consts(D, a, g)
    unscale(x) = isfinite(x) ? divu(x, c) : x
    # The vector eq:farpower consumes, from the same sweep and at no further cost. `F_up` bounds
    # Σ_i‖B^j e_i‖² with the rounding allowances of the certified pair, so its square root bounds
    # ‖B^j‖_F and hence ‖B^j‖. Without it farpower_lo has nothing to work with and σ22 is always
    # the order-one form |z| − ḡ, which is the weaker of the two whenever the far block is
    # non-normal enough for its powers to decay.
    gp = Float64[]
    if S.finite
        for j in 1:m
            fj = F_up(S, j, εB, ηB, g)            # Frobenius, loose by up to sqrt(rank)
            ij = I_up(S, j, εB, ηB)               # interpolation, sharp when B^j is sparse
            bj = min(isfinite(fj) && fj >= 0 ? fj : Inf, isfinite(ij) && ij >= 0 ? ij : Inf)
            isfinite(bj) || break
            push!(gp, divu(sqrtu(bj), powd(c, j)))
        end
    end
    return FarBlockCertificate(true, unscale(g), unscale(a), m, which, t, unscale.(gt),
        unscale.(at), unscale(gi), unscale(ai), trfp ./ c, ipfp / c, est, c, εB, divu(ηB, c),
        εP, gp, S, S.nmatvec + m * nsamples, te, S.time, tc)
end
