#=
Krylov resolvent floor — a matrix-free certified lower bound on σ_min(A − zI), and
the spectral-gap certificate on a circle around a captured eigenvalue.

This is the reference implementation of the matvec certificate developed in the draft
(§ "Matrix-free operators: certifying the far block by matvec") and prototyped in
`bench/ulam_circle_near1.jl` / `bench/gkw_matfree.jl`. It is written numerically
(Float64 / ComplexF64) — the candidate estimate. The rigorous upgrade replaces the
Float64 matvec by a `BallArithmetic` ball matvec (so the per-step defect ‖f_k‖ and the
matvecs are enclosed); the control flow is identical.

THE METHOD (block resolvent, one captured eigenvalue λ₁)
--------------------------------------------------------
Resolvent ↔ smallest singular value:  ‖(A−zI)⁻¹‖₂ = 1/σ_min(A−zI), so a lower bound on
σ_min over a curve Γ is an upper bound on the resolvent there. We want this on a circle
Γ = { center + ρ·e^{iθ} } that encloses ONLY λ₁ (0 < ρ < dist(λ₁, σ(A)∖{λ₁})); a
positive floor on all of Γ ⇒ exactly one eigenvalue inside ⇒ a certified spectral gap.

Capture λ₁ by its (approximate) eigenvector w₁ and the orthogonal rank-1 deflation
        P₂₂ = I − w₁w₁* .
In the orthonormal basis [w₁ | U] (U spanning ran P₂₂) the operator is block UPPER
triangular because U*Aw₁ = U*(λ₁w₁) = 0:
        A ≅ [ G11  g* ]      G11 = w₁*Aw₁ ≈ λ₁ ,   g* = w₁*AU  (the coupling),
            [ 0    G22 ]      G22 = U*AU = P₂₂ A P₂₂ |_{ran P₂₂}.
For a block-triangular matrix the Schur-complement bound gives, for σ₁₁ = |G11−z| and
σ₂₂ = σ_min(G22 − zI),
        σ_min(A − zI)  ≥  σ₁₁ σ₂₂ / (σ₁₁ + σ₂₂ + γ),      γ = ‖g‖ = coupling,           (★)
positive whenever σ₁₁, σ₂₂ > 0 — for ANY coupling γ (the crude Weyl bound min(σ₁₁,σ₂₂)−γ
instead dies once γ is large, which it is for non-normal transfer operators).

THE FAR BLOCK BY MATVEC — σ₂₂ = σ_min(G22 − zI)
-----------------------------------------------
σ_min(G22−zI) = √(λ_min(M_z)) with the HERMITIAN  M_z = P₂₂ (G22−zI)'(G22−zI) P₂₂.
M_z is applied purely by matvecs of A and its adjoint Aᵀ (plus the rank-1 P₂₂); G22 is
never formed. λ_min(M_z) is obtained by Lanczos on M_z. Two subtleties, both handled:

 1. M_z's structural KERNEL is the deflated direction w₁ (P₂₂w₁ = 0 ⇒ M_z w₁ = 0). A
    ~1e-16 leak of w₁ would give Lanczos a fake converged eigenvalue 0. We SHIFT it out:
    add `nullshift·w₁w₁*` so w₁'s eigenvalue moves from 0 to `nullshift` (above ‖M_z‖),
    and the true σ_min is the genuine minimum.

 2. LOSS OF ORTHOGONALITY. Plain three-term Lanczos loses orthogonality the moment the
    bottom Ritz value converges, and ‖R₂‖ = ‖QᵀQ − I‖ blows up (we measured ~1e38),
    which destroys the certificate (β₂₂ = ‖R₁‖/(1−‖R₂‖)). Two devices fix it WITHOUT
    storing the full basis:
      • Paige's recurrence streams a rigorous bound on ‖R₂‖ from the scalar coefficients
        (α_j, β_j, defect norms ‖f_j‖, adjacent Gram seeds) — no stored basis, no inner
        products (the `W` recurrence inlined in `lanczos_pro`).
      • Driven by that same ω estimate, we reorthogonalize the new vector against the FEW
        already-converged Ritz directions whenever semi-orthogonality (ω > √u) is lost
        (Simon 1984 / Parlett–Scott selective orthogonalization). In practice only a
        handful of directions ever converge, so the retained set is tiny — the cost is
        nearly memory-free and keeps ‖R₂‖ ≲ √u < 1 out to a full/breakdown basis.

THE FLOOR (draft Prop. "matvec floor on the far block")
-------------------------------------------------------
With ‖R₂‖ ≤ ρ < 1, the deviation lemma gives: every eigenvalue of M_z lies within
β₂₂ = ‖R₁‖/(1−ρ) of spec(T_k) (T_k the computed tridiagonal), where ‖R₁‖ is the
eigen-residual. Hence
        λ_min(M_z) ≥ θ_min − β₂₂,   θ_min = λ_min(T_k),
        σ₂₂ = σ_min(G22−zI) ≥ √(max(0, θ_min − β₂₂)).
‖R₁‖ also streams: r_i = β_k (s_i)_k q_{k+1} + F_k s_i, so ‖R₁‖ is bounded by the
last-component term β_k|(s_i)_k| plus the accumulated local defects Σ_l |(s_i)_l|‖f_l‖.

WHOLE CIRCLE — ONE Krylov run.  G₂₂ does not depend on z, so σ_min(G₂₂ − zI) is itself a
1-Lipschitz function of z. We therefore run the expensive far-block Lanczos ONCE, at the
circle centre z₀ = λ₁, and bound σ₂₂(z) ≥ σ₂₂(z₀) − radius over the whole disk. The captured
block σ₁₁(z) = σ_min(H − zI) is a cheap small (m×m) SVD, minimised over the circle (samples +
a 1-Lipschitz arc-cover). Since the assembly (★) increases in both σ₁₁ and σ₂₂, the
whole-circle floor is (★) evaluated at the two circle-minimum lower bounds — so the expensive
certification is run once per captured eigenvalue, not once per sample. (A single far-block
call at z₀ in fact certifies σ₂₂ > 0 on the entire ball |z−z₀| < σ₂₂(z₀), so one run also
covers any smaller circle / nearby |z| — it need only be repeated per spectral scale.)
=#

using LinearAlgebra
using BallArithmetic

export krylov_resolvent_floor, certify_gap_on_circle, KrylovCircleCertificate

# ----------------------------------------------------------------------------------------
# Selective-reorthogonalization Lanczos on a Hermitian operator given as a matvec `Mz`.
#
# Returns the tridiagonal coefficients (α, β), the per-step local defect norms `fn`, a
# streamed upper bound `R2` on ‖QᵀQ − I‖₂ (Paige), and the number of steps `k`. The Lanczos
# basis is NOT returned; reorthogonalization touches only the few flagged (converged) columns.
# ----------------------------------------------------------------------------------------
struct LanczosRun{R <: Real}     # R = real precision (Float64 or BigFloat)
    α::Vector{R}            # tridiagonal diagonal
    β::Vector{R}            # tridiagonal off-diagonal (β[k] is the residual norm at step k)
    fn::Vector{R}           # local rounding-defect norms ‖f_k‖ (streamed)
    wlast::Vector{R}        # Paige estimate of |⟨q_i, q_{k+1}⟩| (orthogonality of the new vector)
    R2::R                   # streamed Paige bound on ‖QᵀQ − I‖₂
    k::Int                  # steps actually taken (≤ kmax; fewer on breakdown)
    nreorth::Int            # number of reorthogonalization events (diagnostic)
end

"""
    lanczos_pro(Mz, n, kmax; v0=nothing, reorth=true) -> LanczosRun

Lanczos on the Hermitian matvec `Mz` (an `n`-vector → `n`-vector closure) with online
Paige loss-of-orthogonality tracking and Simon-style partial reorthogonalization.

`v0` seeds the first vector (e.g. a vector inside the working subspace); `reorth=false`
gives the plain three-term recurrence (orthogonality is lost — diagnostic only).
"""
function lanczos_pro(Mz, n::Int, kmax::Int; v0 = nothing, reorth::Bool = true)
    # generic precision: T = complex vector type, R = its real part (Float64 or BigFloat).
    # BigFloat lowers the semi-orthogonality floor √u, so a far block whose σ_min² sits below
    # the Float64 certificate floor (strongly non-normal, e.g. contracting Lorenz) clears it.
    T = v0 === nothing ? ComplexF64 : eltype(v0)
    R = real(T)
    u = eps(R)
    trig = sqrt(u)          # semi-orthogonality threshold (reorth is triggered above this)
    keepθ = u^(R(3) / 4)    # flag columns above this for the reorth set
    reset = 2u              # ω reset value after reorthogonalizing against a column
    # relative breakdown / numerical-rank threshold: stop when the residual is tiny vs the
    # running ‖T‖. For a far block with low effective rank (M_z = |z|²I + low-rank, the noise-
    # smoothed regime) the Krylov subspace becomes invariant early — a HAPPY breakdown, which is
    # completeness (sound). The threshold tracks the OPERATOR's precision, not the working one: a
    # Float64-precision matvec (e.g. a Float64 transfer matrix promoted to BigFloat) floors the
    # residual at ~1e-15·‖T‖, so a BigFloat √u (≈1e-24) would never fire and the run grinds to
    # k≈n with an O(n²) basis. `max(√u, 1e-12)` fires at the operator's numerical rank instead
    # (still sound: σ_min is certified down to ~1e-12·‖T‖, below the Float64 operator's own floor).
    breaktol = max(trig, R(1e-12))
    Tnorm = zero(R)

    α = zeros(R, kmax)
    β = zeros(R, kmax)
    fn = zeros(R, kmax)
    # W[i,j] ≥ |ω_ij| = |⟨q_i,q_j⟩−δ_ij|, propagated online by Paige's recurrence.
    W = zeros(R, kmax + 1, kmax + 1)
    nreorth = 0
    nrestart = 0        # invariant-subspace restarts (block boundaries β_k = 0 in T)

    q = v0 === nothing ? randn(ComplexF64, n) : copy(v0)
    q ./= norm(q)
    qprev = zeros(T, n)
    βprev = zero(R)
    kact = kmax

    # MEMORY NOTE. The ‖R₂‖ bound and the floor are STREAMED (Paige scalars + the tridiagonal),
    # needing no basis. Reorthogonalization touches earlier columns, so the basis `cols` is kept
    # only when `reorth=true`. With the relative breakdown above, a low-rank far block stops at
    # its effective rank (small `kact`), so `cols` is O(n·rank), not O(n²); for a genuinely high-
    # rank far block at large n this still grows and wants thick-restart (separate work). With
    # `reorth=false` (pure 3-term, e.g. BigFloat that stays semi-orthogonal) nothing is stored.
    cols = reorth ? Vector{Vector{T}}(undef, kmax) : Vector{Vector{T}}()

    for k in 1:kmax
        reorth && (cols[k] = q)
        W[k, k] = u
        w = Mz(q)
        a = real(dot(q, w))
        α[k] = a
        r = w .- a .* q .- (βprev .* qprev)

        # ---- online Paige ω update: propagate the (k+1)-th column of |ω| from k, k−1 ----
        if k == 1
            W[1, 2] = W[2, 1] = u
        else
            W[k, k + 1] = W[k + 1, k] = u
            ξ = u * (abs(α[k]) + norm(r))                 # rounding injection ~ u·‖T‖
            for i in (k - 1):-1:1
                num = β[i] * W[i + 1, k] + abs(α[i] - α[k]) * W[i, k] +
                      (i > 1 ? β[i - 1] * W[i - 1, k] : zero(R)) +
                      β[k - 1] * W[i, k - 1] + ξ + u * (abs(α[i]) + β[i])
                W[i, k + 1] = W[k + 1, i] = num / max(norm(r), u)
            end
            # ---- selective reorthogonalization: if semi-orthogonality is lost, project the
            #      residual against the flagged (near-converged) columns and reset their ω ----
            if reorth && maximum(@view W[1:k, k + 1]) > trig
                idx = findall(i -> W[i, k + 1] > keepθ, 1:k)
                for _ in 1:2                                # twice for numerical stability
                    for i in idx
                        r .-= cols[i] .* (cols[i]' * r)
                    end
                end
                for i in idx
                    W[i, k + 1] = W[k + 1, i] = reset
                end
                nreorth += 1
            end
        end

        b = norm(r)
        β[k] = b
        Tnorm = max(Tnorm, abs(a) + b + βprev)
        # BREAKDOWN — the Krylov space has closed on itself before spanning the invariant
        # subspace. This is NOT completeness: spec(T_k) is then a near-SUBSET of spec(M), and
        # λ_min may live entirely on the untouched orthogonal complement (a "happy breakdown"
        # yields the full spectrum only under an uncertifiable genericity assumption on the
        # start vector). So we do NOT stop: we set the block boundary β_k = 0 and RESTART from
        # a fresh vector orthogonal to everything so far, so the basis still reaches dimension
        # `kmax`. T becomes block-tridiagonal and its spectrum is the union of the blocks —
        # the full-basis similarity of Prop. 5 then applies verbatim.
        if !isfinite(b) || b <= u * abs(a) || b <= breaktol * Tnorm
            if !reorth || k >= kmax
                kact = k
                break
            end
            β[k] = zero(R)                       # block boundary: column k of S is just r
            fn[k] = norm(a .* q .+ βprev .* qprev .- w)     # ‖S[:,k]‖ with β_k q_{k+1} dropped
            qn = randn(T, n)
            for _ in 1:2, i in 1:k               # orthogonalize against the whole basis

                qn .-= cols[i] .* (cols[i]' * qn)
            end
            nn = norm(qn)
            if nn <= sqrt(u) * sqrt(R(n))        # complement exhausted — basis is complete
                kact = k
                break
            end
            nrestart += 1
            for i in 1:k                         # fresh column is orthogonal to all of them
                W[i, k + 1] = W[k + 1, i] = u
            end
            qprev, βprev, q = q, zero(R), qn ./ nn
            continue
        end
        qnext = r ./ b
        # realized local defect  f_k = a·q + βprev·qprev + b·qnext − M_z q   (here ≈ rounding)
        fn[k] = norm(a .* q .+ βprev .* qprev .+ b .* qnext .- w)
        qprev, βprev, q = q, b, qnext
    end

    # streamed ‖R₂‖₂ ≤ ‖W‖₁ = ‖W‖_∞ (W symmetric, nonnegative)
    R2 = maximum(sum(@view W[i, 1:kact]) for i in 1:kact)
    wlast = W[1:kact, kact + 1]               # |⟨q_i, q_{kact+1}⟩| estimate (orthogonality)
    return LanczosRun(α[1:kact], β[1:kact], fn[1:kact], wlast, R2, kact, nreorth)
end

# ----------------------------------------------------------------------------------------
# Far-block floor σ_min(G22 − zI) = √(max(0, θ_min − β₂₂)) from a LanczosRun (draft Prop. 5).
# Returns (σ₂₂, θ_min, ‖S‖, ‖R₂‖); `N` is the dimension of the invariant subspace the Krylov
# run lives in (for the deflated far block, N = n − m). Also see `lanczos_deviation`.
#
# SOUNDNESS (the full-basis similarity — this is the whole argument).  A lower bound on
# λ_min(M) needs the COMPLETENESS direction  spec(M) ⊆ spec(T_k) + [−β,β].  Ritz values are
# Rayleigh quotients, so θ_min ≥ λ_min(M) ALWAYS (Ritz converges from above): a PARTIAL basis
# gives NO lower bound — the deficit θ_min − λ_min is governed by the angle between the
# minimizing eigenvector and the Krylov space, which no residual can see.  At k = N the basis
# Q̄ is SQUARE on that subspace, and the residual identity becomes a SIMILARITY:
#       M Q̄ = Q̄ T + S    ⟹   Q̄⁻¹ M Q̄ = T + E,   E = Q̄⁻¹S,
# so spec(M) = spec(T + E) EXACTLY, and Bauer–Fike for the normal (real symmetric) T gives
#       spec(M) ⊆ spec(T) + [−‖E‖, ‖E‖],   ‖E‖ ≤ ‖S‖/√(1−‖R₂‖),
# using σ_min(Q̄)² = λ_min(Q̄*Q̄) ≥ 1 − ‖R₂‖.  THAT is the completeness certificate; it needs no
# genericity, no invariance assumption, no angle condition.
#
# The residual must enter BARE.  From the recurrence, S = M Q̄ − Q̄T = −F + β_k q_{k+1}eₖᵀ, so
#       ‖S‖ ≤ β_k + ‖F‖_F,   ‖F‖_F = √(Σ_j ‖f_j‖²)      (all streamed scalars).
# The previous version used R₁ = Q̄*S — the residual PROJECTED BACK onto the Krylov space,
# where S (being ⊥ to it) has almost nothing, so ‖R₁‖ ~ √u·β_k for EVERY k.  That hid the one
# quantity measuring incompleteness and made the floor look k-independent.  It was unsound in
# proof (though hard to break numerically: rounding reinjects missing directions and Lanczos
# converges to extremal eigenvalues anyway — the Paige ghost phenomenon).
#
# At k < N the bare β_k is O(‖T‖), so β₂₂ swamps θ_min and the floor goes ≤ 0: the bound is
# VACUOUS but SOUND, and it is self-policing — no separate validity test is needed beyond the
# `full` gate below. At k = N the Krylov space is exhausted, β_N collapses to rounding level,
# and the floor is both rigorous and TIGHTER than the old projected-R₁ bound.
#
# NOTE (early "happy breakdown" is NOT completeness). β_k ≈ 0 at k < N means ran(Q̄_k) is nearly
# INVARIANT — spec(T_k) is then a near-SUBSET of spec(M), which is again the wrong direction:
# λ_min(M) may live entirely on the untouched orthogonal complement. Breakdown gives the full
# spectrum only under a genericity assumption on the start vector (that it has a nonzero
# component on every eigenspace), which we do not certify. Hence the strict `k ≥ N` gate.
function far_floor(L::LanczosRun, N::Int = 0)
    k = L.k
    k < 2 && return (0.0, Float64(L.α[1]), Inf, Float64(L.R2))
    # θ_min from the tridiagonal in Float64 — eigenvalues are well conditioned and θ_min ~1e-7
    # is representable, so the slow generic BigFloat symmetric eigensolve is unnecessary.
    θmin = minimum(eigvals(SymTridiagonal(Float64.(L.α[1:k]), Float64.(L.β[1:(k - 1)]))))
    S = lanczos_deviation(L)
    β22 = L.R2 < 1 ? Float64(S / sqrt(1 - L.R2)) : Inf
    # FULL-BASIS GATE: without a complete basis of the invariant subspace there is no
    # completeness certificate and hence no lower bound. Refuse rather than report a number.
    full = N > 0 && k >= N
    σ = full ? sqrt(max(0.0, θmin - β22)) : 0.0
    return (σ, θmin, Float64(S), Float64(L.R2))
end

# ‖S‖ ≤ β_k + ‖F‖_F — the BARE deviation of the computed decomposition M Q̄ = Q̄T + S, from
# streamed scalars only. Divided by √(1−‖R₂‖) it is the Bauer–Fike radius of Prop. 5.
function lanczos_deviation(L::LanczosRun)
    k = L.k
    return L.β[k] + sqrt(sum(abs2, @view L.fn[1:k]))
end

# Rigorous UPPER bound on λ_max(M) from the same full-basis similarity (Prop. 5, other side):
# λ_max(M) ≤ θ_max + ‖S‖/√(1−‖R₂‖). Returns `Inf` unless the basis is complete (k ≥ N) — the
# completeness direction is exactly what an upper bound needs too (a missed LARGE eigenvalue
# is invisible to a partial basis).
function lanczos_lambda_max(L::LanczosRun, N::Int)
    k = L.k
    k < 1 && return Inf
    (L.R2 >= 1 || !(N > 0 && k >= N)) && return Inf
    θmax = k == 1 ? Float64(L.α[1]) :
           maximum(eigvals(SymTridiagonal(Float64.(L.α[1:k]), Float64.(L.β[1:(k - 1)]))))
    return θmax + Float64(lanczos_deviation(L) / sqrt(1 - L.R2))
end

# ----------------------------------------------------------------------------------------
# Capture frame and far block. `W` (n×m) captures the eigenvalue(s) of interest; the
# orthogonal deflation is P₂₂ = I − WW*, the captured block is H = W*AW (m×m), and the
# coupling is γ = max(‖P₂₂AW‖, ‖P₂₂A*W‖). A vector `w1` is treated as a rank-1 frame.
# ----------------------------------------------------------------------------------------
_frame(W::AbstractVector) = reshape(W ./ norm(W), :, 1)
_frame(W::AbstractMatrix) = Matrix(qr(Matrix(W)).Q)[:, 1:size(W, 2)]

# apply an operator to every column of W → an n×m matrix (works for m=1, unlike reduce(hcat,·)).
_applycols(op, W) = hcat((op(Vector{ComplexF64}(W[:, j])) for j in axes(W, 2))...)

_captured_block(Amul, W) = W' * _applycols(Amul, W)            # H = W*AW  (m×m)

function _coupling(Amul, Aadj, W)
    defl(M) = M .- W * (W' * M)
    return max(opnorm(defl(_applycols(Amul, W)), 2),           # ‖P₂₂ A W‖
        opnorm(defl(_applycols(Aadj, W)), 2))                  # ‖P₂₂ A* W‖
end

# VERIFIED σ_min lower bound of a (complex) matrix via the rigorous SVD (BallArithmetic.svdbox).
_sigma_min_cert(M::AbstractMatrix) = _sigmin_lower(BallMatrix(Matrix(M)))

# Certified lower bound on σ_min(M − wI) over the WHOLE circle {center + radius·e^{iθ}}, via an
# ADAPTIVE PEARL cover. At each pearl centre z a point svdbox gives σ(z)=σ_min(M−zI); since
# σ_min(M−wI) is 1-Lipschitz in w, the pearl {|w−z|≤ρ} is certified at `σ(z)−ρ`, and we advance
# Δθ = 2ρ/radius so the pearls overlap. The radius ρ = safety·σ(z) is set by the LOCAL σ_min — big
# pearls where σ_min is large, small where small — so smooth arcs cost one SVD and the whole circle
# takes O(∮ dθ·radius/σ) SVDs instead of a fixed uniform nθ. (Using `M−Ball(z,ρ)I` instead would
# perturb each diagonal INDEPENDENTLY — a far bigger set than the common shift — and is much looser.)
function _sigmin_circle(M::AbstractMatrix, center::Number, radius::Real;
        safety::Real = 0.5, ρfloor::Real = 0.02, maxpearls::Int = 4096)
    T = eltype(M)
    R = real(T)                               # Float64 or BigFloat (svdbox precision)
    c = T(center)
    rad = R(radius)
    saf = R(safety)
    ρf = R(ρfloor)
    smin = R(Inf)
    θ = zero(R)
    np = 0
    twoπ = 2 * R(π)
    while θ < twoπ && np < maxpearls
        z = c + rad * (cos(θ) + im * sin(θ))
        σz = _sigma_min_cert(M - z * I)                       # point svdbox (common shift, exact)
        ρ = max(saf * max(σz, zero(R)), ρf * rad)             # pearl radius from the local σ_min
        smin = min(smin, σz - ρ)                              # 1-Lipschitz cover of |w−z| ≤ ρ
        θ += 2ρ / rad
        np += 1
    end
    return smin
end

# CERTIFIED upper bound on the coupling γ = max(‖P₂₂AW‖₂, ‖P₂₂A*W‖₂) (verified L² opnorm).
# Tight AND cheap via the small m×m GRAM: γ = ‖C‖₂ = √σ_max(C*C) with C = P₂₂·(op)·W the n×m WIDE
# coupling matrix. We must NOT use the cheap min(Collatz, √(‖·‖₁‖·‖∞)) — on the wide matrix it
# overestimates by ~√rank (measured 5.34 vs true 1.33), which sank the assembly floor. And we
# should NOT verified-SVD the WIDE n×m C directly (3s at n=4096, m=256); instead svd_bound the
# `m×m` Hermitian `C*C` (0.5s, identical value) — ‖C‖₂² = ‖C*C‖₂. γ is z-independent, computed once.
# The ASYMMETRIC coupling pair (g12, g21) = (‖G12‖, ‖G21‖) = (‖P₂₂A*W‖, ‖P₂₂AW‖): the two block
# rows the block-dominance floor (`_assemble2`) charges SEPARATELY. For a non-normal A the two
# differ — the adjoint coupling ‖G12‖ = ‖P₂₂A*W‖ typically dominates — and charging each block only
# its own row is the Feingold–Varga sharpening. `_coupling_cert` is their max (the assembly γ).
function _coupling_pair(Amul, Aadj, W)
    defl(M) = M .- W * (W' * M)
    gram(op) = (
        Cb = BallMatrix(defl(_applycols(op, W))); sqrt(svd_bound_L2_opnorm(Cb' * Cb)))
    return (gram(Aadj), gram(Amul))                            # (‖G12‖, ‖G21‖)
end
_coupling_cert(Amul, Aadj, W) = maximum(_coupling_pair(Amul, Aadj, W))

"""
    far_block_sigma_min(Amul, Aadj, n, W, z; prec=Float64, kmax=0, nullshift=10.0, reorth=true)
        -> (σ22, info)

Certified lower bound on `σ_min(G₂₂ − zI)`, where `G₂₂ = P₂₂ A P₂₂` is the far block after
deflating the captured frame `W` (`P₂₂ = I − WW*`), by ONE selective-reorthogonalization
Lanczos on the Hermitian `M_z`. This is THE expensive step. `info` carries `θmin, R1, R2,
k, nreorth`. `kmax=0` ⇒ full basis `n − m − 1` (required for soundness).

`prec` is the real working precision of the Lanczos (`Float64` or `BigFloat`). BigFloat
lowers the semi-orthogonality floor `√u`, certifying a far block whose `σ_min²` sits below
the Float64 floor (strongly non-normal, e.g. contracting Lorenz). The candidate matvecs
`Amul`/`Aadj` may stay Float64 — only the Lanczos arithmetic and the orthogonality bound
need the higher precision.

Because `G₂₂` does not depend on `z`, `σ_min(G₂₂ − zI)` is a fixed 1-Lipschitz function of
`z`: a single call at `z₀` lower-bounds it on the whole disk `|z − z₀| ≤ r` by
`σ_min(G₂₂ − z₀I) − r` — so the circle certificate needs only ONE such call (at the centre).
"""
function far_block_sigma_min(Amul, Aadj, n::Int, W::AbstractMatrix, z::Number;
        prec::Type = Float64, kmax::Int = 0, nullshift::Real = 10.0, reorth::Bool = true)
    m = size(W, 2)
    # The Krylov run from a P₂₂-start lives in ran(P₂₂), of dimension N = n − m (M_z maps it to
    # itself). A COMPLETE basis of that invariant subspace is N vectors — required for the
    # full-basis similarity of Prop. 5 (`far_floor`'s `N` gate). This is the price of rigor: a
    # certified floor needs N matvecs, because only a complete basis guarantees no direction —
    # in particular the minimizing one — was missed.
    N = n - m
    kmax <= 0 && (kmax = N)
    Wt = prec == Float64 ? W : Matrix{Complex{prec}}(W)
    zz = Complex{prec}(z)
    ns = prec(nullshift)
    P22(x) = x .- Wt * (Wt' * x)
    Mz = function (x)
        v1 = P22(x)
        v2 = Complex{prec}.(Amul(v1)) .- zz .* v1
        v3 = P22(v2)
        v4 = Complex{prec}.(Aadj(v3)) .- conj(zz) .* v3
        P22(v4) .+ ns .* (Wt * (Wt' * x))
    end
    L = lanczos_pro(Mz, n, kmax; v0 = P22(randn(Complex{prec}, n)), reorth = reorth)
    σ22, θmin, R1, R2 = far_floor(L, N)
    return σ22, (; θmin, R1, R2, k = L.k, nreorth = L.nreorth, N, full = L.k >= N)
end

# Outer 2-block resolvent floor — the MAX of two certified lower bounds on σ_min(A − zI):
#   • Schur-complement assembly (★): σ11·σ22/(σ11+σ22+γ), γ=max(‖G12‖,‖G21‖) — positive for ANY
#     coupling, never dies;
#   • block-Geršgorin / Feingold–Varga block-dominance (draft.tex Prop. blockgersg/blockdom):
#     min(σ11,σ22)·(1−τ), τ² = (‖G12‖/σ11)² + (‖G21‖/σ22)². κ-FREE — the captured/far outer basis
#     is unitary, so no transport constant (the same footing the assembly already assumes). It is
#     ≈2× tighter than the assembly when the couplings are small vs the gaps (the far field), but
#     dies once τ≥1 (near the spectrum, where σ11→0); the assembly is the survivor there.
# σ11 ↔ captured block G11 (its off-block row is G12), σ22 ↔ far block G22 (off-block row G21).
# Both are valid lower bounds, so their max is too; feeding LOWER bounds of σ11,σ22 stays valid
# (each term is increasing in σ11,σ22 on the live region).
function _assemble2(σ11, σ22, g12, g21)
    (σ11 <= 0 || σ22 <= 0) && return 0.0
    γ = max(g12, g21)
    schur = σ11 * σ22 / (σ11 + σ22 + γ)
    τ = sqrt((g12 / σ11)^2 + (g21 / σ22)^2)
    bdom = τ < 1 ? min(σ11, σ22) * (1 - τ) : 0.0
    return max(schur, bdom)
end
# Symmetric back-compat: a scalar coupling γ charges both off-block rows the same (a valid, if
# looser, over-estimate of each), so every legacy call site gains the block-dominance sharpening.
_assemble(σ11, σ22, γ) = _assemble2(σ11, σ22, γ, γ)

"""
    krylov_resolvent_floor(Amul, Aadj, n, W, z; kmax=0, nullshift=10.0, reorth=true)
        -> (sigma_min_lower, info)

Certified lower bound on `σ_min(A − zI)` at a single `z`, via the block resolvent (★):
small dense SVD for the captured block `σ11 = σ_min(H − zI)`, one matvec Lanczos for the far
block `σ22`, assembled with the coupling `γ`. `W` is the captured frame (an eigenvector for
rank-1). `info` has `σ11, σ22, γ, R2, θmin, R1, k, nreorth`.
"""
function krylov_resolvent_floor(Amul, Aadj, n::Int, W0, z::Number; kwargs...)
    W = _frame(W0)
    H = _captured_block(Amul, W)
    g12, g21 = _coupling_pair(Amul, Aadj, W)
    γ = max(g12, g21)
    σ22, info = far_block_sigma_min(Amul, Aadj, n, W, z; kwargs...)
    σ11 = _sigma_min_cert(H - z * I)                        # VERIFIED small SVD (svdbox)
    floor = info.R2 >= 1 ? 0.0 : _assemble2(σ11, σ22, g12, g21)
    return floor, (; σ11, σ22, γ, info.R2, info.θmin, info.R1, info.k, info.nreorth)
end

# ----------------------------------------------------------------------------------------
# Whole-circle gap certificate — ONE far-block Krylov (at the centre) + Lipschitz.
# ----------------------------------------------------------------------------------------
"""
    KrylovCircleCertificate

Result of `certify_gap_on_circle`. `floor` is the certified lower bound on `σ_min(A − zI)`
over the WHOLE circle `Γ = { center + radius·e^{iθ} }`; `certified = floor > 0` ⇒ Γ encloses
exactly one eigenvalue (the captured `center`), so the spectral gap is `≥ radius` and the
resolvent there is `≤ 1/floor`. `σ11lo`/`σ22lo` are the circle-minimum lower bounds of the
captured / far blocks; `γ` the coupling; `R2` the far-block orthogonality residual.
"""
struct KrylovCircleCertificate
    center::ComplexF64
    radius::Float64
    floor::Float64
    σ11lo::Float64
    σ22lo::Float64
    γ::Float64
    R2::Float64
    certified::Bool
end

"""
    certify_gap_on_circle(Amul, Aadj, n, W, center, radius;
                          prec=Float64, auto_radius=false, nθ=64, kmax=0, nullshift=10.0, reorth=true)
        -> KrylovCircleCertificate

Certify `σ_min(A − zI) > 0` over the whole circle `Γ = { center + radius·e^{iθ} }`.

The far block `σ_min(G₂₂ − zI)` is `z`-independent up to translation, hence 1-Lipschitz in
`z`, so the expensive Lanczos runs ONCE — at the centre — and `σ22(z) ≥ σ22(center) − radius`
covers the whole disk. The captured block `σ11(z) = σ_min(H − zI)` is a cheap `m×m` SVD,
minimised over the circle (`nθ` samples + a 1-Lipschitz arc-cover). Since the assembly (★)
increases in both, the whole-circle floor is `_assemble(σ11lo, σ22lo, γ)`. `center` should be
the captured Rayleigh quotient `≈ λ₁` and `0 < radius < dist(λ₁, σ(A)∖{captured})`.

`prec` sets the far-block Lanczos precision (`BigFloat` for strongly non-normal far blocks).
`auto_radius=true` shrinks `radius` to the far Lipschitz reach `½·σ22c` when the requested
radius exceeds it (non-normality caps how wide a circle can be certified) — the used radius
is returned in the certificate. The captured frame `W` should be the cluster's columns of the
REFINED Miyajima–Rump frame `mr.W` (the §4e Sylvester–Newton basis), which decouples the
block far better than raw Ritz vectors — essential for the less-isolated modes.
"""
function certify_gap_on_circle(Amul, Aadj, n::Int, W0,
        center::Number, radius::Real;
        prec::Type = Float64, auto_radius::Bool = false,
        nθ::Int = 64, kmax::Int = 0, nullshift::Real = 10.0, reorth::Bool = true)
    W = _frame(W0)
    H = _captured_block(Amul, W)
    g12, g21 = _coupling_pair(Amul, Aadj, W)
    γ = max(g12, g21)

    # ONE expensive far-block Krylov, at the centre; 1-Lipschitz over the disk.
    σ22c, info = far_block_sigma_min(Amul, Aadj, n, W, center;
        prec = prec, kmax = kmax, nullshift = nullshift, reorth = reorth)
    r = (auto_radius && info.R2 < 1) ? min(radius, 0.5 * σ22c) : radius
    σ22lo = info.R2 < 1 ? σ22c - r : -Inf

    # captured-block σ11 over the circle: VERIFIED small SVDs (svdbox) + 1-Lipschitz arc-cover
    σ11s = [_sigma_min_cert(H - (center + r * cis(2π * t / nθ)) * I) for t in 0:(nθ - 1)]
    σ11lo = minimum(σ11s) - 2 * r * sin(π / (2nθ))

    floor = _assemble2(σ11lo, σ22lo, g12, g21)
    return KrylovCircleCertificate(ComplexF64(center), Float64(r), floor,
        σ11lo, σ22lo, γ, info.R2, floor > 0)
end

# Convenience methods for a dense/sparse matrix A (derives the two matvecs).
function krylov_resolvent_floor(A::AbstractMatrix, W0, z::Number; kwargs...)
    krylov_resolvent_floor(x -> A * x, x -> A' * x, size(A, 1), W0, z; kwargs...)
end
function certify_gap_on_circle(
        A::AbstractMatrix, W0, center::Number, radius::Real; kwargs...)
    certify_gap_on_circle(
        x -> A * x, x -> A' * x, size(A, 1), W0, center, radius; kwargs...)
end

export far_block_sigma_min, certify_leading_eigs, certify_gap, certify_gap_locked

# ----------------------------------------------------------------------------------------
# Krylov–Rump–Miyajima decoupling: certify the leading eigenvalues of A isolated.
# ----------------------------------------------------------------------------------------

# matvec Arnoldi: A Vₙ = Vₙ Hₙ + β_N q_{N+1} eₙᵀ. Returns (H, V, β_N).
function _arnoldi(Amul, n::Int, N::Int; v0 = randn(ComplexF64, n))
    V = zeros(ComplexF64, n, N + 1)
    H = zeros(ComplexF64, N + 1, N)
    V[:, 1] = v0 ./ norm(v0)
    for k in 1:N
        w = ComplexF64.(Amul(V[:, k]))
        for _ in 1:2, i in 1:k                          # modified GS + one reorthogonalisation

            c = dot(V[:, i], w)
            H[i, k] += c
            w .-= c .* V[:, i]
        end
        H[k + 1, k] = norm(w)
        abs(H[k + 1, k]) < 1e-13 && return H[1:k, 1:k], V[:, 1:k], abs(H[k + 1, k])
        V[:, k + 1] = w ./ H[k + 1, k]
    end
    return H[1:N, 1:N], V[:, 1:N], abs(H[N + 1, N])
end

# Verified block diagonalization of the Krylov compression via BallArithmetic's VBD
# (`schur_newton_vbd`: block-orthonormal Schur+Newton frame + Miyajima certification + disc
# clustering + per-block coupling — κ(W) governed by benign inter-block angles). The VBD lives
# in BallArithmetic; here we only adapt its result to the (centres, clusters, frame W, discs,
# κ₂, raw `res`) the Krylov certificate reads. `res` feeds `vbd_resolvent` for the block floor.
function _certify_block(Hc::AbstractMatrix, maxsteps::Integer)
    res = BallArithmetic.schur_newton_vbd(BallMatrix(Matrix(Hc));
        maxsteps = maxsteps, kappa_mode = :svdbox)
    return (centres = res.eigenvalues, clusters = [collect(c) for c in res.clusters],
        W = mid(res.basis), radii = [rad(d) for d in res.cluster_intervals],
        kappa = res.kappa, res = res)
end

# ----------------------------------------------------------------------------------------
# Shared setup + per-circle assembly — the ONE primitive behind BOTH the leading-eigenvalue
# isolation circles AND the good/bad separating circle (guide §5 / the matvec §matfree).
#
# THREE blocks: GOOD (leading `neig`) + BAD (rest of the certified Arnoldi block) live together
# in the dense VBD block G₁₁ = W*AW; UGLY = the complement of the whole Krylov basis V (the ball
# near 0). σ_min(A−zI) is assembled per circle from
#   • σ₁₁ = block resolvent of G₁₁ (good+bad) — `block_estimate` ⇒ the Miyajima–Rump row-residual
#     per-block floor `sigma_min_lower` (κ₂-transported; benign for the block-orthonormal VBD
#     frame), else the exact `svdbox(H−zI)` (κ-free, tight, O(nblock³)/z);
#   • σ₂₂ = the ONE ugly block — z-INDEPENDENT, so `‖G₂₂‖` is bounded ONCE (a single Lanczos on
#     G₂₂*G₂₂) and every circle reuses `σ₂₂(z) ≥ |z|−‖G₂₂‖` (1-Lipschitz). The per-z far-block
#     Lanczos re-triggers ONLY if the ugly block reaches the circle (`|z| ≤ ‖G₂₂‖`).
#   • γ = the block↔ugly coupling (svd_bound_L2_opnorm), bounded ONCE.
# So the expensive ugly-block work is done ONCE and shared across all `neig`+1 circles.
# ----------------------------------------------------------------------------------------
function _krylov_gap_setup(Amul, Aadj, n::Int; nblock::Int, refine::Integer, reorth::Bool)
    H, V, βN = _arnoldi(Amul, n, min(nblock, n - 2))
    Hc = ComplexF64.(H)
    mr = _certify_block(Hc, refine)
    vr = vbd_resolvent(mr.res)                            # good+bad block resolvent (row-residual)
    λ = mr.centres
    ord = sortperm(abs.(λ); rev = true)
    g12, g21 = _coupling_pair(Amul, Aadj, V)             # block↔ugly coupling pair, ONCE (tight svd)
    γ = max(g12, g21)
    # ONE ugly-block ‖G₂₂‖ (z-independent): G₂₂ = P_V^⊥ A P_V^⊥, spectrum = the ugly ball near 0.
    Pv(x) = x .- V * (V' * x)
    GtG(x) = Pv(ComplexF64.(Aadj(Pv(ComplexF64.(Amul(Pv(x)))))))
    # ‖G₂₂‖ is an UPPER bound on λ_max, which needs the SAME completeness direction as the floor
    # (a missed large eigenvalue is invisible to a partial basis) ⇒ FULL basis of ran(P_V),
    # N = n − dim V. `lanczos_lambda_max` returns Inf if the basis is not complete.
    Ng = n - size(V, 2)
    Lg = lanczos_pro(GtG, n, Ng; v0 = Pv(randn(ComplexF64, n)), reorth = reorth)
    _, _, R1g, R2g = far_floor(Lg, Ng)
    G22n = sqrt(max(0.0, lanczos_lambda_max(Lg, Ng)))                # ‖G₂₂‖ upper bound
    return (; H, Hc, V, βN, mr, vr, λ, ord, γ, g12, g21, G22n, R2g)
end

# Certify σ_min(A − zI) > 0 on ONE circle Γ = {center + radius·e^{iθ}}, reusing the shared setup.
function _certify_shared_circle(su, Amul, Aadj, n::Int, center::Number, radius::Real;
        nθ::Int, block_estimate::Bool, prec::Type, kmax::Int, nullshift::Real, reorth::Bool)
    r = Float64(radius)
    floors = Float64[]
    R2max = su.R2g
    σ11min = Inf
    σ22min = Inf
    ntrig = 0                                             # far-block re-triggers (Lipschitz failed)
    for t in 0:(nθ - 1)
        z = center + r * cis(2π * t / nθ)
        σ11 = block_estimate ? sigma_min_lower(su.vr, z; near = true) :  # good+bad row-residual floor
              _sigma_min_cert(su.Hc - z * I)                            # or exact svdbox (κ-free)
        local σ22
        if su.G22n < abs(z)
            σ22 = abs(z) - su.G22n                        # ugly deep ⇒ Lipschitz, no per-z Lanczos
        else
            σ22, info = far_block_sigma_min(Amul, Aadj, n, su.V, z;     # re-trigger: ugly reaches Γ
                prec = prec, kmax = kmax, nullshift = nullshift, reorth = reorth)
            R2max = max(R2max, info.R2)
            ntrig += 1
        end
        σ11min = min(σ11min, σ11)
        σ22min = min(σ22min, σ22)
        push!(floors, R2max < 1 ? _assemble2(σ11, σ22, su.g12, su.g21) : 0.0)
    end
    fl = minimum(floors) - 2 * r * sin(π / (2nθ))         # 1-Lipschitz arc-cover
    return fl,
    (; floor = fl, radius = r, center = ComplexF64(center),
        γ = su.γ, R2 = R2max, σ11min = σ11min, σ22min = σ22min, G22norm = su.G22n,
        ntrigger = ntrig, certified = (fl > 0 && R2max < 1))
end

"""
    certify_leading_eigs(Amul, Aadj, n; nblock=256, neig=5, refine=6, prec=Float64,
                         frac=0.4, nθ=24, block_estimate=false) -> (mr, results, …)

Certify the leading eigenvalues of `A` (matvecs `Amul`, `Aadj`) isolated, sharing ONE ugly-block
Lanczos across all circles (the good/bad/ugly architecture — the same primitive as `certify_gap`):

  1. ONE Arnoldi compresses `A` to a BIG `nblock × nblock` block `H` (basis `V`); `miyajima_rump`
     block-diagonalizes + certifies it (frame `mr.W`, discs, clusters). GOOD = the leading `neig`,
     BAD = the rest of `H`, together the dense block `G₁₁`; UGLY = the complement of `V` (ball near 0).
  2. Bound ONCE (z-independent): the coupling `γ` (block↔ugly, tight `svd_bound_L2_opnorm`) and the
     ugly-block `‖G₂₂‖` (a single Lanczos on `G₂₂*G₂₂`).
  3. For each leading eigenvalue, certify `σ_min(A−zI) > 0` on its isolation circle `Γᵢ` by the shared
     assembly (`_certify_shared_circle`): `σ₁₁` = block resolvent of `G₁₁` on `Γᵢ` (exact `svdbox(H−zI)`,
     κ-free — default; or the κ₂-transported `block_estimate`), `σ₂₂` = `|z|−‖G₂₂‖` (the ONE ugly block,
     1-Lipschitz — re-triggers a per-z far Lanczos only if `|z| ≤ ‖G₂₂‖`), assembled with `γ`, closed by
     the arc-cover. A positive floor ⇒ `Γᵢ` encloses exactly that eigenvalue ⇒ isolated.

`block_estimate=false` (default) uses the κ-free exact `svdbox(H−zI)` for `σ₁₁` — tight, since the
global κ₂ of the block resolvent (Lemma blockres, `/κ₂(W)`) is charged even to well-isolated modes
when the spectrum has an ill-conditioned cluster (e.g. a fast cloud), so `svdbox` certifies where
`block_estimate` fails by ≈κ₂. `results[k]` carries `(λ, disc, cluster, cert)`; `cert.ntrigger` counts
any ugly-block re-triggers. `prec=BigFloat` for strongly non-normal ugly blocks.

REMARK (isolation vs the spectral gap). This certifies each leading eigenvalue ISOLATED. The gap
separating the leading block from the rest is the SAME assembly on the good/bad separating circle —
`certify_gap`, sharing this setup (one Arnoldi block, one VBD, one ugly Lanczos).
"""
function certify_leading_eigs(Amul, Aadj, n::Int; nblock::Int = 256, neig::Int = 5,
        refine::Integer = 6, prec::Type = Float64, frac::Real = 0.4, nθ::Int = 24,
        block_estimate::Bool = false, kmax::Int = 120, nullshift::Real = 10.0, reorth::Bool = true)
    # ONE big Arnoldi block + VBD + block resolvent + coupling + one ugly-block ‖G₂₂‖ Lanczos,
    # shared across all `neig` isolation circles (the ugly block is z-independent — Lipschitz).
    su = _krylov_gap_setup(Amul, Aadj, n; nblock = nblock, refine = refine, reorth = reorth)
    mr, V, λ, ord = su.mr, su.V, su.λ, su.ord
    done = Set{Int}()
    leadcols = Int[]                                     # all certified-leading H-columns
    results = NamedTuple[]
    for i in ord[1:min(neig, length(ord))]
        i in done && continue
        cl = mr.clusters[findfirst(c -> i in c, mr.clusters)]
        union!(done, cl)
        append!(leadcols, cl)
        c = λ[i]
        gap = minimum((abs(c - λ[j]) for j in eachindex(λ) if !(j in cl));
            init = 2 * maximum(abs, λ))                  # to the nearest DISTINCT eigenvalue
        # isolation circle Γᵢ certified by the SAME shared assembly (good+bad block resolvent +
        # the one ugly block), NOT a per-eigenvalue far-block Lanczos — the ugly Lanczos is reused.
        _, cert = _certify_shared_circle(su, Amul, Aadj, n, c, frac * gap;
            nθ = nθ, block_estimate = block_estimate, prec = prec,
            kmax = kmax, nullshift = nullshift, reorth = reorth)
        push!(results, (; λ = c, disc = mr.radii[i], cluster = length(cl), cert))
    end
    # the certified-leading frame, for a subsequent certify_gap (separation from the fast spectrum)
    Wlead = V * mr.W[:, sort(unique(leadcols))]
    return (; mr, βN = su.βN, results, V, Wlead, Hritz = λ[ord])
end

function certify_leading_eigs(A::AbstractMatrix; kwargs...)
    certify_leading_eigs(x -> A * x, x -> A' * x, size(A, 1); kwargs...)
end

export certify_leading_eigs_sing, pseudospectrum_outside

# ----------------------------------------------------------------------------------------
# Singular block by matvec: top right singular vectors of A with Ritz σ ≥ `thresh`, via Lanczos
# on M(x)=A*(A x) (two matvecs/step, full reorthogonalization, basis stored to form Ritz vectors).
# Grows the basis until the smallest converged Ritz singular value falls below `thresh` — the
# stopping rule "peel singular vectors until σ < the smallest good eigenvalue" — then keeps the
# Ritz vectors above `thresh`. These are the "bad" directions the eigenvector frame misses; the
# far block deflated by them has ‖G₂₂‖ ≈ σ_{k+1} < thresh (deep — the matvec far certifier then
# converges easily, with a well-separated bottom).
# ----------------------------------------------------------------------------------------
function _singular_block(Amul, Aadj, n::Int, thresh::Real; kmax::Int = 256,
        v0 = randn(ComplexF64, n))
    Q = zeros(ComplexF64, n, kmax)
    α = zeros(kmax)
    β = zeros(kmax)
    q = ComplexF64.(v0)
    q ./= norm(q)
    Q[:, 1] = q
    qp = zeros(ComplexF64, n)
    βp = 0.0
    kact = kmax
    thr2 = thresh^2                                               # work in θ = σ² units
    maxm = 0
    since = 0                                           # plateau tracker for #{σ ≥ thresh}
    for j in 1:kmax
        w = ComplexF64.(Aadj(ComplexF64.(Amul(Q[:, j]))))         # A*A q  (two matvecs)
        a = real(dot(Q[:, j], w))
        α[j] = a
        w .-= a .* Q[:, j] .+ βp .* qp
        for _ in 1:2, i in 1:j                                    # full reorthogonalization

            c = dot(Q[:, i], w)
            w .-= c .* Q[:, i]
        end
        b = norm(w)
        β[j] = b
        # Stop when the COUNT of Ritz values ≥ thresh plateaus — all σ ≥ thresh are captured. Robust
        # to the large A*A nullspace (its zero Ritz values are < thresh, so they never inflate the
        # count) — unlike a "converged value < thresh" test, which the nullspace triggers instantly.
        if j >= 8
            m = count(≥(thr2), eigvals(SymTridiagonal(α[1:j], β[1:(j - 1)])))
            m > maxm ? (maxm = m; since = 0) : (since += 1)
            (since >= 20 && j >= 2maxm + 10) && (kact = j; break)
        end
        (b < 1e-12 || j == kmax) && (kact = j; break)
        Q[:, j + 1] = w ./ b
        qp = Q[:, j]
        βp = b
    end
    F = eigen(SymTridiagonal(α[1:kact], β[1:(kact - 1)]))
    σ = sqrt.(max.(0.0, F.values))
    keep = findall(≥(thresh), σ)                                  # singular dirs above the eig scale
    isempty(keep) && (keep = [argmax(σ)])
    return Matrix(qr(Q[:, 1:kact] * F.vectors[:, keep]).Q)[:, 1:length(keep)]
end

"""
    certify_leading_eigs_sing(Amul, Aadj, n; neig=5, nblock_eig=40, margin=0.8, ...) -> (mr, results, …)

Leading-eigenvalue isolation via the **consolidated good + deep-far** architecture (Arnoldi +
`A*A` Lanczos): a SMALL eigenvalue block enclosed by Rump–Miyajima, and the far block made deep by
a **singular** deflation sized by the stopping rule `σ_singular < |λ_min good|`. Distinct from
`certify_leading_eigs` (per-eigenvalue far Lanczos), which is kept for the ghost-free tight σ11.

  1. Arnoldi → small eigenvalue block `V_g` (`nblock_eig`); `|λ_g|` = the `neig`-th |eigenvalue|.
  2. `A*A` Lanczos → singular block `U` with `σ ≥ margin·|λ_g|` (`_singular_block`, stopping rule).
  3. captured frame `Q = orth(V_g ∪ U)` (project `U` against the orthonormal `V_g`, drop the overlap).
  4. `G₁₁ = Q*AQ` (a FAITHFUL compression: `σ_min(G₁₁−zI) ≈ σ_min(A−zI)`); VBD certifies its eigenvalues.
  5. far block `‖G₂₂‖` (one Lanczos) + coupling `γ`, both z-independent — the far block is deep
     (`‖G₂₂‖ < |λ_g|`) so `σ₂₂ = |z|−‖G₂₂‖ > 0` on every good circle (no per-z re-trigger).
  6. per eigenvalue: adaptive radius (best of `fracs`), `σ₁₁ = svdbox(G₁₁−zI)`, assemble + arc-cover.

`results[k].cert` carries `(floor, radius, σ11, σ22, γ, G22norm, certified)`; `nsing` is the singular
block size chosen by the stopping rule.
"""
function certify_leading_eigs_sing(Amul, Aadj, n::Int; neig::Int = 5, nblock_eig::Int = 40,
        Rcirc::Union{Nothing, Real} = nothing, far_frac::Real = 0.9, refine::Integer = 6,
        margin::Real = 0.8, fracs = (0.30, 0.20, 0.12), prec::Type = Float64, ρfloor::Real = 0.004,
        tighten::Bool = true, kmax::Int = 256, reorth::Bool = true)
    # 1. eigenvalue block — the good block to enclose
    Hg, Vg, βN = _arnoldi(Amul, n, min(nblock_eig, n - 2))
    λg = eigvals(ComplexF64.(Hg))
    ordg = sortperm(abs.(λg); rev = true)
    λ_min_good = abs(λg[ordg[min(neig, length(ordg))]])
    # 2. singular block driven by the stopping scale: `Rcirc` FIXES a target circle and certifies
    # ALL eigenvalues with |λ| > Rcirc (deflate until ‖G₂₂‖ < Rcirc/2); else the leading `neig`
    # (deflate until σ < margin·|λ_g|). The former is the automatic "certify everything outside R".
    thresh = Rcirc === nothing ? margin * λ_min_good : far_frac * Rcirc
    U = _singular_block(Amul, Aadj, n, thresh; kmax = kmax)
    # 3. augmented captured frame Q = orth(V_g ∪ U); V_g orthonormal ⇒ clean project-and-drop
    Up = U .- Vg * (Vg' * U)
    keep = [j for j in axes(Up, 2) if norm(Up[:, j]) > 1e-8]
    Qf = Matrix(qr(hcat(Vg, Up[:, keep])).Q)[:, 1:(size(Vg, 2) + length(keep))]
    # 4. faithful captured block G11 = Q*AQ + its VBD (Rump–Miyajima) — certified eigenvalues.
    # The block entries come from Float64 matvecs; the VBD/candidates stay Float64, but the σ11
    # svdbox may run in `prec` (BigFloat) — a targeted boost (`Hcp`) that certifies σ_min at scales
    # below the Float64 floor without any BigFloat matvec (strongly non-normal, e.g. Lorenz).
    G11 = Qf' * _applycols(Amul, Qf)
    Hc = ComplexF64.(G11)
    Hcp = prec == Float64 ? Hc : Complex{prec}.(G11)
    mr = _certify_block(Hc, refine)
    λ = mr.centres
    ord = sortperm(abs.(λ); rev = true)
    Winv = inv(mr.W)                                      # left eigenvectors (rows) — for κ(λᵢ)
    # 5. deep far block ‖G₂₂‖ and coupling — computed ONCE (z-independent). The ASYMMETRIC pair
    # (‖G12‖,‖G21‖) feeds the block-dominance sharpening in `_assemble2`; γ=max is the display value.
    g12, g21 = _coupling_pair(Amul, Aadj, Qf)
    γ = max(g12, g21)
    Pq(x) = x .- Qf * (Qf' * x)
    GtG(x) = Pq(ComplexF64.(Aadj(Pq(ComplexF64.(Amul(Pq(x)))))))
    # FULL basis of ran(P_Q) (N = n − dim Q) — the completeness certificate for the ‖G₂₂‖ upper
    # bound (Prop. 5). N matvecs: the price of rigour.
    Ng = n - size(Qf, 2)
    Lg = lanczos_pro(GtG, n, Ng; v0 = Pq(randn(ComplexF64, n)), reorth = reorth)
    _, _, R1g, R2g = far_floor(Lg, Ng)
    G22n = sqrt(max(0.0, lanczos_lambda_max(Lg, Ng)))
    # 6. per target eigenvalue: adaptive-radius isolation. Targets = the leading `neig`, or (when
    # `Rcirc` is set) EVERY certified block eigenvalue with |λ| > Rcirc — the automatic outside-R set.
    targets = Rcirc === nothing ? ord[1:min(neig, length(ord))] :
              [i for i in ord if abs(λ[i]) > Rcirc]
    results = NamedTuple[]
    for i in targets
        c = λ[i]
        gap = minimum((abs(c - λ[j]) for j in eachindex(λ) if j != i); init = 2 *
                                                                              maximum(abs, λ))
        best = (; floor = -Inf, radius = 0.0, σ11 = 0.0, σ22 = 0.0)
        for fr in fracs
            r = fr * gap
            # σ11 certified over the WHOLE circle by the pearl cover (no arc-cover needed); σ22 from
            # the exact circle-minimum |z| (analytic) minus the z-independent deep ‖G₂₂‖.
            σ11 = Float64(_sigmin_circle(Hcp, c, r; ρfloor = ρfloor))  # svdbox in `prec`, Float64 out
            minz = abs(c) - r                                 # min |z| on the circle around c
            σ22 = G22n < minz ? minz - G22n : 0.0
            fl = _assemble2(σ11, σ22, g12, g21)
            fl > best.floor && (best = (; floor = fl, radius = r, σ11, σ22))
        end
        # Enclosure of A's eigenvalue, in TWO steps: a cheap heuristic to find the SCALE, then ONE
        # rigorous verification there (not a blind bisection). (1) Bauer–Fike estimate `r_heur =
        # κ(λᵢ)·‖lifted Ritz residual‖` — κ from the frame (one inv), residual of the lifted
        # eigenvector `Q·W[:,i]` against A (one matvec). (2) starting a few × `r_heur`, GROW the
        # circle until one pearl cover certifies floor>0: `σ_min(A−zI)>0` + count 1 ⇒ A's eigenvalue
        # lies inside — a rigorous enclosure `renc`, at ~1–3 pearl covers instead of 14 blind ones.
        κi = norm(@view mr.W[:, i]) * norm(@view Winv[i, :])
        qi = Qf * mr.W[:, i]
        qi ./= norm(qi)
        r_heur = κi * norm(ComplexF64.(Amul(qi)) .- λ[i] .* qi)     # Bauer–Fike scale estimate
        renc = NaN
        if tighten && best.floor > 0 && R2g < 1
            # (a) CHEAP Float64 probe: grow r from the heuristic scale until the (non-rigorous) σ_min
            # at `c+r` matches the linear estimate `r/κ` — i.e. it has cleared the numerical noise
            # floor (below that, svdvals is just machine noise, not the true σ_min).
            r = max(r_heur, 1e-14 * max(abs(c), 1.0))
            for _ in 1:60
                (minimum(svdvals(Hc - (c + r) * I)) >= 0.5 * r / κi) && break
                r *= 1.5
            end
            # (b) TRIPLE the distance for margin, then do the RIGOROUS pearl cover there — growing ×3
            # if it still fails. Floor the rigorous radius at ~1e-7·|c|: the verified svdbox floor
            # `κ(G₁₁)·eps` is well above the Float64 svdvals floor, and svdbox of a NEAR-singular block
            # (tiny r) is extremely slow — so we never probe below a svdbox-safe scale in Float64.
            # `prec=BigFloat` lowers both floors for a tighter enclosure.
            r = clamp(3 * r, 1e-7 * max(abs(c), 1.0), best.radius)
            for _ in 1:10
                σ11 = Float64(_sigmin_circle(Hcp, c, r; ρfloor = ρfloor))
                minz = abs(c) - r
                σ22 = G22n < minz ? minz - G22n : 0.0
                (_assemble2(σ11, σ22, g12, g21) > 0) && (renc = r; break)
                r >= best.radius && break
                r = min(3 * r, best.radius)
            end
        end
        push!(results,
            (; λ = c, disc = mr.radii[i], cluster = 1,
                cert = (; floor = best.floor, radius = best.radius,
                    enclosure = renc, r_heur = r_heur,
                    center = ComplexF64(c), σ11min = best.σ11, σ22min = best.σ22, γ = γ, G22norm = G22n,
                    certified = best.floor > 0 && R2g < 1)))
    end
    return (; mr, βN, results, V = Vg, U, Q = Qf, G11 = Hc,
        λ_min_good, G22norm = G22n, γ, g12, g21,
        R2g, Rcirc, far_deep = (Rcirc === nothing || G22n < Rcirc),
        nsing = length(keep), dimQ = size(Qf, 2), Hritz = λ[ord])
end

"""
    pseudospectrum_outside(out, z) -> certified lower bound on σ_min(A − zI), for |z| > ‖G₂₂‖

Certified `σ_min(A − zI)` for a point `z` OUTSIDE the far circle, from a `certify_leading_eigs_sing`
result `out`. Outside the far circle the far block is deep, so `σ₂₂ = |z| − ‖G₂₂‖` (z-independent),
and `σ₁₁ = svdbox(G₁₁ − zI)` is the FAITHFUL block resolvent (`σ_min(G₁₁−zI) ≈ σ_min(A−zI)`);
assembled with the coupling `γ`. Returns `0` inside the far circle (where this bound is vacuous).
Use over a grid to plot the rigorous ε-pseudospectrum outside the far circle.
"""
function pseudospectrum_outside(out, z::Number)
    r = abs(z)
    r <= out.G22norm && return 0.0
    σ11 = _sigma_min_cert(out.G11 - z * I)
    σ22 = r - out.G22norm
    return (out.R2g < 1) ? _assemble2(σ11, σ22, out.g12, out.g21) : 0.0
end

function certify_leading_eigs_sing(A::AbstractMatrix; kwargs...)
    certify_leading_eigs_sing(x -> A * x, x -> A' * x, size(A, 1); kwargs...)
end

# ----------------------------------------------------------------------------------------
# Spectral-gap certificate (Nisoli, Gauss paper arXiv:2602.19435). Workflow: ONE large Krylov
# block H_{n+k} (k ≫ n, e.g. nblock=256) is BLOCK-DIAGONALIZED + certified by miyajima_rump; an
# excluding circle separates the leading `nsep` (≈5–10) eigenvalues from the rest. Then on the
# circle, σ_min(A−zI) is assembled from (a) the BLOCK-DIAGONAL resolvent estimate on the certified
# block — a small SVD per diagonal block (vbd_resolvent / sigma_min_lower), NOT one full svdbox —
# and (b) the matvec far block (z−G₂₂)'(z−G₂₂) on the residual beyond the block, whose spectrum is
# DEEP inside the circle (≈0 for the noise-smoothed regime) so σ₂₂ is large and certifies trivially.
# ----------------------------------------------------------------------------------------
"""
    certify_gap(Amul, Aadj, n; nsep=5, nblock=256, radius=nothing, center=0, refine=6,
                block_estimate=false, nθ=64, prec=Float64, ...) -> (floor, info)

Certify the SPECTRAL GAP separating the leading `nsep` eigenvalues of `A` (matvecs `Amul`,
`Aadj`) from the rest of `σ(A)`, on a circle `Γ = {center + radius·e^{iθ}}` that leaves
`λ₁,…,λ_{nsep}` OUTSIDE and encloses `λ_{nsep+1},…`. RIGOROUS, matvec-only:

  1. Arnoldi compresses `A` to a LARGE block `H = H_{nblock}` (`nblock ≫ nsep`); `miyajima_rump`
     (frame=:schur) BLOCK-DIAGONALIZES + certifies it (`Ã = W⁻¹HW`, clusters, `κ₂`).
  2. `radius` defaults to `√(|λ_{nsep}|·|λ_{nsep+1}|)` from the certified spectrum (a circle in the
     gap below the leading `nsep`); pass `radius` to choose it yourself.
  3. On each of `nθ` samples `z_j` on Γ:
       `σ₁₁ = σ_min(H − z_jI)` — by default the EXACT `svdbox(H−z_jI)` (κ₂-free, `O(nblock³)` ≈
              ms/sample for moderate `nblock`). `block_estimate=true` uses the per-block estimate
              `sigma_min_lower` (a small SVD per diagonal block) — cheaper for a HUGE block but it
              pays a `κ₂(W)` transport factor (looser; only worth it when the full svdbox is too big);
       `σ₂₂ = σ_min(z_j − G₂₂)` — the residual far block (deflate the whole Krylov basis `V`):
              `≥ |z_j| − ‖G₂₂‖` (deep/low-rank block ⇒ `‖G₂₂‖≈0`), else `far_block_sigma_min`;
       assembled with the certified coupling `γ`. Closed by the 1-Lipschitz arc-cover.

`info` carries `radius, nsep, nblock, γ, R2, σ22min, G22norm, βN, mr, spectrum` (sorted `|λ|`)
and `block_estimate`. `prec=BigFloat` for the far block when the residual is strongly non-normal.
"""
function certify_gap(Amul, Aadj, n::Int; nsep::Int = 5, nblock::Int = 256,
        radius::Union{Nothing, Real} = nothing, center::Number = 0, refine::Integer = 6,
        block_estimate::Bool = false, nθ::Int = 64, prec::Type = Float64,
        kmax::Int = 0, nullshift::Real = 10.0, reorth::Bool = true)
    # Same shared setup as the isolation circles (one Arnoldi block, VBD, block resolvent,
    # coupling, one ugly-block ‖G₂₂‖ Lanczos) — here applied to the good/bad SEPARATING circle.
    su = _krylov_gap_setup(Amul, Aadj, n; nblock = nblock, refine = refine, reorth = reorth)
    λ, ord = su.λ, su.ord
    # radius defaults to √(|λ_{nsep}|·|λ_{nsep+1}|): a circle in the gap below the leading `nsep`.
    r = radius === nothing ?
        sqrt(abs(λ[ord[nsep]]) * abs(λ[ord[min(nsep + 1, length(ord))]])) : Float64(radius)
    fl, ci = _certify_shared_circle(su, Amul, Aadj, n, center, r;
        nθ = nθ, block_estimate = block_estimate, prec = prec,
        kmax = kmax, nullshift = nullshift, reorth = reorth)
    return fl,
    (; ci..., nsep = nsep, nblock = size(su.V, 2), βN = su.βN, mr = su.mr,
        block_estimate = block_estimate, spectrum = abs.(λ[ord]))
end

function certify_gap(A::AbstractMatrix; kwargs...)
    certify_gap(x -> A * x, x -> A' * x, size(A, 1); kwargs...)
end

# ----------------------------------------------------------------------------------------
# LOCKED gap certificate — the sound, tight variant (the singular-direction lock).
#
# The obstruction to a matvec gap floor on a circle Γ of radius r is that σ_min(z−A) is SMALL
# on Γ (the circle threads between eigenvalues), and a random-seeded far_floor on the whole
# operator is UNSOUND there: if the Krylov space misses the minimizing direction, θ_min sits
# ABOVE the true σ_min (the {1,100} counterexample). The fix (user's construction):
#
#   σ_min(z−A) on Γ is attained along the RIGHT SINGULAR VECTOR of (z − H_kk), where H_kk is
#   the NEAR-CIRCLE block (the certified eigenvalues within a band of |z|=r). We:
#     • capture that band as an orthonormal frame  W = orth(V · mr.W[:,kk])  (its right singular
#       vectors of (z−H_kk) attain the minimum — locking W removes exactly the small direction);
#     • take σ₁₁ = σ_min(z − H_kk) from the EXACT verified svdbox of the SMALL block W*AW (well
#       conditioned — unlike svdbox of the full nblock block, which can't certify σ>0);
#     • take σ₂₂ = σ_min(z − G₂₂) on the COMPLEMENT (deflate W): now the complement has NO tiny
#       eigenvalue (we removed it), so its σ_min is large & well-separated ⇒ far_floor captures
#       it SOUNDLY from a random seed (no completeness gap — the deviation lemma applies to a
#       well-separated bottom). γ = coupling. Assembled by (★), closed by the 1-Lipschitz arc.
#
# This is `krylov_resolvent_floor`'s block resolvent with the captured frame = the near-circle
# band (NOT the leading eigenvalues). `band` (relative) sets the lock width: every eigenvalue
# with ||λᵢ|−r| ≤ band·r is locked. Wider band ⇒ more directions locked (safer, larger σ₂₂)
# at the cost of a bigger small-SVD. `info` carries σ11min/σ22min/γ/R2 and the lock size `m`.
# ----------------------------------------------------------------------------------------
"""
    certify_gap_locked(Amul, Aadj, n; nsep=5, nblock=256, radius=nothing, center=0,
                       refine=6, band=0.2, nθ=64, prec=Float64, kmax=0, reorth=true)
        -> (floor, info)

Certified `σ_min(A − zI) > 0` on the circle `Γ = {center + radius·e^{iθ}}` separating the
leading `nsep` eigenvalues from the rest, by the **singular-direction lock**: capture the
near-circle eigenvalue band `H_kk` (whose right singular vectors of `z−H_kk` attain the
minimum), certify `σ₁₁ = σ_min(z−H_kk)` by the exact small `svdbox`, and certify the deflated
complement `σ₂₂` by one streamed-`R₁/R₂` far-block Lanczos — SOUND because deflating the lock
removes the only small singular value, leaving a well-separated bottom. Assembled by (★).
`band` (relative to `radius`) is the lock width. See the block comment for the rigor argument.
"""
function certify_gap_locked(Amul, Aadj, n::Int; nsep::Int = 5, nblock::Int = 256,
        radius::Union{Nothing, Real} = nothing, center::Number = 0, refine::Integer = 6,
        band::Real = 0.2, nθ::Int = 64, prec::Type = Float64, kmax::Int = 0,
        reorth::Bool = true)
    H, V, βN = _arnoldi(Amul, n, min(nblock, n - 2))
    Hc = ComplexF64.(H)
    mr = _certify_block(Hc, refine)
    λ = mr.centres
    ord = sortperm(abs.(λ); rev = true)
    r = radius === nothing ?
        sqrt(abs(λ[ord[nsep]]) * abs(λ[ord[min(nsep + 1, length(ord))]])) : Float64(radius)
    # near-circle band: every certified eigenvalue within `band` (relative) of the circle radius.
    kk = [i for i in eachindex(λ) if abs(abs(λ[i]) - r) <= band * r]
    isempty(kk) && (kk = [ord[min(nsep + 1, length(ord))]])   # at least the first inside eigenvalue
    W = _frame(V * mr.W[:, kk])                                # lifted near-circle frame (the lock)
    Hk = _captured_block(Amul, W)                              # H_kk = W*AW (small, well conditioned)
    g12, g21 = _coupling_pair(Amul, Aadj, W)
    γ = max(g12, g21)
    floors = Float64[]
    R2max = 0.0
    σ11min = Inf
    σ22min = Inf
    for t in 0:(nθ - 1)
        z = center + r * cis(2π * t / nθ)
        σ11 = _sigma_min_cert(Hk - z * I)                     # exact small svdbox — the minimizer
        σ22, info = far_block_sigma_min(Amul, Aadj, n, W, z;  # complement: minimizer removed ⇒ sound
            prec = prec, kmax = kmax, reorth = reorth)
        R2max = max(R2max, info.R2)
        σ11min = min(σ11min, σ11)
        σ22min = min(σ22min, σ22)
        push!(floors, R2max < 1 ? _assemble2(σ11, σ22, g12, g21) : 0.0)
    end
    fl = minimum(floors) - 2 * r * sin(π / (2nθ))
    return fl,
    (; floor = fl, radius = r, center = ComplexF64(center), nsep = nsep, m = length(kk),
        γ = γ, R2 = R2max, σ11min = σ11min, σ22min = σ22min, βN = βN, mr = mr,
        spectrum = abs.(λ[ord]), certified = (fl > 0 && R2max < 1))
end

function certify_gap_locked(A::AbstractMatrix; kwargs...)
    certify_gap_locked(x -> A * x, x -> A' * x, size(A, 1); kwargs...)
end
