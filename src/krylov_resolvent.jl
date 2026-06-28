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
        # breakdown: absolute (a true zero residual) OR relative (numerical rank / happy
        # breakdown) — the Krylov subspace is invariant, hence COMPLETE for the deviation lemma.
        if !isfinite(b) || b <= u * abs(a) || b <= breaktol * Tnorm
            kact = k
            break
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
# Far-block floor σ_min(G22 − zI) = √(max(0, θ_min − β₂₂)) from a LanczosRun (draft Prop.).
# Returns (σ₂₂, θ_min, ‖R₁‖, ‖R₂‖).
#
# Deviation lemma: spec(M_z) ⊂ ⋃ D(θ_i, β₂₂),  β₂₂ = ‖R₁‖₂/(1−‖R₂‖),  R₁ = Q*(M_zQ − QT_k).
# Now  M_zQ − QT_k = β_k q_{k+1} eₖᵀ + F_k, so R₁ = β_k (Q*q_{k+1}) eₖᵀ + Q*F_k and
#   |R₁[i,j]| ≤ β_k·wlast[i]·[j=k] + |⟨q_i,f_j⟩| ≤ β_k·wlast[i]·[j=k] + fn[j].
# The β_k term is multiplied by the orthogonality wlast = |Q*q_{k+1}| (≈√u with reorth) — it
# must NOT enter bare (that overcount, ~β_k, is what kept the floor pinned to 0 at partial
# basis). ‖R₁‖₂ is bounded by the GEOMETRIC MEAN √(‖R₁‖₁‖R₁‖∞) (1- and ∞-norms), tighter
# than Frobenius and the same bound used for ‖R₂‖.
function far_floor(L::LanczosRun)
    k = L.k
    k < 2 && return (0.0, Float64(L.α[1]), Inf, Float64(L.R2))
    # θ_min from the tridiagonal in Float64 — eigenvalues are well conditioned and θ_min ~1e-7
    # is representable, so the slow generic BigFloat symmetric eigensolve is unnecessary.
    θmin = minimum(eigvals(SymTridiagonal(Float64.(L.α[1:k]), Float64.(L.β[1:(k - 1)]))))
    fns = @view L.fn[1:k]
    wl = @view L.wlast[1:k]
    R1_1 = max(L.β[k] * sum(wl) + k * fns[k], k * maximum(fns))     # max column sum of |R₁|
    R1_inf = L.β[k] * maximum(wl) + sum(fns)                        # max row sum of |R₁|
    R1 = sqrt(R1_1 * R1_inf)                                        # ‖R₁‖₂ ≤ √(‖R₁‖₁‖R₁‖∞)
    # β₂₂ tiny in BigFloat (wlast,fn ~ √u, u) ⇒ clears θ_min where Float64 cannot.
    β22 = L.R2 < 1 ? Float64(R1 / (1 - L.R2)) : Inf
    return (sqrt(max(0.0, θmin - β22)), θmin, Float64(R1), Float64(L.R2))
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

# CERTIFIED upper bound on the coupling γ = max(‖P₂₂AW‖₂, ‖P₂₂A*W‖₂) (verified L² opnorm).
function _coupling_cert(Amul, Aadj, W)
    defl(M) = M .- W * (W' * M)
    return max(upper_bound_L2_opnorm(BallMatrix(defl(_applycols(Amul, W)))),
        upper_bound_L2_opnorm(BallMatrix(defl(_applycols(Aadj, W)))))
end

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
    kmax <= 0 && (kmax = n - m - 1)
    Wt = prec == Float64 ? W : Matrix{Complex{prec}}(W)
    zz = Complex{prec}(z);
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
    σ22, θmin, R1, R2 = far_floor(L)
    return σ22, (; θmin, R1, R2, k = L.k, nreorth = L.nreorth)
end

# block-resolvent assembly (★): increasing in σ11 and σ22, so feeding their lower bounds
# yields a valid lower bound on σ_min(A − zI).
_assemble(σ11, σ22, γ) = (σ11 <= 0 || σ22 <= 0) ? 0.0 : σ11 * σ22 / (σ11 + σ22 + γ)

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
    γ = _coupling_cert(Amul, Aadj, W)
    σ22, info = far_block_sigma_min(Amul, Aadj, n, W, z; kwargs...)
    σ11 = _sigma_min_cert(H - z * I)                        # VERIFIED small SVD (svdbox)
    floor = info.R2 >= 1 ? 0.0 : _assemble(σ11, σ22, γ)
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
    γ = _coupling_cert(Amul, Aadj, W)

    # ONE expensive far-block Krylov, at the centre; 1-Lipschitz over the disk.
    σ22c, info = far_block_sigma_min(Amul, Aadj, n, W, center;
        prec = prec, kmax = kmax, nullshift = nullshift, reorth = reorth)
    r = (auto_radius && info.R2 < 1) ? min(radius, 0.5 * σ22c) : radius
    σ22lo = info.R2 < 1 ? σ22c - r : -Inf

    # captured-block σ11 over the circle: VERIFIED small SVDs (svdbox) + 1-Lipschitz arc-cover
    σ11s = [_sigma_min_cert(H - (center + r * cis(2π * t / nθ)) * I) for t in 0:(nθ - 1)]
    σ11lo = minimum(σ11s) - 2 * r * sin(π / (2nθ))

    floor = _assemble(σ11lo, σ22lo, γ)
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

export far_block_sigma_min, certify_leading_eigs, certify_gap

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

"""
    certify_leading_eigs(Amul, Aadj, n; nblock=16, neig=5, refine=6, prec=Float64,
                         frac=0.4, nθ=24) -> (mr, results)

Certify the leading eigenvalues of `A` (matvecs `Amul`, `Aadj`) isolated, by the
**Krylov–Rump–Miyajima decoupling**:

  1. Arnoldi compresses `A` to the `nblock × nblock` Rayleigh block `H` (and Krylov basis `V`).
  2. `miyajima_rump(H, vbd_solve(H))` block-diagonalizes `H` — the VBD block-orthonormal frame
     produces the decoupled frame `mr.W` (`Ã = mr.W⁻¹ H mr.W` almost block-diagonal), certified
     eigenvalue discs, and the cluster partition.
  3. For each cluster, the lifted REFINED frame `Wᵢ = V·mr.W[:, cluster]` deflates the cluster
     cleanly (coupling = its Ritz residual, ~`E`), and `certify_gap_on_circle` (with
     `auto_radius`) certifies `σ_min(A − zI) > 0` on a circle `Γᵢ` ⇒ `Γᵢ` encloses exactly that
     cluster ⇒ those eigenvalues are isolated with the returned gap.

Using `mr.W` (not raw Ritz vectors) is what decouples the less-isolated modes; `prec=BigFloat`
handles strongly non-normal far blocks. `results[k]` carries `(λ, disc, cluster, cert)`.

REMARK (full spectral gap vs eigenvalue isolation). This certifies each leading eigenvalue
ISOLATED — a small circle `Γᵢ` around it contains no other eigenvalue. That is NOT yet the
SPECTRAL GAP. Once `λ₁,…,λ_k` are the certified leading eigenvalues, the gap separating that
block from the rest requires ONE further far-block certificate: deflate ALL `k` certified modes
(`W = V·mr.W[:, 1:k]`, the refined-frame columns) and certify `σ_min(z − G₂₂) > 0` on a circle
`Γ_gap` enclosing the REMAINING spectrum `λ_{k+1}, λ_{k+2}, …` — a rigorous resolvent bound on a
contour enclosing the whole far spectrum `σ(G₂₂)`, the contour certification of Nisoli (Gauss
paper, arXiv:2602.19435). It is the SAME primitive (`far_block_sigma_min`), run ONCE on the
bulk-enclosing circle rather than the per-eigenvalue circles, and is the binding, harder step
for strongly non-normal operators (the far block's pseudospectrum reaches across the gap).
"""
function certify_leading_eigs(Amul, Aadj, n::Int; nblock::Int = 16, neig::Int = 5,
        refine::Integer = 6, prec::Type = Float64, frac::Real = 0.4, nθ::Int = 24)
    H, V, βN = _arnoldi(Amul, n, nblock)
    Hc = ComplexF64.(H)
    mr = miyajima_rump(Hc, first(vbd_solve(Hc; maxsteps = refine)); kappa_mode = :svdbox)
    λ = mr.centres
    ord = sortperm(abs.(λ); rev = true)
    done = Set{Int}()
    leadcols = Int[]                                     # all certified-leading H-columns
    results = NamedTuple[]
    for i in ord[1:min(neig, length(ord))]
        i in done && continue
        cl = mr.clusters[findfirst(c -> i in c.idx, mr.clusters)].idx
        union!(done, cl)
        append!(leadcols, cl)
        c = λ[i]
        gap = minimum(abs(c - λ[j]) for j in eachindex(λ) if !(j in cl))
        Wi = V * mr.W[:, cl]                             # lift the REFINED cluster frame
        cert = certify_gap_on_circle(Amul, Aadj, n, Wi, c, frac * gap;
            prec = prec, auto_radius = true, nθ = nθ)
        push!(results, (; λ = c, disc = mr.radii[i], cluster = length(cl), cert))
    end
    # the certified-leading frame, for a subsequent certify_gap (separation from the fast spectrum)
    Wlead = V * mr.W[:, sort(unique(leadcols))]
    return (; mr, βN, results, V, Wlead, Hritz = λ[ord])
end

function certify_leading_eigs(A::AbstractMatrix; kwargs...)
    certify_leading_eigs(x -> A * x, x -> A' * x, size(A, 1); kwargs...)
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
    H, V, βN = _arnoldi(Amul, n, min(nblock, n - 2))
    Hc = ComplexF64.(H)
    # frame=:schur ⇒ a BLOCK-ORTHONORMAL frame (orthonormal within each block): κ(W) is then
    # governed by benign inter-block subspace angles, NOT the malignant intra-block non-normality
    # (draft.tex §"block-orthonormal Schur frame"). mr also gives the CERTIFIED block spectrum.
    mr = miyajima_rump(Hc, first(vbd_solve(Hc; maxsteps = refine)); kappa_mode = :svdbox)
    # σ₁₁ = σ_min(H − zI) of the captured block. By default the EXACT `svdbox(H−zI)` — for a
    # moderate block (`nblock` ≲ hundreds) it is O(nblock³) ≈ ms/sample and κ₂-FREE. The
    # `block_estimate` path (`vbd_resolvent`/`sigma_min_lower`: a small SVD per diagonal block)
    # is cheaper for a HUGE block but pays a `κ₂(W)` transport factor — use it only when the
    # full svdbox is too big.
    vr = block_estimate ? vbd_resolvent(Hc, mr) : nothing
    λ = mr.centres
    ord = sortperm(abs.(λ); rev = true)
    r = radius === nothing ?
        sqrt(abs(λ[ord[nsep]]) * abs(λ[ord[min(nsep + 1, length(ord))]])) : Float64(radius)
    γ = _coupling_cert(Amul, Aadj, V)                    # certified coupling (≈ β_N)

    # Certified ‖G₂₂‖ (z-independent): largest eigenvalue of G₂₂*G₂₂ via Lanczos (the MAX
    # converges fast) + deviation lemma. For a low-rank / deep far block (everything captured in
    # the block) G₂₂ ≈ 0, so σ_min(z−G₂₂) ≥ |z| − ‖G₂₂‖ directly — the per-z σ_min Lanczos can't
    # certify a trivial/empty far block (it breaks at k=1 with the complement unseen).
    Pv(x) = x .- V * (V' * x)
    GtG(x) = Pv(ComplexF64.(Aadj(Pv(ComplexF64.(Amul(Pv(x)))))))
    Lg = lanczos_pro(GtG, n, min(40, n - 2); v0 = Pv(randn(ComplexF64, n)), reorth = reorth)
    _, _, R1g, R2g = far_floor(Lg)
    θmaxg = length(Lg.α) <= 1 ? (isempty(Lg.α) ? 0.0 : Lg.α[1]) :
            maximum(eigvals(SymTridiagonal(Float64.(Lg.α), Float64.(Lg.β[1:(end - 1)]))))
    G22n = R2g < 1 ? sqrt(max(0.0, θmaxg + R1g / (1 - R2g))) : Inf   # ‖G₂₂‖ upper bound

    floors = Float64[]
    R2max = R2g
    σ22min = Inf
    for t in 0:(nθ - 1)
        z = center + r * cis(2π * t / nθ)
        σ11 = block_estimate ? sigma_min_lower(vr, z; near = true) :  # per-block, κ₂-transported
              _sigma_min_cert(Hc - z * I)                            # exact full svdbox (default)
        # deep far block ⇒ σ_min(z−G₂₂) ≥ |z| − ‖G₂₂‖ (z-independent, no Lanczos); only if the far
        # block reaches the circle (‖G₂₂‖ ≥ |z|) fall back to the per-z far-block σ_min certificate.
        local σ22
        if G22n < abs(z)
            σ22 = abs(z) - G22n
        else
            σ22, info = far_block_sigma_min(Amul, Aadj, n, V, z;
                prec = prec, kmax = kmax, nullshift = nullshift, reorth = reorth)
            R2max = max(R2max, info.R2)
        end
        σ22min = min(σ22min, σ22)
        push!(floors, R2max < 1 ? _assemble(σ11, σ22, γ) : 0.0)
    end
    fl = minimum(floors) - 2 * r * sin(π / (2nθ))
    return fl,
    (; floor = fl, radius = r, center = ComplexF64(center),
        nsep = nsep, nblock = size(V, 2),
        γ = γ, R2 = R2max, σ22min = σ22min, G22norm = G22n, βN = βN, mr = mr,
        block_estimate = block_estimate, spectrum = abs.(λ[ord]),
        certified = (fl > 0 && R2max < 1))
end

function certify_gap(A::AbstractMatrix; kwargs...)
    certify_gap(x -> A * x, x -> A' * x, size(A, 1); kwargs...)
end
