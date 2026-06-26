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
σ_min(G22−zI) = √(λ_min(M_z)) with the HERMITIAN  M_z = P₂₂ (G22−zI)*(G22−zI) P₂₂.
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
    # needing no basis. Reorthogonalization, however, touches earlier columns. For clarity this
    # reference retains the basis (`cols`, O(nk)); in practice only the FEW directions whose Ritz
    # value converges ever get flagged, so a production variant keeps just those (Parlett–Scott
    # selective orthogonalization against converged Ritz vectors) and is nearly memory-free.
    cols = Vector{Vector{T}}(undef, kmax)

    for k in 1:kmax
        cols[k] = q
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
        if !isfinite(b) || b <= u * abs(a)               # Lanczos breakdown
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
    γ = _coupling(Amul, Aadj, W)
    σ22, info = far_block_sigma_min(Amul, Aadj, n, W, z; kwargs...)
    σ11 = minimum(svdvals(H - z * I))                       # small certified SVD (m×m)
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
    γ = _coupling(Amul, Aadj, W)

    # ONE expensive far-block Krylov, at the centre; 1-Lipschitz over the disk.
    σ22c, info = far_block_sigma_min(Amul, Aadj, n, W, center;
        prec = prec, kmax = kmax, nullshift = nullshift, reorth = reorth)
    r = (auto_radius && info.R2 < 1) ? min(radius, 0.5 * σ22c) : radius
    σ22lo = info.R2 < 1 ? σ22c - r : -Inf

    # cheap captured-block σ11 over the circle: small SVDs + 1-Lipschitz arc-cover
    σ11s = [minimum(svdvals(H - (center + r * cis(2π * t / nθ)) * I)) for t in 0:(nθ - 1)]
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
  2. `miyajima_rump(H; refine)` block-diagonalizes `H` — the §4e Sylvester–Newton refinement
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
    mr = miyajima_rump(ComplexF64.(H); refine = refine, kappa_mode = :svdbox)
    λ = mr.centres
    ord = sortperm(abs.(λ); rev = true)
    done = Set{Int}()
    results = NamedTuple[]
    for i in ord[1:min(neig, length(ord))]
        i in done && continue
        cl = mr.clusters[findfirst(c -> i in c.idx, mr.clusters)].idx
        union!(done, cl)
        c = λ[i]
        gap = minimum(abs(c - λ[j]) for j in eachindex(λ) if !(j in cl))
        Wi = V * mr.W[:, cl]                             # lift the REFINED cluster frame
        cert = certify_gap_on_circle(Amul, Aadj, n, Wi, c, frac * gap;
            prec = prec, auto_radius = true, nθ = nθ)
        push!(results, (; λ = c, disc = mr.radii[i], cluster = length(cl), cert))
    end
    return (; mr, βN, results)
end

function certify_leading_eigs(A::AbstractMatrix; kwargs...)
    certify_leading_eigs(x -> A * x, x -> A' * x, size(A, 1); kwargs...)
end

# ----------------------------------------------------------------------------------------
# Spectral-gap certificate via the far block's DOMINANT SINGULAR SUBSPACE (Nisoli, Gauss
# paper arXiv:2602.19435; the recursion applied to G₂₂*G₂₂).
# ----------------------------------------------------------------------------------------
"""
    certify_gap(Amul, Aadj, n, Wslow, radius; kgram=40, nθ=64, tol=1e-8) -> (floor, info)

Certify the SPECTRAL GAP: `σ_min(A − zI) > 0` on the circle `|z| = radius` enclosing the fast
spectrum, given the certified leading frame `Wslow` (`n×k`, e.g. `V·mr.W[:,1:k]` from
`certify_leading_eigs`). The far block `G₂₂ = P_W A P_W` (`P_W = I − Wslow Wslow*`) is handled
by its dominant SINGULAR subspace rather than a direct `σ_min` Lanczos:

  * run the eigenvalue scheme on the HERMITIAN Gram `G₂₂*G₂₂` (matvec `x ↦ G₂₂*(G₂₂ x)`) to
    capture the `k'` dominant singular values `Σ` and right vectors `V_g`; `U_g = G₂₂ V_g/Σ`.
    The uncaptured singular values are the residual `‖E‖ = σ_{k'+1}` (≈0 for low-rank far blocks).
  * `(U_g Σ V_g* − zI)` is block-diagonal w.r.t. `span[U_g,V_g]` and its complement (where it is
    `−zI`), so `σ_min(U_gΣV_g* − zI) = min(σ_min(B − zI), |z|)`, `B = Q*(U_gΣV_g*)Q` a small
    `≤2k'` SVD. Hence `σ_min(G₂₂ − zI) ≥ min(σ_min(B−zI), |z|) − ‖E‖`.

Then `σ_min(A − zI) ≥ assemble(σ_slow, σ_far, γ)` on the circle (`σ_slow = σ_min(Wslow*A Wslow −
zI)`, `γ` the coupling), Lipschitz arc-covered. This SIDESTEPS the non-normal `σ_min` collapse:
all the structure lives in the few large singular directions (a small dense SVD), the rest is
`≈|z|`. (For a rigorous tail, `‖E‖` needs a certified upper bound on `σ_{k'+1}`; here it is the
Ritz estimate — Float64, the candidate.)
"""
function certify_gap(Amul, Aadj, n::Int, Wslow::AbstractMatrix, radius::Real;
        kgram::Int = 40, nθ::Int = 64, tol::Real = 1e-8)
    W = _frame(Wslow)
    Pw(x) = x .- W * (W' * x)
    G22mul(x) = Pw(Amul(Pw(x)))
    G22adj(x) = Pw(Aadj(Pw(x)))
    Gram(x) = G22adj(G22mul(x))                          # Hermitian PSD, eigenvalues = σ(G₂₂)²

    Hg, Vb, _ = _arnoldi(Gram, n, min(kgram, n - size(W, 2) - 1))
    E = eigen(Hermitian((Hg + Hg') / 2))
    σ2 = real.(E.values)
    ord = sortperm(σ2; rev = true)
    keep = [i for i in ord if σ2[i] > tol * σ2[ord[1]]]  # dominant singular values²
    nrest = length(ord) - length(keep)
    Enorm = sqrt(max(0.0, nrest > 0 ? σ2[ord[length(keep) + 1]] : 0.0))   # ‖E‖ = σ_{k'+1}
    Vg = Vb * E.vectors[:, keep]                         # right singular vectors
    Σg = sqrt.(max.(σ2[keep], 0.0))
    Ug = reduce(hcat, [G22mul(Vector{ComplexF64}(Vg[:, j])) ./ max(Σg[j], eps())
                       for j in eachindex(Σg)])          # left singular vectors
    Q = Matrix(qr(hcat(Ug, Vg)).Q)
    B = Q' * (Ug * Diagonal(complex.(Σg)) * Vg') * Q     # ≤2k'×2k' block of U Σ V*

    Hs = W' * _applycols(Amul, W)                        # slow block
    γ = _coupling(Amul, Aadj, W)

    floors = Float64[]
    for t in 0:(nθ - 1)
        z = radius * cis(2π * t / nθ)
        σlr = min(minimum(svdvals(B - z * I)), abs(z))   # σ_min(U_gΣV_g* − zI)
        σfar = max(0.0, σlr - Enorm)                     # σ_min(G₂₂ − zI)
        σslow = minimum(svdvals(Hs - z * I))
        push!(floors, _assemble(σslow, σfar, γ))
    end
    fl = minimum(floors) - 2 * radius * sin(π / (2nθ))
    return fl,
    (; floor = fl, radius = Float64(radius), Enorm = Enorm,
        ksing = length(keep), γ = γ, certified = fl > 0)
end

function certify_gap(A::AbstractMatrix, Wslow::AbstractMatrix, radius::Real; kwargs...)
    certify_gap(x -> A * x, x -> A' * x, size(A, 1), Wslow, radius; kwargs...)
end
