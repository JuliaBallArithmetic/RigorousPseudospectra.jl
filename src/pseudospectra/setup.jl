# Setup phase — z-independent precomputation (implementation.md §3, paper §1/§3).
#
# FLOAT64 first cut. Every residual quantity the certified version needs is
# computed here, in the SAME structure — only the container is ComplexF64 instead
# of BallMatrix. Nothing is assumed exact: the Schur basis Z is only approximately
# unitary (residual R₂ = YZ − I), the reduced residual S = Z⁻¹AZ − T̃ is carried,
# and the conditioning κ is computed from ‖Z‖, ‖Y‖, ‖R₂‖ — never set to 1. Moving
# to BallArithmetic is a drop-in: swap the matrices for BallMatrix and the opnorms
# for certified `upper_bound_*` bounds.
#
# Collapsed matrix (the bridge identity (A−zI)⁻¹ = X̃(M₀−zI)⁻¹X̃⁻¹):
#     M₀ = Z⁻¹ A Z = B⁻¹C        (= T̃ − S; differs from the triangular T̃ by S).

"""
    CertSetup

z-independent certified data (paper §1/§3). Carries the full residual record so the
σ_min(A−zI) bounds are honest (no exact-unitarity assumption):

- `M0` — collapsed matrix Z⁻¹AZ = B⁻¹C (recentred-gap core; contains S)
- `centres`/`radrow`/`radcol` — diag(M0) and its off-diagonal row/col abs-sums
- `eigs`/`tailrow`/`tailcol` — diag(T̃) and strict-upper row/col tails (free gap, on T̃)
- `beta_*` — solve-residual slacks for M0 (Lemma `Sresidual`)
- `kappa_1`/`kappa_inf`/`kappa_2` — ‖Z‖_p·‖Z⁻¹‖_p conditioning prefactors
- `normS_2` — ‖S‖₂ = ‖T̃ − M0‖₂, the residual charged when transporting the free
  (triangular) gap on T̃ to a σ_min(A−zI) bound
"""
struct CertSetup{M}
    A::M
    Z::M            # Schur vectors (approximately unitary), X̃
    T::M            # upper-triangular Schur factor, T̃
    Y::M            # inv(Z) ≈ X̃⁻¹
    M0::M           # collapsed matrix Z⁻¹AZ = B⁻¹C (= T̃ − S)
    nrmR2_inf::Float64
    nrmR2_1::Float64
    nrmR2_2::Float64
    centres::Vector{ComplexF64}
    radrow::Vector{Float64}
    radcol::Vector{Float64}
    eigs::Vector{ComplexF64}
    tailrow::Vector{Float64}
    tailcol::Vector{Float64}
    beta_inf::Float64
    beta_1::Float64
    kappa_inf::Float64
    kappa_1::Float64
    kappa_2::Float64
    normS_2::Float64
    AZ::M                       # A*Z, z-independent (inclusion-side σ_min UPPER bound)
    Zcolnorm::Vector{Float64}   # ‖Z[:,i]‖₂
    frame::Symbol               # :eigen (rung 1, diagonal-residue) or :schur (legacy fallback)
end

# off-diagonal absolute row/column sums of a square matrix
function _offdiag_sums(M)
    n = size(M, 1)
    row = zeros(Float64, n)
    col = zeros(Float64, n)
    @inbounds for j in 1:n, i in 1:n
        if i != j
            a = abs(M[i, j])
            row[i] += a
            col[j] += a
        end
    end
    return row, col
end

# strict-upper-triangular row/column tails of an upper-triangular T
function _triangular_tails(T)
    n = size(T, 1)
    row = zeros(Float64, n)   # τ_i = Σ_{j>i}|T_ij|
    col = zeros(Float64, n)   # Σ_{i<j}|T_ij|
    @inbounds for j in 2:n, i in 1:(j - 1)
        a = abs(T[i, j])
        row[i] += a
        col[j] += a
    end
    return row, col
end

"""
    _certify_from_basis(Ac, Z, T) -> CertSetup

Build the residual record for a given (approximate) Schur basis `Z` and triangular
factor `T` of `Ac`. Shared by `certified_setup` and by `reorder_cluster`, which
re-certifies the reordered basis the same way. Verifies the Neumann condition
`‖R₂‖ < 1` (ℓ¹ and ℓ∞) and throws otherwise.
"""
function _certify_from_basis(Ac::AbstractMatrix, Z::AbstractMatrix, T::AbstractMatrix;
        frame::Symbol = :schur)
    Y = inv(Z)
    B = Y * Z                # = I + R₂
    R2 = B - I
    nrmR2_inf = opnorm(R2, Inf)
    nrmR2_1 = opnorm(R2, 1)
    nrmR2_2 = opnorm(R2, 2)
    if !(nrmR2_inf < 1) || !(nrmR2_1 < 1)
        error("Neumann condition ‖R₂‖ < 1 failed (‖R₂‖_∞=$nrmR2_inf, " *
              "‖R₂‖_1=$nrmR2_1): certificate unavailable for this basis.")
    end

    C = Y * Ac * Z
    M0 = B \ C               # = Z⁻¹AZ, the collapsed matrix (contains S)
    res = B * M0 - C         # solve residual (≈ 0 in Float64)
    beta_inf = opnorm(res, Inf) / (1 - nrmR2_inf)
    beta_1 = opnorm(res, 1) / (1 - nrmR2_1)

    centres = diag(M0)
    radrow, radcol = _offdiag_sums(M0)
    eigs = diag(T)
    tailrow, tailcol = _triangular_tails(T)

    kappa_inf = opnorm(Z, Inf) * opnorm(Y, Inf) / (1 - nrmR2_inf)
    kappa_1 = opnorm(Z, 1) * opnorm(Y, 1) / (1 - nrmR2_1)
    # 2-norm conditioning, tight for a near-unitary Schur basis (κ₂≈1) — NOT 1 by
    # assumption, and NOT the loose sqrt(κ₁κ∞): computed from the 2-norm residual.
    kappa_2 = opnorm(Z, 2) * opnorm(Y, 2) / (1 - nrmR2_2)
    normS_2 = opnorm(T - M0, 2)          # ‖S‖₂, S = T̃ − M0

    AZ = Ac * Z
    Zcolnorm = [norm(view(Z, :, i)) for i in 1:size(Z, 2)]

    return CertSetup{typeof(M0)}(Ac, Matrix(Z), Matrix(T), Y, M0,
        nrmR2_inf, nrmR2_1, nrmR2_2,
        centres, radrow, radcol, eigs, tailrow, tailcol,
        beta_inf, beta_1, kappa_inf, kappa_1, kappa_2, normS_2,
        AZ, Zcolnorm, frame)
end

# eigenvector (diagonal-residue) frame, rung 1: T = diag(eigenvalues), so the
# off-diagonal "tails" vanish and the collapsed M₀ = W⁻¹AW carries the TIGHT
# residue (radii O(‖R₁‖)). κ comes from the residuals (‖W‖‖Y‖/(1−‖R₂‖)), never cond.
function _eigen_setup(Ac)
    E = eigen(Ac)
    return _certify_from_basis(Ac, Matrix(E.vectors), Matrix(Diagonal(E.values));
        frame = :eigen)
end

# Schur frame (rung 3 / legacy fallback): unitary basis, κ≈1, but O(1) tails.
function _schur_setup(Ac)
    F = schur(Ac)
    return _certify_from_basis(Ac, Matrix(F.Z), Matrix(F.T); frame = :schur)
end

"""
    certified_setup(A; frame = :auto, kappa_max = 1.0e2) -> CertSetup

Run the once-only O(n³) setup. The committed default is the **eigenvector
(diagonal-residue) frame** (rung 1): for a (near-)diagonalizable `A` the residue
`M₀ − D` is tiny, so the recentred gap is `≈ dist/κ` — tight when the eigenbasis
conditioning `κ₂(X̃)` is benign. `κ₂` is certified from the residuals
(`‖X̃‖₂‖Ỹ‖₂/(1−‖R₂‖₂)`), never via `cond`. The `:auto` gate keeps the eigen frame
when `‖R₂‖ < 1` **and** `κ₂ < kappa_max`, else falls to the legacy **Schur frame**
(rung 3: unitary, κ≈1, but loose O(1) tails — relies on the cluster path). Per-cell
escalation can still reach the Schur cluster path even from an eigen-frame setup
(see `bisect`). `frame = :eigen`/`:schur` force the choice.
"""
function certified_setup(A::AbstractMatrix; frame::Symbol = :auto,
        kappa_max::Real = 1.0e2)
    Ac = ComplexF64.(Matrix(A))
    frame === :schur && return _schur_setup(Ac)
    frame === :eigen && return _eigen_setup(Ac)
    # :auto — eigenvector frame when its conditioning is benign, else Schur.
    Se = try
        _eigen_setup(Ac)
    catch
        nothing
    end
    (Se !== nothing && Se.kappa_2 < kappa_max) && return Se
    return _schur_setup(Ac)
end
