# SVD-frame certificate (svd_frame_certificate.tex, paper §"SVD-frame").
#
# An ALTERNATIVE certificate layer: a certified lower bound on σ_min(A − zI)²,
# valid for all z, from a SINGLE (verified) SVD of A. Targets σ_min² directly via
# the normal-equations matrix M(z) = (A − zI)ᴴ(A − zI), instead of the triangular
# surrogate of the Schur gaps — no κ(Z) transport, overshoot governed by the
# singular-vector misalignment W = VᴴU rather than by triangular tails.
#
# Frame (exact SVD A = UΣVᴴ): conjugating by V,
#     VᴴM(z)V = (Σ² + |z|²I) − (z̄ WΣ + z ΣWᴴ),     W = VᴴU.
# Constant + quadratic parts are diagonal (σ_i² + |z|², growing with |z| — far
# region is easy); the whole z-linear off-diagonal carries the non-normality.
# A Hermitian Gershgorin bound on λ_min, minus a z-uniform deviation from the
# approximate-SVD residuals, gives the certificate (Thm `svdmain`).
#
# FLOAT64 first cut. With a LAPACK SVD the residuals P, Q, E are ~machine-eps, so
# the deviation is negligible; the rigorous version takes a verified SVD and
# certified residual norms (BallArithmetic `rigorous_svd`).
#
# NOTE vs. the .tex: the paper's G(z) as printed keeps the diagonal as σ̃_i² + |z|²
# and subtracts only the off-diagonal row sum. That drops the diagonal part of the
# z-linear term, −2σ̃_i Re(z̄ W̃_ii), which is generally nonzero (it is the entire
# bound for Hermitian-PSD A). We include it — the correct Hermitian-Gershgorin
# left endpoint — so the bound is a genuine lower bound. See the paper section,
# which is corrected to match.

"""
    SVDFrame

z-independent data for the SVD-frame certificate (Thm `svdmain`). Built from one
SVD of `A`: singular values, the misalignment `W = VᴴU`, the precomputed linear
off-diagonal coefficient matrices, and the deviation polynomial coefficients
`e0 + e1|z| + e2|z|²` bounding `‖𝓔(z)‖₂` from the SVD residuals.
"""
struct SVDFrame{M}
    A::M
    U::M
    V::M
    S::Vector{Float64}             # singular values σ̃ (descending)
    Wdiag::Vector{ComplexF64}      # diag(W), W = VᴴU
    Acoef::Matrix{ComplexF64}      # Acoef[i,j] = σ̃_j W_ij
    Bcoef::Matrix{ComplexF64}      # Bcoef[i,j] = σ̃_i conj(W_ji)
    e0::Float64
    e1::Float64
    e2::Float64
    normP::Float64
    normQ::Float64
    normE::Float64
end

"""
    svd_frame(A) -> SVDFrame

Build the SVD-frame certificate data with one O(n³) SVD (Float64 first cut; the
rigorous version uses a verified SVD and certified residual norms).
"""
function svd_frame(A::AbstractMatrix)
    Ac = ComplexF64.(Matrix(A))
    F = svd(Ac)
    U, V = Matrix(F.U), Matrix(F.V)
    S = F.S
    W = V' * U

    # z-independent residuals and their 2-norms
    E = U' * Ac * V - Diagonal(complex(S))
    P = V' * V - I
    Q = U' * U - I
    normE = opnorm(E, 2)
    normP = opnorm(P, 2)
    normQ = opnorm(Q, 2)
    normSig = isempty(S) ? 0.0 : S[1]      # ‖Σ̃‖₂ = σ_max

    e0 = 2 * normSig * normE + normE^2 + normSig^2 * normQ
    e1 = 2 * normE + 2 * normSig * normQ
    e2 = normP

    n = size(Ac, 1)
    Acoef = Matrix{ComplexF64}(undef, n, n)
    Bcoef = Matrix{ComplexF64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        Acoef[i, j] = S[j] * W[i, j]
        Bcoef[i, j] = S[i] * conj(W[j, i])
    end

    return SVDFrame{typeof(Ac)}(Ac, U, V, S, diag(W), Acoef, Bcoef,
        e0, e1, e2, normP, normQ, normE)
end

"""
    svdframe_gamma(z, F::SVDFrame) -> Float64

The certified lower bound `γ(z)` on `σ_min(A − zI)² = λ_min(M(z))` (Thm `svdmain`):
the Hermitian-Gershgorin lower bound of the ideal frame minus the deviation
`‖𝓔(z)‖₂`. May be negative (inconclusive).
"""
function svdframe_gamma(z, F::SVDFrame)
    az = abs(z)
    cz = conj(z)
    n = length(F.S)
    G = Inf
    @inbounds for i in 1:n
        # Hermitian-Gershgorin diagonal entry (real): σ_i² + |z|² − 2σ_i Re(z̄ W_ii)
        di = F.S[i]^2 + az^2 - 2 * F.S[i] * real(cz * F.Wdiag[i])
        Ri = 0.0
        for j in 1:n
            j == i && continue
            Ri += abs(cz * F.Acoef[i, j] + z * F.Bcoef[i, j])
        end
        G = min(G, di - Ri)
    end
    dev = F.e0 + F.e1 * az + F.e2 * az^2
    return G - dev
end

"""
    gap_svd(z, F::SVDFrame) -> Float64

Certified lower bound on `σ_min(A − zI)` from the SVD-frame certificate:
`sqrt(γ(z))` where `γ(z) = svdframe_gamma(z, F)`, or `-Inf` if `γ(z) ≤ 0`
(inconclusive). Cost O(n²) per z (O(n) with per-row envelopes).
"""
function gap_svd(z, F::SVDFrame)
    γ = svdframe_gamma(z, F)
    return γ > 0 ? sqrt(γ) : -Inf
end
