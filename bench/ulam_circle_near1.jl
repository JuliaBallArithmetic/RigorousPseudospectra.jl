# Can we bound the resolvent on a circle near 1 that encloses ONLY λ₁ = 1?
#
# This is the spectral-gap-from-1 certificate. ‖(P−zI)⁻¹‖₂ = 1/σ_min(P−zI), so a
# resolvent bound on a curve = a POSITIVE lower bound on σ_min over that curve.
#
# Circle Γ = { z = 1 + ρ·e^{iθ} } encloses only λ₁ = 1 iff  0 < ρ < |1 − λ₂|
# (ρ > 0 keeps Γ off the pole at 1; ρ < |1−λ₂| keeps the rest of σ(P) outside).
#
# Matrix-free certificate (adapts bench/ulam_matfree.jl), block-triangular:
#   • capture λ₁ ONLY  →  rank-1 deflation P22 = I − w₁w₁*   (λ₂… stay in the far block)
#   • captured 1×1 block:  σ₁₁ = |G11 − z|,  G11 = w₁*Pw₁ ≈ 1   ⇒  σ₁₁ ≈ ρ
#   • far floor σ₂₂ = σ_min(P22(P−zI)·|_ran P22) via matvec-Lanczos on M_z
#                                                  ⇒  σ₂₂ ≈ dist(z, σ(P)∖{1}) ≳ |1−λ₂|−ρ
#   • assemble  σ_min(P−zI) ≥ σ₁₁σ₂₂/(σ₁₁+σ₂₂+γ) > 0,  γ = coupling of the capture
#   • WHOLE circle by Lipschitz arc-covering: σ_min is 1-Lipschitz in z, so
#         min_i σ_min(zᵢ) − 2ρ·sin(π/2n) > 0   certifies all of Γ.
# A positive bound ⇒ exactly one eigenvalue (λ₁=1) inside Γ ⇒ spectral gap ≥ ρ.
#
# Run with the RIM project (sparse 1-D Ulam + Arpack):
#   julia --project=$HOME/Code/RigorousInvariantMeasures.jl bench/ulam_circle_near1.jl [N] [frac]
using RigorousInvariantMeasures, Arpack, LinearAlgebra, SparseArrays, Printf
import IntervalArithmetic

build_ulam(N) = sparse(map(IntervalArithmetic.mid,
    DiscretizedOperator(Ulam(N), mod1_dynamic(x -> 2 * x + 0.5 * x * (1 - x))).L))

# Lanczos on the Hermitian M_z (matvec-only); full reorth (window=kmax) stands in for the
# orthogonality-retained regime (storage-free Float64 loses it before a full basis).
function lanczos(Mz, N, kmax; window::Int = 0)
    Q = zeros(ComplexF64, N, kmax)
    α = zeros(Float64, kmax); β = zeros(Float64, kmax)
    q = randn(ComplexF64, N); q ./= norm(q)
    qprev = zeros(ComplexF64, N); βprev = 0.0
    kact = kmax
    for k in 1:kmax
        Q[:, k] = q
        w = Mz(q)
        a = real(dot(q, w)); α[k] = a
        r = w - a * q - βprev * qprev
        if window > 0
            lo = max(1, k - window + 1)
            r -= Q[:, lo:k] * (Q[:, lo:k]' * r)
        end
        b = norm(r)
        (!isfinite(b) || b <= 1e-13 * abs(a)) && (kact = k; break)
        qprev, βprev, q = q, b, r / b
        β[k] = b
    end
    return Q[:, 1:kact], α[1:kact], β[1:kact], kact
end

R2norm(Q, k) = opnorm(Q[:, 1:k]' * Q[:, 1:k] - I, 2)

# far floor σ_min(G22−zI) at step k: smallest Ritz value θ_min of T_k, certified down by the
# Ritz residual, inflated by 1/(1−‖R2‖).
function far_floor(Mz, Q, α, β, k)
    Tk = SymTridiagonal(α[1:k], β[1:(k - 1)])
    E = eigen(Tk)
    i = argmin(E.values); θ = E.values[i]
    y = Q[:, 1:k] * E.vectors[:, i]
    ρ1 = norm(Mz(y) - θ * y)
    r2 = R2norm(Q, k)
    β22 = r2 < 1 ? ρ1 / (1 - r2) : Inf
    return sqrt(max(0.0, θ - β22)), θ, ρ1, r2
end

function main(N, frac)
    P = build_ulam(N)
    @printf("Ulam (Lanford map) N=%d  nnz=%d (%.1f/row)\n", N, nnz(P), nnz(P) / N)
    vals, V = eigs(P; nev = 6, which = :LM)
    o = sortperm(abs.(vals); rev = true)
    λ = vals[o]; V = V[:, o]
    @printf("leading λ: %s\n", join(round.(λ[1:5]; digits = 5), ", "))

    # distance from z=1 to the rest of the spectrum sets the admissible radius
    d2 = minimum(abs.(1 .- λ[2:end]))
    ρ = frac * d2
    @printf("dist(1, σ∖{1}) = %.5f   ⇒  radius ρ = %.2f·dist = %.5f  (encloses only λ₁=1)\n\n",
        d2, frac, ρ)

    w1 = V[:, 1]; w1 ./= norm(w1)        # rank-1 capture of λ₁
    WC = reshape(w1, N, 1)
    P22(x) = x - WC * (WC' * x)
    G11 = (WC' * (P * WC))[1, 1]         # scalar Rayleigh quotient ≈ 1
    γ = max(opnorm(P22(P * WC), 2), opnorm(P22(P' * WC), 2))
    @printf("G11 = w₁*Pw₁ = %.6f   coupling γ = %.3e\n\n", real(G11), γ)

    kmax = min(N - 2, 400)
    nθ = 12
    floors = Float64[]
    for t in 0:(nθ - 1)
        θ = 2π * t / nθ
        z = 1 + ρ * cis(θ)
        Mz(x) = (v1 = P22(x); v2 = P * v1 - z * v1; v3 = P22(v2);
        v4 = P' * v3 - conj(z) * v3; P22(v4))
        Q, α, β, k = lanczos(Mz, N, kmax; window = kmax)
        fl, _, _, r2 = far_floor(Mz, Q, α, β, k)
        σ11 = abs(G11 - z)
        σA = (σ11 <= 0 || fl <= 0) ? 0.0 : σ11 * fl / (σ11 + fl + γ)
        push!(floors, σA)
        tru = N <= 4096 ? minimum(svdvals(Matrix(P) - z * I)) : NaN
        ok = isnan(tru) ? "  " : (σA <= tru + 1e-9 ? "sound" : "VIOLATION")
        @printf("θ=%4.2fπ  σ11=%.4e far σ22=%.4e (k=%d,‖R2‖=%.1e) → σ_min≥%.4e  true=%.4e %s\n",
            θ / π, σ11, fl, k, r2, σA, tru, ok)
    end

    # whole-circle floor via Lipschitz arc-covering
    cover = 2ρ * sin(π / (2nθ))
    circ = minimum(floors) - cover
    println()
    @printf("min over %d samples = %.4e ;  Lipschitz cover 2ρsin(π/2n) = %.4e\n",
        nθ, minimum(floors), cover)
    if circ > 0
        @printf("✓ σ_min(P−zI) ≥ %.4e on ALL of Γ  ⇒  resolvent ≤ %.3e there\n", circ, 1 / circ)
        @printf("  ⇒ λ₁=1 isolated, spectral gap ≥ ρ = %.5f (exactly one eigenvalue inside Γ)\n", ρ)
    else
        @printf("✗ cover exceeds the sampled floor (need more samples nθ, or smaller ρ)\n")
    end
end

main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1024,
    length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.5)
