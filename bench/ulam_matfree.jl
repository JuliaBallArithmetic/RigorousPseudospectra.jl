# Matrix-free spectral-gap / mixing-rate certificate on a big SPARSE Ulam matrix
# (RigorousInvariantMeasures.jl). Run with the RIM project (has Arpack + the Ulam assembler):
#   julia --project=$HOME/Code/RigorousInvariantMeasures.jl bench/ulam_matfree.jl [N]
#
# Capture the 2 leading eigenpairs (λ1=1, λ2) of the sparse transfer matrix P via Arpack,
# form the far block G22 = P22 P P22 (P22 = I − WC WC*) applied MATVEC-ONLY (sparse mat-vec
# + rank-2 projection; G22 never formed), and certify σ_min(P − zI) on the separating circle
# |z|=ρ=(|λ2|+|λ3|)/2 by Lanczos on M_z=(G22−zI)*(G22−zI). We measure the real questions for
# big Ulam: (a) steps for the far floor to stabilize, (b) where storage-free ‖R2‖ crosses 1
# vs the orthogonality-retained floor, (c) the certified gap vs the true σ_min.
using RigorousInvariantMeasures, Arpack, LinearAlgebra, SparseArrays, Printf
import IntervalArithmetic

build_ulam(N) = sparse(map(IntervalArithmetic.mid,
    DiscretizedOperator(Ulam(N), mod1_dynamic(x -> 2 * x + 0.5 * x * (1 - x))).L))

# Lanczos on the Hermitian M_z (matvec-only), storing Q to MEASURE ‖R2‖ and the Ritz residual.
# window = trailing reorth width (0 = storage-free, ≥kmax = full).
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
# Ritz residual ‖r_min‖ = ‖M_z y − θ_min y‖ (y = Q s_min), inflated by 1/(1−‖R2‖).
function far_floor(Mz, Q, α, β, k)
    Tk = SymTridiagonal(α[1:k], β[1:(k - 1)])
    E = eigen(Tk)
    i = argmin(E.values); θ = E.values[i]
    y = Q[:, 1:k] * E.vectors[:, i]
    ρ1 = norm(Mz(y) - θ * y)                 # true Ritz residual (matvec)
    r2 = R2norm(Q, k)
    β22 = r2 < 1 ? ρ1 / (1 - r2) : Inf
    return sqrt(max(0.0, θ - β22)), θ, ρ1, r2
end

function main(N)
    P = build_ulam(N)
    @printf("Ulam (Lanford map) N=%d  nnz=%d (%.1f/row)\n", N, nnz(P), nnz(P) / N)
    vals, V = eigs(P; nev = 6, which = :LM)
    o = sortperm(abs.(vals); rev = true)
    λ = vals[o]; V = V[:, o]
    av = abs.(λ)
    ρ = (av[2] + av[3]) / 2
    @printf("leading |λ|: %s   mixing rate |λ2|=%.5f   gap circle ρ=%.5f\n",
        join(round.(av[1:5]; digits = 5), ", "), av[2], ρ)

    WC = Matrix(qr(V[:, 1:2]).Q)             # orthonormal capture of λ1, λ2
    P22(x) = x - WC * (WC' * x)
    G11 = WC' * (P * WC)                     # 2×2 captured block
    γ = max(opnorm(P22(P * WC), 2), opnorm(P22(P' * WC), 2))   # coupling (Rayleigh–Ritz resid.)
    @printf("coupling γ=max(‖G21‖,‖G12‖)=%.3e   (capture is approximate ⇒ G21≠0)\n\n", γ)

    kmax = min(N - 2, 400)
    for θ in (0.0, π / 2, π)                  # a few points on the circle |z|=ρ
        z = ρ * cis(θ)
        Mz(x) = (v1 = P22(x); v2 = P * v1 - z * v1; v3 = P22(v2);
        v4 = P' * v3 - conj(z) * v3; P22(v4))
        # orthogonality-retained (full reorth): the usable floor + steps to stabilize
        Q, α, β, k = lanczos(Mz, N, kmax; window = kmax)
        fl, _, _, r2 = far_floor(Mz, Q, α, β, k)
        # steps to stabilize: first k0 where the floor is within 1% of the final
        k0 = k
        for kk in 5:k
            f, = far_floor(Mz, Q, α, β, kk)
            if abs(f - fl) <= 0.01 * max(fl, 1e-12)
                k0 = kk; break
            end
        end
        # storage-free: where does ‖R2‖ cross 1?
        Qs, _, _, ks = lanczos(Mz, N, kmax; window = 0)
        kc = findfirst(j -> R2norm(Qs, j) >= 1, 1:ks)
        σ11 = minimum(svdvals(G11 - z * I))
        # block-triangular assembly (G21≈0 for a well-converged capture): POSITIVE for any γ,
        # unlike the crude Weyl min(σ11,σ22)−γ which dies on the large Ulam coupling. NB the far
        # floor fl must itself be > 0, which needs the SMALLEST eigenvalue of M_z converged in the
        # matvec-Lanczos — the slow bottom-of-spectrum convergence is the real matvec bottleneck.
        σA = (σ11 <= 0 || fl <= 0) ? 0.0 : σ11 * fl / (σ11 + fl + γ)
        tru = minimum(svdvals(Matrix(P) - z * I))     # dense reference (N≤~4k)
        @printf("z=ρ·e^{i%.2fπ}: far σ22=%.4e (k=%d, stabilizes@k=%d, ‖R2‖=%.1e) | σ11=%.4e\n",
            θ / π, fl, k, k0, r2, σ11)
        @printf("   assembled σ_min(P−zI)≥%.4e   true=%.4e   %s   storage-free ‖R2‖>1 @k=%s\n",
            σA, tru, σA <= tru + 1e-9 ? "sound" : "VIOLATION", string(kc))
    end
end

main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1024)
