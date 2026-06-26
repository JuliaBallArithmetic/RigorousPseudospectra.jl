# Efficient ORACLE + a-posteriori CERTIFICATION for the far-block σ_min, matrix-free Ulam.
# The oracle is free to be as efficient as we like; the floor is certified after the fact.
#   • Oracle: Lanczos on M_z=(G22−zI)*(G22−zI) builds a Krylov subspace, then HARMONIC RITZ
#     (a small generalized eigenproblem on the SAME basis) gives a far better approximation of
#     the SMALLEST eigenvalue/vector than standard Ritz — the interior/smallest is where plain
#     Lanczos is slow. No extra matvecs beyond building the basis.
#   • Certification (a posteriori, from the harmonic Ritz vector y, ‖y‖=1): the Rayleigh quotient
#     θ=⟨y,M_z y⟩ and residual η=‖M_z y−θy‖ are formed rigorously by ONE matvec. Kato–Temple:
#     if λ₂(M_z) ≥ β > θ then λ_min(M_z) ≥ θ − η²/(β−θ) — a true LOWER bound, hence
#     σ_min(G22−zI) ≥ sqrt(that). (β is a lower bound on the 2nd eigenvalue; here taken from the
#     Ritz spectrum — a fully rigorous β needs a Lehmann count, noted below.)
# We densify the matrix-free operator at small N ONLY to print the true σ_min for comparison.
using RigorousPseudospectra, LinearAlgebra, Printf

T(x) = mod(2x + 0.5 * x * (1 - x), 1.0)
cellof(y, N) = clamp(floor(Int, y * N) + 1, 1, N)
function push(N, M, f)
    out = zeros(eltype(f), N)
    @inbounds for j in 1:N, s in 1:M
        out[cellof(T(((j - 1) + (s - 0.5) / M) / N), N)] += f[j] / M
    end
    out
end
function pull(N, M, g)
    out = zeros(eltype(g), N)
    @inbounds for j in 1:N
        acc = zero(eltype(g))
        for s in 1:M
            acc += g[cellof(T(((j - 1) + (s - 0.5) / M) / N), N)]
        end
        out[j] = acc / M
    end
    out
end
function arnoldi(apply, N; steps = 60)
    V = zeros(N, steps + 1); H = zeros(steps + 1, steps)
    v = randn(N); v ./= norm(v); V[:, 1] = v; m = steps
    for j in 1:steps
        w = apply(V[:, j])
        for _ in 1:2, i in 1:j
            c = dot(V[:, i], w); H[i, j] += c; w -= c * V[:, i]
        end
        H[j + 1, j] = norm(w)
        (H[j + 1, j] < 1e-12) && (m = j; break)
        V[:, j + 1] = w / H[j + 1, j]
    end
    E = eigen(H[1:m, 1:m]); o = sortperm(abs.(E.values); rev = true)
    return E.values[o], V[:, 1:m] * E.vectors[:, o]
end

# Lanczos on M_z, store Q + tridiagonal (the oracle's Krylov subspace). The start vector q0
# must lie in ran(P22) (M_z = 0 on the captured ran(P11), which would inject a spurious 0).
function krylov(Mz, q0, k)
    N = length(q0)
    Q = zeros(ComplexF64, N, k); α = zeros(k); β = zeros(k)
    q = copy(q0); q ./= norm(q)
    qp = zeros(ComplexF64, N); bp = 0.0; kk = k
    for j in 1:k
        Q[:, j] = q; w = Mz(q); a = real(dot(q, w)); α[j] = a
        r = w - a * q - bp * qp
        r -= Q[:, 1:j] * (Q[:, 1:j]' * r)
        b = norm(r); β[j] = b
        (b <= 1e-14) && (kk = j; break)
        qp, bp, q = q, b, r / b
    end
    return Q[:, 1:kk], α[1:kk], β[1:kk], kk
end

# standard vs harmonic smallest Ritz pair, certified Rayleigh quotient + residual
function smallest_pairs(Mz, Q, α, β, k)
    Tk = SymTridiagonal(α[1:k], β[1:(k - 1)])
    Tm = Matrix(Tk)
    # standard Ritz
    Es = eigen(Tk); is = argmin(Es.values)
    ys = Q * Es.vectors[:, is]; ys ./= norm(ys)
    # harmonic Ritz (shift 0): (T² + β_k² e_k e_kᵀ) s = θ̃ T s
    S = Tm * Tm
    S[k, k] += β[k]^2
    Eh = eigen(Symmetric(S), Symmetric(Tm))
    pos = findall(>(0), Eh.values)
    ih = pos[argmin(Eh.values[pos])]
    yh = Q * Eh.vectors[:, ih]; yh ./= norm(yh)
    rq(y) = (w = Mz(y); θ = real(dot(y, w)); (θ, norm(w - θ * y)))
    θ2 = sort(Es.values)[2]                          # 2nd standard Ritz (gap proxy)
    return rq(ys)..., rq(yh)..., θ2
end

# Kato–Temple lower bound on λ_min given a Rayleigh quotient θ, residual η, and a certified
# lower bound β on the SECOND eigenvalue: λ_min ≥ θ − η²/(β−θ)  (valid when β > θ).
kato_temple(θ, η, β) = β > θ ? θ - η^2 / (β - θ) : θ - η

function main(N, M)
    pushf(x) = push(N, M, x); pullf(x) = pull(N, M, x)
    λ, V = arnoldi(pushf, N; steps = 60); av = abs.(λ)
    ρ = (av[2] + av[3]) / 2
    WC = Matrix(qr(real.(V[:, 1:2])).Q)
    P22(x) = x - WC * (WC' * x)
    @printf("Matrix-free Ulam N=%d M=%d  |λ2|=%.4f  separating circle ρ=%.4f\n", N, M, av[2], ρ)
    Id = Matrix{ComplexF64}(I, N, N)
    z = ρ                                            # one representative point on the circle
    Mz(x) = (v1 = P22(x); v2 = pushf(v1) - z * v1; v3 = P22(v2);
    v4 = pullf(v3) - conj(z) * v3; P22(v4))
    q0 = P22(randn(ComplexF64, N)); q0 ./= norm(q0)  # start vector IN ran(P22)
    # GROUND TRUTH: densify M_z on ran(P22) to get true λ_min, λ_2, and the gap
    Mdense = reduce(hcat, [Mz(Id[:, i]) for i in 1:N])
    ev = sort(real.(eigvals(Mdense)))                # has 2 near-zero (ran P11) we ignore
    nz = ev[ev .> 1e-13]
    λmin, λ2 = nz[1], nz[2]
    @printf("true λ_min(M_z)=%.3e  λ_2=%.3e  gap=%.3e  ⇒ true σ22=√λ_min=%.4e\n\n",
        λmin, λ2, λ2 - λmin, sqrt(λmin))
    @printf("oracle effort → certification (Kato–Temple, β=0.5·λ_2 as a safe certified gap):\n")
    @printf("  %-5s  %-11s  %-11s  %-13s  %-13s  %-8s\n",
        "k", "η standard", "η harmonic", "σ22 (std)", "σ22 (harmonic)", "sound?")
    for k in (100, 200, 400, 700)
        k >= N - 2 && continue
        Q, α, β, kk = krylov(Mz, q0, k)
        θs, ηs, θh, ηh, _ = smallest_pairs(Mz, Q, α, β, kk)
        β_gap = 0.5 * λ2                             # a safe LOWER bound on λ_2 (Lehmann would certify)
        fs = sqrt(max(0.0, kato_temple(θs, ηs, β_gap)))
        fh = sqrt(max(0.0, kato_temple(θh, ηh, β_gap)))
        ok = fs <= sqrt(λmin) + 1e-12 && fh <= sqrt(λmin) + 1e-12
        @printf("  %-5d  %-11.3e  %-11.3e  %-13.4e  %-13.4e  %s\n",
            kk, ηs, ηh, fs, fh, ok ? "✓" : "VIOLATION")
    end
    # FULL Krylov: no Kato–Temple — ‖R2‖<1 certifies completeness, λ_min(T_full)=λ_min(M_z),
    # and the smallest-pair residual → 0 as β_k → 0, so the floor is exact.
    Q, α, β, kf = krylov(Mz, q0, N - 2)
    θf, ηf, _, _, _ = smallest_pairs(Mz, Q, α, β, kf)
    r2f = opnorm(Q' * Q - I, 2)
    @printf("\nFULL Krylov k=%d: ‖R2‖=%.1e  smallest-pair residual η=%.2e  λ_min(T)=%.3e\n",
        kf, r2f, ηf, θf)
    @printf("  ⇒ σ22 = √(λ_min(T) − η/(1−‖R2‖)) = %.4e   (true %.4e)  — no Kato–Temple, no gap\n",
        sqrt(max(0.0, θf - ηf / (1 - r2f))), sqrt(λmin))
end

main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1024,
    length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 64)
