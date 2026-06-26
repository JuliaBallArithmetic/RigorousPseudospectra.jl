# MATRIX-FREE Ulam: the transfer operator is applied on the fly from the map T (mass is
# pushed through T by subsampling each cell) — the N×N matrix is NEVER formed, O(N) memory.
# Non-rigorous (M subsamples per cell ⇒ quadrature error), but lets us push N large and test
# the spectral-gap / mixing-rate certificate matrix-free.
#   julia --project=bench bench/ulam_free.jl [N] [M]
using RigorousPseudospectra, LinearAlgebra, Printf

# Lanford map T(x) = (2x + 0.5 x(1−x)) mod 1
T(x) = mod(2x + 0.5 * x * (1 - x), 1.0)

# matrix-free transfer operator P (pushforward, column-stochastic) and its adjoint P', via M
# uniform subsamples per cell. cell(y)=⌈yN⌉. No storage: T is re-evaluated each matvec.
struct UlamFree
    N::Int
    M::Int
end
cellof(y, N) = clamp(floor(Int, y * N) + 1, 1, N)
function push(U::UlamFree, f)                       # (Pf)_i = Σ_j frac(j→i) f_j
    N, M = U.N, U.M
    out = zeros(eltype(f), N)
    @inbounds for j in 1:N
        fj = f[j] / M
        for s in 1:M
            x = ((j - 1) + (s - 0.5) / M) / N
            out[cellof(T(x), N)] += fj
        end
    end
    out
end
function pull(U::UlamFree, g)                        # (P'g)_j = (1/M) Σ_s g_{cell(T(x_s))}
    N, M = U.N, U.M
    out = zeros(eltype(g), N)
    @inbounds for j in 1:N
        acc = zero(eltype(g))
        for s in 1:M
            x = ((j - 1) + (s - 0.5) / M) / N
            acc += g[cellof(T(x), N)]
        end
        out[j] = acc / M
    end
    out
end

# matrix-free Arnoldi → top Ritz pairs (real operator; leading eigenvalues are real)
function arnoldi(apply, N; steps = 60)
    V = zeros(Float64, N, steps + 1); H = zeros(Float64, steps + 1, steps)
    v = randn(N); v ./= norm(v); V[:, 1] = v
    local m = steps
    for j in 1:steps
        w = apply(V[:, j])
        for _ in 1:2, i in 1:j                       # modified GS, reorth twice
            c = dot(V[:, i], w); H[i, j] += c; w -= c * V[:, i]
        end
        H[j + 1, j] = norm(w)
        (H[j + 1, j] < 1e-12) && (m = j; break)
        V[:, j + 1] = w / H[j + 1, j]
    end
    E = eigen(H[1:m, 1:m])
    o = sortperm(abs.(E.values); rev = true)
    return E.values[o], V[:, 1:m] * E.vectors[:, o]
end

# Hermitian Lanczos on M_z (full reorth), storing Q; returns the far floor data
function far_floor(Mz, N, kmax)
    Q = zeros(ComplexF64, N, kmax); α = zeros(kmax); β = zeros(kmax)
    q = randn(ComplexF64, N); q ./= norm(q)
    qprev = zeros(ComplexF64, N); βprev = 0.0; k = kmax
    for j in 1:kmax
        Q[:, j] = q
        w = Mz(q); a = real(dot(q, w)); α[j] = a
        r = w - a * q - βprev * qprev
        r -= Q[:, 1:j] * (Q[:, 1:j]' * r)            # full reorth
        b = norm(r)
        (!isfinite(b) || b <= 1e-13 * abs(a)) && (k = j; break)
        β[j] = b; qprev, βprev, q = q, b, r / b
    end
    Tk = SymTridiagonal(α[1:k], β[1:(k - 1)])
    Eg = eigen(Tk); i = argmin(Eg.values); θ = Eg.values[i]
    y = Q[:, 1:k] * Eg.vectors[:, i]
    ρ1 = norm(Mz(y) - θ * y)
    r2 = opnorm(Q[:, 1:k]' * Q[:, 1:k] - I, 2)
    return sqrt(max(0.0, θ - ρ1 / (1 - min(r2, 0.999)))), θ, ρ1, k, r2
end

function main(N, M)
    U = UlamFree(N, M)
    @printf("MATRIX-FREE Ulam (Lanford map)  N=%d  M=%d subsamples/cell  (O(N) memory)\n", N, M)
    λ, V = arnoldi(x -> push(U, x), N; steps = 60)
    av = abs.(λ)
    @printf("leading |λ| (matrix-free Arnoldi): %s\n", join(round.(av[1:5]; digits = 5), ", "))
    @printf("mixing rate |λ2|=%.5f  (RIM dense ref at N=1024: 0.57712)\n", av[2])
    ρ = (av[2] + av[3]) / 2
    WC = Matrix(qr(real.(V[:, 1:2])).Q)              # capture λ1, λ2
    P22(x) = x - WC * (WC' * x)
    G11 = WC' * hcat(push(U, WC[:, 1]), push(U, WC[:, 2]))
    γ = max(norm(P22(push(U, WC[:, 1]))), norm(P22(push(U, WC[:, 2]))),
        norm(P22(pull(U, WC[:, 1]))), norm(P22(pull(U, WC[:, 2]))))
    @printf("capture m=2  coupling γ=%.3e  gap circle ρ=%.5f\n\n", γ, ρ)

    kmax = min(N - 2, 300)
    for θ in (0.0, π / 2, π)
        z = ρ * cis(θ)
        Mz(x) = (v1 = P22(x); v2 = push(U, v1) - z * v1; v3 = P22(v2);
        v4 = pull(U, v3) - conj(z) * v3; P22(v4))
        fl, θm, ρ1, k, r2 = far_floor(Mz, N, kmax)
        σ11 = minimum(svdvals(G11 - z * I))
        σ22est = sqrt(max(0.0, θm))                  # NON-rigorous estimate sqrt(λ_min Ritz)
        σAest = σ11 * σ22est / (σ11 + σ22est + γ)     # triangular assembly on the estimate
        @printf("z=ρ·e^{i%.2fπ}: est σ22≈%.3e (cert floor=%.1e; ‖rmin‖=%.1e@k=%d) σ11=%.3e → est σ_min(P−zI)≈%.3e\n",
            θ / π, σ22est, fl, ρ1, k, σ11, σAest)
    end
end

main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4096,
    length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16)
