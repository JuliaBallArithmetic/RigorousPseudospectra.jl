# Test-matrix gallery (implementation.md §8). Each entry returns a dense matrix
# plus metadata; `gallery()` lists them for the benchmark driver. Kept dependency-
# free (plain Base/LinearAlgebra) so it can be `include`d from bench scripts or
# tests. Expand per §8 (Demmel, Jordan/coalescing, Kahan/Frank, companion,
# convection–diffusion, the Gauss witness) as the engine matures.

using LinearAlgebra
using Random

"""
    grcar(n, k = 3) -> Matrix{Float64}

Grcar matrix: 1 on the subdiagonal, 1 on the diagonal and the first `k`
superdiagonals. The canonical non-normal pseudospectrum benchmark.
"""
function grcar(n::Integer, k::Integer = 3)
    A = zeros(Float64, n, n)
    for i in 1:n
        if i + 1 <= n
            A[i + 1, i] = -1.0
        end
        for d in 0:k
            (i + d <= n) && (A[i, i + d] = 1.0)
        end
    end
    return A
end

"""
    normal_random(n; seed) -> Matrix{ComplexF64}

Q·diag(λ)·Qᴴ with Q a random unitary — a normal matrix whose pseudospectrum is an
ε-neighbourhood of its spectrum. Sanity/tightness floor (free gap is ~exact).
"""
function normal_random(n::Integer; seed::Integer = 1)
    rng = MersenneTwister(seed)
    Q, _ = qr(randn(rng, ComplexF64, n, n))
    λ = randn(rng, ComplexF64, n)
    return Matrix(Q) * Diagonal(λ) * Matrix(Q)'
end

"""
    jordan_block(m, λ) -> Matrix{ComplexF64}

Single Jordan block: σ_min(J − zI) ∼ |z−λ|^m, so the certified bound must track
the tiny σ_min where eigenvalue distance would report |z−λ| (paper Remark 2.3).
"""
function jordan_block(m::Integer, λ = 0.0 + 0.0im)
    J = zeros(ComplexF64, m, m)
    for i in 1:m
        J[i, i] = λ
        (i + 1 <= m) && (J[i, i + 1] = 1.0)
    end
    return J
end

"""
    two_clusters(; seed = 1) -> Matrix{ComplexF64}

Two tight, non-normal eigenvalue clusters (near 0 and near 2.5) in a random
unitary frame — exercises the cluster region and the encircling-curve certificate.
"""
function two_clusters(; seed::Integer = 1)
    rng = MersenneTwister(seed)
    λ = ComplexF64[0, 0.12, -0.1im, 2.5, 2.5 + 0.13im, 2.6 - 0.05im]
    n = length(λ)
    T = Matrix(Diagonal(λ))
    for j in 1:n, i in 1:(j - 1)
        T[i, j] = 0.25 * (randn(rng) + im * randn(rng))
    end
    Q, _ = qr(randn(rng, ComplexF64, n, n))
    return Matrix(Q) * T * Matrix(Q)'
end

"""
    graded_chain(n; s = 0.7) -> Matrix{ComplexF64}

Weakly-coupled chain: distinct eigenvalues `1..n` with a small superdiagonal `s`.
The coupling graph makes it ONE global cluster (consecutive eigenvalues linked
transitively, m = n), but at each z only a local window of eigenvalues is active —
the case where the z-adaptive active-set escalation beats the global cluster.
"""
function graded_chain(n::Integer; s::Real = 0.7)
    A = zeros(ComplexF64, n, n)
    for i in 1:n
        A[i, i] = i
        (i + 1 <= n) && (A[i, i + 1] = s)
    end
    return A
end

"""
    gallery() -> Vector{NamedTuple}

Benchmark matrices with metadata: `name`, `A`, `box` = (lo, hi) corners,
`ε` level, and `h` target half-diagonal for the sweep. Each is chosen to show a
distinct regime (normal / non-normal / defective / clustered).
"""
function gallery()
    return [
        (name = "normal16", A = normal_random(16),
            box = (-3.0 - 3.0im, 3.0 + 3.0im), ε = 0.4, h = 0.04),
        (name = "grcar32", A = grcar(32),
            box = (-2.5 - 3.5im, 3.5 + 3.5im), ε = 0.5, h = 0.07),
        (name = "jordan8", A = jordan_block(8),
            box = (-1.1 - 1.1im, 1.1 + 1.1im), ε = 0.2, h = 0.02),
        (name = "two_clusters", A = two_clusters(),
            box = (-1.3 - 1.3im, 3.8 + 1.3im), ε = 0.3, h = 0.03),
        (name = "chain", A = graded_chain(12; s = 0.7),
            box = (0.0 - 1.5im, 13.0 + 1.5im), ε = 0.3, h = 0.12),
    ]
end
