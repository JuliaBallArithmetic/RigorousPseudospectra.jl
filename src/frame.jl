# §5.1 (sec:frame) and §5.3 (sec:depth): building the frame, in floating point. Nothing in this file
# is certified, and nothing needs to be (sec:sound): the frame enters the certificate only through
# P⊥, α, the compression and the couplings, all of which are certified for whatever frame is given;
# a poor frame makes the method decline, not assert.

# Matrix-free application of the operator to one vector, promoted to the working type.
_mv(f, x, T) = Vector{T}(f(Vector{T}(x)))

"""
    arnoldi(Amul, n, L; v0, T) -> (V, H)

`L` steps of Arnoldi with full reorthogonalisation (twice), `A V[:, 1:L] ≈ V H`, `V` of size
n×(L+1) and `H` (L+1)×L; stops early at a breakdown.
"""
function arnoldi(Amul, n::Integer, L::Integer; T = ComplexF64,
        v0 = nothing, rng = Random.MersenneTwister(1))
    L = min(L, n)
    V = zeros(T, n, L + 1)
    H = zeros(T, L + 1, L)
    v = v0 === nothing ? (T <: Complex ? randn(rng, ComplexF64, n) : randn(rng, n)) : Vector{T}(v0)
    V[:, 1] = v / norm(v)
    Lact = L
    for j in 1:L
        w = _mv(Amul, V[:, j], T)
        for _ in 1:2
            h = V[:, 1:j]' * w
            w .-= V[:, 1:j] * h
            H[1:j, j] .+= h
        end
        H[j + 1, j] = norm(w)
        if abs(H[j + 1, j]) <= 1e-14 * max(norm(H[1:(j + 1), 1:j]), eps())
            Lact = j
            break
        end
        V[:, j + 1] = w / H[j + 1, j]
    end
    return V[:, 1:(Lact + 1)], H[1:(Lact + 1), 1:Lact]
end

"""
    eigendirections(Amul, n, R; L = 60) -> (W, θ)

Ritz vectors of `L` Arnoldi steps for the Ritz values of modulus `> R`: the "Arnoldi
eigendirections with |λ| > R" of Definition def:Q.
"""
function eigendirections(Amul, n::Integer, R::Real; L::Integer = 60, T = ComplexF64,
        rng = Random.MersenneTwister(2))
    V, H = arnoldi(Amul, n, L; T, rng)
    Lh = size(H, 2)
    Lh == 0 && return zeros(T, n, 0), ComplexF64[]
    F = eigen(H[1:Lh, 1:Lh])
    sel = findall(θ -> abs(θ) > R, F.values)
    W = V[:, 1:Lh] * F.vectors[:, sel]
    # for a real operator, span{v, v̄} = span{Re v, Im v}: keep the frame real
    T <: Real && (W = hcat(real.(W), imag.(W)))
    return W, F.values[sel]
end

"""
    gkl(Amul, Aadj, n, L; T) -> (Vr, Ul, σ)

`L` steps of Golub–Kahan–Lanczos bidiagonalisation with full reorthogonalisation (twice), and the
Ritz singular triplets of the bidiagonal: right vectors `Vr`, left vectors `Ul` (columns ordered
by decreasing Ritz singular value `σ`). "A Lanczos on A*A and on AA* at two matvecs per step"
(sec:frame).
"""
function gkl(Amul, Aadj, n::Integer, L::Integer; T = ComplexF64,
        rng = Random.MersenneTwister(3))
    L = min(L, n)
    V = zeros(T, n, L + 1)
    U = zeros(T, n, L)
    α = zeros(L)
    β = zeros(L)
    v = T <: Complex ? randn(rng, ComplexF64, n) : randn(rng, n)
    V[:, 1] = v / norm(v)
    big = 0.0
    Lact = L
    for j in 1:L
        u = _mv(Amul, V[:, j], T)
        j > 1 && (u .-= β[j - 1] .* U[:, j - 1])
        for _ in 1:2
            j > 1 && (Uj = view(U, :, 1:(j - 1)); u .-= Uj * (Uj' * u))
        end
        α[j] = norm(u)
        big = max(big, α[j])
        if α[j] <= 1e-13 * big
            Lact = j - 1
            break
        end
        U[:, j] = u / α[j]
        w = _mv(Aadj, U[:, j], T)
        w .-= α[j] .* V[:, j]
        for _ in 1:2
            Vj = view(V, :, 1:j)
            w .-= Vj * (Vj' * w)
        end
        β[j] = norm(w)
        if β[j] <= 1e-13 * big || j == L
            Lact = j
            break
        end
        V[:, j + 1] = w / β[j]
    end
    Lact == 0 && return zeros(T, n, 0), zeros(T, n, 0), Float64[]
    Bd = Bidiagonal(α[1:Lact], β[1:(Lact - 1)], :U)
    F = svd(Matrix(Bd))
    return V[:, 1:Lact] * F.V, U[:, 1:Lact] * F.U, F.S
end

"""
    rrqr_frame(M; droptol = 1e-8, method = :qr) -> Q

The orthonormal factor of a rank-revealing factorisation of `M` (Definition def:Q), the directions
whose diagonal falls below `droptol` times the largest being dropped. `method = :qr` is
column-pivoted QR, `|R_ii| < droptol |R_11|`; `:svd` keeps the left singular vectors with
`s_i >= droptol s_1`; `:qlp` is Stewart's QLP, a second column-pivoted QR applied to the transpose of
`R`, whose diagonal tracks the singular values more closely than that of `R`: the rank is read off
it, and the leading columns of the first orthonormal factor are kept.
"""
function rrqr_frame(M::AbstractMatrix; droptol::Real = 1e-8, method::Symbol = :qr)
    size(M, 2) == 0 && return zeros(eltype(M), size(M, 1), 0)
    if method === :svd
        F = svd(M)
        r = count(>=(droptol * F.S[1]), F.S)
        Q = F.U[:, 1:r]
    elseif method === :qr || method === :qlp
        F = qr(M, ColumnNorm())
        d = abs.(diag(F.R))
        if method === :qlp
            G = qr(Matrix(F.R)', ColumnNorm())
            d = abs.(diag(G.R))
        end
        r = count(>=(droptol * d[1]), d)
        Q = Matrix(F.Q)[:, 1:r]
    else
        error("method must be :qr, :svd or :qlp")
    end
    return Matrix(qr(Q).Q)[:, 1:r]           # one more Householder pass
end

"""
    two_sided_frame(Amul, Aadj, n, R; ks, keig = true, Leig = 60, Lsv = 2ks + 20, droptol,
                    sides = :both) -> (Q, info)

Definition def:Q: `Q` is the orthonormal factor of a rank-revealing factorisation of
`[W_λ | V_σ | U_σ]`, with `W_λ` the Arnoldi eigendirections of modulus `> R` and `V_σ`, `U_σ` the
leading `ks` right and left Ritz singular vectors. `sides = :right` or `:left` keeps one side only
(the "eigen + right", "eigen + left" rows of the table of sec:frame).
"""
function two_sided_frame(Amul, Aadj, n::Integer, R::Real; ks::Integer,
        keig::Bool = true, Leig::Integer = 60, Lsv::Integer = 2ks + 20, droptol::Real = 1e-8,
        sides::Symbol = :both, T = Float64, extra = nothing, method::Symbol = :qr)
    Wλ = keig ? first(eigendirections(Amul, n, R; L = Leig, T)) : zeros(T, n, 0)
    Vr, Ul, σ = gkl(Amul, Aadj, n, Lsv; T)
    ks = min(ks, size(Vr, 2))
    cols = Any[Wλ]
    sides in (:both, :right) && push!(cols, Vr[:, 1:ks])
    sides in (:both, :left) && push!(cols, Ul[:, 1:ks])
    extra === nothing || push!(cols, extra)
    M = hcat(cols...)
    Q = rrqr_frame(M; droptol, method)
    return Q, (; neig = size(Wλ, 2), ks, dimQ = size(Q, 2), ritz_sv = σ)
end

"""
    lanczos_frame(Amul, n, L; tol = 1e-10, kmax = L) -> (Q, info)

Remark rem:lanczos, for Hermitian A: `L` Lanczos steps (full reorthogonalisation, twice), Ritz
pairs `(θ_i, s_i)` of `T_L`, residuals `r_i = β_L |(s_i)_L|`, and `Q = Q_L S_C` for the Ritz
vectors with `r_i ≤ tol · max|θ|`, largest `|θ|` first, at most `kmax`. `Q` is not
re-orthonormalised: its defect α is certified downstream, and γ̄ is taken from the Gram bound on
the frame actually computed, not from the `r_i`.
"""
function lanczos_frame(Amul, n::Integer, L::Integer; tol::Real = 1e-10, kmax::Integer = L,
        T = Float64, rng = Random.MersenneTwister(4))
    L = min(L, n - 1)
    Qk = zeros(T, n, L + 1)
    a = zeros(L)
    bb = zeros(L)
    q = T <: Complex ? randn(rng, ComplexF64, n) : randn(rng, n)
    Qk[:, 1] = q / norm(q)
    Lact = L
    for j in 1:L
        w = _mv(Amul, Qk[:, j], T)
        a[j] = real(dot(Qk[:, j], w))
        w .-= a[j] .* Qk[:, j]
        j > 1 && (w .-= bb[j - 1] .* Qk[:, j - 1])
        for _ in 1:2
            Qj = view(Qk, :, 1:j)
            w .-= Qj * (Qj' * w)
        end
        bb[j] = norm(w)
        if bb[j] <= 1e-14 * maximum(abs, a[1:j])
            Lact = j
            break
        end
        Qk[:, j + 1] = w / bb[j]
    end
    E = eigen(SymTridiagonal(a[1:Lact], bb[1:(Lact - 1)]))
    θ = E.values
    r = bb[Lact] .* abs.(E.vectors[Lact, :])
    scale = maximum(abs, θ)
    conv = findall(i -> r[i] <= tol * scale, eachindex(θ))
    conv = conv[sortperm(abs.(θ[conv]); rev = true)]
    C = conv[1:min(kmax, length(conv))]
    Q = Qk[:, 1:Lact] * E.vectors[:, C]
    return Q, (; L = Lact, k = length(C), theta = θ[C], resid = r[C], betaL = bb[Lact])
end
