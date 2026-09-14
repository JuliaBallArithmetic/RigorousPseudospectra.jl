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
    rrqr_frame(M; droptol = 1e-8) -> Q

The orthonormal factor of a rank-revealing (column-pivoted) QR of `M`, the columns with
`|R_ii| < droptol |R_11|` dropped (Definition def:Q).
"""
function rrqr_frame(M::AbstractMatrix; droptol::Real = 1e-8)
    size(M, 2) == 0 && return zeros(eltype(M), size(M, 1), 0)
    F = qr(M, ColumnNorm())
    d = abs.(diag(F.R))
    r = count(>=(droptol * d[1]), d)
    Q = Matrix(F.Q)[:, 1:r]
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
        sides::Symbol = :both, T = Float64, extra = nothing)
    Wλ = keig ? first(eigendirections(Amul, n, R; L = Leig, T)) : zeros(T, n, 0)
    Vr, Ul, σ = gkl(Amul, Aadj, n, Lsv; T)
    ks = min(ks, size(Vr, 2))
    cols = Any[Wλ]
    sides in (:both, :right) && push!(cols, Vr[:, 1:ks])
    sides in (:both, :left) && push!(cols, Ul[:, 1:ks])
    extra === nothing || push!(cols, extra)
    M = hcat(cols...)
    Q = rrqr_frame(M; droptol)
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

# ------------------------------------------------------------------------------------------------
# The depth (sec:depth), in floating point
# ------------------------------------------------------------------------------------------------

"""
    float_split_estimates(op, Q, R; nθ = 32, mmax = 8) -> NamedTuple

Floating-point estimates for the depth rule eq:depthrule on the circle `|z| = R`: `ḡ` from the
stochastic trace estimate ([`estimate_orders`](@ref), least over orders ≤ mmax), `γ1`, `γ2` as
the largest singular values of `(I − QQ*)A*Q` and `(I − QQ*)AQ`, `σ11` as the least over `nθ`
points of the circle of `σ_min(Q*AQ − zQ*Q)`. Not certified.
"""
function float_split_estimates(op::LinOp, Q::AbstractMatrix, R::Real; nθ::Integer = 32,
        mmax::Integer = 8, nsamples::Integer = 6)
    X = apply_cols(op, Q, false)
    Xt = apply_cols(op, Q, true)
    C = X - Q * (Q' * X)
    D = Xt - Q * (Q' * Xt)
    γ2 = size(Q, 2) == 0 ? 0.0 : opnorm(C)
    γ1 = size(Q, 2) == 0 ? 0.0 : opnorm(D)
    G = Q' * X
    M = Q' * Q
    σ11 = size(Q, 2) == 0 ? Inf :
          minimum(svdvals(G - R * cis(2π * t / nθ) * M)[end] for t in 0:(nθ - 1))
    g = minimum(estimate_orders(op, Q, mmax; nsamples))
    return (; g, γ1, γ2, σ11, rule = γ1 * γ2 < σ11 * (R - g))
end

"""
    choose_frame(op, Amul, Aadj, R; ks0 = 4, kmax, grow = 1.5, Leig = 60, droptol = 1e-8,
                 sides = :both, nθ = 32, verbose = false) -> (Q, info)

The adaptive depth search of sec:depth, run in floating point: frames of Definition def:Q with a
growing number `ks` of singular directions, each tested against the depth rule eq:depthrule
`γ1γ2 < σ11 (R − ḡ)` with the estimates of [`float_split_estimates`](@ref); after each failure a
short Arnoldi on G22 is run, and a Ritz value of modulus `> R` with small residual adds its Ritz
vector to the eigendirections. Stops at the first frame passing the rule or at `dim Q > kmax`
(Remark rem:limits: declare rather than diverge); `info.rule` says which.
"""
function choose_frame(op::LinOp, Amul, Aadj, R::Real; ks0::Integer = 4,
        kmax::Integer = min(op.n - 1, 200), grow::Real = 1.5, Leig::Integer = 60,
        droptol::Real = 1e-8, sides::Symbol = :both, nθ::Integer = 32, verbose::Bool = false)
    n = op.n
    T = op.T
    Wλ = first(eigendirections(Amul, n, R; L = min(Leig, n), T))
    Lsv = min(n, 2 * kmax + 20)
    Vr, Ul, σ = gkl(Amul, Aadj, n, Lsv; T)
    extra = zeros(T, n, 0)
    ks = min(ks0, size(Vr, 2))
    history = NamedTuple[]
    Q = zeros(T, n, 0)
    est = (; g = Inf, γ1 = Inf, γ2 = Inf, σ11 = 0.0, rule = false)
    while true
        cols = Any[Wλ, extra]
        sides in (:both, :right) && push!(cols, Vr[:, 1:ks])
        sides in (:both, :left) && push!(cols, Ul[:, 1:ks])
        Q = rrqr_frame(hcat(cols...); droptol)
        est = float_split_estimates(op, Q, R; nθ)
        push!(history, (; ks, dimQ = size(Q, 2), est...))
        verbose && @info "depth search" ks dimQ = size(Q, 2) est.g est.γ1 est.γ2 est.σ11 est.rule
        est.rule && break
        # an eigenvalue of modulus > R still uncaptured? short Arnoldi on G22
        Pq(x) = x - Q * (Q' * x)
        G22mul(x) = Pq(_mv(Amul, Pq(x), T))
        V2, H2 = arnoldi(G22mul, n, min(30, n); T)
        L2 = size(H2, 2)
        if L2 > 0
            F2 = eigen(H2[1:L2, 1:L2])
            res = abs.(H2[L2 + 1, L2] .* F2.vectors[L2, :])
            new = findall(i -> abs(F2.values[i]) > R && res[i] < 1e-6 * abs(F2.values[i]),
                eachindex(F2.values))
            if !isempty(new)
                Wn = V2[:, 1:L2] * F2.vectors[:, new]
                T <: Real && (Wn = hcat(real.(Wn), imag.(Wn)))
                extra = hcat(extra, Wn)
                continue
            end
        end
        ks >= size(Vr, 2) && break
        size(Q, 2) >= kmax && break
        ks = min(size(Vr, 2), max(ks + 1, ceil(Int, grow * ks)))
    end
    return Q, (; ks, dimQ = size(Q, 2), rule = est.rule, estimates = est, history,
        ritz_sv = σ)
end
