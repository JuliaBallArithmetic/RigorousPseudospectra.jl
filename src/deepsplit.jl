# §3.2 continued: the three-tier split, in which the deepest tier leaves the compression.
#
# The split by modulus of `src/compression.jl` keeps the whole compression and certifies the part
# inside the circle through the powers of the trailing block. Here the compression is cut twice,
#
#     W⁻¹ G11 W = diag(E, N, D) + Δ,     ‖Δ‖ ≤ β_Λ,
#
# with E the eigenvalues of modulus above the circle, N the ones just below it that the conditions
# have to add, and D the deep ones; E ∪ N stays in the compression and is resolved by a verified
# SVD per z, whose cost is the only consequence of its size,
# and D is moved out of the frame altogether. What is required of D is only that the far block it
# joins stay below R.
#
# ------------------------------------------------------------------------------------------------
# The derivation (an isometry Q first; the almost-orthonormal case follows below)
#
# Let Z be the isometry with ran Z = ran Q, P⊥ = I − ZZ*, and let the four certified numbers of the
# old frame be ḡ ≥ ‖G22‖, γ1 ≥ ‖G12‖, γ2 ≥ ‖G21‖, s ≥ ‖G11‖. Let U ∈ ℂ^{k×k_EN} have full column
# rank (the leading Schur vectors below) and put
#
#     Q' = Q U,     Z' the isometry with ran Z' = ran Q',     Z_D an isometry onto ran Z ⊖ ran Z'.
#
# Then ran Z' ⊆ ran Z, so ran P'⊥ = ran P⊥ ⊕ ran Z_D as an ORTHOGONAL direct sum, and in that
# decomposition the new blocks are
#
#     G'11 = Z'*AZ',
#     G'12 = [ Z'*AZ_D ,  Z'*AP⊥ ],          G'21 = [ Z_D*AZ' ; P⊥AZ' ],
#     G'22 = [ Z_D*AZ_D   Z_D*AP⊥ ;  P⊥AZ_D   G22 ].
#
# Write X' = Z*Z' and X_D = Z*Z_D, isometries of ℂ^k with [X', X_D] unitary. Every block that meets
# P⊥ is an old coupling compressed by a contraction:
#
#     ‖Z'*AP⊥‖ = ‖X'*G12‖ ≤ γ1,   ‖Z_D*AP⊥‖ ≤ γ1,   ‖P⊥AZ'‖ ≤ γ2,   ‖P⊥AZ_D‖ ≤ γ2,
#
# so with ḡ_D ≥ ‖X_D*G11X_D‖, c₁₂ ≥ ‖X'*G11X_D‖ and c₂₁ ≥ ‖X_D*G11X'‖,
#
#     ‖G'12‖ ≤ √(c₁₂² + γ1²),   ‖G'21‖ ≤ √(c₂₁² + γ2²),
#     ‖(G'22)^m‖ ≤ ‖ N^m ‖,     N = [ ḡ_D  γ1 ; γ2  ḡ ]  (nonnegative 2×2).
#
# The first two are `‖[B, C]‖² ≤ ‖B‖² + ‖C‖²`. The third is the monotonicity of the spectral norm
# in the norms of the blocks: for a 2×2 block operator matrix, ‖Mx‖² ≤ ‖N(‖x1‖,‖x2‖)‖² with N the
# matrix of block norms, and the block norms of a product are at most the product of the block-norm
# matrices entrywise, while the spectral norm of a nonnegative matrix is monotone in its entries.
# So one 2×2 power gives every order of `eq:farpower` for the merged far block, at no cost.
#
# ------------------------------------------------------------------------------------------------
# The three inner norms
#
# Let M = Q*Q, S_p = M^{1/2}, so Z = QS_p⁻¹ and G11 = S_p⁻¹ G̃ S_p⁻¹ with G̃ = Q*AQ. Then
# ran X' = ran(S_p U), so X' = S_p U K with K = (U*MU)^{-1/2}, and X'*S_p⁻¹ = K*U*, S_p⁻¹X' = UK.
# Let S̃ ∈ ℂ^{k×k_D} have full column rank with ran S̃ = (ran U)^⊥. Since v ⊥ S_pUx for all x iff
# S_pv ⊥ ran U, ran X_D = S_p⁻¹ ran S̃, so X_D = S_p⁻¹ S̃ L with L = (S̃*M⁻¹S̃)^{-1/2} and
# S_p⁻¹X_D = M⁻¹S̃L. Hence
#
#     X'*G11X'  = K*(U*G̃U)K,
#     X'*G11X_D = K*(U*G̃M⁻¹S̃)L,
#     X_D*G11X' = L*(S̃*M⁻¹G̃U)K,
#     X_D*G11X_D = L*(S̃*M⁻¹G̃M⁻¹S̃)L,
#
# and with ‖I − M‖ ≤ α < 1: ‖M⁻¹‖ ≤ 1/(1−α), ‖M⁻¹ − I‖ ≤ ε_α := α/(1−α),
# ‖K‖ ≤ 1/(√(1−α) σ_min(U)) and ‖L‖ ≤ √(1+α)/σ_min(S̃). Writing the ball norms
# `u = ‖U*G̃S̃‖`, `v = ‖S̃*G̃U‖`, `w = ‖S̃*G̃S̃‖`, `nG = ‖G̃‖`,
#
#     s'   ≤ ‖K‖² ‖U*G̃U‖,
#     c₁₂  ≤ ‖K‖‖L‖ ( u + ‖U‖ nG ε_α ‖S̃‖ ),
#     c₂₁  ≤ ‖K‖‖L‖ ( v + ‖S̃‖ nG ε_α ‖U‖ ),
#     ḡ_D  ≤ ‖L‖²  ( w + ‖S̃‖² nG ε_α (1/(1−α) + 1) ).
#
# Nothing here applies A: every quantity comes from the ball `G̃`, the ball `M` and the floating
# point U and S̃, so the deep tier costs no operator application and no new sweep. With α at the
# 1e-13 of the runs, every factor above is 1 + O(1e-12).
#
# This is the constant the brief asked to be checked. The route through the candidate,
# `ḡ_D ≤ ‖W_D‖‖Y_D‖‖D‖ + κβ_Λ`, is also valid and is the one to state when `G̃` is not available;
# it is weaker here by the factor κ, and the code uses the direct enclosure above.
#
# ------------------------------------------------------------------------------------------------
# What is NOT small
#
# `c₂₁` is of the size of β_Λ, because ran Z' is an approximately invariant subspace. `c₁₂` is the
# off-diagonal block of the Schur form of the compression and is of the size of ‖G11‖: an invariant
# subspace makes one of the two off-diagonal blocks small and, unless G11 is normal, not the other.
# So the deep tier trades γ1 for √(c₁₂² + γ1²) and buys the smaller compression with it; whether
# the product γ'₁γ'₂ falls below γ1γ2 is a measurement, not a theorem, and the code reports both.

"""
    DeepSplit

The three-tier cut of a compression: `kE` eigenvalues of modulus above `rE`, `kN` between `rD` and
`rE`, `kD` at or below `rD`, the first two kept in the frame and the third moved into the far
block, `E` fixed by the circle and `N` by the conditions. Certified: `gD ≥ ‖Z_D*AZ_D‖`, `c12 ≥ ‖Z'*AZ_D‖`, `c21 ≥ ‖Z_D*AZ'‖`, the new couplings
`gamma1n ≥ ‖G'12‖` and `gamma2n ≥ ‖G'21‖`, `snormn ≥ ‖G'11‖`, and `gbars[j] ≥ ‖(G'22)^j‖` from the
2×2 matrix `N = [gD γ1; γ2 ḡ]`; `gbarn = min_j gbars[j]^{1/j}`. `U` is the k×k_EN block of Schur
vectors that spans the new frame and `S` its complement. `ok` says that the cut closes. `reason` names the obstruction
otherwise: `:farblock` (the gate `γ1γ2 < R(R − ḡ)` fails, so no cut can help), `:nocut` (no cut
passes the conditions) or `:cost` (the conditions pass at a `k_EN` past the stopping rule;
`inner.kwanted` and `inner.tz` say which and at what cost per `z`).
"""
struct DeepSplit
    ok::Bool
    reason::Symbol
    k::Int
    kE::Int
    kN::Int
    kD::Int
    R::Float64
    rE::Float64
    rD::Float64
    U::Matrix{ComplexF64}
    S::Matrix{ComplexF64}
    α::Float64
    gD::Float64
    c12::Float64
    c21::Float64
    gamma1::Float64
    gamma2::Float64
    gbar::Float64
    gamma1n::Float64
    gamma2n::Float64
    snormn::Float64
    gbars::Vector{Float64}
    gbarn::Float64
    m::Int
    ntrials::Int
    inner::NamedTuple
    time::NamedTuple
end

function Base.show(io::IO, ds::DeepSplit)
    print(io, "DeepSplit(ok = ", ds.ok, " (", ds.reason, "), E|N|D = ", ds.kE, "|", ds.kN, "|",
        ds.kD, " of ", ds.k, ", ḡ_D ≤ ", ds.gD, ", ḡ' ≤ ", ds.gbarn, " against R = ", ds.R,
        ", γ'₁ ≤ ", ds.gamma1n, ", γ'₂ ≤ ", ds.gamma2n, ")")
end

"""
    deep_reason(ds) -> String

One line naming the obstruction, with the numbers.
"""
function deep_reason(ds::DeepSplit)
    γγ = ds.gamma1 * ds.gamma2
    rr = ds.R * (ds.R - ds.gbar)
    if ds.reason === :ok
        return "the cut closes at k_EN = $(ds.kE + ds.kN) of $(ds.k), " *
               "ḡ' ≤ $(ds.gbarn) against R = $(ds.R)"
    elseif ds.reason === :cost
        kw = ds.inner.kwanted
        tz = ds.inner.tz
        lim = ds.inner.budget === nothing ? "the ceiling k_EN ≤ $(ds.inner.kcap)" :
              "the budget of $(ds.inner.budget) s per z"
        return "a cost decline, not a failure of the method: the conditions choose k_EN = $kw " *
               "of $(ds.k) (E has $(ds.kE) columns, N has $(kw - ds.kE)), which costs $tz s per " *
               "z for one verified SVD of that size, and that is past $lim"
    elseif ds.reason === :farblock
        return "with the deep tier empty the merged far block is already at or above the " *
               "circle: ḡ' ≤ $(ds.gbarn) against R = $(ds.R). The spectral radius of " *
               "[0 γ1; γ2 ḡ] is below R exactly when γ1γ2 < R(R − ḡ), and here γ1γ2 = $γγ " *
               "against R(R − ḡ) = $rr; no cut can help"
    elseif ds.reason === :nocut
        return "no admissible cut: at the deepest admissible one, k_EN = $(ds.kE + ds.kN), the " *
               "merged far block has ḡ' ≤ $(ds.gbarn) against R = $(ds.R) (ḡ_D ≤ $(ds.gD))"
    end
    return string(ds.reason)
end

# ------------------------------------------------------------------------------------------------

"""
    _deep_blocks(G, M, α, U, S) -> NamedTuple

The four inner norms of the header, certified: `sn ≥ ‖X'*G11X'‖`, `gD ≥ ‖X_D*G11X_D‖`,
`c12 ≥ ‖X'*G11X_D‖`, `c21 ≥ ‖X_D*G11X'‖`, together with the raw ball norms `u`, `v`, `w` and the
factors. `G` is the ball enclosing `Q*AQ`, `M` the ball enclosing `Q*Q` (`nothing` for `M = I`),
`α ≥ ‖I − Q*Q‖`, and `U`, `S` floating-point matrices of full column rank with `ran S = (ran U)^⊥`.
"""
function _deep_blocks(G, M, α::Real, U::AbstractMatrix, S::AbstractMatrix)
    α = Float64(α)
    Gb = _promote_ball(_ball(G))
    Ub = BallMatrix(ComplexF64.(Matrix(U)))
    Sb = BallMatrix(ComplexF64.(Matrix(S)))
    Uh = BallMatrix(ComplexF64.(Matrix(U')))
    Sh = BallMatrix(ComplexF64.(Matrix(S')))
    GU = _bmul(Gb, Ub)
    nUGU = opnorm_up(_bmul(Uh, GU))
    nG = opnorm_up(Gb)
    nU = opnorm_up(Ub)
    sU = sigmin_lo(Ub)
    oma = subd(1.0, α)
    opa = addu(1.0, α)
    (oma > 0 && sU > 0) || return (; ok = false, sn = Inf, gD = Inf, c12 = Inf, c21 = Inf,
        u = Inf, v = Inf, w = Inf, nG, nU, nS = Inf, sU, sS = 0.0, nK = Inf, nL = Inf)
    nK = divu(1.0, muld(sqrtd(oma), sU))
    sn = mulu(mulu(nK, nK), nUGU)
    if size(S, 2) == 0
        return (; ok = true, sn, gD = 0.0, c12 = 0.0, c21 = 0.0, u = 0.0, v = 0.0, w = 0.0,
            nG, nU, nS = 0.0, sU, sS = 0.0, nK, nL = 0.0)
    end
    GS = _bmul(Gb, Sb)
    u = opnorm_up(_bmul(Uh, GS))
    v = opnorm_up(_bmul(Sh, GU))
    w = opnorm_up(_bmul(Sh, GS))
    nS = opnorm_up(Sb)
    sS = sigmin_lo(Sb)
    sS > 0 || return (; ok = false, sn, gD = Inf, c12 = Inf, c21 = Inf, u, v, w, nG, nU, nS, sU,
        sS, nK, nL = Inf)
    nL = divu(sqrtu(opa), sS)
    εα = M === nothing ? 0.0 : divu(α, oma)
    KL = mulu(nK, nL)
    c12 = mulu(KL, addu(u, mulu(mulu(nU, nG), mulu(εα, nS))))
    c21 = mulu(KL, addu(v, mulu(mulu(nS, nG), mulu(εα, nU))))
    f = addu(divu(1.0, oma), 1.0)
    gD = mulu(mulu(nL, nL), addu(w, mulu(mulu(mulu(nS, nS), nG), mulu(εα, f))))
    return (; ok = true, sn, gD, c12, c21, u, v, w, nG, nU, nS, sU, sS, nK, nL)
end

"""
    merged_far_bounds(gD, γ1, γ2, ḡ, m) -> (gbars, gbarn)

`gbars[j] ≥ ‖(G'22)^j‖` for `j = 1:m` from the nonnegative 2×2 matrix `N = [ḡ_D γ1; γ2 ḡ]`, and
`gbarn = min_j gbars[j]^{1/j}`, every operation rounded upward. The powers of a nonnegative 2×2
are exact arithmetic on four numbers, so every order of `eq:farpower` for the merged far block is
free once `ḡ_D` is known.
"""
function merged_far_bounds(gD::Real, γ1::Real, γ2::Real, gbar::Real, m::Integer;
        empty::Bool = false)
    # with the deep tier empty there is nothing to merge: G'22 = G22 and the bound is ḡ^j
    if empty
        d = Float64(gbar)
        gs = [powu(d, j) for j in 1:max(Int(m), 1)]
        return (gs, d)
    end
    a, b, c, d = Float64(gD), Float64(γ1), Float64(γ2), Float64(gbar)
    (isfinite(a) && isfinite(b) && isfinite(c) && isfinite(d)) ||
        return (fill(Inf, max(Int(m), 1)), Inf)
    P = (a, b, c, d)
    gs = Float64[]
    best = Inf
    for j in 1:max(Int(m), 1)
        if j > 1
            P = (addu(mulu(P[1], a), mulu(P[2], c)), addu(mulu(P[1], b), mulu(P[2], d)),
                addu(mulu(P[3], a), mulu(P[4], c)), addu(mulu(P[3], b), mulu(P[4], d)))
        end
        g = norm2x2_up(P...)
        push!(gs, g)
        isfinite(g) && (best = min(best, rootu(g, j)))
    end
    return (gs, best)
end

"""
    _deep_cost_per_z(sc, U, R) -> seconds

The measured cost of one `z` for the reduced frame: one `svdbox` of the `k_EN × k_EN` ball pencil
`G̃' − zM'`, timed (three repetitions, the least kept). That is the only per-`z` work the deep tier
does, and it is the only consequence of the size of `E ∪ N`, so the stopping rule is expressed in
it.
"""
function _deep_cost_per_z(sc::SplitCertificate, U::AbstractMatrix, R::Real)
    size(U, 2) == 0 && return 0.0
    Ub = BallMatrix(ComplexF64.(Matrix(U)))
    Uh = BallMatrix(ComplexF64.(Matrix(U')))
    Gn = _bmul(Uh, _bmul(_promote_ball(_ball(sc.G)), Ub))
    Mn = _bmul(Uh, _bmul(_promote_ball(_ball(sc.M)), Ub))
    z = ComplexF64(R)
    best = Inf
    for _ in 1:3
        t = time()
        sigmin_lo(_bsub(Gn, _zm_ball(Mn, z)))
        best = min(best, time() - t)
    end
    return best
end

# The leading `kEN` Schur vectors of the compression and their orthogonal complement, from one
# `ordschur` on a copy of the Schur pair.
function _deep_frames(T0, V0, kEN::Integer)
    k = size(T0, 1)
    kEN >= k && return (Matrix(V0), zeros(ComplexF64, k, 0))
    kEN <= 0 && return (zeros(ComplexF64, k, 0), Matrix(V0))
    T = copy(T0)
    V = copy(V0)
    _ordschur!(T, V, _topk_mask(abs.(diag(T)), kEN))
    return (V[:, 1:kEN], V[:, (kEN + 1):k])
end

"""
    deep_split(sc::SplitCertificate, R; m = 8, margin = 1.02, kmax = nothing, budget = nothing,
               frac = 0.9, criterion = :assembly, amargin = 1.5, nθ = 24, scan = nothing)
        -> DeepSplit

The three-tier cut for the certificate `sc` and the circle of radius `R`. The compression is Schur
factorised once; the cut `k_EN` is the SMALLEST admissible one passing the criterion, found by
bisection on the number of columns kept (admissible means that it does not split a pair of equal
moduli). The three groups define themselves: `E = {|λ| > frac·R}` is fixed by the circle, `N` is
whatever the conditions have to add, and the size of `E ∪ N` is an outcome. Its only consequence is
the per-`z` cost of one verified SVD of that size, so the search is capped only by a STOPPING RULE:
`kmax` (a ceiling on `k_EN`, by default the frame size, so no ceiling) or `budget` (seconds per `z`,
measured by [`_deep_cost_per_z`](@ref)). When the conditions want more than the stopping rule
allows, the call declines with `reason = :cost` and `inner.kwanted`, `inner.tz` record what the
conditions wanted and what it would cost. That is a statement about cost, not a failure of the
method.

Two criteria:

* `:farblock` — `min_{j≤m} ‖(G'22)^j‖^{1/j} · margin < R`, the merged far block below the circle.
  This is the condition the deep tier exists to satisfy, and it is satisfied by the deepest cuts,
  which are also the ones with the largest `ḡ_D` and so the smallest `R − ḡ'`;
* `:assembly` (the default) — that, and in addition a floating-point estimate of the hypothesis of
  thm:assembly on the circle, `σ̂11(R)(R − ḡ') > amargin·γ'₁γ'₂`, with `σ̂11(R)` the least
  `σ_min(G̃' − zM')` over `nθ` samples of `|z| = R`. Nothing about that estimate is certified; it
  only chooses the cut, and the certificate is the arc cover afterwards.

`scan` (a list of `k_EN` values) evaluates the criterion at those cuts as well and records the row
in `inner.scan`, so that the trade between the size of the frame and `R − ḡ'` is visible.

Only cuts that remove at least one column are searched: `k_EN = k` is the identity and is not a
cut, so `ok = true` always means a frame that is strictly smaller.

There is always a necessary condition, checked first: with the deep tier empty the merged far block
is `[0 γ1; γ2 ḡ]`, whose spectral radius is below `R` exactly when `γ1γ2 < R(R − ḡ)`, which is the
hypothesis of thm:assembly at `|z| = R` with `σ11` replaced by `R`. If that fails, no cut helps
and the call declines with `reason = :farblock`.

The bisection assumes that the criterion is monotone in `k_EN`, which it is not in general (the
Schur subspaces are not nested as the cut moves); the value returned is verified to pass and the
value one admissible step below it is verified to fail, so what is returned is a local minimum and
is reported as such.
"""
function deep_split(sc::SplitCertificate, R::Real; m::Integer = 8, margin::Real = 1.02,
        kmax = nothing, budget = nothing, frac::Real = 0.9, criterion::Symbol = :assembly,
        amargin::Real = 1.5, nθ::Integer = 24, scan = nothing)
    t0 = time()
    R = Float64(R)
    Gm = ComplexF64.(Matrix(mid(sc.G)))
    Mm = ComplexF64.(Matrix(mid(sc.M)))
    Gc = Mm \ Gm
    k = size(Gc, 1)
    F = schur(Gc)
    T0 = Matrix(F.T)
    V0 = Matrix(F.Z)
    a = sort(abs.(diag(T0)); rev = true)
    rE = Float64(frac) * R
    kE = count(>(rE), a)
    kcap = kmax === nothing ? k : clamp(Int(kmax), 1, k)
    t_schur = time() - t0
    kw = Ref(-1)          # the k_EN the conditions want, whatever the stopping rule
    tzw = Ref(NaN)        # the measured seconds per z that k_EN implies
    bad(reason, kEN, blocks, gs, gn, ntr) = DeepSplit(false, reason, k, kE,
        max(kEN - kE, 0), k - kEN, R, rE, kEN < k ? sqrt(a[kEN] * a[min(kEN + 1, k)]) : 0.0,
        zeros(ComplexF64, k, 0), zeros(ComplexF64, k, 0), sc.α,
        blocks === nothing ? Inf : blocks.gD, blocks === nothing ? Inf : blocks.c12,
        blocks === nothing ? Inf : blocks.c21, sc.γ1, sc.γ2, sc.gbar, Inf, Inf, Inf,
        gs, gn, Int(m), ntr,
        (; u = NaN, v = NaN, w = NaN, nG = NaN, nU = NaN, nS = NaN, sU = NaN, sS = NaN,
            nK = NaN, nL = NaN, s11 = NaN, marg = NaN, criterion, local_min = false,
            scan = NamedTuple[], cands = 0, kcap, kwanted = kw[], tz = tzw[], budget),
        (; schur = t_schur, total = time() - t0))
    # With the deep tier empty the merged far block is [0 γ1; γ2 ḡ], whose spectral radius is
    # below R exactly when γ1γ2 < R(R − ḡ), which is the hypothesis of thm:assembly at |z| = R
    # with σ11 replaced by R. Nothing deeper can help if that already fails.
    gs0, gn0 = merged_far_bounds(0.0, sc.γ1, sc.γ2, sc.gbar, m)
    if !(mulu(gn0, Float64(margin)) < R)
        return bad(:farblock, k, nothing, gs0, gn0, 0)
    end

    # only cuts that actually remove a column: k_EN = k is the identity and is not a cut
    cands = [j for j in max(kE, 1):(k - 1) if a[j] > a[j + 1] * (1 + 1e-12)]
    isempty(cands) && return bad(:nocut, k, nothing, gs0, gn0, 0)
    scanrows = NamedTuple[]
    cache = Dict{Int, Any}()
    ntr = Ref(0)
    zs = ComplexF64[R * cis(2π * t / nθ) for t in 0:(Int(nθ) - 1)]
    function trial(kEN)
        get!(cache, kEN) do
            ntr[] += 1
            U, S = _deep_frames(T0, V0, kEN)
            b = _deep_blocks(sc.G, sc.M, sc.α, U, S)
            gs, gn = b.ok ? merged_far_bounds(b.gD, sc.γ1, sc.γ2, sc.gbar, m;
                empty = kEN >= k) : (fill(Inf, Int(m)), Inf)
            far = b.ok && mulu(gn, Float64(margin)) < R
            γ1n = sqrtu(addu(mulu(b.c12, b.c12), mulu(sc.γ1, sc.γ1)))
            γ2n = sqrtu(addu(mulu(b.c21, b.c21), mulu(sc.γ2, sc.γ2)))
            # a floating-point estimate of σ11 on the circle for the reduced compression, used
            # only to choose the cut; nothing about it is certified
            s11 = 0.0
            if far
                Gn = U' * (Gm * U)
                Mn = U' * (Mm * U)
                s11 = minimum(minimum(svdvals(Gn - z * Mn)) for z in zs)
            end
            marg = far ? s11 * (R - gn) / max(mulu(γ1n, γ2n), floatmin(Float64)) : 0.0
            pass = criterion === :farblock ? far : (far && marg > Float64(amargin))
            (; U, S, b, gs, gn, γ1n, γ2n, s11, marg, far, pass)
        end
    end
    if scan !== nothing
        for kEN in scan
            kEN in cands || continue
            t = trial(kEN)
            push!(scanrows, (; kEN, gD = t.b.gD, gbarn = t.gn, γ1n = t.γ1n, γ2n = t.γ2n,
                s11 = t.s11, margin = t.marg, far = t.far, pass = t.pass))
        end
    end
    # Bisection on the index into `cands` for the SMALLEST k_EN passing the conditions, with no
    # cap: E is fixed by R and N is whatever the conditions ask for, so the size of E ∪ N is an
    # outcome. The stopping rule is applied afterwards and is a statement about cost, not about
    # the method.
    lo = 1
    hi = length(cands)
    if !trial(cands[hi]).pass
        t = trial(cands[hi])
        return bad(:nocut, cands[hi], t.b, t.gs, t.gn, ntr[])
    end
    if trial(cands[lo]).pass
        hi = lo
    else
        while hi - lo > 1
            mid_ = (lo + hi) ÷ 2
            trial(cands[mid_]).pass ? (hi = mid_) : (lo = mid_)
        end
    end
    kEN = cands[hi]
    tr = trial(kEN)
    kw[] = kEN
    tzw[] = _deep_cost_per_z(sc, tr.U, R)
    # the stopping rule: a ceiling on k_EN, or a budget in seconds per z
    over = kEN > kcap || (budget !== nothing && tzw[] > Float64(budget))
    if over
        return bad(:cost, kEN, tr.b, tr.gs, tr.gn, ntr[])
    end
    U, S, b = tr.U, tr.S, tr.b
    γ1n, γ2n = tr.γ1n, tr.γ2n
    rD = kEN < k ? sqrt(a[kEN] * a[kEN + 1]) : 0.0
    return DeepSplit(true, :ok, k, kE, kEN - kE, k - kEN, R, rE, rD, U, S, sc.α, b.gD, b.c12,
        b.c21, sc.γ1, sc.γ2, sc.gbar, γ1n, γ2n, b.sn, tr.gs, tr.gn, Int(m), ntr[],
        (; b.u, b.v, b.w, b.nG, b.nU, b.nS, b.sU, b.sS, b.nK, b.nL, tr.s11, tr.marg,
            criterion, local_min = hi == 1 || !trial(cands[hi - 1]).pass, scan = scanrows,
            cands = length(cands), kcap, kwanted = kw[], tz = tzw[], budget),
        (; schur = t_schur, total = time() - t0))
end

"""
    deep_certificate(sc::SplitCertificate, ds::DeepSplit; compression = :full, kwargs...)
        -> SplitCertificate

The certificate of §5 for the reduced frame `Q' = Q·U`, built from `sc` and `ds` with no
application of the operator and no new sweep: `G̃' = U*G̃U` and `M' = U*MU` enclose `Q'*AQ'` and
`Q'*Q'`, `α'` is read off `M'`, the couplings and the far block are the derived bounds of
[`deep_split`](@ref), and `gbars_power` carries every order of `eq:farpower` for the merged far
block. `compression` is passed to the compression layer for the reduced compression, which is
small; `:full` (the default) is the block diagonalisation of sec:candidate, `:split` the split by
modulus, `:none` neither.
"""
function deep_certificate(sc::SplitCertificate, ds::DeepSplit; compression = :full,
        vbd_method::Symbol = :sylvester, xmax::Real = 1e8, kappa_mode::Symbol = :svdbox,
        split_frac::Real = 0.9, split_m::Integer = 8, split_kmax = nothing,
        split_budget = nothing)
    t0 = time()
    ds.ok || error("the deep split did not close: " * deep_reason(ds))
    U = ds.U
    kEN = size(U, 2)
    Ub = BallMatrix(ComplexF64.(U))
    Uh = BallMatrix(ComplexF64.(Matrix(U')))
    Gn = _bmul(Uh, _bmul(_promote_ball(_ball(sc.G)), Ub))
    Mn = _bmul(Uh, _bmul(_promote_ball(_ball(sc.M)), Ub))
    αn = opnorm_up(_bsub(Mn, Matrix{ComplexF64}(I, kEN, kEN)))
    αn < 1 || error("the reduced frame is not certified to have full rank: α' = $αn")
    Qn = sc.Q * U
    fd = FrameData(sc.n, kEN, αn, Gn, Mn, (0.0, 0.0), (ds.snormn, 0.0), (ds.gamma1n, 0.0),
        (ds.gamma2n, 0.0), frobenius_up(Qn), sc.fd.stored, sc.fd.coupling, 0.0)
    mode = _compression_mode(compression, ds.R, kEN, 0)
    comp = nothing
    ms = nothing
    if mode === :split
        ms = split_compression(Gn, ds.R; M = Mn, frac = split_frac, m = split_m,
            kmax = split_kmax, budget = split_budget, kappa_mode, xmax)
    elseif mode === :full
        comp = miyajima_rump(Gn; M = Mn, method = vbd_method, xmax, kappa_mode)
    end
    return SplitCertificate(sc.n, kEN, Qn, αn, Gn, Mn, ds.gbars[1], sc.abar, ds.gamma1n,
        ds.gamma2n, ds.snormn, nothing, fd, comp, ms, sc.hermitian, copy(ds.gbars),
        (; frame = 0.0, far = 0.0, compression = time() - t0, total = time() - t0, mode,
            deep = true))
end
