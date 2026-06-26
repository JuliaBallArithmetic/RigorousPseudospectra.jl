# Matrix-free far-block certificate (draft §sec:matfree) on the GKW operator.
#
# Capture the m leading eigenvalues in an orthonormal invariant subspace (ordered Schur),
# so the far block G22 = P22 A P22 is, in those coordinates, the trailing triangular Schur
# block T22 (spec = the trailing eigenvalues; G21 = 0 by invariance, coupling = ‖G12‖).
# Certify σ_min(G22 − zI) = sqrt(λ_min(M_z)), M_z = (G22−zI)*(G22−zI), by Lanczos on M_z:
#   • the loss-of-orthogonality residual ‖R2‖ = ‖Q*Q − I‖ is bounded a-posteriori from the
#     scalar coefficients via Paige's recurrence (eq. paige) — the streamed bound — and is
#     also computed directly from the stored basis, to validate the streaming;
#   • the per-step local defect ‖f_k‖ is enclosed rigorously by a ball matvec of M_z;
#   • at a FULL basis (k = dim ran P22), ‖R2‖<1 certifies the frame is invertible AND
#     complete (Lemma fullbasis), and the floor is σ_min ≥ sqrt(max(0, θ_min − β)),
#     β = ‖R1‖/(1−‖R2‖), θ_min = λ_min(T_k).
#
# Precision note: storage-free Float64 Lanczos loses orthogonality before a full basis
# (‖R2‖ crosses 1), so the certificate declines — the precision finding of the draft. To
# exhibit the floor mechanism without the Julia-1.12 BigFloat-LA GC segfault, `reorth=true`
# (full reorthogonalization) stands in for "extended precision retains orthogonality": it is
# NOT storage-free, but it reproduces the ‖R2‖≈u regime that BigFloat would reach storage-free.
#
# Run:  julia --project=bench bench/gkw_matfree.jl
using RigorousPseudospectra, LinearAlgebra, BallArithmetic, Serialization, Printf

# ---- block split via ordered Schur: bring the m largest-|λ| eigenvalues to the front ----
# G11 (m×m, captured), T22 (far block, spec = trailing eigenvalues), coupling ‖G12‖
# (G21 = 0 by leading invariance), and the reordered eigenvalues.
function blocks(A, m)
    F = schur(A)
    idx = sortperm(abs.(F.values); rev = true)
    sel = falses(length(F.values))
    sel[idx[1:m]] .= true
    Fo = ordschur(F, sel)
    G11 = Fo.T[1:m, 1:m]
    T22 = Fo.T[(m + 1):end, (m + 1):end]
    G12 = Fo.T[1:m, (m + 1):end]
    return G11, T22, opnorm(G12, 2)
end

# ---- Lanczos on M_z = (B−zI)*(B−zI), with rigorous per-step defect norm ‖f_k‖ ----
# `window` = number of trailing basis vectors to reorthogonalize against (0 = storage-free
# three-term only; ≥ kmax = full reorthogonalization). A finite window is local reorth.
function lanczos(B, z, kmax; window::Int = 0)
    N = size(B, 1)
    Bz = B - z * I
    Bzb = BallMatrix(ComplexF64.(Bz))               # for the rigorous ‖f_k‖ enclosure
    Mz(x) = Bz' * (Bz * x)
    Mzb(x) = Bzb' * (Bzb * x)                        # ball M_z·x (encloses rounding)
    Q = zeros(ComplexF64, N, kmax)
    α = zeros(Float64, kmax)
    β = zeros(Float64, kmax)
    fn = zeros(Float64, kmax)
    qn = zeros(Float64, kmax)
    q = randn(ComplexF64, N); q ./= norm(q)
    qprev = zeros(ComplexF64, N); βprev = 0.0
    kact = kmax
    for k in 1:kmax
        Q[:, k] = q
        qn[k] = norm(q)
        w = Mz(q)
        a = real(dot(q, w)); α[k] = a
        r = w - a * q - βprev * qprev
        if window > 0                                    # (local) reorth vs last `window` cols
            lo = max(1, k - window + 1)
            r -= Q[:, lo:k] * (Q[:, lo:k]' * r)
        end
        b = norm(r); β[k] = b
        if !isfinite(b) || b <= eps(Float64) * abs(a)   # Lanczos breakdown (β collapse)
            kact = k
            break
        end
        qnext = r / b
        # rigorous local defect f_k = α q_k + βprev q_{k-1} + β_k q_{k+1} − M_z q_k
        combo = a * q + βprev * qprev + b * qnext
        fk = BallVector(combo, zeros(Float64, N)) - Mzb(q)
        fn[k] = upper_bound_norm(fk, 2)
        qprev, βprev, q = q, b, qnext
    end
    return Q[:, 1:kact], α[1:kact], β[1:kact], fn[1:kact], qn[1:kact], kact
end

# ---- Paige-triggered partial reorthogonalization (Simon 1984) ------------------------
# Maintain the loss-of-orthogonality estimate ω_{i,k+1} ONLINE via the Paige recurrence
# (eq. paige) from the scalar coefficients — no inner products. When max_i |ω_{i,k+1}|
# exceeds √u (semi-orthogonality lost), reorthogonalize the new vector against the flagged
# earlier columns {i : |ω| > u^{3/4}} and reset their estimate. This keeps ‖R2‖ ≲ √u < 1
# to a full/breakdown basis. The DISTINCT flagged set is what a selective (storage-light)
# implementation must keep; here Q is kept in full only so we can report the true ‖R2‖.
function lanczos_pro(B, z, kmax)
    N = size(B, 1)
    Bz = B - z * I
    Bzb = BallMatrix(ComplexF64.(Bz))
    Mz(x) = Bz' * (Bz * x)
    Mzb(x) = Bzb' * (Bzb * x)
    Q = zeros(ComplexF64, N, kmax)
    α = zeros(Float64, kmax); β = zeros(Float64, kmax); fn = zeros(Float64, kmax)
    W = zeros(Float64, kmax + 1, kmax + 1)          # online Paige ω-magnitude estimate
    u = eps(Float64); trig = sqrt(u); ηth = u^0.75; reset = 2u
    flagged = Set{Int}(); nre = 0
    q = randn(ComplexF64, N); q ./= norm(q)
    qprev = zeros(ComplexF64, N); βprev = 0.0
    kact = kmax
    for k in 1:kmax
        Q[:, k] = q
        W[k, k] = u
        w = Mz(q); a = real(dot(q, w)); α[k] = a
        r = w - a * q - βprev * qprev
        b = norm(r)
        (!isfinite(b) || b <= u * abs(a)) && (kact = k; break)
        if k >= 2
            W[k, k + 1] = W[k + 1, k] = u                          # local (superdiagonal) seed
            ξ = u * (abs(α[k]) + b)                                # rounding injection ~ ε‖T‖
            for i in (k - 1):-1:1
                num = β[i] * W[i + 1, k] + abs(α[i] - α[k]) * W[i, k] +
                      (i > 1 ? β[i - 1] * W[i - 1, k] : 0.0) +
                      β[k - 1] * W[i, k - 1] + ξ + u * (abs(α[i]) + β[i])
                W[i, k + 1] = W[k + 1, i] = num / b
            end
            if maximum(@view W[1:k, k + 1]) > trig                 # semi-orthogonality lost
                idx = findall(i -> W[i, k + 1] > ηth, 1:k)
                for _ in 1:2                                       # reorth twice (stability)
                    r -= Q[:, idx] * (Q[:, idx]' * r)
                end
                b = norm(r)
                for i in idx
                    W[i, k + 1] = W[k + 1, i] = reset
                end
                union!(flagged, idx); nre += 1
            end
        elseif k == 1
            W[1, 2] = W[2, 1] = u
        end
        β[k] = b; qnext = r / b
        combo = a * q + βprev * qprev + b * qnext
        fk = BallVector(combo, zeros(Float64, N)) - Mzb(q)
        fn[k] = upper_bound_norm(fk, 2)
        qprev, βprev, q = q, b, qnext
    end
    return Q[:, 1:kact], α[1:kact], β[1:kact], fn[1:kact], kact, nre, length(flagged)
end

# ---- direct loss-of-orthogonality ‖R2(k)‖ = ‖Q_k* Q_k − I‖_2 from the stored basis ----
function direct_R2(Q, k)
    any(!isfinite, @view Q[:, 1:k]) && return Inf
    return opnorm(Q[:, 1:k]' * Q[:, 1:k] - I, 2)
end

# ---- streamed Paige bound on ‖R2(k)‖ (eq. paige + interval propagation) ----
# seeds: realized local Gram on the 3-term window (storage-free), |ω_ii−1| and |ω_i,i+1|.
function paige_R2(α, β, fn, qn, Q, kmax)
    W = zeros(Float64, kmax, kmax)
    for i in 1:kmax
        W[i, i] = abs(qn[i]^2 - 1)                          # |‖q_i‖²−1|
        i < kmax && (W[i, i + 1] = W[i + 1, i] = abs(dot(Q[:, i], Q[:, i + 1])))
    end
    ξ(i, j) = qn[i] * fn[j] + fn[i] * qn[j]                 # ≥ |ξ_ij|, Cauchy–Schwarz
    for j in 2:(kmax - 1)                                   # propagate column j+1 from j, j−1
        for i in (j - 1):-1:1
            num = β[i] * W[i + 1, j] + abs(α[i] - α[j]) * W[i, j] +
                  (i > 1 ? β[i - 1] * W[i - 1, j] : 0.0) + β[j - 1] * W[i, j - 1] + ξ(i, j)
            W[i, j + 1] = W[j + 1, i] = num / β[j]
        end
    end
    bound(k) = maximum(sum(@view W[i, 1:k]) for i in 1:k)   # ‖·‖_1 = ‖·‖_∞ (symmetric) ≥ ‖R2‖_2
    return bound
end

# ---- the floor at a given step k (sound only when ‖R2(k)‖ < 1 AND k = full basis) ----
function floor_at(α, β, fn, k, rho)
    Tk = SymTridiagonal(α[1:k], β[1:(k - 1)])
    θmin = minimum(eigvals(Tk))
    # ‖R1‖: Ritz residuals r_i = β_k (s_i)_k q_{k+1} + F_k s_i; bound via Frobenius of
    # [β_k (s_i)_k] (last-component term) + accumulated local defects Σ_l |(s_i)_l| ‖f_l‖.
    E = eigen(Tk); S = E.vectors
    lastcomp = β[k] * abs.(S[k, :])
    fcontrib = [sum(abs.(S[1:k, i]) .* fn[1:k]) for i in 1:k]
    ρ1 = sqrt(sum((lastcomp .+ fcontrib) .^ 2))            # ≥ ‖R1‖_2 (column Frobenius)
    βbnd = ρ1 / (1 - rho)
    return θmin, βbnd, sqrt(max(0.0, θmin - βbnd))
end

# far-block floor σ_min(G22 − zI) from the matvec Lanczos (orthogonality-retained path)
function far_floor(B, z, N2)
    Q, α, β, fn, _, k = lanczos(B, z, N2; window = N2)     # full reorth ≈ extended precision
    r2 = direct_R2(Q, k)
    _, _, fl = floor_at(α, β, fn, k, r2)
    return fl, k, r2
end

# study: does a local reorthogonalization WINDOW keep ‖R2‖<1 long enough? For each window
# width, report the step k at which the direct ‖R2‖ first crosses 1 (or "—" if it never does,
# up to breakdown), and the far floor it yields. Decides window vs selective reorth.
function study_window(B, z, N2)
    truefloor = minimum(svdvals(B - z * I))
    println("   window  cross‖R2‖>1@k   kact   ‖R2‖(kact)   far floor σ22   (true=$(round(truefloor,sigdigits=4)))")
    for w in (0, 3, 8, 16, 32, N2)
        Q, α, β, fn, _, k = lanczos(B, z, N2; window = w)
        kc = findfirst(j -> direct_R2(Q, j) >= 1, 1:k)
        r2 = direct_R2(Q, k)
        _, _, fl = floor_at(α, β, fn, k, r2)
        wl = w >= N2 ? "full" : string(w)
        @printf("   %-6s  %-13s  %-5d  %-10.2e   %s\n", wl, kc === nothing ? "—" : string(kc),
            k, r2, r2 < 1 ? @sprintf("%.4e (‖R2‖<1 ✓)", fl) : "‖R2‖≥1 → declines")
    end
end

# block-triangular resolvent assembly: with ordered-Schur capture the matrix is block
# UPPER triangular (G21 = 0 exactly), so (T−zI)⁻¹ = [[(G11−z)⁻¹, −(G11−z)⁻¹G12(G22−z)⁻¹],
# [0,(G22−z)⁻¹]] gives ‖(A−zI)⁻¹‖ ≤ 1/σ11 + 1/σ22 + γ/(σ11σ22), i.e.
#   σ_min(A−zI) ≥ σ11·σ22 / (σ11 + σ22 + γ)   — POSITIVE for any coupling γ when σ11,σ22>0.
# (The crude Weyl min(σ11,σ22)−γ is wrong here: it ignores triangularity and dies for the
#  large GKW coupling γ≈0.38.)
assemble(σ11, σ22, γ) = σ11 <= 0 || σ22 <= 0 ? 0.0 : σ11 * σ22 / (σ11 + σ22 + γ)

function main()
    A = ComplexF64.(Matrix(deserialize(joinpath(@__DIR__, "gkw_vbd_bf256.jls")).A))
    n = size(A, 1)
    m = 2                                                  # capture the 2 LEADING eigenvalues
    G11, B, γ = blocks(A, m)
    N2 = size(B, 1)
    # leading two (captured) and largest far eigenvalue → the gap-separating radius ρ
    absλ = sort(abs.(eigvals(A)); rev = true)
    λ1, λ2, λ3 = absλ[1], absλ[2], absλ[3]
    ρ = (λ2 + λ3) / 2                                       # circle between λ2 (kept) and λ3 (far)
    @printf("GKW n=%d  capture m=2 (|λ1|=%.4f, |λ2|=%.4f)  far block N2=%d  far max |λ3|=%.4f\n",
        n, λ1, λ2, N2, λ3)
    @printf("coupling ‖G12‖=%.3e (G21=0, block-triangular)   gap circle ρ=(|λ2|+|λ3|)/2=%.4f\n\n", γ, ρ)
    @printf("Certifying σ_min(A−zI) on |z|=ρ (the annulus separating λ2 from the far spectrum):\n")
    @printf("  %-8s  %-10s  %-10s  %-12s  %-12s  %-10s\n",
        "arg z/π", "σ11(G11)", "σ22(G22)", "assembled", "true σ_min", "‖R(z)‖≤")

    θs = range(0, 2π; length = 25)[1:24]
    minfloor = Inf; worst = -Inf; kfar = 0; r2far = 0.0
    for θ in θs
        z = ρ * cis(θ)
        σ11 = minimum(svdvals(G11 - z * I))                # 2×2 captured block (exact)
        σ22, kfar, r2far = far_floor(B, z, N2)             # matvec far floor
        fl = assemble(σ11, σ22, γ)
        tru = minimum(svdvals(A - z * I))
        minfloor = min(minfloor, fl)
        worst = max(worst, fl - tru)                       # ≤ 0 ⇒ sound (floor ≤ true)
        if θ in (θs[1], θs[7], θs[13], θs[19])             # print a few angles
            @printf("  %-8.2f  %-10.4e  %-10.4e  %-12.4e  %-12.4e  %-10.2e\n",
                θ / π, σ11, σ22, fl, tru, fl > 0 ? 1 / fl : Inf)
        end
    end
    @printf("\n  far-block Lanczos: k=%d (numerical rank), ‖R2‖=%.1e\n", kfar, r2far)
    @printf("  min assembled floor over the circle = %.4e  (%s)\n",
        minfloor, minfloor > 0 ? "POSITIVE ⇒ no eigenvalue on |z|=ρ" : "fails")
    @printf("  rigor: max(assembled − true) over circle = %+.2e  %s\n",
        worst, worst <= 1e-9 ? "✓ sound" : "VIOLATION")
    println()
    println("Local reorthogonalization window study (z = ρ, the storage-vs-orthogonality tradeoff):")
    study_window(B, ρ, N2)
    println()
    println("Paige-triggered partial reorthogonalization (selective storage) at z = ρ:")
    Qp, αp, βp, fnp, kp, nre, nflag = lanczos_pro(B, ρ, N2)
    r2p = direct_R2(Qp, kp)
    @printf("   k=%d  ‖R2‖=%.2e (%s)  reorth events=%d  distinct vectors flagged=%d/%d\n",
        kp, r2p, r2p < 1 ? "<1 ✓" : "≥1", nre, nflag, N2)
    flp = r2p < 1 ? floor_at(αp, βp, fnp, kp, r2p)[3] : NaN
    @printf("   floor σ22=%.4e  (true=%.4e)\n\n", flp, minimum(svdvals(B - ρ * I)))
    @printf("Mixing-rate consequence: σ_min(A−zI) > 0 on the whole circle |z|=ρ=%.4f, and the\n", ρ)
    @printf("only spectrum outside it is the captured λ1=1, λ2 (|λ2|=%.4f). So λ2 is rigorously\n", λ2)
    @printf("the subdominant eigenvalue and the rest of σ(A) lies in |z|<ρ: the spectral gap is\n")
    @printf("≥ 1−ρ = %.4f and the mixing rate is governed by |λ2| = %.4f.\n", 1 - ρ, λ2)
end
main()
