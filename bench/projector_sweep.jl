# prop:projector as a function of the contour. The paragraph after the proposition says the contour
# should be taken as large as the hypothesis of prop:count allows: the quantity bounded does not
# depend on it, while M and M₀ grow like the reciprocal of the distance from Γ to the spectrum. This
# script sweeps the radius of each isolation disc from the radius `certify_exterior` returns up to
# the largest admissible one, and the radius of the exterior circle over the gap it sits in,
# printing the certified bound `r M M₀ max(γ1, γ2)` beside the dense ‖P − P₀‖.
#
#   julia --project=. bench/projector_sweep.jl
using RigorousPseudospectra, LinearAlgebra, Random, Printf

# singular values decaying geometrically in random frames, plus a few large eigenvalues
function decaying(n; ρ = 0.6, seed = 1, cplx = false, nbig = 3)
    rng = MersenneTwister(seed)
    T = cplx ? ComplexF64 : Float64
    U = Matrix(qr(randn(rng, T, n, n)).Q)
    V = Matrix(qr(randn(rng, T, n, n)).Q)
    A = U * Diagonal(ρ .^ (0:(n - 1))) * V'
    X = randn(rng, T, n, nbig)
    λ = [1.0, -0.8, 0.6, 0.5, -0.45][1:nbig]
    return A * 0.3 + X * Diagonal(λ) * pinv(X)
end
epsA_width(w) = RigorousPseudospectra.mulu(RigorousPseudospectra.sqrtu(Float64(w)),
    RigorousPseudospectra.gammau(w))
function riesz(M, sel)
    F = eigen(ComplexF64.(Matrix(M)))
    return F.vectors * Diagonal(ComplexF64.(sel.(F.values))) * inv(F.vectors)
end

function sweep(n; cplx = false, seed = 41, nr = 14)
    A = decaying(n; seed, cplx)
    ev = eigvals(A)
    ma = sort(abs.(ev); rev = true)
    R = (ma[3] + ma[4]) / 2
    L = certify_leading_eigs(x -> A * x, x -> A' * x, n; epsA = epsA_width(n), R)
    L.ok || (println("  the split did not certify at R = $R"); return)
    sc = L.split
    Z = sc.Q / sqrt(Hermitian(sc.Q' * sc.Q))
    G11 = Z' * A * Z
    @printf("\n%s n = %d, %s: dim Q = %d, ḡ = %.3e, γ1 = %.3e, γ2 = %.3e\n", "operator", n,
        cplx ? "complex" : "real", sc.k, sc.gbar, sc.γ1, sc.γ2)
    @printf("  %-18s %9s %9s %9s %11s %11s %10s\n", "contour", "r", "M", "M₀", "bound",
        "‖P − P₀‖", "ratio")
    # the exterior circle, over the gap between |λ3| and |λ4|
    for Rr in range(ma[4] * 1.05, ma[3] * 0.95; length = 6)
        pc = certify_projector(sc, 0.0, Rr; region = :exterior, subspace = false)
        pc.ok || continue
        sel = λ -> abs(λ) > Rr
        t = opnorm(riesz(A, sel) - Z * riesz(G11, sel) * Z')
        @printf("  %-18s %9.4g %9.4g %9.4g %11.3e %11.3e %10.1f\n", "exterior", Rr, pc.M, pc.M0,
            pc.bound, t, pc.bound / t)
    end
    for d in L.discs
        c = d.center
        far = [abs(λ - c) for λ in ev if abs(λ - c) > 2d.radius]
        rmax = min(abs(c) - R, 0.45 * minimum(far))
        rmax > d.radius || continue
        best = (; ratio = Inf, r = NaN)
        for r in exp.(range(log(d.radius), log(rmax); length = nr))
            pc = certify_projector(sc, c, r; subspace = false)
            pc.ok || continue
            sel = λ -> abs(λ - c) < r
            t = opnorm(riesz(A, sel) - Z * riesz(G11, sel) * Z')
            rr = pc.bound / t
            rr < best.ratio && (best = (; ratio = rr, r))
            @printf("  %-18s %9.4g %9.4g %9.4g %11.3e %11.3e %10.1f\n",
                "disc λ ≈ $(round(real(c); sigdigits = 3))", r, pc.M, pc.M0, pc.bound, t, rr)
        end
        @printf("  %-18s smallest ratio %.1f at r = %.4g (largest admissible %.4g)\n", "",
            best.ratio, best.r, rmax)
    end
end

println("prop:projector: the certified bound r·M·M₀·max(γ1, γ2) against the dense ‖P − P₀‖")
for (n, cplx) in ((60, false), (120, false), (120, true))
    sweep(n; cplx)
end
