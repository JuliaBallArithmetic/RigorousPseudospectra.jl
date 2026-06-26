# Head-to-head: certified polynomial range enclosure by
#   A. interval branch-and-bound  (range_bisect)   — cost grows with tolerance
#   B. critical points + Miyajima–Rump (range_critical) — one eigenproblem, ε-free
# across the regimes that separate them: well-separated vs clustered critical
# points, low vs high degree.
#   julia --project=. bench/poly_range_compare.jl
using RigorousPseudospectra, Printf

# Chebyshev T_n coefficients (ascending). On [-1,1], range(T_n) = [-1,1] EXACTLY,
# with n-1 interior critical points all attaining ±1 — a clean known answer and a
# non-normal companion of degree n-1 for the critical method.
function chebyshev_coeffs(n)
    n == 0 && return [1.0]
    n == 1 && return [0.0, 1.0]
    Tm, T = [1.0], [0.0, 1.0]
    for _ in 2:n
        Tn = zeros(length(T) + 1)
        for i in eachindex(T)
            Tn[i + 1] += 2T[i]
        end
        for i in eachindex(Tm)
            Tn[i] -= Tm[i]
        end
        Tm, T = T, Tn
    end
    return T
end

width(R) = R.max_hi - R.min_lo

# Per row: bisection at a MODEST tolerance with a hard eval cap (it would otherwise
# blow up on oscillatory cases — see the sweep below), vs the ε-free critical method.
function row(name, c, a, b; truth = nothing, tol = 1e-3, cap = 200_000, bisect = true)
    tc = @elapsed Rc = range_critical(c, a, b; refine = 10)
    @printf("%-22s deg=%2d\n", name, length(c) - 1)
    if bisect
        tb = @elapsed Rb = range_bisect(c, a, b; tol = tol, maxeval = cap)
        hit = Rb.diag.evals >= cap - 2
        @printf("   bisect   tol=%.0e range ⊆ [% .5e,% .5e] width=%.2e evals=%-7d %s %6.1f ms\n",
            tol, Rb.min_lo, Rb.max_hi, width(Rb), Rb.diag.evals,
            hit ? "(CAPPED)" : "        ", 1e3tb)
    end
    @printf("   critical          range ⊆ [% .5e,% .5e] width=%.2e ncrit=%d maxdisc=%.1e κ=%.1e %6.2f ms\n",
        Rc.min_lo, Rc.max_hi, width(Rc), Rc.diag.ncrit, Rc.diag.maxdisc, Rc.diag.kappa,
        1e3tc)
    if truth !== nothing
        tlo, thi = truth
        okc = Rc.min_lo <= tlo + 1e-9 && Rc.max_hi >= thi - 1e-9
        @printf("   truth             [% .5e,% .5e]   critical %s\n",
            tlo, thi, okc ? "✓sound" : "VIOLATION")
    end
    println()
end

println("="^78)
println("Certified polynomial range — bisection vs critical-point (Miyajima–Rump)")
println("="^78, "\n")

# 1. low degree, well-separated extrema at the endpoints (bisection's easy case)
row("x³ − x  on [-2,2]", [0.0, -1, 0, 1], -2.0, 2.0; truth = (-6.0, 6.0))

# 2. many well-separated interior critical points (oscillatory) — Chebyshev.
#    Bisection cannot prune (all n−1 extrema are equal at ±1), so its cost explodes;
#    at T_20 it is impractical, so we run critical only (high-degree non-normal companion).
row("Chebyshev T_5 on [-1,1]", chebyshev_coeffs(5), -1.0, 1.0; truth = (-1.0, 1.0))
row("Chebyshev T_10 on [-1,1]", chebyshev_coeffs(10), -1.0, 1.0; truth = (-1.0, 1.0))
row("Chebyshev T_20 on [-1,1]", chebyshev_coeffs(20), -1.0, 1.0; truth = (-1.0, 1.0),
    bisect = false)

# 3. clustered critical points: p = (x²−ε)²·… → p' has near-coincident roots, so
#    the companion is hard to diagonalise (large maxdisc/κ). p(x) = (x²−1e-4)².
let ε = 1e-4
    # (x²−ε)² = x⁴ − 2ε x² + ε² ; ascending [ε², 0, −2ε, 0, 1]
    c = [ε^2, 0.0, -2ε, 0.0, 1.0]
    # extrema on [-1,1]: min 0 at x=±√ε, max (1−ε)² at x=±1
    row("(x²−1e-4)² on [-1,1]", c, -1.0, 1.0; truth = (0.0, (1 - ε)^2))
end

# 4. tolerance sweep on T_5: bisection's geometrically growing cost vs the ε-free method.
println("-"^78)
println("Tolerance sweep — bisection evals/time to tighten the range on Chebyshev T_5")
println("-"^78)
c5 = chebyshev_coeffs(5)
Rcrit = range_critical(c5, -1.0, 1.0)
@printf("critical-point: width=%.2e in one eigenproblem (degree-4 companion)\n",
    width(Rcrit))
for tol in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
    t = @elapsed R = range_bisect(c5, -1.0, 1.0; tol = tol, maxeval = 5_000_000)
    @printf("  bisect tol=%.0e → width=%.2e  evals=%-8d %7.1f ms\n",
        tol, width(R), R.diag.evals, 1e3t)
end
