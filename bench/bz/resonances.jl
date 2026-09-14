# The certified discs of `cor:agree`, read as Ruelle-Pollicott resonances.
#
# Each disc D(c, ρ) of `cor:agree` holds exactly `count` eigenvalues of A. For a disc with
# `count = 1` and ρ < |c| every eigenvalue λ in it satisfies
#
#     |λ| ∈ [|c| − ρ, |c| + ρ],        arg λ ∈ arg(c) ± asin(ρ/|c|),
#
# the second because the disc subtends that half-angle at the origin. From those two enclosures,
#
#     κ = −ln|λ|        (decay rate),
#     T = 2π/|arg λ|    (characteristic period, in iterates of the map),
#
# both evaluated in interval arithmetic, so the printed brackets are outward rounded. A disc whose
# closure meets the real axis (|Im c| ≤ ρ) is reported as "real or unresolved": no period follows
# from it. A disc with |Im c| > ρ is a genuine conjugate pair together with its mirror image, and
# only those give a Lorentzian peak.
#
# The projector bound of `prop:projector` for the same disc, ‖P − P₀‖ ≤ r M M₀ max(γ1, γ2), is
# printed beside it: it is what bounds the error in the amplitude of the peak.
#
# Nothing about the power spectrum is computed here.
#
# Usage: julia --project=. resonances.jl

include("bzcommon.jl")
using Serialization
using IntervalArithmetic: interval, inf, sup, mid as imid

const II = interval

"Outward-rounded enclosures of |λ|, arg λ, κ = −ln|λ| and T = 2π/|arg λ| over the disc D(c, ρ)."
function disc_resonance(c::Complex, ρ::Real)
    x = II(real(c)); y = II(imag(c)); r = II(ρ)
    m = sqrt(x * x + y * y)                       # |c|
    modlo = inf(m) - sup(r)
    mod = II(max(modlo, 0.0), sup(m) + sup(r))    # |λ|
    κ = -log(mod)                                 # decay rate
    real_or_unresolved = abs(inf(y)) <= sup(r) || abs(sup(y)) <= sup(r) ||
                         (inf(y) <= 0 <= sup(y)) || sup(r) >= inf(m)
    if real_or_unresolved
        return (; mod, κ, arg = nothing, period = nothing, pair = false)
    end
    half = asin(r / m)                            # half-angle subtended at the origin
    a0 = atan(y, x)
    arg = II(inf(a0) - sup(half), sup(a0) + sup(half))
    aabs = II(min(abs(inf(arg)), abs(sup(arg))), max(abs(inf(arg)), abs(sup(arg))))
    period = 2 * II(π) / aabs
    return (; mod, κ, arg, period, pair = true)
end

fmt(iv) = @sprintf("[%.10f, %.10f]", inf(iv), sup(iv))

files = sort(filter(f -> startswith(f, "cert_") && endswith(f, ".jls"), readdir(OUT)))
rs = [deserialize(joinpath(OUT, f)) for f in files]
rs = [r for r in rs if haskey(r, :discs) && !isempty(r.discs) &&
      any(d -> isfinite(d.radius), r.discs)]
sort!(rs; by = r -> (r.ξ, r.neig, r.ke, r.dimQ))

println("## Certified resonances from the discs of cor:agree\n")
for r in rs
    @printf("\n### n = 2^%d, ξ = %.0e, neig = %d, dim Q = %d, R = %.6f\n\n",
        r.ke, r.ξ, r.neig, r.dimQ, r.R)
    pj = Dict{ComplexF64, Any}()
    for p in r.projs
        p.region === :interior && (pj[ComplexF64(p.c)] = p)
    end
    for d in r.discs
        isfinite(d.radius) || continue
        q = disc_resonance(ComplexF64(d.center), d.radius)
        pb = get(pj, ComplexF64(d.center), nothing)
        @printf("disc centre %+.10f %+.10fim, radius %.4e, eigenvalues inside: %d\n",
            real(d.center), imag(d.center), d.radius, d.count)
        @printf("  |λ| ∈ %s\n", fmt(q.mod))
        @printf("  κ = −ln|λ| ∈ %s\n", fmt(q.κ))
        if q.pair
            @printf("  arg λ ∈ %s  (conjugate pair with its mirror image)\n", fmt(q.arg))
            @printf("  period 2π/|arg λ| ∈ %s iterates\n", fmt(q.period))
        else
            println("  real or unresolved in argument: the disc meets the real axis, no period")
        end
        if pb !== nothing
            @printf("  prop:projector on this disc: M ≤ %.4e, ‖P − P₀‖ ≤ %.4e, ‖P‖ ≤ %.4e\n",
                pb.M, pb.bound, pb.normP)
        end
        # the floating-point baseline: the nearest Ritz value, and the nearest dense eigenvalue
        if haskey(r, :λ) && !isempty(r.λ)
            j = argmin(abs.(r.λ .- d.center))
            z = r.λ[j]
            @printf("  baseline (Arnoldi Ritz): %+.10f %+.10fim, |λ| = %.10f, κ = %.10f",
                real(z), imag(z), abs(z), -log(abs(z)))
            q.pair && @printf(", period = %.6f", 2π / abs(angle(z)))
            println()
        end
        if haskey(r, :dtruth) && r.dtruth !== nothing
            j = argmin(abs.(r.dtruth.ev .- d.center))
            z = r.dtruth.ev[j]
            @printf("  baseline (dense eigen): %+.10f %+.10fim, |λ| = %.10f, κ = %.10f",
                real(z), imag(z), abs(z), -log(abs(z)))
            q.pair && @printf(", period = %.6f", 2π / abs(angle(z)))
            println()
        end
    end
end
