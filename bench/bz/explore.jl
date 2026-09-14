# Floating-point exploration, no certificate: how the depth rule γ1γ2 < σ11(R − ḡ) behaves as the
# frame grows, for several noise sizes and several R (5 and 10 leading eigenvalue classes, a
# conjugate pair counted once). Chooses the sizes for the certified runs.
#
# Usage: julia -t 128 --project=. explore.jl 12 1e-1,1e-2,1e-3,1e-4

include("bzcommon.jl")

ke = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
ξs = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ',')) : [1e-1, 1e-2, 1e-3, 1e-4]
kss = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ',')) : [16, 32, 64, 128, 200]

k = 2^ke
L, binfo = build_L(k)
@info "built" k binfo.nnz binfo.t_assemble w = stencil_width(L)

"""
Classes of Ritz values by modulus, a conjugate pair counted once; returns the moduli of the
distinct classes in decreasing order.
"""
function classes(λ; tol = 1e-8)
    a = sort(abs.(λ); rev = true)
    out = Float64[]
    for x in a
        (isempty(out) || abs(x - out[end]) > tol * max(1.0, x)) && push!(out, x)
    end
    return out
end

println()
@printf("%8s %8s %6s %9s %9s %9s | %6s %7s %11s %11s %11s %11s %8s\n",
    "ξ", "l", "ncls", "R5", "R10", "‖A‖≈", "ks", "dimQ", "ĝ", "γ1", "γ2", "σ11(R)", "rule")
for ξ in ξs
    l = noise_halfwidth(ξ, k)
    Amul, Aadj, η, oi = make_operator(L, l)
    op = RP.LinOp(Amul, Aadj, k; epsA = 0.0, etaA = η)
    λ = RP._ritz_leading(Amul, k, Float64; L = 120)
    cl = classes(λ)
    R5 = length(cl) > 5 ? sqrt(cl[5] * cl[6]) : NaN
    R10 = length(cl) > 10 ? sqrt(cl[10] * cl[11]) : NaN
    _, _, σ = RP.gkl(Amul, Aadj, k, 300; T = Float64)
    for (tag, R) in (("R5", R5), ("R10", R10))
        isfinite(R) || continue
        for ks_ in kss
            Qf, _ = RP.two_sided_frame(Amul, Aadj, k, R; ks = ks_, Leig = 120,
                Lsv = 2ks_ + 40, T = Float64)
            est = RP.float_split_estimates(op, Qf, R; nθ = 24, mmax = 8, nsamples = 6)
            @printf("%8.0e %8d %6d %9.5f %9.5f %9.4f | %6d %7d %11.4e %11.4e %11.4e %11.4e %8s  %s\n",
                ξ, l, length(cl), R5, R10, σ[1], ks_, size(Qf, 2), est.g, est.γ1, est.γ2,
                est.σ11, est.rule, tag)
            flush(stdout)
            est.rule && break
        end
    end
    @printf("   σ decay ξ=%.0e: ", ξ)
    for i in (1, 8, 16, 32, 64, 128, 200, 256, 300)
        i <= length(σ) && @printf("σ%d=%.3e ", i, σ[i])
    end
    println()
    @printf("   classes ξ=%.0e: ", ξ)
    for i in 1:min(13, length(cl)); @printf("%.4f ", cl[i]); end
    println("\n")
end
