# Read every cert_*.jls in OUT and print the report tables.
# Usage: julia --project=. tables.jl

include("bzcommon.jl")
using Serialization

files = sort(filter(f -> startswith(f, "cert_") && endswith(f, ".jls"), readdir(OUT)))
rs = [deserialize(joinpath(OUT, f)) for f in files]
sort!(rs; by = r -> (r.neig, r.ke, -r.ξ, r.dimQ))

println("\n## the operator and its rounding constant\n")
@printf("| 2^k | ξ | l | nnz(L) | w | ‖L‖ ≤ | ε_L | γ_{2l+2} | η |\n|---|---|---|---|---|---|---|---|---|\n")
seen = Set()
for r in rs
    key = (r.ke, r.ξ); key in seen && continue; push!(seen, key)
    @printf("| 2^%d | %.0e | %d | %d | %d | %.2f | %.3e | %.3e | %.3e |\n",
        r.ke, r.ξ, r.l, r.oi.nnzL, r.oi.w, r.oi.normL, r.oi.εL, r.oi.γN, r.η)
end

println("\n## the certified split\n")
@printf("| 2^k | ξ | neig | dim Q | α ≤ | R | window | ḡ ≤ | ā ≤ | γ1 ≤ | γ2 ≤ | m | σ11 ≥ | γ1γ2 | R−ḡ | assembly |\n")
println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rs
    @printf("| 2^%d | %.0e | %d | %d | %.1e | %.5f | (%.4f, %.4f) | %.5f | %.4f | %.3e | %.3e | %d | %.3e | %.3e | %.4f | %s |\n",
        r.ke, r.ξ, r.neig, r.dimQ, r.α, r.R, r.Rwin[1], r.Rwin[2], r.gbar, r.abar, r.γ1, r.γ2,
        r.m, r.s11_svd, r.γ1 * r.γ2, r.R - r.gbar,
        r.s11_svd * (r.R - r.gbar) > r.γ1 * r.γ2 ? "holds" : "fails")
end

println("\n## cost\n")
@printf("| 2^k | ξ | neig | dim Q | frame s | sweep s | split s | mem MB | matvecs | ε_P | cover pts | cover s |\n")
println("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rs
    @printf("| 2^%d | %.0e | %d | %d | %.1f | %.1f | %.1f | %.0f | %d | %.2e | %d | %.0f |\n",
        r.ke, r.ξ, r.neig, r.dimQ, r.t_frame, r.t_sweep, r.t_split, r.mem_MB, r.nmatvec,
        r.epsP, haskey(r,:ncover) ? r.ncover : -1, haskey(r,:tcover) ? r.tcover : NaN)
end

println("\n## outputs (full runs only)\n")
for r in rs
    (haskey(r, :full) && r.full) || continue
    @printf("\n### 2^%d, ξ = %.0e, neig = %d, dim Q = %d\n", r.ke, r.ξ, r.neig, r.dimQ)
    @printf("circles: %s\n", (haskey(r.circles[1], :cover) && r.circles[1].cover) ? "certified arc cover" : "pointwise samples")
    @printf("| r | ok | region | floor ≥ | ‖(A−zI)⁻¹‖ ≤ | min σ11 | min σ22 | evals | s |\n|---|---|---|---|---|---|---|---|---|\n")
    for c in r.circles
        @printf("| %.5f | %s | %s | %.4e | %.4e | %.3e | %.3e | %d | %.0f |\n",
            c.r, c.ok, c.region_ok, c.floor, c.resolvent, c.minσ11, c.minσ22, c.nevals, c.t)
    end
    @printf("prop:complete: ")
    for i in r.incl
        i.ok && @printf("ε=%.0e→δ=%.3e (δ/ε=%.3f)  ", i.ε, i.δ, i.ratio)
    end
    println()
    @printf("count outside R: %d, isolation discs ok = %s\n", r.count, r.ext_ok)
    for d in r.discs
        @printf("  disc at %+.6f%+.6fim, radius %.4e, count %d\n",
            real(d.center), imag(d.center), d.radius, d.count)
    end
    for p in r.projs
        @printf("  projector %s r=%.4e: M ≤ %.3e, ‖P−P₀‖ ≤ %.3e, ‖P‖ ≤ %.3e, dim %d, subspace gap %.2e\n",
            p.region, p.r, p.M, p.bound, p.normP, p.dim, p.sgap)
    end
    @printf("smallest R still passing with this frame: %.6f (ḡ = %.6f)\n", r.Rmin, r.gbar)
    if haskey(r, :dtruth) && r.dtruth !== nothing
        @printf("dense: ‖G22‖ = %.6f (ḡ/‖G22‖ = %.3f), ‖A‖ = %.6f (ā/‖A‖ = %.4f), eigs outside R: A %d, G11 %d\n",
            r.dtruth.nG22, r.gbar / r.dtruth.nG22, r.dtruth.nA, r.abar / r.dtruth.nA,
            r.dtruth.nout, haskey(r.dtruth, :nout11) ? r.dtruth.nout11 : -1)
    end
end

println("\n## leading certified eigenvalue enclosures\n")
for r in rs
    @printf("\n2^%d, ξ = %.0e, neig = %d, dim Q = %d:\n", r.ke, r.ξ, r.neig, r.dimQ)
    for C in (haskey(r, :lead) ? r.lead : r.comps)
        @printf("   %+.10f %+.10fim  ± %.3e  (count %d, |c| = %.8f)\n",
            real(C.center), imag(C.center), C.radius, C.count, abs(C.center))
    end
    if haskey(r, :dtruth) && r.dtruth !== nothing
        @printf("   dense eigenvalues of A: ")
        for z in r.dtruth.ev[1:min(6, length(r.dtruth.ev))]
            @printf("%.6f%+.6fim  ", real(z), imag(z))
        end
        println()
    end
end
