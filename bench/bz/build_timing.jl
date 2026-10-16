# Time and memory to build mid(L) at the sizes of the brief, and the sparse structure that the
# rounding constant depends on. Writes the matrices to OUT so the certified runs need not
# re-assemble.
#
# Usage: julia --project=. build_timing.jl 14,16,18,20

include("bzcommon.jl")
using Serialization

kes = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ',')) : [14, 16, 18, 20]
rows = NamedTuple[]
for ke in kes
    k = 2^ke
    GC.gc(); m0 = Sys.maxrss()
    t0 = time()
    L, binfo = build_L(k)
    t = time() - t0
    GC.gc(); m1 = Sys.maxrss()
    w = RigorousPseudospectra._stencil_width(L)
    n1 = opnorm(L, 1); ninf = opnorm(L, Inf)
    u = eps(Float64) / 2; γ(m) = (m * u) / (1 - m * u)
    εL = sqrt(Float64(w)) * γ(w)
    r = (; ke, k, nnz = binfo.nnz, t, L_MB = Base.summarysize(L) / 2^20,
        peak_MB = (m1 - m0) / 2^20, w, n1, ninf, normL = sqrt(n1 * ninf), εL,
        maxrad = binfo.maxrad)
    push!(rows, r)
    serialize(joinpath(OUT, "L_$(ke).jls"), L)
    @printf("2^%-2d nnz=%10d t=%8.1fs L=%8.1fMB peak=%9.1fMB w=%8d ‖L‖₁=%.4f ‖L‖∞=%10.1f ‖L‖≤%9.2f ε_L=%.3e maxrad=%.2e\n",
        ke, binfo.nnz, t, r.L_MB, r.peak_MB, w, n1, ninf, r.normL, εL, binfo.maxrad)
    flush(stdout)
    L = nothing; GC.gc()
end
serialize(joinpath(OUT, "build_timing.jls"), rows)
