# Step 1 of the brief: floating point, no certificate.
#
#   * time and memory to build mid(L) at 2^14, 2^16, 2^18, 2^20;
#   * the leading eigenvalues of A = N∘L by Arnoldi, for each noise size;
#   * the decay of the singular values of A (Golub-Kahan-Lanczos Ritz values);
#   * a floating-point estimate of ‖G22‖ for frames of growing size.
#
# Usage: julia -t 128 --project=. preview.jl [k_exponents] [xis]
# e.g.   julia -t 128 --project=. preview.jl 14,16 1e-2,1e-3

include("bzcommon.jl")
using Serialization

ks = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ',')) : [14, 16, 18, 20]
ξs = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ',')) : [1e-2, 1e-3, 1e-4]
nev = 12

rows = NamedTuple[]
build = NamedTuple[]

for ke in ks
    k = 2^ke
    GC.gc()
    m0 = Sys.maxrss()
    L, binfo = build_L(k)
    GC.gc()
    m1 = Sys.maxrss()
    Lbytes = Base.summarysize(L)
    push!(build, (; ke, k, t_assemble = binfo.t_assemble, nnz = binfo.nnz,
        L_MB = Lbytes / 2^20, peak_MB = (m1 - m0) / 2^20, w = stencil_width(L),
        n1 = opnorm(L, 1), ninf = opnorm(L, Inf), maxrad = binfo.maxrad))
    @info "built" ke nnz = binfo.nnz t = binfo.t_assemble w = stencil_width(L)
    flush(stderr)

    for ξ in ξs
        l = noise_halfwidth(ξ, k)
        Amul, Aadj, η, oi = make_operator(L, l)
        op = RP.LinOp(Amul, Aadj, k; epsA = 0.0, etaA = η)

        t0 = time()
        λ = RP._ritz_leading(Amul, k, Float64; L = 80)
        t_arn = time() - t0

        t0 = time()
        Lsv = 80
        _, _, σ = RP.gkl(Amul, Aadj, k, Lsv; T = Float64)
        t_gkl = time() - t0

        # ‖G22‖ estimated (stochastic trace, floating point) for frames of growing size
        gest = NamedTuple[]
        R0 = length(λ) > 5 ? sqrt(abs(λ[5]) * abs(λ[6])) : abs(λ[end])
        for ks_ in (4, 8, 16, 24, 32)
            t0 = time()
            Qf, finfo = RP.two_sided_frame(Amul, Aadj, k, R0; ks = ks_, Leig = 80,
                Lsv = 2ks_ + 20, T = Float64)
            est = RP.float_split_estimates(op, Qf, R0; nθ = 24, mmax = 6, nsamples = 6)
            push!(gest, (; ks = ks_, dimQ = size(Qf, 2), g = est.g, γ1 = est.γ1, γ2 = est.γ2,
                σ11 = est.σ11, rule = est.rule, t = time() - t0))
        end

        push!(rows, (; ke, k, ξ, l, η, w = oi.w, εL = oi.εL, normL = oi.normL, γN = oi.γN,
            λ = λ[1:min(nev, length(λ))], σ, R0, gest, t_arn, t_gkl))
        @info "noise" ke ξ l η λ1 = λ[1] λ5 = (length(λ) >= 5 ? λ[5] : NaN) t_arn t_gkl
        flush(stderr)
    end
    L = nothing
    GC.gc()
end

serialize(joinpath(OUT, "preview_$(join(ks, "_")).jls"), (; build, rows))

println("\n## build of mid(L)\n")
@printf("%4s %9s %10s %9s %9s %7s %10s %10s %10s\n",
    "2^k", "nnz", "t_asm(s)", "L(MB)", "peak(MB)", "w", "‖L‖₁", "‖L‖∞", "maxrad")
for b in build
    @printf("%4d %9d %10.2f %9.1f %9.1f %7d %10.4f %10.1f %10.2e\n",
        b.ke, b.nnz, b.t_assemble, b.L_MB, b.peak_MB, b.w, b.n1, b.ninf, b.maxrad)
end

println("\n## the operator A = N∘L\n")
@printf("%4s %8s %8s %11s %11s %11s %9s %9s\n",
    "2^k", "ξ", "l", "η", "ε_L", "‖L‖≤", "t_arn", "t_gkl")
for r in rows
    @printf("%4d %8.0e %8d %11.3e %11.3e %11.3e %9.1f %9.1f\n",
        r.ke, r.ξ, r.l, r.η, r.εL, r.normL, r.t_arn, r.t_gkl)
end

println("\n## leading Ritz values (Arnoldi, 80 steps)\n")
for r in rows
    @printf("2^%d, ξ = %.0e:\n", r.ke, r.ξ)
    for (i, z) in enumerate(r.λ)
        @printf("  λ%-2d  %+.6f %+.6fim   |λ| = %.6f\n", i, real(z), imag(z), abs(z))
    end
end

println("\n## singular values (GKL Ritz, 80 steps)\n")
for r in rows
    @printf("2^%d, ξ = %.0e:  ", r.ke, r.ξ)
    for i in (1, 2, 4, 8, 16, 32, 64)
        i <= length(r.σ) && @printf("σ%-3d=%.3e ", i, r.σ[i])
    end
    println()
end

println("\n## floating-point ‖G22‖ estimate and the depth rule at R0\n")
@printf("%4s %8s %10s %6s %7s %11s %11s %11s %11s %6s %7s\n",
    "2^k", "ξ", "R0", "ks", "dimQ", "ĝ", "γ1", "γ2", "σ11(R0)", "rule", "t(s)")
for r in rows, g in r.gest
    @printf("%4d %8.0e %10.5f %6d %7d %11.4e %11.4e %11.4e %11.4e %6s %7.1f\n",
        r.ke, r.ξ, r.R0, g.ks, g.dimQ, g.g, g.γ1, g.γ2, g.σ11, g.rule, g.t)
end
