# Correctness of the split by modulus (`split_compression`) against dense truth, and its cost
# against the old path.
#
# For each matrix G of the gallery and of a family of random matrices, and each radius R:
#
#   * `sigma_split_lo(ms, z)` against `minimum(svdvals(G − zI))` at `nz` points of an annulus
#     around |z| = R and on the circle itself;
#   * `sigma_blocks_lo` (the full block diagonalisation, eq:blockres) and one `svdbox` of the whole
#     matrix, on the same points, for the ratio and the per-z cost;
#   * the certified components of `split_components` against the eigenvalues of G: every eigenvalue
#     in some component, and each component's certified count equal to the number of eigenvalues
#     inside its disc;
#   * the certified count outside R, when it can be read, against the dense count.
#
# Usage: julia --project=. bench/split_check.jl [ntrials]

using RigorousPseudospectra
using LinearAlgebra, Random, Printf
using BallArithmetic: BallMatrix
const RP = RigorousPseudospectra
include(joinpath(@__DIR__, "matrices.jl"))

const TOL = 1e-9
le(a, b) = a <= b + TOL * max(1.0, abs(b))

"A matrix with a few eigenvalues of modulus above R and a non-normal interior."
function planted(n::Integer, nout::Integer; seed::Integer = 1, T = ComplexF64, scale = 5.0)
    rng = MersenneTwister(seed)
    A = randn(rng, T, n, n) / sqrt(n)
    F = eigen(A)
    d = F.values
    p = sortperm(abs.(d); rev = true)
    d[p[1:nout]] .*= scale
    return F.vectors * Diagonal(d) / F.vectors
end

function cases(ntr::Integer)
    out = NamedTuple[]
    for (nm, n, nout, seed) in [("planted40", 40, 3, 1), ("planted80", 80, 5, 2),
        ("planted150", 150, 4, 3), ("planted200", 200, 8, 4)]
        push!(out, (; name = nm, G = planted(n, nout; seed)))
    end
    for s in 1:ntr
        n = 40 + 20 * (s % 6)
        push!(out, (; name = "randn$(n)_s$s", G = planted(n, 2 + s % 5; seed = 100 + s)))
    end
    push!(out, (; name = "grcar60", G = ComplexF64.(grcar(60))))
    push!(out, (; name = "grcar120", G = ComplexF64.(grcar(120))))
    push!(out, (; name = "normal64", G = normal_random(64; seed = 7)))
    push!(out, (; name = "two_clusters", G = ComplexF64.(two_clusters())))
    push!(out, (; name = "chain60", G = ComplexF64.(graded_chain(60))))
    # a defective interior cluster with three well separated outliers outside it
    Jd = ComplexF64.(jordan_block(18, 0.5))
    Gd = zeros(ComplexF64, 21, 21)
    Gd[1:18, 1:18] .= Jd
    Gd[19, 19] = 3.0
    Gd[20, 20] = 2.0 + 1.0im
    Gd[21, 21] = -2.5
    Gd[1:18, 19:21] .= 0.3 .* randn(MersenneTwister(9), ComplexF64, 18, 3)
    push!(out, (; name = "defective21", G = Gd))
    return out
end

# R placed between the moduli of the `nout`-th and (nout+1)-th eigenvalues when that is possible,
# else at the geometric mean of the extreme moduli.
function radii(G)
    a = sort(abs.(eigvals(G)); rev = true)
    out = Float64[]
    for j in (2, 4, 8)
        j < length(a) || continue
        a[j] > a[j + 1] * (1 + 1e-8) && push!(out, sqrt(a[j] * a[j + 1]))
    end
    isempty(out) && push!(out, sqrt(a[1] * a[end]))
    return unique(out)
end

function main(ntr::Integer = 12)
    Random.seed!(20260912)
    nsig = 0
    nbad_sig = 0
    worst_ratio = Inf
    ncomp = 0
    nbad_comp = 0
    ncount = 0
    nbad_count = 0
    ncase = 0
    rows = NamedTuple[]
    for cs in cases(ntr)
        G = cs.G
        ev = eigvals(G)
        for R in radii(G)
            ncase += 1
            ms = RP.split_compression(G, R)
            ms.cert_ok || (@warn "split: ‖R2‖ < 1 not certified" cs.name R ms.nrmR2; continue)
            ms.ok || @info "split declines" cs.name R reason = RP.split_reason(ms)
            mr = RP.miyajima_rump(G)
            # --- σ_min at many z
            zs = ComplexF64[]
            for ρ in (0.85, 1.0, 1.15, 1.6), t in 0:23
                push!(zs, ρ * R * cis(2π * t / 24))
            end
            tsp = @elapsed lo_sp = [RP.sigma_split_lo(ms, z) for z in zs]
            tsv = @elapsed lo_sv = [RP.sigmin_lo(RP._bsub(BallMatrix(G),
                Matrix{ComplexF64}(z * I, size(G)...))) for z in zs]
            tbl = @elapsed lo_bl = mr.ok ? [RP.sigma_blocks_lo(mr.cert, z) for z in zs] :
                  fill(0.0, length(zs))
            tru = [minimum(svdvals(G - z * I)) for z in zs]
            for i in eachindex(zs)
                nsig += 1
                le(lo_sp[i], tru[i]) || (nbad_sig += 1;
                @warn "split σ above truth" cs.name R zs[i] lo_sp[i] tru[i])
                lo_sp[i] > 0 && (worst_ratio = min(worst_ratio, lo_sp[i] / tru[i]))
            end
            # --- the components against the eigenvalues
            comps = RP.split_components(ms)
            covered = trues(length(ev))
            for (j, λ) in enumerate(ev)
                covered[j] = any(abs(λ - C.center) <= C.radius * (1 + 1e-10) for C in comps)
            end
            ncomp += length(ev)
            nbad_comp += count(!, covered)
            all(covered) || @warn "eigenvalue outside every component" cs.name R
            for C in comps
                inside = count(λ -> abs(λ - C.center) <= C.radius, ev)
                C.count == inside ||
                    @warn "component count differs" cs.name R C.count inside C.radius
            end
            # --- the count outside R
            cnt, okc = RP.eigencount_outside(comps, R)
            dense_out = count(λ -> abs(λ) > R, ev)
            if okc
                ncount += 1
                cnt == dense_out || (nbad_count += 1;
                @warn "certified count differs" cs.name R cnt dense_out)
            end
            push!(rows, (; cs.name, k = size(G, 1), R, ms.kplus, nout_fp = ms.nout_fp,
                ms.tries, ms.reason, ms.kwanted, ms.tz, split_ok = ms.ok,
                nrmX = ms.nrmX, nrmR2 = ms.nrmR2, betaΛ = ms.betaΛ, ms.kappa,
                gbarJ = ms.gbarJ, rho_fp = ms.rho_fp, rhoJ = ms.rhoJ,
                nblocks_full = mr.ok ? length(mr.clusters) : -1,
                kappa_full = mr.ok ? mr.kappa : Inf, beta_full = mr.ok ? mr.betaΛ : Inf,
                cnt = okc ? cnt : -1, dense_out,
                minratio_split = minimum(lo_sp ./ tru), minratio_svd = minimum(lo_sv ./ tru),
                minratio_blocks = minimum(lo_bl ./ tru),
                out_split = minimum(lo_sp[i] / tru[i] for i in eachindex(zs)
                                    if abs(zs[i]) >= R),
                out_svd = minimum(lo_sv[i] / tru[i] for i in eachindex(zs)
                                  if abs(zs[i]) >= R),
                t_split_per_z = tsp / length(zs), t_svd_per_z = tsv / length(zs),
                t_blocks_per_z = tbl / length(zs),
                t_build_split = sum(values(ms.time)), t_build_full = mr.ok ? mr.cert.time : NaN))
        end
    end
    println()
    @printf("%-14s %5s %10s %5s %5s %4s %9s %9s %9s %9s %9s %9s %6s %5s %5s %10s %10s %10s %10s %10s %10s %14s %5s\n",
        "matrix", "k", "R", "k₊", "out", "try", "‖X‖", "‖R2‖", "β_Λ", "κ", "ḡ_J", "ρ(J)fp",
        "blks", "cnt", "dns", "r split", "r svd", "r blk", "r sp≥R", "t split/z", "t svd/z",
        "ok", "need")
    for r in rows
        @printf("%-14s %5d %10.4g %5d %5d %4d %9.2e %9.2e %9.2e %9.2e %9.4g %9.4g %6d %5d %5d %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %14s %5d\n",
            r.name, r.k, r.R, r.kplus, r.nout_fp, r.tries, r.nrmX, r.nrmR2, r.betaΛ, r.kappa,
            r.gbarJ, r.rho_fp, r.nblocks_full, r.cnt, r.dense_out, r.minratio_split,
            r.minratio_svd, r.minratio_blocks, r.out_split, r.t_split_per_z, r.t_svd_per_z,
            string(r.reason), r.kwanted)
    end
    println()
    @printf("cases %d; σ comparisons %d, above truth %d; eigenvalues %d, uncovered %d; certified counts %d, wrong %d\n",
        ncase, nsig, nbad_sig, ncomp, nbad_comp, ncount, nbad_count)
    @printf("smallest positive ratio split/truth over all points: %.3e\n", worst_ratio)
    nok = count(r -> r.split_ok, rows)
    @printf("both conditions (k₊ ≤ 20 and ḡ_J·margin < R) held on %d of %d cases; the others: %s\n",
        nok, length(rows),
        join(sort(unique(string(r.reason) for r in rows if !r.split_ok)), ", "))
    return rows
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(isempty(ARGS) ? 12 : parse(Int, ARGS[1]))
end
