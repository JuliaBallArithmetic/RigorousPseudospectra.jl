# The triangular route of sec:triangular through the harness of bench/split_check.jl: the same
# fifteen matrices and the same radii, with the certified floor compared against dense σ_min at 96
# points per case, the exterior count against the dense count, and the cost per z against the
# two-tier split and against one verified SVD of the whole matrix.
#
#   julia --project=. -t 8 bench/tri_gallery.jl [ntrials]

using RigorousPseudospectra
using LinearAlgebra, Random, Printf, Serialization
import BallArithmetic
using BallArithmetic: BallMatrix, mid
const RP = RigorousPseudospectra
include(joinpath(@__DIR__, "matrices.jl"))

# the gallery of bench/split_check.jl
function planted(n::Integer, nout::Integer; seed::Integer = 1, T = ComplexF64, scale = 5.0)
    Random.seed!(seed)
    A = randn(T, n, n) / sqrt(n)
    F = schur(A)
    Tm = Matrix(F.T)
    for i in 1:nout
        Tm[i, i] *= scale
    end
    return Matrix(F.Z) * Tm * Matrix(F.Z)'
end

function cases(ntr::Integer)
    out = NamedTuple[]
    for (nm, n, nout, seed) in [("planted40", 40, 3, 1), ("planted80", 80, 5, 2),
        ("planted150", 150, 4, 3), ("planted200", 200, 8, 4)]
        push!(out, (; name = nm, G = planted(n, nout; seed)))
    end
    for s in 1:ntr
        n = (40, 60, 80, 100, 120, 140)[1 + s % 6]
        push!(out, (; name = "randn$(n)_s$s", G = planted(n, 2 + s % 5; seed = 100 + s)))
    end
    push!(out, (; name = "grcar60", G = ComplexF64.(grcar(60))))
    push!(out, (; name = "grcar120", G = ComplexF64.(grcar(120))))
    push!(out, (; name = "normal64", G = ComplexF64.(normal_random(64))))
    push!(out, (; name = "two_clusters", G = ComplexF64.(two_clusters())))
    push!(out, (; name = "chain60", G = ComplexF64.(graded_chain(60))))
    Random.seed!(7)
    Gd = zeros(ComplexF64, 21, 21)
    for i in 1:18
        Gd[i, i] = 0.5
        i < 18 && (Gd[i, i + 1] = 1.0)
    end
    Gd[19, 19] = 3.0; Gd[20, 20] = -2.5; Gd[21, 21] = 2.0im
    U = qr(randn(ComplexF64, 21, 21)).Q
    push!(out, (; name = "defective21", G = Matrix(U) * Gd * Matrix(U)'))
    return out
end

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

le(a, b) = a <= b * (1 + 1e-10) + 1e-300

function main(ntr::Integer = 12)
    Random.seed!(20260915)
    nsig = 0; nbad = 0; ncount = 0; nbadcount = 0; ncase = 0; ndecl = 0
    worst = Inf
    rows = NamedTuple[]
    for cs in cases(ntr)
        G = cs.G
        ev = eigvals(G)
        Gb = BallMatrix(Matrix(G))
        Mb = BallMatrix(Matrix{ComplexF64}(I, size(G)...))
        for R in radii(G)
            ncase += 1
            t_set = @elapsed ts = RP.triangular_split(Gb, Mb, R; factor = 2.0)
            if ts === nothing
                ndecl += 1
                @printf("%-14s R = %10.4g : declined\n", cs.name, R); flush(stdout)
                push!(rows, (; cs.name, R, ok = false))
                continue
            end
            zs = ComplexF64[]
            for ρ in (0.85, 1.0, 1.15, 1.6), t in 0:23
                push!(zs, ρ * R * cis(2π * t / 24))
            end
            t_z = @elapsed lo = [RP.sigma_tri_lo(ts, z) for z in zs]
            t_z /= length(zs)
            t_sv = @elapsed sv = [RP.sigmin_lo(RP._bsub(Gb,
                Matrix{ComplexF64}(z * I, size(G)...))) for z in zs]
            t_sv /= length(zs)
            tru = [minimum(svdvals(G - z * I)) for z in zs]
            for i in eachindex(zs)
                nsig += 1
                le(lo[i], tru[i]) || (nbad += 1;
                    @warn "above the truth" cs.name R z = zs[i] lo = lo[i] tru = tru[i])
                lo[i] > 0 && abs(zs[i]) >= R && (worst = min(worst, lo[i] / tru[i]))
            end
            cv = RP.cover_tri(ts; η = 0.5, samples = 64, maxarcs = 20_000)
            dense = count(z -> abs(z) > R, ev)
            if cv.ok
                ncount += 1
                cv.count == dense || (nbadcount += 1;
                    @warn "count mismatch" cs.name R cv.count dense)
            end
            @printf("%-14s R = %10.4g : cut %4d, γ₁ = %.2e, ḡ₂₂ = %.2e, floor %.3e, %s (dense %d), %6.2f ms/z (svd %6.2f), setup %.1f s\n",
                cs.name, R, ts.c, ts.γ1, ts.gbar, cv.ok ? cv.floor : 0.0,
                cv.ok ? string(cv.count) : "no count", dense, 1000t_z, 1000t_sv, t_set)
            flush(stdout)
            push!(rows, (; cs.name, R, ok = true, c = ts.c, γ1 = ts.γ1, gbar = ts.gbar,
                floor = cv.ok ? cv.floor : 0.0, count = cv.ok ? cv.count : -1, dense,
                t_z, t_sv, t_set, npoints = cv.npoints, nbisect = cv.nbisect))
        end
    end
    @printf("\n%d cases, %d declined; %d σ comparisons, %d above the truth; smallest ratio to truth %.3e\n",
        ncase, ndecl, nsig, nbad, worst)
    @printf("%d covers certified, %d counts wrong\n", ncount, nbadcount)
    out = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra",
        "tri_gallery.$(gethostname()).jls")
    mkpath(dirname(out)); serialize(out, rows)
    println("rows serialised to $out")
end

main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12)
