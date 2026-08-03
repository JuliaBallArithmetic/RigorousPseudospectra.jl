# Real-application sparse non-normal operators from the Matrix Market NEP collection (paper §6).
#
# These are the honest test of the KRYLOV layer, as opposed to the small dense gallery which tests
# the enclosure: they are large, sparse, genuinely non-normal, and arise from applications rather
# than being constructed to be nasty. The question for each is the single hypothesis of the method,
#     ‖G₂₂‖ < R,
# which holds exactly when the singular values decay enough for a modest deflation to make the far
# block deep. We report the decay, the deflation needed, and whether a spectral gap certificate
# results (exactly one eigenvalue outside R ⇒ |λ₂| ≤ R by the exact-count proposition).
#
# NOTE ON AVAILABILITY: the paper drafted a wish-list of Tolosa / Olmstead / rdb. Tolosa is not
# served at any `pub/MatrixMarket2/NEP/tolosa/...` path we could find (all 404); Olmstead, the
# Brusselator `rdb` family and the Dwave set are. We report what is actually reachable.
#
#   julia --project=. bench/matrixmarket.jl
using RigorousPseudospectra, LinearAlgebra, SparseArrays, Printf, Downloads

const CACHE = joinpath(@__DIR__, "mm_cache")
mkpath(CACHE)

# minimal Matrix Market coordinate reader (real/general) — avoids adding a dependency
function read_mtx(path)
    open(path) do io
        hdr = readline(io)
        occursin("coordinate", hdr) || error("only coordinate format supported: $hdr")
        sym = occursin("symmetric", lowercase(hdr))
        line = readline(io)
        while startswith(line, "%"); line = readline(io); end
        m, n, nnz = parse.(Int, split(line))
        I = Int[]; J = Int[]; V = Float64[]
        for _ in 1:nnz
            p = split(readline(io))
            i = parse(Int, p[1]); j = parse(Int, p[2])
            v = length(p) >= 3 ? parse(Float64, p[3]) : 1.0
            push!(I, i); push!(J, j); push!(V, v)
            sym && i != j && (push!(I, j); push!(J, i); push!(V, v))
        end
        sparse(I, J, V, m, n)
    end
end

function fetch_mtx(set, name)
    gz = joinpath(CACHE, "$name.mtx.gz"); mtx = joinpath(CACHE, "$name.mtx")
    isfile(mtx) && return read_mtx(mtx)
    url = "https://math.nist.gov/pub/MatrixMarket2/NEP/$set/$name.mtx.gz"
    Downloads.download(url, gz)
    run(pipeline(`gunzip -c $gz`, stdout = mtx))
    read_mtx(mtx)
end

function run_case(set, name)
    local S
    try
        S = fetch_mtx(set, name)
    catch e
        @printf("%-10s FETCH FAILED: %s\n", name, first(sprint(showerror, e), 60)); return
    end
    n = size(S, 1)
    A = ComplexF64.(Matrix(S))                      # dense reference (these are moderate n)
    ev = eigvals(A); o = sortperm(abs.(ev); rev = true)
    sv = svdvals(A)
    decay = sv[min(50, n)] / sv[1]
    λ1, λ2 = abs(ev[o[1]]), abs(ev[o[2]])
    @printf("\n%s (%s)  n=%d  nnz=%d  ‖A‖₂=%.3e\n", name, set, n, nnz(S), sv[1])
    @printf("   σ₅₀/σ₁ = %.2e   |λ₁|=%.4e  |λ₂|=%.4e   ratio |λ₂/λ₁|=%.4f\n",
            decay, λ1, λ2, λ2 / λ1)
    # R strictly between |λ₂| and |λ₁| ⇒ a successful run certifies the SPECTRAL GAP
    R = sqrt(λ1 * λ2)
    t0 = time()
    local out
    try
        out = certify_leading_eigs_sing(A; Rcirc = R, nblock_eig = min(80, n - 2), margin = 0.8)
    catch e
        @printf("   DECLINED: %s\n", first(replace(sprint(showerror, e), "\n" => " "), 80)); return
    end
    el = time() - t0
    nout = length(out.results)
    @printf("   R=%.4e: nsing=%d dim(Q)=%d ‖G₂₂‖=%.4e  <R? %s   #outside=%d   (%.1fs)\n",
            R, out.nsing, out.dimQ, out.G22norm, out.far_deep ? "yes ✓" : "NO ✗", nout, el)
    for r in out.results
        @printf("      λ=%+.4e%+.4ei  σ_min≥%.3e %s  enclosure=%.1e\n",
                real(r.λ), imag(r.λ), r.cert.floor, r.cert.certified ? "✓" : "✗", r.cert.enclosure)
    end
    if nout == 1 && out.far_deep && all(r -> r.cert.certified, out.results)
        @printf("   ⇒ CERTIFIED SPECTRAL GAP: every other eigenvalue has |λ| ≤ %.4e  (|λ₂/λ₁| ≤ %.4f)\n",
                R, R / λ1)
    end
end

println("Matrix Market NEP — the Krylov layer on real sparse non-normal operators")
println("="^80)
run_case("brussel",  "rdb200")
run_case("olmstead", "olm500")
run_case("brussel",  "rdb800l")
run_case("olmstead", "olm1000")
println("\n" * "="^80)
println("The method certifies exactly when the singular values decay enough that a modest")
println("deflation drives ‖G₂₂‖ below R; where they do not, the hypothesis fails and it declines.")
