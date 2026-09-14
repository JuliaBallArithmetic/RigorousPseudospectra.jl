# Sparse non-normal operators from the Matrix Market NEP collection, stored as sparse matrices
# (so eq:roundA is computed, ε_A = √w γ_w, and the small blocks are ball products).
#
# For each: the moduli of the two leading eigenvalues (dense reference), R = √(|λ1||λ2|), and the
# whole pipeline of §5 (`certify_leading_eigs`: frame by the depth rule in floating point, one
# certified split, the exterior at R with isolation discs). Reported: dim Q, the certified ḡ, the
# couplings, the exterior count and whether cor:gap applies, times.
#
# Availability (as found when the script was first written): Tolosa is not served at any
# `pub/MatrixMarket2/NEP/tolosa/...` path tried; Olmstead and the Brusselator `rdb` are. Cached
# files are in bench/mm_cache.
#
#   julia --project=. bench/matrixmarket.jl
using RigorousPseudospectra, LinearAlgebra, SparseArrays, Printf, Downloads

const CACHE = joinpath(@__DIR__, "mm_cache")
mkpath(CACHE)

# minimal Matrix Market coordinate reader (real/general), to avoid a dependency
function read_mtx(path)
    open(path) do io
        hdr = readline(io)
        occursin("coordinate", hdr) || error("only coordinate format supported: $hdr")
        sym = occursin("symmetric", lowercase(hdr))
        line = readline(io)
        while startswith(line, "%")
            line = readline(io)
        end
        m, n, nnz = parse.(Int, split(line))
        I = Int[]
        J = Int[]
        V = Float64[]
        for _ in 1:nnz
            p = split(readline(io))
            i = parse(Int, p[1])
            j = parse(Int, p[2])
            v = length(p) >= 3 ? parse(Float64, p[3]) : 1.0
            push!(I, i)
            push!(J, j)
            push!(V, v)
            sym && i != j && (push!(I, j); push!(J, i); push!(V, v))
        end
        sparse(I, J, V, m, n)
    end
end

function fetch_mtx(set, name)
    gz = joinpath(CACHE, "$name.mtx.gz")
    mtx = joinpath(CACHE, "$name.mtx")
    isfile(mtx) && return read_mtx(mtx)
    url = "https://math.nist.gov/pub/MatrixMarket2/NEP/$set/$name.mtx.gz"
    Downloads.download(url, gz)
    run(pipeline(`gunzip -c $gz`, stdout = mtx))
    read_mtx(mtx)
end

function run_case(set, name; kmax = 200)
    local S
    try
        S = fetch_mtx(set, name)
    catch e
        @printf("%-10s FETCH FAILED: %s\n", name, first(sprint(showerror, e), 60))
        return
    end
    n = size(S, 1)
    ev = eigvals(Matrix(S))
    o = sortperm(abs.(ev); rev = true)
    sv = svdvals(Matrix(S))
    λ1, λ2 = abs(ev[o[1]]), abs(ev[o[2]])
    @printf("\n%s (%s)  n=%d  nnz=%d  ‖A‖=%.3e  s_50/s_1=%.2e  |λ1|=%.4e  |λ2|=%.4e\n", name,
        set, n, nnz(S), sv[1], sv[min(50, n)] / sv[1], λ1, λ2)
    R = sqrt(λ1 * λ2)
    t = @elapsed out = certify_gap(S; R, kmax)
    sc = out.split
    @printf("   R=%.4e  dim Q=%d  ḡ=%.4e  γ1=%.3e  γ2=%.3e  α=%.1e  exterior ok=%s count=%d  gap certified=%s  (%.1fs)\n",
        R, sc.k, sc.gbar, sc.γ1, sc.γ2, sc.α, out.exterior.ok && out.exterior.region_ok,
        out.count, out.certified, t)
    for d in out.discs
        @printf("      disc centre %+.6e%+.6ei radius %.2e count %d\n", real(d.center),
            imag(d.center), d.radius, d.count)
    end
end

println("Matrix Market NEP: the pipeline of §5 on stored sparse non-normal operators")
println("="^80)
run_case("brussel", "rdb200")
run_case("olmstead", "olm500")
run_case("brussel", "rdb800l")
run_case("olmstead", "olm1000")
