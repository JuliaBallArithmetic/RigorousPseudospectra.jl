# Non-rigorous reference for benchmarking / testing ONLY (implementation.md §7).
# Float64-only; never feed its output into a certified bound.

"""
    baseline_sigma_min(A, z; method = :svd) -> Float64

Reference σ_min(A − zI). `:svd` = `svdvals(A − zI)[end]` (O(n³)/point), the exact
value used as ground truth for containment tests and tightness plots.

`:lanczos` (inverse-Lanczos, the classical EigTool inner loop, O(k n²)/point) is
not yet implemented; it will share a single `schur(A)` and do two triangular
solves per Lanczos step.
"""
function baseline_sigma_min(A, z; method::Symbol = :svd)
    if method === :svd
        M = ComplexF64.(Matrix(A)) - z * I
        return svdvals(M)[end]
    elseif method === :lanczos
        error(":lanczos baseline not implemented yet — see implementation.md §7")
    else
        error("unknown method $method (use :svd or :lanczos)")
    end
end
