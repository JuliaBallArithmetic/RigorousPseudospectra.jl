# Inverse iteration on the triangular factor, to estimate sigma_11 on a circle without
# certifying anything. Used by the cut scan of triangular.jl and by the planner.


"""
    sigma11_estimate(T, z; iters = 12) -> Float64

A floating-point estimate of `σ_min(T − zI)` for triangular `T`, by inverse iteration with two
triangular solves per step. Returns 0 when the solves overflow.
"""
function sigma11_estimate(T::AbstractMatrix, z::Number; iters::Integer = 12)
    k = size(T, 1)
    k == 0 && return 0.0
    A = UpperTriangular(T - ComplexF64(z) * I)
    v = normalize!(randn(ComplexF64, k))
    s = 0.0
    for _ in 1:iters
        w = try
            A' \ (A \ v)
        catch
            return 0.0
        end
        nw = norm(w)
        (isfinite(nw) && nw > 0) || return 0.0
        v = w / nw
        s = 1 / sqrt(nw)
    end
    return s
end
