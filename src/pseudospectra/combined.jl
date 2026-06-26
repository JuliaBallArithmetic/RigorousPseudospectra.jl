# Combining certificates: pointwise maximum of certified σ_min floors
# (paper §5, Lemma `maxfloor`).
#
# The free triangular (Schur) gap and the SVD-frame bound are BOTH O(n²) direct
# lower bounds on σ_min(A − zI), and lower bounds compose by taking the max. Their
# looseness is complementary: the triangular gap loses where the Schur off-diagonal
# tails τ_i are large; the SVD frame loses where the singular-vector misalignment
# W − I is large. These are distinct signatures of non-normality, so on a given
# matrix one floor is often tight exactly where the other is loose — their max
# settles strictly more of the plane than either, at the same O(n²) per-z cost.
#
# A Schur factorization and an SVD are each one O(n³) setup, negligible against a
# sweep over thousands of cells; compute both once and keep the larger floor.

"""
    CheapFloor

Bundles the two cheap, z-reusable certificates for a matrix `A`: the Schur-based
`CertSetup` (free triangular gap) and the `SVDFrame` (SVD-frame bound). Build once
with `cheap_floor(A)`, then evaluate the combined floor as `f(z)`.
"""
struct CheapFloor{S <: CertSetup, F <: SVDFrame}
    setup::S
    frame::F
end

"""
    cheap_floor(A) -> CheapFloor

Build both cheap certificates (one Schur factorization + one SVD, each O(n³)).
"""
cheap_floor(A::AbstractMatrix) = CheapFloor(certified_setup(A), svd_frame(A))

"""
    (f::CheapFloor)(z) -> Float64

Combined certified lower bound on σ_min(A − zI): `max(gap_tri, gap_svd)`
(Lemma `maxfloor`). Wherever this exceeds ε, `z` is certified outside the
ε-pseudospectrum.
"""
(f::CheapFloor)(z) = max(gap_tri(z, f.setup), gap_svd(z, f.frame))

"""
    binding_floor(f::CheapFloor, z) -> (value, which)

The combined floor value and which certificate is binding: `:triangular`, `:svd`,
or `:none` when both are non-positive (inconclusive). This is the per-cell routing
datum for the benchmark — its frequency measures whether a matrix's non-normality
shows up more in Schur tails or in singular-vector misalignment.
"""
function binding_floor(f::CheapFloor, z)
    gt = gap_tri(z, f.setup)
    gs = gap_svd(z, f.frame)
    if gt <= 0 && gs <= 0
        return (max(gt, gs), :none)
    elseif gt >= gs
        return (gt, :triangular)
    else
        return (gs, :svd)
    end
end
