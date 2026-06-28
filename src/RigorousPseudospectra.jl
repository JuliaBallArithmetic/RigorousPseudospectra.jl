"""
    RigorousPseudospectra

Currently exposes the **Miyajima–Rump certified eigenvalue enclosure**
(`miyajima_rump`): a verified eigendecomposition of a dense matrix via the
eigenvector (diagonal-residue) frame and a single two-residual certification, with
separation clustering of coupled eigenvalues. The mathematics is in
`miyajima_pseudospectra.tex` and the routine map in `implementation.md` (§3, §4c–4e).

The pseudospectrum engine (gap ladder, box-bisection sweep, contours, plotting) is
quarantined in `src/pseudospectra/` — dormant, not loaded — and will return once the
eigenvalue-enclosure core is settled.
"""
module RigorousPseudospectra

include("miyajima_rump.jl")
include("resolvent.jl")
include("krylov_resolvent.jl")
include("schur_resolvent.jl")
include("vbd.jl")
include("poly_range.jl")

export miyajima_rump, refine_basis, MiyajimaRump, EigCluster
export range_bisect, range_critical, ball_horner, PolyRange
export free_resolvent, FreeResolvent, sigma_min_lower, resolvent_bound, row_col_gaps
export schur_resolvent, SchurResolvent
export vbd_solve, vbd_resolvent, VBDResolvent, block_enclosure

end # module
