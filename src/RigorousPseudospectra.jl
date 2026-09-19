"""
    RigorousPseudospectra

Companion code of *Certified pseudospectra and faithful Krylov reductions for implicitly defined
linear operators on Hilbert spaces* (I. Nisoli, `~/Write/KrylovPseudospectra/draft_lean.tex`).
The source files follow the sections of the paper; the docstrings cite its labels.

    rounding.jl      directed rounding for scalars (nextfloat/prevfloat)
    normbounds.jl    certified upper bounds on spectral norms of ball matrices
    svdenclosure.jl  §3.1 thm:svdbox (BallArithmetic's svdbox)
    twoblock.jl      §2 thm:assembly, prop:blockdom, eq:farshort, eq:farpower, prop:complete,
                     prop:count (circle and union covers), prop:projector, cor:selfadj, cor:weyl
    compression.jl   §3.2 sec:candidate (Sylvester/Bavely–Stewart and Newton), sec:certification
                     (lem:neumann, lem:deviation, thm:enclosure, cor:clusters, prop:blockgersg),
                     sec:transport (eq:blockres), prop:projector on the regions of thm:enclosure;
                     `miyajima_rump`
    operator.jl      eq:roundA, the operator and its rounding model
    couplings.jl     §3.3 eq:gram; α, the pencil and the factors of prop:almostortho
    farblock.jl      §4 eq:mincert, prop:ladder, eq:sweep, sec:fpsweep (eq:epsP, lem:onestep,
                     eq:FI, prop:fpchecks, rem:impl, alg:fpsweep)
    frame.jl         §5.1 def:Q (two-sided frame), rem:lanczos, §5.3 eq:depthrule (floating point)
    splitting.jl     §5 the certified splitting and the entry points
    deepsplit.jl     §3.2 the three-tier split: the deep tier leaves the compression and joins the
                     far block (the monotone 2×2 bound and the reduced frame)
    radius_sweep.jl  how far in the circle can be pushed at a fixed frame, by inverse
                     iteration on the Schur factor; nothing certified
    triangular.jl    the triangular route: the two-block floor in the unitary Schur basis,
                     the adaptive cover and the count by homotopy in the coupling
    planner.jl       the planning phase: floating-point estimates over a grid of (R, m, dim Q) and
                     the feasible region of the gate and of thm:assembly; nothing certified
    poly_range.jl    certified range of a polynomial (not part of the paper)
"""
module RigorousPseudospectra

using LinearAlgebra
using SparseArrays
using Random
using BallArithmetic

include("rounding.jl")
include("svdenclosure.jl")
include("normbounds.jl")
include("twoblock.jl")
include("compression.jl")
include("operator.jl")
include("couplings.jl")
include("farblock.jl")
include("frame.jl")
include("splitting.jl")
include("deepsplit.jl")
include("triangular.jl")
include("radius_sweep.jl")
include("planner.jl")
include("poly_range.jl")

# §2
export assembly_floor, blockdom_floor, farshort_lo, farpower_lo, inclusion_delta
export CircleCheck, check_circle, EigInterval, selfadj_enclosure, selfadj_count, weyl_pairing
# §3.1 and norm bounds
export svd_enclosure, sigmin_lo, sigmax_up, opnorm_up, frobenius_up, interp_up
# §3.2
export CompressionCandidate, candidate_sylvester, candidate_newton, cluster_by_distance
export CompressionCertificate, EigComponent, certify_compression, outer_radius, sigma_blocks_lo
export eigencount_in_disc, eigencount_outside, circle_min, projector_bound
export BlockProjectorCertificate, certify_block_projector
export miyajima_rump, MiyajimaRump, EigCluster
export ModulusSplit, split_compression, sigma_split_lo, sigma_J_lo, split_components
export split_sigma_blocks_lo, split_block_discs, split_reason
export DeepSplit, deep_split, deep_certificate, deep_reason, merged_far_bounds
export SplitPlan, plan_split, plan_table, candidate_radii
export TriangularSplit, triangular_split, sigma_tri_lo, count_tri, cover_tri, certify_exterior_tri
export radius_sweep, radius_table, sigma11_estimate, plan_radii
# §3.3, §4
export LinOp, FrameData, frame_data, frame_alpha, isometry_bounds, compression_ball
export epsP_up, etaP_up, onestep_up, trace_sweep, SweepSums, estimate_orders
export FarBlockCertificate, certify_far_block
# §5
export arnoldi, gkl, eigendirections, rrqr_frame, two_sided_frame, lanczos_frame
export float_split_estimates, choose_frame
export SplitCertificate, certify_split, sigma11_lo, sigma22_lo, resolvent_floor
export pseudospectrum_outside, count_disc, count_exterior, count_union, certify_exterior
export certify_gap, union_boundary_arcs, ProjectorCertificate, certify_projector
export certify_leading_eigs, krylov_resolvent_floor, certify_selfadjoint, eig_intervals
export coupling_hermitian
# polynomial range (kept, not part of the paper)
export range_bisect, range_critical, ball_horner, PolyRange

end # module
