"""
    RigorousPseudospectra

Companion code of *The reverse Ritz inclusion: computable bounds on the pseudospectra of an
operator by those of a finite-rank compression* (I. Nisoli, `~/Write/KrylovPseudospectra/
draft_dry.tex`). The source files follow the sections of the paper; the docstrings cite its labels.
Everything that is not specific to the paper, the directed rounding, the norm bounds, the singular
value enclosures, the ball products and the eigenvalue inclusions (`verifyeigall`), is
BallArithmetic's.

    twoblock.jl      §2 thm:assembly, prop:blockdom, eq:farshort, prop:complete, prop:count
                     (circle and union covers), prop:projector, cor:selfadj, cor:weyl
    operator.jl      eq:roundA, the operator and its rounding model
    couplings.jl     §3.3 eq:gram; α, the pencil and the factors of prop:almostortho
    farblock.jl      §4 eq:mincert, prop:ladder, eq:sweep, sec:fpsweep (eq:epsP, lem:onestep,
                     eq:FI, prop:fpchecks, rem:impl, alg:fpsweep)
    frame.jl         §5.1 def:Q (two-sided frame), rem:lanczos (floating point)
    splitting.jl     §5 the certified splitting, the counts through verifyeigall and the
                     homotopy of prop:almostortho, cor:agree, cor:gap, the self-adjoint case
"""
module RigorousPseudospectra

using LinearAlgebra
using SparseArrays
using Random
using BallArithmetic

include("twoblock.jl")
include("operator.jl")
include("couplings.jl")
include("farblock.jl")
include("frame.jl")
include("splitting.jl")

export assembly_floor, blockdom_floor, farshort_lo, inclusion_delta
export CircleCheck, check_circle, union_boundary_arcs, projector_bound
export EigInterval, selfadj_enclosure, selfadj_count, weyl_pairing
export LinOp, FrameData, frame_data, frame_alpha, isometry_bounds, compression_ball
export epsP_up, etaP_up, onestep_up, trace_sweep, SweepSums, estimate_orders
export FarBlockCertificate, certify_far_block
export arnoldi, gkl, eigendirections, rrqr_frame, two_sided_frame, lanczos_frame
export SplitCertificate, certify_split, sigma11_lo, sigma22_lo, resolvent_floor
export pseudospectrum_outside, count_disc, count_exterior, count_union, certify_exterior
export certify_gap, ProjectorCertificate, certify_projector
export certify_selfadjoint, eig_intervals, coupling_hermitian

end
