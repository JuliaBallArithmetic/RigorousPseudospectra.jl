# RigorousPseudospectra.jl

Companion code of *Certified pseudospectra and faithful Krylov reductions for implicitly defined
linear operators on Hilbert spaces* (I. Nisoli). For a linear operator on ℂⁿ given by
`x ↦ Ax` and `x ↦ A*x` and a frame `Q`, the package certifies the four numbers of the paper's
Main Theorem, `σ_min(G11 − zI)`, the couplings `‖G12‖`, `‖G21‖`, and `‖G22‖`, and from them a
lower bound on `σ_min(A − zI)` outside the disc of radius `‖G22‖`, the inclusion of pseudospectra,
the count of eigenvalues outside a circle, isolation discs and spectral gaps.

```
src/rounding.jl      directed rounding for scalars
src/normbounds.jl    certified norm bounds for ball matrices
src/svdenclosure.jl  §3.1 the economy-SVD enclosure (thm:svdbox)
src/twoblock.jl      §2 the two-block bound, contours, self-adjoint corollaries
src/compression.jl   §3.2 the compression: candidates, certification, transport; miyajima_rump
src/operator.jl      the operator and its rounding model (eq:roundA)
src/couplings.jl     §3.3 the couplings (eq:gram) and the almost-orthonormal frame
src/farblock.jl      §4 the far block: the sweep in floating point, certified afterwards
src/frame.jl         §5.1, §5.3 the frame and the depth search (floating point)
src/splitting.jl     §5 the certified splitting and the entry points
src/poly_range.jl    certified range of a polynomial (not part of the paper)
bench/               benchmarks and figures
test/runtests.jl     tests against dense ground truth
```

`implementation.md` maps every label of the paper to the code and lists the departures.

```julia
using RigorousPseudospectra
# stored matrix (dense or sparse): the rounding model is computed
out = certify_gap(A; R = 0.9)                       # cor:gap
out = certify_leading_eigs(A; R = 0.3)              # cor:agree: counts and isolation discs
# matrix-free: the constant ε_A of eq:roundA must be supplied
out = certify_leading_eigs(x -> A * x, x -> A' * x, n; R = 0.3, epsA = 1e-14)
# a given frame
sc = certify_split(A, Q)
resolvent_floor(sc, z)                              # thm:assembly at one point
pseudospectrum_outside(sc, ε, R)                    # prop:complete
certify_exterior(sc, R)                             # prop:count, cor:agree
certify_projector(sc, 0, R; region = :exterior)     # prop:projector: ‖P − P₀‖ and ran P₀
```

Depends on [`BallArithmetic`](https://github.com/JuliaBallArithmetic/BallArithmetic.jl) 0.3 (the
registered release) for ball matrices, their products and `svdbox`, and on the standard libraries
`LinearAlgebra`, `SparseArrays`, `Random`.

```bash
julia --project=. test/runtests.jl          # about 75 s on one thread
julia --project=. -t 8 test/runtests.jl     # the sweep runs on the threads
```

## The drafts live elsewhere

The LaTeX moved out of this repository on 2026-08-15, into `~/Write/KrylovPseudospectra`; the
paper this code follows is `draft_lean.tex` there. `bench/` generates the figures into
`bench/figures/`, untracked because they are regenerable. The dated session logs stay here.
