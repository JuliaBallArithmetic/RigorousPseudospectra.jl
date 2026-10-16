# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

The companion code of `~/Write/KrylovPseudospectra/draft_lean.tex` (Nisoli, *Certified pseudospectra
and faithful Krylov reductions for implicitly defined linear operators on Hilbert spaces*). The
source is laid out by the sections of the paper and every docstring cites the labels it
implements; `implementation.md` is the label-by-label map, with every departure from the paper.
When the paper and the code disagree, the paper is the specification: change the code, or report
the place in the paper, never both silently.

## Layout

| file | paper |
|---|---|
| `src/rounding.jl` | directed rounding for scalars (`addu`, `mulu`, `sqrtd`, …, `gammau`) |
| `src/normbounds.jl` | `opnorm_up`, `frobenius_up`, `interp_up` for ball matrices |
| `src/svdenclosure.jl` | §3.1 `thm:svdbox` (BallArithmetic's `svdbox`, `apply_vbd = false`) |
| `src/twoblock.jl` | §2 `thm:assembly`, `prop:blockdom`, `eq:farshort`, `eq:farpower`, `prop:complete`, `prop:count` covers, `prop:projector`, `cor:selfadj`, `cor:weyl` |
| `src/compression.jl` | §3.2 candidates (Sylvester/Bavely–Stewart, Newton), certification, transport; `miyajima_rump` |
| `src/operator.jl` | `LinOp`, `eq:roundA` |
| `src/couplings.jl` | §3.3 `eq:gram`, `eq:alpha`, `prop:almostortho` factors |
| `src/farblock.jl` | §4 the sweep (`trace_sweep`) and `prop:fpchecks` (`certify_far_block`) |
| `src/frame.jl` | §5.1 `def:Q`, `rem:lanczos`, §5.3 depth search (floating point) |
| `src/splitting.jl` | §5 `certify_split`, `resolvent_floor`, `certify_exterior`, `certify_projector`, entry points |
| `src/poly_range.jl` | not part of the paper; kept unchanged at the author's request |
| `bench/` | `matrices.jl` gallery, `candidates_compare.jl`, `projector_sweep.jl`, `figures.jl`, `vbd_gallery_plot.jl`, `free_resolvent_plot.jl`, `regime_figure.jl`, `matrixmarket.jl`, `poly_range_compare.jl` |

## Rigor invariants

- A certified number is an outward-rounded ball quantity (BallArithmetic products and sums,
  `svdbox`) or an a priori bound evaluated with the directed rounding of `src/rounding.jl`. A
  Float64 quantity never enters a certificate unless it is enclosed first.
- Scalar directed rounding is round-to-nearest plus one `nextfloat`/`prevfloat`; it does not touch
  the hardware rounding mode, so it is safe inside threads. Do not use `setrounding` in this
  package.
- Every hypothesis is checked: `σ11σ22 > γ1γ2` before thm:assembly, `‖R2‖ < 1` before
  lem:neumann, `α < 1`, `D ∪ Γ ⊆ {|z| > ḡ}` for prop:count, the two checks of prop:fpchecks. A
  routine that cannot certify returns `ok = false` and a zero bound.
- γ1 and γ2 are kept separate; never replace them by their maximum.
- The false forms of the old code must not come back: the floor σ11σ22/(σ11+σ22+max γ) without a
  hypothesis, the localised radius ‖Ñ_{C_i,:}‖ + β₂ (the paper's is Σ_{j}‖Ñ_{C_iC_j}‖ + qβ₂), an
  ∞-norm slack used as a 2-norm one.
- From BallArithmetic use only the ball types, ball products and sums, and `svdbox`. Its VBD
  (`schur_newton_vbd`, `miyajima_vbd`, `block_enclosure`), block Schur, projector routines and
  `upper_bound_L2_opnorm` are not called: the author considers them unreliable (an audit found a
  false localised radius and an ∞-norm slack used as a 2-norm slack).
- In `Threads.@threads` loop bodies, declare `local` every accumulator whose name is also assigned
  outside the loop; otherwise the tasks share it (this happened once in `trace_sweep`).
- No Ulam operators as test cases or benchmarks.

## Commands

```bash
julia --project=. test/runtests.jl            # the test suite, about 75 s on one thread
julia --project=. -t 8 test/runtests.jl       # with the threaded sweep
julia --project=. bench/candidates_compare.jl # Sylvester vs Newton table
julia --project=bench bench/figures.jl        # figures (CairoMakie, bench environment)
```

Dependencies: the registered BallArithmetic 0.3.0 (not the local checkout), and the standard
libraries LinearAlgebra, SparseArrays, Random. The `bench/` environment adds CairoMakie and
`dev`s this package.

## Data

`bench/results/` and `bench/mm_cache/` are data and stay. The files `bench/results/*certified*`
were produced by the old code through the false assembly (σ11σ22/(σ11+σ22+max γ) with no
hypothesis check) and are not certificates.
