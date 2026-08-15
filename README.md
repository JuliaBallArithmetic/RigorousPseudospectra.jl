# RigorousPseudospectra.jl

Certified eigenvalue enclosures and resolvent bounds for large non-normal and
matrix-free operators: the Miyajima–Rump verified block diagonalization, a
certified Lanczos decomposition, and the certified Krylov method that joins
them.

```
src/miyajima_rump.jl     verified block diagonalization, enclosure + resolvent
src/krylov_resolvent.jl  the certified Krylov method (captured + far block)
src/vbd.jl               block-diagonalization driver
src/schur_resolvent.jl   resolvent bounds from a Schur form
src/poly_range.jl        certified range of a polynomial
bench/                   benchmark harness, figures, cached results
```

Depends on [`BallArithmetic`](https://github.com/JuliaBallArithmetic/BallArithmetic.jl),
which owns `schur_newton_vbd` — the verified kernel this package layers a method
on top of.

## The drafts live elsewhere

The LaTeX moved out of this repository on 2026-08-15, into

    ~/Write/KrylovPseudospectra

(`draft_krylov.tex` — the paper, `draft.tex`, `miyajima_pseudospectra.tex`, and
what was `bench/report.tex`). The commits that touched them were carried across
with `git filter-branch`, so they are in both histories.

`bench/` still generates the figures, into `bench/figures/`, untracked because
they are regenerable. The paper repo keeps its own committed copies; after
regenerating one here, copy it over.

The dated session logs and `implementation.md` stay here — they are engineering
notes, not paper text.
