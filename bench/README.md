# Benchmarks

The benchmark suite that produces the paper's figures and tables
(`implementation.md` §8–9). It consumes `RigorousPseudospectra` (the engine) and
the non-rigorous baselines, never implementing algorithms itself.

## Enclosure report (`figures.jl`)

`figures.jl` sweeps each gallery matrix with `bisect`, renders the certified
enclosure (with the true σ_min = ε contour overlaid as dense-SVD ground truth),
and writes `bench/REPORT.md` + `bench/figures/*.png`. It runs in the `bench/`
environment, which `dev`s the local `RigorousPseudospectra` and `BallArithmetic`
and adds `CairoMakie`:

```bash
# one-time setup
julia -e 'using Pkg; Pkg.activate("bench");
          Pkg.develop(path="/home/isaia/Code/BallArithmetic.jl");
          Pkg.develop(path="."); Pkg.add("CairoMakie")'
# generate the report
julia --project=bench bench/figures.jl
```

Open `bench/REPORT.md` to view the figures and the cell-count / routing table.

### Consolidated LaTeX report

`bench/report.tex` is a self-contained LaTeX writeup pulling the whole story
together — method, certified enclosures (Fig. + table), the free-estimate reach,
worst-block removal, and the localization (removal-fraction) fingerprint. It
`\includegraphics` from `figures/`, so compile from `bench/`:

```bash
cd bench && pdflatex report.tex && pdflatex report.tex   # twice for cross-refs
```

Regenerate the figures it references first (`figures.jl`, `free_estimates.jl`,
`free_reduced.jl`, `free_sweep.jl`) if the code has changed.

## Free-estimate field report (`free_estimates.jl`)

`free_estimates.jl` renders, per gallery matrix, the cheap certified lower bound
`g(z) = max(gap_tri, gap_svd, gap_recentred)` as a field (left) next to the true
`σ_min` (right), and writes `bench/REPORT_FREE.md`. At each `z`, `g(z)` is the
*best ε certifiable from the formulas alone* (`z ∉ σ_ε` for every `ε < g(z)`); the
red `g(z)=0` contour is the failure boundary. The table reports the box-fraction
where the free formulas certify anything and the max free ε vs the true max — i.e.
how much the almost-free layer buys per matrix (everything for normal, a few %
for strongly non-normal `grcar`).

```bash
julia --project=bench bench/free_estimates.jl
```

### Note on the Makie install
If Makie fails to precompile with `invalid UTF-8 sequence "\xa0"` in
`Makie/.../basic_recipes/contours.jl`, the installed source has a stray
non-breaking space in an indentation; repair it with
`perl -i -pe 's/\xa0/ /g' <that file>` (the file may need `chmod u+w` first).

Planned files:

- `matrices.jl` — test-matrix gallery (`implementation.md` §8): Grcar, Demmel,
  Jordan/coalescing, Kahan/Frank, companion, convection–diffusion, the Gauss
  witness. Each returns a `Matrix{ComplexF64}` plus metadata
  (`name`, `true_spectrum`, `suggested_box`, `suggested_ε`, `expects_failure`),
  with a `gallery()` listing all of them.
- `run_benchmarks.jl` — the routing/cell histogram (primary metric), cumulative
  cost decomposition, cell-count-vs-grid-count, and tightness-vs-true-σ_min
  tables (`implementation.md` §9).
- `figures.jl` — the free-gap contour overlay and refinement-region figures
  (via the Plots/Makie extensions).
