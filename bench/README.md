# Benchmarks

Scripts that run the package on the gallery and on stored sparse operators. They consume
`RigorousPseudospectra`; none implements an algorithm of its own.

| script | environment | what it does |
|---|---|---|
| `matrices.jl` | none | the gallery (Grcar, normal, Jordan, two clusters, graded chain), dependency-free; also used by the tests |
| `candidates_compare.jl` | `--project=.` | Sylvester (xmax 1e4 and 1e8) against Newton candidates of sec:candidate: β_Λ, κ(W), blocks, radii, iterations, times |
| `matrixmarket.jl` | `--project=.` | the pipeline of §5 (`certify_gap`) on Matrix Market NEP matrices stored sparse (cache in `mm_cache/`) |
| `projector_sweep.jl` | `--project=.` | prop:projector as a function of the contour radius, beside the dense ‖P − P₀‖ |
| `poly_range_compare.jl` | `--project=.` | certified polynomial range, bisection against critical points |
| `figures.jl` | `--project=bench` | the certified floor of the split on the gallery; writes `REPORT.md` and `figures/split_*.png` |
| `vbd_gallery_plot.jl` | `--project=bench` | the transport eq:blockres with the outer discs of both enclosure families; `figures/vbd_*.png` |
| `free_resolvent_plot.jl` | `--project=bench` | the region where eq:blockres gives no bound, against the outer discs; `figures/blockres_region_*.png` |
| `regime_figure.jl` | `--project=bench` | singular-value decay and the certified ḡ for a fixed frame; `figures/regime_decay.png` |

The `bench/` environment uses the registered BallArithmetic 0.3.0, `dev`s this package (`..`)
and adds CairoMakie:

```bash
julia --project=bench -e 'using Pkg; Pkg.instantiate()'
julia --project=bench bench/figures.jl
```

`results/` holds data produced by the old code; the files `results/*certified*` were produced
through the false assembly and are not certificates.

If Makie fails to precompile with `invalid UTF-8 sequence "\xa0"` in
`Makie/.../basic_recipes/contours.jl`, the installed source has a stray non-breaking space in an
indentation; repair it with `perl -i -pe 's/\xa0/ /g' <that file>` (after `chmod u+w`).
