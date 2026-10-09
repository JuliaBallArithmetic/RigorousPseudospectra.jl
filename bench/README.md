# Benchmarks

The experiments of the numerics section of the paper. They consume `RigorousPseudospectra` and
BallArithmetic; none implements an algorithm of its own. Every output goes to
`~/ResearchData/KrylovPseudospectra_Data/pseudospectra/`, named
`<script>.<configuration>.<host>.<ext>`, and the frame each one reports is saved with
`save_certificate.jl` to `frames/<script>.<configuration>.N<N>.<host>.jld2` (HDF5).

| script | experiment | what it does |
|---|---|---|
| `reach_diagnostic.jl` | 1 | the necessary condition: least frame dimension from the dense singular values and eigenvalues |
| `ideal_sides.jl` | 2 | frames of right, left and two-sided dense singular vectors against the hypothesis, uncertified |
| `effective_frame.jl` | 3 | the certified ḡ against the dense ‖G22‖, the frame size found by doubling and bisection |
| `frame_exact_sv.jl` | 3 | ‖G22‖ for frames from bidiagonalisation and from dense singular vectors |
| `certify_feasible.jl` | 4 | the full certificate on the circle, with the count, where experiment 3 found ḡ < R |
| `paper_maps.jl` | 4 | the certified floor on a grid at a radius and frame given by experiment 4 |
| `bz/exp5_bz.jl` | 5 | the certificate and the isolation discs on the Belousov-Zhabotinsky operator, applied as two maps |
| `bz/paper_map_bz.jl`, `draw_bz_map.jl` | 5 | the certified floor on a grid for experiment 5, computed and drawn |
| `bz/build_timing.jl` | 5 | assembles and caches the Ulam matrices |

`pseudospectra_ops.jl` is the catalogue of operators, `matrices.jl` the gallery used by the tests,
`bz/bzcommon.jl` the BZ operator and its rounding constant. `REPLAY=1` makes
`certify_feasible.jl` and `effective_frame.jl` rebuild and save only the frames of their earlier
output.

The environments `bench/` and `bench/bz/` `dev` this package and BallArithmetic; instantiate them
with `julia --project=bench -e 'using Pkg; Pkg.instantiate()'`.
