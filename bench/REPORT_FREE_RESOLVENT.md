# Free resolvent bound on the legacy gallery

`julia --project=bench bench/free_resolvent_plot.jl` — builds `miyajima_rump → free_resolvent`
for each gallery matrix and plots the certified resolvent upper bound
`‖(A−zI)⁻¹‖₂ ≤ 1/σ_min_lower(z)`: **grey** where the free bound is undefined
(`σ_min_lower = 0`, the near field), a `log₁₀` heatmap/contour where defined. Cyan dashed
circles are the predicted grey discs `⋃ⱼ disc(cⱼ, ‖Nⱼ‖ + ‖E‖₂ + β₂)`.

| matrix | n | #clusters | max‖N‖ | ‖E‖₂ | β₂ | κ₂ | grey % |
|---|---|---|---|---|---|---|---|
| normal16     | 16 | 16 | 1e-14 | 9e-14 | 2e-14 | 1.0    | 0.0% |
| grcar32      | 32 | 32 | 1e-14 | 3e-11 | 3e-11 | 9.8e4  | 0.0% |
| jordan8 (W=I)|  8 |  1 | **1** | 2e-162| 4e-16 | 1.0    | 46.2% |
| two_clusters |  6 |  6 | 4e-14 | 2e-13 | 6e-14 | 26     | 0.0% |
| chain        | 12 | 12 | 2e-14 | 4e-14 | 1e-14 | 3.9    | 0.0% |

## Is the grey region "balls of radius ‖Nⱼ‖ around the cluster barycenter"?

**Yes — exactly, by construction.** `σ_min_lower(z) = max(0, minⱼ(|z−cⱼ| − ‖Nⱼ‖) − ‖E‖₂ − β₂)/κ₂`,
so it vanishes (grey) iff `∃j: |z−cⱼ| ≤ ‖Nⱼ‖ + ‖E‖₂ + β₂`. The grey set is therefore the
**union of discs of radius `‖Nⱼ‖` about the barycenters `cⱼ`**, uniformly inflated by the
global coupling `‖E‖₂` and slack `β₂` (both `≲1e-11` for a refined basis — negligible). In
`jordan8` the dashed circle (`‖N‖ = ‖shift‖₂ = 1`) sits exactly on the grey boundary.

**But a grey ball only appears where a non-normal block is KEPT TOGETHER.** The gallery
splits cleanly into two regimes:

- **Defective / coincident block** (`jordan8`): the eigenvalues cannot be separated, so the
  whole block is one cluster with `‖N‖ = 1` → a genuine grey ball of radius 1. Here the
  non-normality lives in `‖N‖` and the intuition is literal.
- **Separable spectrum** (`grcar32`, `two_clusters`, `chain`, `normal16`): `eigen`/refinement
  splits it into singletons, so `‖Nⱼ‖ ≈ 0` and there is **no grey** — the non-normality
  shows up instead in the global prefactor `κ₂` (grcar `9.8e4`, two_clusters `26`). The bound
  is defined everywhere but loose by the factor `κ₂`.

So: grey balls ⇔ defective/coupled clusters (`‖N‖`); a separable but non-normal matrix
(grcar) has **no grey** and pays its non-normality in `κ₂`. The two channels —
per-block `‖Nⱼ‖` and global `κ₂` — are exactly where the resolvent looseness can hide.

Figures: `bench/figures/free_resolvent_<name>.png`.
