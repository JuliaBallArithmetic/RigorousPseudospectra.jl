# Design note: certifying along the σ_min = ε level set (moving-centre pearl cover)

Status: **scoping** (2026-07-28). Prereq shipped: `certify_gap` with the tight-coupling fix
(RPS f4760fb) — the circle certificate is now the better+cheaper three-block one.

## Motivation

### Primary: a FLAT integrand for resolvent-integral / projector bounds

The spectral projector for a cluster is `P = (1/2πi)∮_Γ (zI−A)⁻¹ dz`, so
`‖P‖ ≤ (L_Γ/2π)·max_{z∈Γ}‖(zI−A)⁻¹‖ = (L_Γ/2π)·(1/min_Γ σ_min)`. On a **circle** `σ_min`
spikes toward 0 wherever the curve runs near an eigenvalue or into the pseudospectral bulge —
near an eigenvalue the pseudospectrum varies extremely fast — so `1/min_Γ σ_min` blows up and
the bound is hostage to one near-singular arc. On the **σ_min = ε level set** `σ_min ≡ ε`
*everywhere* by construction, so `‖(zI−A)⁻¹‖ ≡ 1/ε` is **uniform** and `‖P‖ ≤ L_ε/(2πε)`: the
fast variation is absorbed into the contour *shape* (it bulges exactly where σ_min would dive),
leaving a **flat integrand**. Trade a spiky integrand on a short curve for a flat one on a
longer curve — a large net win on ‖P‖ for non-normal clusters. This is the natural contour for
*any* resolvent-integral bound; gap separation is the special case below.

Rigorous caveats: the traced curve is a Float64 candidate at level ε; the cover certifies
`σ_min ≥ ε'` (slightly below ε) on the pearls, so the integrand is pinned to a **narrow band
[ε', ε_max]**, not a literal constant — still flat vs a circle's spike, with `1/ε'` the certified
factor. `L_ε` must be rigorously over-estimated (arc length of the certified cover), and the
trace must not silently skip a bulge (coverage-cap logging, below).

### Secondary: separating good from bad where no circle can

A circle is also a crude *enclosure* of a non-normal spectrum: for the Hénon depth-6 gap we
measured `σ_min(A−zI) = 0.034` where the nearest eigenvalue is `0.159` away — a **4.6×
pseudospectral bulge** (full-operator materialization, examples/14i). When the bad block's
pseudospectrum *reaches across* the radius one wants to cut, **no circle separates good from
bad, but a contour that goes around the bulge does.** (Not needed for the plain mixing rate —
shrinking circles already reach rate ≤0.72.)

### What the circle gives cheaply (and the level set must recover)

On a **circle** `Γ = {c + r·e^{iθ}}` the certificate is cheap to cover adaptively because
(a) the geometry is analytic and (b) **every point is equidistant (r) from one far-block
centre**, so a single far-block Lanczos at `c` covers the whole curve with uniform slack
`σ₂₂(z) ≥ σ₂₂(c) − r`.

## What carries over from the circle, and what breaks

The certificate is `σ_min(A−zI) ≥ σ₁₁σ₂₂/(σ₁₁+σ₂₂+γ)` with
- `σ₁₁ = σ_min(G₁₁−zI)` — exact svdbox of the captured Arnoldi block (cheap per z),
- `σ₂₂ = σ_min(G₂₂−zI)` — matvec far block, **z-independent for a fixed captured set** ⇒
  1-Lipschitz in z,
- `γ` — the certified coupling (now `svd_bound_L2_opnorm`, z-independent, computed once per
  captured set).

Along a level set, for a **fixed** captured set nothing about `G₂₂` or `γ` changes — they stay
z-independent, `σ₂₂` stays 1-Lipschitz. **The far block is NOT recomputed per point.** What
breaks is only the *equidistance*: a snaking curve is not a fixed distance from any single
centre, so the single-run slack `|z − c|` grows and `σ₂₂(c) − |z−c|` eventually goes useless.

### The three real difficulties (in order of bite)

1. **Moving far-block centres.** Replace "one centre, slack r" by a *sequence* of far-block
   centres `c_k` along the curve, each covering the arc where `|z − c_k|` keeps `σ₂₂(c_k) − |z−c_k|`
   above threshold. Placement is adaptive: dense where `σ₂₂` decays fast, sparse where flat.
   This is the pearl-necklace (`_certify_circle`/`_compute_exclusion_set`, the 5/8 pearl radius)
   generalized from **one centre** to a **moving centre**.
2. **Tracing the curve.** The level set isn't given analytically — trace it by predictor–corrector
   (tangent step + Newton back onto σ_min=ε). Step size interleaves with the cover: chosen from
   local Lipschitz/curvature so each segment fits inside one pearl. Guard against the curve
   pinching near an eigenvalue (σ_min→0 pole) — refuse/branch there ([[pole-obstruction-encircling-curve]]).
3. **z-local active band.** Which eigenvalues belong in the captured block `G₁₁` (the "bad" near
   the curve) is **z-local** and varies along the curve, unlike the ~constant band on a circle.
   Re-form the captured block per segment — exactly the dormant sweep's `adaptive=true`
   z-adaptive active set (`_active_set`/`_adaptive_bracket`), to be ported onto the matvec
   certificate.

## Proposed build (incremental, each independently testable)

- **P0 — moving-centre cover on a circle (no tracing yet).** Prove the multi-centre pearl cover
  reproduces the single-centre circle result: split `Γ` into arcs, one far-block run per arc,
  Lipschitz-cover each arc, assemble. Regression: must match `certify_gap` on |z|=0.834. This
  isolates difficulty (1) from tracing.
- **P1 — predictor–corrector tracer of σ_min=ε** (candidate Float64 curve; rigor comes from the
  cover, not the trace). Reuse the far-block/σ₁₁ evaluators as the σ_min oracle.
- **P2 — per-segment active band** (difficulty 3), ported from the dormant sweep.
- **P3 — closed-contour certificate:** a positive floor on the whole traced loop ⇒ encloses a
  certified count (argument principle), the inclusion half ([[gauss-contour-certification]]).

## Cost / honesty

Per-segment cost = one far-block Lanczos (moving centre) + a small svdbox per pearl sample.
The far block is the expensive part; the win over a uniform circle is that pearls are placed by
local Lipschitz, so smooth stretches use few centres. **Log any coverage cap** (max segments,
trace step floor) — a silently truncated trace reads as "certified the whole contour" when it
didn't. The tracer is a *candidate*; only the pearl cover is rigorous.

## Non-goals

Not for the plain mixing rate (shrinking circles suffice, rate ≤0.72 already). Not sparse/large-
scale beyond the matvec regime already in scope. This revives `src/pseudospectra/` (currently
dormant) on the matvec certificate — coordinate with that engine's contour tracing rather than
duplicating it.
