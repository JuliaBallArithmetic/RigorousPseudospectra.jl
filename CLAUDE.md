# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠ Current focus (refactor in progress)

The package has been **refocused onto the Miyajima–Rump certified eigenvalue
enclosure** (`src/miyajima_rump.jl`, exported `miyajima_rump`); the pseudospectrum
engine is **quarantined and dormant** in `src/pseudospectra/` (not loaded, not
exported) and will return once the eigenvalue-enclosure core is settled. So:

- **Active code:** `src/RigorousPseudospectra.jl` (module, loads only the core) +
  `src/miyajima_rump.jl`; tests in `test/runtests.jl`. Run `julia --project=. test/runtests.jl`.
- **Dormant:** `src/pseudospectra/` (gaps, sweep, svd_frame, combined, cluster,
  coupling, setup, baseline, ext_dormant), `test/pseudospectra/`, `bench/` — all
  reference the shelved pseudospectrum API and won't load/run as-is.
- The sections below describing the gap ladder / box-bisection / cluster
  escalation document that **dormant** engine — accurate for when it returns, not
  the current public surface.

**The Miyajima–Rump algorithm (the current deliverable, implementation.md §3, §4c–4e):**
`eigen(B)` candidate → transform to almost-diagonal `Ã = W⁻¹BW = D + E` → **one**
two-residual certification (`R₂ = YW − I`, slack `β`) → recentred Gershgorin discs
of `Ã` (centre `Ãᵢᵢ`, radius `Σ_{j≠i}|Ãᵢⱼ| + β`) that rigorously enclose `σ(B)` →
separation clustering by overlapping discs (a connected component of k discs holds
exactly k eigenvalues). **`Y` is an approximate inverse** (`Y ≈ W⁻¹`, floating
point) — `W⁻¹` is never formed; the certificate is valid for any `Y` with
`‖R₂‖ < 1` ("Y is free"), and `‖W⁻¹‖ ≤ ‖Y‖/(1−‖R₂‖)`.

**Status: RIGOROUS.** The certification runs in midpoint–radius ball arithmetic
(`BallArithmetic.BallMatrix`): `R₂ = YW−I` and `Ã = YBW` are ball products (all
rounding bounded), norms via certified `upper_bound_L_inf_opnorm`. Candidate
production (`eigen`, `inv`, refinement) stays Float64. **§4e refinement is
implemented** (`refine_basis`, float-Newton `Xᵢⱼ = −Ãᵢⱼ/(dᵢ−dⱼ)` — note the **minus**;
the §4e guide snippet had `+`, which *doubles* the off-diagonal — quadratic
convergence to the κ·eps floor) with a **dynamic working clustering** recomputed each
step by the contraction criterion `|Ãᵢⱼ| ≥ θ|dᵢ−dⱼ|` (θ<1; the §4e merge-detection):
non-contracting pairs (Newton would diverge) are neutralized, the rest separate; as
the residue falls, pairs become contracting and split, coincident/too-coupled stay
together — split AND merge handled numerically. Driver `miyajima_rump(B; refine, θ)`:
`eigen` → up to `refine` dynamic steps → ONE rigorous certification (disc-overlap =
the arbiter: over-split → discs overlap → re-merged; divergent split → `‖R₂‖≥1`
Neumann fail, caught). `refine_basis(B,W,clusters;steps)` is the fixed-clustering
variant for the manual workflow.

**κ₂ — cheap (Collatz) by DEFAULT, svdbox only on request (`kappa_mode`).** `κ₂ =
‖W‖₂‖W⁻¹‖₂` (from residuals, never `cond`) is a **scale-invariant diagnostic** — NOT used
in the eigenvalue enclosure — and the prefactor of the free resolvent layer
(`src/resolvent.jl`). The verified `svdbox(W)` is the ONLY `O(n³)` step beyond the ball
products, so `_certify`/`miyajima_rump` take `kappa_mode`:
- `:cheap` (**default**) — no SVD: `upper_bound_L2_opnorm` (`min(Collatz, √(‖·‖₁‖·‖∞))`,
  `O(n²)`) for `‖W‖₂`, `‖Y‖₂` + the residual `‖W⁻¹‖₂ ≤ ‖Y‖₂/(1−‖R₂‖₂)`. Overestimates
  `‖W‖₂` by ≈`√n` on orthonormal bases (so `κ₂ ≈ n` for a *normal* matrix, true 1), but
  keeps the whole pipeline `O(n³ matmul)` — **cheap even in BigFloat**. κ₂ rarely needs to
  be tight (the resolvent small-gain typically has orders of headroom, e.g. GKW `ε~1e-45`).
- `:svdbox` (opt-in, tight) — a **single** `svdbox(W)`: `κ₂ = σ_max(W)/σ_min(W)` (largest
  ball upper end / smallest ball lower end), `W⁻¹` never formed; falls back to the residual
  when the verified SVD can't certify `σ_min(W) > 0` (badly ill-conditioned `W` — e.g. GKW
  `cond(W)~1e13` at Float64; BigFloat certifies it directly). Use when a tight κ₂ matters.

`free_resolvent(B, R; tight=false)` mirrors this for `‖Nⱼ‖`/`‖E‖` (Collatz default, svdbox
when `tight=true`). The whole pipeline is generic in the float type (`Float64`/`BigFloat`),
BallArithmetic's products/svdbox/opnorms working transparently in `BigFloat`.

**Split/merge are numerical; correctness is the single end certification (§4e — DONE).**
Following §4e's asymmetry, neither split nor merge is certified — both are decided
numerically by the dynamic clustering inside `refine_basis`, and the lone rigorous
step is the final certification, whose disc-overlap clustering **self-corrects**: an
over-optimistic split whose certified discs still overlap is **re-merged**, and trying
to separate genuinely-coincident eigenvalues ill-conditions `W` (Newton blow-up) →
`‖R₂‖≥1` → **Neumann fail** (caught). So `svdbox` is **not** needed for clustering — it
stays **only** for the `κ₂` diagnostic.

**Deferred** (§4e refinements / second pass):
- **Local per-block refinement** (`O(n m²)`): the current `refine_basis` does the full
  `W⁻¹BW` solve + full-`W` update each step (`O(n³)`/step). The localized form would
  operate only on the `m×m` block `Ã_CC` and update only the `n×m` columns `W_C` via an
  `m×m` similarity (new blocks from old small blocks, no full inverse; far block
  untouched), making "only the `m×m` cluster is expensive" literal.
- the extra merge-detection surrogates (separation below `u^{1/k}‖A‖`; sub-block
  conditioning) — the contraction criterion (signal 2) is the one implemented;
- an explicit `:remerge` re-solve and extended precision on a bad block (§4d, BigFloat);
- `prodK` accurate residuals (shelved; revisit only if residual accuracy is the wall for
  high-κ/near-defective). The committed-frame / κ-gate / eigen-vs-Schur discussion is
  pseudospectrum-era and on hold.

## What this repository is

**`RigorousPseudospectra.jl`** — a Julia package implementing a certified
ε-pseudospectrum engine, plus the paper draft and benchmark suite that document
and exercise it. It is the rigorous counterpart to `Pseudospectra.jl` (the Julia
EigTool port): same job — enclose `σ_ε(A)` — but with *verified* bounds instead
of floating-point estimates. Intended to live under the **`JuliaBallArithmetic`**
GitHub organization, alongside `BallArithmetic.jl`.

The package name is `RigorousPseudospectra`. The on-disk directory is still
`RigPseudospectra.jl` (kept to preserve session/memory continuity); rename it to
`RigorousPseudospectra.jl` with `mv` whenever convenient — nothing depends on the
folder name.

### Layering (the key architectural decision)

```
BallArithmetic.jl   — kernel: verified numerical linear algebra (ball arithmetic,
                      rigorous opnorm/SVD/Sylvester/eigenvalue enclosures)
        ↓ depends on
RigorousPseudospectra.jl — engine: the pseudospectral method built ON the kernel
        ↓ package extensions
Plots / Makie       — visualization (contour overlays, routing cell maps)
```

This package was split out from `BallArithmetic.jl` deliberately: the kernel
stays a lean verified-linalg library, and the pseudospectral *method* +
*plotting* + *benchmarks* live here. Plotting especially belongs here as
`Plots`/`Makie` **package extensions** (`weakdeps` in `Project.toml`, code in
`ext/`) — the EigTool/`Pseudospectra.jl` model — so the kernel never grows a
plotting dependency.

### Cut line (engine-only migration — current policy)

- **Implemented HERE** (the method): the gap-function ladder (free / uniform /
  recentred / interpolated-ℓ² / cluster), the box-bisection sweep, contour
  tracing, benchmarks, and the plotting extensions.
- **Stays in / comes from `BallArithmetic.jl`** (verified primitives): `Ball` /
  `BallMatrix`, `upper_bound_L1/L2/Linf_opnorm`, `rigorous_svd` / `svdbox` /
  `SVDMethod`, `sylvester_miyajima_enclosure`, the Miyajima/Rump eigenvalue
  enclosures, `ordschur_ball`, block Schur.
- The **existing** pseudospectra code in `BallArithmetic.jl/src/pseudospectra/`
  (`CertifScripts.jl`, `sylvester_resolvent_bound.jl`, `rigorous_contour.jl`)
  is **left in place for now**. It may migrate here later, but the current task
  is to build the new engine cleanly against BallArithmetic's *public* API —
  do not reach for its underscore-private internals (e.g. `_certify_svd`); if you
  need something unexported, export it from BallArithmetic with intent.

## Source-of-truth documents (in this repo)

- `miyajima_pseudospectra.tex` — the math paper ("Enclosure of Eigenvalues via
  Approximate Diagonalization"). Develops the certified `σ_min(A−zI)` bound on
  Miyajima's residual machinery, sharpened into a pseudospectral engine. **Every
  routine is derived from a numbered theorem/lemma here** — cite it in the code.
- `implementation.md` — the implementation guide: maps each result to a concrete
  routine, fixes the rigor conventions, and lays out the gap ladder (§4),
  coupling-graph clustering (§4b), cluster path (§5), sweep (§6), baselines (§7),
  test-matrix gallery (§8), benchmarks (§9), and rigor regression tests (§10).
  **Note its one anachronism**: it is
  written against `IntervalArithmetic.jl`; translate all interval wrapping to
  `BallArithmetic`'s `BallMatrix`. The two-phase pattern is identical; only the
  rigorous container changes.

## Package layout

```
src/RigorousPseudospectra.jl   module, imports BallArithmetic, includes + exports
src/setup.jl                   CertSetup + certified_setup — z-independent precompute (§3)
src/gaps.jl                    gap_tri / gap_uniform / gap_recentred (§4)
src/svd_frame.jl               SVDFrame + svd_frame + gap_svd — SVD-frame certificate (paper §5)
src/combined.jl                CheapFloor + cheap_floor + binding_floor — max(tri, svd) (paper §5.x)
src/coupling.jl                cluster_components + percolation_scan — coupling-graph clustering (§4b)
src/cluster.jl                 reorder_cluster + sigma_min_block + gap_cluster — block path (§5)
src/sweep.jl                   bisect + PseudospectrumEnclosure — box bisection (§6)
src/baseline.jl                non-rigorous inverse-Lanczos / full-SVD reference (§7)
ext/PlotsExt.jl                Plots visualization (loaded when Plots present)
ext/MakieExt.jl                Makie visualization (loaded when Makie present)
test/runtests.jl               smoke + rigor regressions (§10)
bench/                         test-matrix gallery, routing/cost tables, figures (§8–9)
```

**Implementation status (Float64 first):** `setup.jl`, `gaps.jl`, `svd_frame.jl`,
and the `:svd` baseline are implemented in plain `ComplexF64` — the algorithm
skeleton is correct and tested (containment, normal/Hermitian-PSD tightness,
Gerschgorin ordering, eigenvalue sanity; ~3200 assertions in `test/runtests.jl`).
The deliberate next step is to swap the rigorous container in:
`Matrix{ComplexF64}` → `BallMatrix` and `opnorm` →
BallArithmetic's `upper_bound_L1/Linf_opnorm` / `rigorous_svd`. The algebra does
not change. Still stubbed: only the `:lanczos` baseline.

**Box-bisection sweep (`sweep.jl`, guide §6).** `bisect(A; ε, min_halfdiag, cluster=true, θ=0.25)`
(or `bisect(cf::CheapFloor; box, ε, min_halfdiag, ...)`) classifies cells into
`excluded` / `included` / `boundary` (the ε-contour), returning a
`PseudospectrumEnclosure`. **Tiered, cheapest-first** (this is the routing the
paper is about):
1. *cheap exclusion* — the floor `max(gap_tri, gap_svd, gap_recentred)`, using the
   resolvent **maximum principle** on eigenvalue-free cells ([[resolvent-maximum-principle]]):
   `‖(A−zI)⁻¹‖` subharmonic ⇒ certifying `σ_min ≥ ε` on the 1-D boundary certifies
   the whole cell (`_boundary_lower` + `_eigfree` Gershgorin guard), much tighter
   than centre-Lipschitz `σ_min(z₀)−r>ε`;
2. *cheap inclusion* — `sigma_min_upper(z₀)+r<ε` (certified σ_min *upper* bound from
   the Schur-vector residuals, `A·Z` precomputed);
3. *escalation (block tier)* — **only on cells the cheap layers leave unresolved**:
   `gap_cluster` on a cluster, **cached** (O(n³) reorder/re-cert once, not per cell).
   Two modes:
   - `:global` (default) — `gap_cluster` on each coupling-graph component
     (`_build_sweep_clusters`, ≥2-eigenvalue components);
   - `adaptive=true` — **z-adaptive active set** (`_active_set`/`_adaptive_bracket`):
     the failing/almost-failing gaps at `z` (`|z−t̃ᵢ|−τᵢ ≤ active_margin`), cached by
     active-set signature. The far set is then automatically separated, so the block
     is as small as the *local* non-normality. This localizes chain/banded operators
     (which the global graph lumps into one `m=n` cluster) — the transfer-operator
     regime; `active_margin` defaults to the matrix's max tail. A generous margin
     makes the active set vary slowly in `z`, amortizing the reorder over many cells.
   For a whole-matrix cluster the block SVD is the exact σ_min, giving a tight
   bracket for *both* exclusion and inclusion.

`routing_histogram` shows which tier settled each excluded cell. **This is what
makes the non-normal cases work**: without escalation, Grcar/Jordan are ~96%
unresolved (the Gershgorin floor is loose); with it they resolve to a thin contour
(`routing` becomes `cluster`-dominated). Normal matrices resolve entirely on
`triangular`. Cost: escalation pays a per-cell m×m SVD on unresolved cells — cheap
for small clusters, baseline-cost when the cluster is the whole matrix (globally
non-normal). `cluster=false` disables escalation (cheap floors only).

**Plotting** (`ext/MakieExt.jl`, `ext/PlotsExt.jl`): `plot_enclosure(enc; by_layer)`
renders the enclosure (excluded grey / by-layer, included blue, ε-contour red,
eigenvalues ×). Loaded as a package extension when Makie or Plots is present.

**Coupling-graph clustering (`coupling.jl`, guide §4b).** Block membership for the
(future) cluster path is decided once, from the Schur factor, by the off-diagonal
coupling mass — *not* by modulus/annulus. `cluster_components(S; θ, ε_target)`
returns connected components of the graph `|T̃ᵢⱼ| > θ·min(|t̃ᵢ−t̃ⱼ|, ε_target)`
(transitive coupling ⇒ shared block); `percolation_scan` is the θ-selection
diagnostic (stop just above the giant-component transition). This is pure
graph-structural Float64 — no certified arithmetic decides membership; the
certified work happens later on the resulting blocks. The components are exactly
the `select` masks for `ordschur`. Real (quasi-triangular) Schur would also need
conjugate-pair edges; we use complex Schur, so that step is a no-op.

**Cluster / block path (`cluster.jl`, guide §5 / paper §4).** `reorder_cluster(S, idx)`
brings a cluster to the leading block via `ordschur` and **re-certifies the
reordered basis** (`_certify_from_basis` — its `R₂`, `S`, `β`, `κ` are recomputed,
never assumed). It partitions the collapsed `M₀ = Z⁻¹AZ` into `[[P,Q],[R,W]]`;
`gap_cluster(z, cb)` assembles the block-triangular resolvent bound (Thm
`assembly`): tight per-`z` SVD on the small cluster block `P` (`abs`/2×2-Kahan/
`svdvals` — **not** the SVD-frame, which overshoots near the cluster spectrum),
the cheap triangular far gap for `‖W⁻¹‖₂`, and **everything carried**: `κ₂(X̃)`,
the `β` slacks, and the dropped lower-left `R=−S_FC` via the Schur-complement
correction `σ_min(P_SC) ≥ σ_min(P) − ‖Q‖₂‖W⁻¹‖₂‖R‖₂`.

**Pole obstruction → encircling curve** ([[pole-obstruction-encircling-curve]]).
An eigenvalue is a resolvent pole (`σ_min→0`), so no positive `σ_min` lower bound
is certifiable at interior points near a cluster — any cell reaching the pole
contains `σ_min=0`. Never certify exclusion there; instead `curve_lower_bound` /
`encircle_cluster` give a rigorous `σ_min` lower bound over a *whole* pole-free
encircling circle (samples + Lipschitz arc-covering, `min_i f(zᵢ) − 2ρ·sin(π/2n)`).
A positive on-curve bound encloses the cluster (inclusion / eigenvalue count). This
is the contour/inclusion half of the sweep; it ties into BallArithmetic's existing
`rigorous_contour.jl`.

**Residual honesty is non-negotiable** ([[carry-residuals-never-assume-exactness]]).
The Schur basis is only approximately unitary; `S`, `R₂`, `κ₂` are real and are
computed and carried in the Float64 flow with the same structure the BallArithmetic
version uses — so the rigorous port is a drop-in (`Matrix{ComplexF64}`→`BallMatrix`,
`opnorm`→`upper_bound_*`, `svdvals`→`svdbox`). Do **not** set `κ₂=1`, drop `S`, or
treat the Schur factor as exactly unitary. `gap_tri` is honestly transported:
`(σ_min(T̃−zI) − ‖S‖₂)/κ₂` (the raw triangular screen is `gap_tri_schur`).

**SVD-frame certificate (paper §5, `svd_frame.jl`).** An *alternative* σ_min layer:
one verified SVD of `A`, reused across all `z` via the normal-equations frame
`M(z)=(A−zI)ᴴ(A−zI)`. Bounds `σ_min(A−zI)²` directly (no κ transport), with a
Hermitian-Gershgorin gap on `λ_min` minus a z-uniform deviation from the SVD
residuals `P,Q,E`. Distinct from the Schur gaps (overshoot governed by
singular-vector misalignment `W=VᴴU`, not triangular tails). **Correctness note:**
the standalone SVD-frame `.tex` (now deleted, folded into paper §5) printed `G(z)`
*without* the diagonal part of the z-linear term (`−2σ̃ᵢ Re(z̄ W̃ᵢᵢ)`); that term is
essential (it is the whole bound for Hermitian-PSD `A`) and is included in both
the code and the integrated paper §5.

**Combined cheap floor (`combined.jl`, paper §5 "maximum of floors").** Since the
triangular gap and the SVD-frame bound are both certified lower bounds on the same
`σ_min(A−zI)`, their pointwise max is too (Lemma `maxfloor`) — `cheap_floor(A)`
builds both (one Schur + one SVD) and `f(z) = max(gap_tri, gap_svd)`;
`binding_floor` reports which won (the per-cell routing datum). **Empirical
reality** (measured, reflected honestly in the paper): the triangular gap (κ=1,
sandwich-tight) dominates most of the plane; the SVD frame contributes near
**defective/clustered** spectra (e.g. Jordan blocks), rarely elsewhere. The max
never hurts and is the cheap-layer driver for the sweep.

**Key framing decision (from the paper, not implementation.md):** the recentred
gap uses the *z-independent collapsed matrix* `M₀ = Z⁻¹AZ = B⁻¹C` and applies
Gerschgorin to `M₀ − zI`, so per-`z` cost is `O(n)` (centres/radii are fixed).
This is cheaper than implementation.md's affine-`S(z) = S_C − zS_B` framing
(`O(n²)`/point) and exactly equivalent. The free triangular gap is a *direct*
σ_min lower bound (no κ factor) because the Schur basis `Z` is unitary; the
recentred gap pays `κ(Z)` and wins only where the free gap is loose, so
`sigma_min_lower_bound = max(gap_tri, gap_recentred)`.

## Commands

```bash
cd <this repo>

# First time: link the local BallArithmetic (dev, not the registered version,
# so the engine tracks in-progress kernel work)
julia --project=. -e 'using Pkg; Pkg.develop(path="/home/isaia/Code/BallArithmetic.jl")'

# Load / REPL
julia --project=. -e 'using RigorousPseudospectra'

# Tests — direct run is primary (Pkg.test sandboxing re-resolves and can fail
# when the depot has no General registry; the direct run avoids the sandbox)
julia --project=. test/runtests.jl

# Format (match BallArithmetic: SciML style)
julia -e 'using JuliaFormatter; format(".")'
```

Requires Julia ≥ 1.10. The kernel's own verified-linalg tests are run inside
`~/Code/BallArithmetic.jl` (`Pkg.test()` there).

## The algorithm, in one breath (see `implementation.md` §3–6)

Goal: a certified `z ↦ lower bound on σ_min(A − zI)`, cheap over a region, in
ℓ¹, ℓ∞, and interpolated ℓ². Scope is **dense, moderate n**; sparse/large-scale
is explicitly out of scope.

Two-phase rigor pattern throughout: (a) an approximate Float64 phase (`schur`,
`eigen`, `inv`, `\`) produces *candidates* `X̃, Ỹ, T̃, S̃`; (b) a **ball-arithmetic**
phase bounds the *residual* of those candidates. Never form a certified inverse —
inverses enter only via Neumann/residual bounds (`‖(I+R₂)⁻¹‖ ≤ 1/(1−‖R₂‖)`, valid
only when `‖R₂‖ < 1`; **verify this and refuse to proceed otherwise**).

Efficiency core (bridge theorem): `R₁(z) = Y(A−zI)X̃ = C − zB`. Form `B = YX̃` and
`C = YAX̃` **once** (`O(n³)`, z-independent); per-`z` work is the affine update
`C − zB` (`O(n²)`), never a matrix product. Recentring data is also affine:
precompute `S_C = (I+R₂)⁻¹C`, `S_B = (I+R₂)⁻¹B`, so `S(z) = S_C − z S_B` with no
solve in the loop.

Gap ladder (each a certified lower bound on `σ_min`), cheapest-first:
1. **Free triangular** (Lemma 2.1, `O(n)`) — diagonal + fixed tails, no residual.
2. **Uniform Miyajima** (Thm `diag`, `O(n²)`).
3. **Recentred per-row** (Thm `resolvent-row`, `O(n²)`) — sharpest scalar layer.
4. **Interpolated ℓ²** (Riesz–Thorin from ℓ¹/ℓ∞; `max(0,·)` before any sqrt).
5. **Cluster / block path** (§5) — reorder cluster to leading block, re-certify,
   far-gap + localization, σ_min of the small `m×m` block (Kahan for `m=2`,
   cancellation-free; verified SVD for `m≥3`), block-triangular assembly.

Sweep is **box bisection, not a uniform grid** (§6): `z ↦ σ_min` is 1-Lipschitz,
so one certified centre evaluation settles a whole cell (`σ_min ≥ σ_lo(z₀) − r`).
The free layer settles most cells; expensive layers run only on a thin annulus
around the ε-contour. The central benchmark is the **routing histogram** — what
fraction of cells/area each layer settles (expect free ≫ recentred ≫ cluster).

## Rigor invariants (correctness, not style)

- Every quantity feeding a certified bound is computed in ball arithmetic with
  outward rounding; Float64 is for candidates only.
- Certified `σ_min` lower bound must **never exceed** `svdvals(A−zI)[end]` — the
  primary regression test.
- At a true eigenvalue the certified gap must be `≤ 0`.
- Routines must refuse when `‖R₂‖ ≥ 1`.
- Affine consistency: `S_C − z S_B` matches a direct `(I+R₂)S = R₁(z)` solve within `β`.
- Defective matrix (Jordan block size `m`): the bound tracks `|z−λ|^m`, not
  `|z−λ|` — the whole reason for using `σ_min` over eigenvalue distance.

See `implementation.md` §10 for the full regression list and §8 for the
test-matrix gallery (Grcar, Demmel, Jordan/Kahan/Frank, convection–diffusion, the
Gauss witness) — each stresses one specific feature.
