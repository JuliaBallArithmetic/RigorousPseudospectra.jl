# Implementation guide — certified pseudospectrum via Miyajima certification

This document specifies a Julia package that implements the certified
pseudospectrum engine of `miyajima_pseudospectra.tex` and benchmarks it against
the classical grid / inverse-Lanczos approach. It is written to map each result
in the note to a concrete routine, and to fix the rigor conventions so the
output is a genuine enclosure rather than an approximation.

The deliverable of the whole package, matching the note's bridge theorem, is a
**certified function `z ↦ lower bound on σ_min(A − zI)`** (equivalently an upper
bound on the resolvent norm), evaluated cheaply over a grid of `z`, in the
`ℓ¹`, `ℓ∞`, and interpolated `ℓ²` norms.

Scope: **dense, moderate `n`**. The sparse / large-scale case is explicitly out
of scope (the eigenvector/Schur basis densifies; see the note's scope remark).
The operator-level transport (transfer operators, DFLY) is a separate paper and
is *not* part of this package.

---

## 1. Rigor conventions

Everything that contributes to a certified bound must be computed with
outward-rounded interval/ball arithmetic. Approximate (Float64) computation is
used only to produce *candidates* (`X̃`, `Ỹ`, `T̃`, `S̃`) that are then bounded
rigorously.

- Use `IntervalArithmetic.jl` for the certified layer. Wrap candidate matrices
  as `Matrix{Interval{Float64}}` (or `Complex{Interval{Float64}}`) only at the
  point where a rigorous bound is formed.
- Two-phase pattern throughout: **(a)** approximate phase in Float64
  (`LinearAlgebra`: `schur`, `inv`, `\`), then **(b)** certification phase in
  interval arithmetic that bounds the residual of the approximate object.
- Operator norms: `opnorm(·, Inf)` = max abs row sum, `opnorm(·, 1)` = max abs
  column sum. Implement certified versions that take `abs` of interval entries
  and sum with directed rounding (do **not** trust a non-interval `opnorm`).
- Never form a certified inverse directly. Inverses enter only through
  Neumann/residual bounds (`‖(I+R₂)⁻¹‖ ≤ 1/(1−‖R₂‖)` when `‖R₂‖<1`).
- Beware the `NG` (non-guaranteed) flag pitfalls discussed in the
  RigorousInvariantMeasures.jl / IntervalArithmetic.jl interaction: assert
  guaranteed intervals at the boundaries of certified routines.

---

## 2. Package layout

```
CertifiedPseudospectra/
├── Project.toml
├── src/
│   ├── CertifiedPseudospectra.jl     # module, exports
│   ├── setup.jl                      # z-independent precomputation (§ bridge)
│   ├── residual.jl                   # R₁(z)=C−zB, R₂, S̃, β  (Lemma 1.3)
│   ├── gaps.jl                       # all the g(z) variants
│   ├── coupling.jl                   # §4c Rump conncomp clustering + R̃ precond
│   │                                 #   (§4b Schur-frame coupling graph = fallback)
│   ├── solve.jl                      # §4e float Newton + merge detect + certify
│   ├── diag_bound.jl                 # Thm 1.1 uniform + Cor 1.2 individual
│   ├── recentred.jl                  # Thm 1.5 recentred per-row (+ ℓ¹, ℓ²)
│   ├── triangular.jl                 # Lemma 2.1 free triangular gap
│   ├── cluster.jl                    # §3 far gap, localization, σ_min block
│   ├── assembly.jl                   # Thm 3.x block-triangular full resolvent
│   ├── twobytwo.jl                   # closed-form σ_min (Kahan) for 2×2 blocks
│   ├── sweep.jl                      # grid evaluation, adaptive refinement
│   └── baseline_lanczos.jl           # classical inverse-Lanczos (NON-rigorous)
├── bench/
│   ├── matrices.jl                   # test matrices (Trefethen gallery + Gauss)
│   ├── run_benchmarks.jl             # cost + tightness tables
│   └── figures.jl                    # free-plot + refinement-region figures
└── test/
    └── runtests.jl                   # rigor regression tests
```

---

## 3. Setup phase (compute once) — maps to the bridge theorem

`setup.jl` builds all `z`-independent data. For a finite matrix `A ∈ ℂ^{n×n}`:

```
struct CertSetup
    A                       # interval matrix (the certified input)
    X̃                      # approx eigenvectors OR Schur basis (Float64)
    Tdiag                   # diagonal d̃ (diagonalizable) or full T̃ (Schur)
    Y                       # approx inverse of X̃  (Float64)
    B                       # interval enclosure of Y*X̃           (= I + R₂)
    C                       # interval enclosure of Y*A*X̃
    R2                      # interval enclosure of B − I
    nrmR2_inf, nrmR2_1      # certified ‖R₂‖_∞ , ‖R₂‖_1
    S̃                      # approx solve of (I+R₂) S = R₁(0)  ... see note
    β_inf, β_1              # certified residual slacks (Lemma 1.3)
    κ_inf, κ_1              # certified ‖X̃‖·‖X̃⁻¹‖ via ‖Ỹ‖/(1−‖R₂‖)
    tails_inf, tails_1      # off-diagonal row/col tails of T̃ (triangular case)
    Tij                     # strict upper part, for tails and cluster coupling
end
```

Key efficiency point (note §1.2, §3.3): `R₁(z) = Y(A − zI)X̃ = C − zB`, so `C`
and `B` are formed **once** here (each one `O(n³)`), and per-`z` work is the
affine update `C − zB` (`O(n²)`), never a matrix product.

Steps:
1. Approximate: `F = schur(A_float)` → `X̃ = F.Z`, `T̃ = F.T` (Schur), or
   `eigen` for the diagonalizable variant. `Y = inv(X̃)` (Float64).
2. Certify `B = Y*X̃`, `C = Y*A*X̃` in interval arithmetic (directed rounding).
   `R2 = B − I`. Compute `nrmR2_inf`, `nrmR2_1` (certified).
3. **Verify** `nrmR2_* < 1`; if not, the certificate is unavailable (report,
   do not proceed — mirrors Miyajima's Algorithm 1 Step 4 failure mode).
4. `S̃`: approximate solve `(I + R₂_float) S̃ = R₁(0)_float` where
   `R₁(0) = C` (note `R₁(0)=YAX̃`). Then certify the residual
   `T_resid = (I+R₂) S̃ − C` and set `β_p = ‖T_resid‖_p /(1 − ‖R₂‖_p)`
   (Lemma 1.3). **Note:** `S̃` here is the proxy for `S=(I+R₂)⁻¹R₁(0)`; for the
   `z`-dependent recentring you need `S(z)=(I+R₂)⁻¹R₁(z)`. Two options:
   - (cheap) recompute `S̃(z)` per `z` by one approximate solve — `O(n²)` if you
     keep a factorization of `I+R₂_float` from setup;
   - (cheapest) since `R₁(z)=C−zB`, linearity gives `S(z)=S_C − z S_B` with
     `S_C=(I+R₂)⁻¹C`, `S_B=(I+R₂)⁻¹B`. Precompute `S̃_C, S̃_B` once and certify
     both residuals → recentring data is then itself affine in `z`. **Prefer
     this**; it keeps per-`z` at `O(n²)` with no solve in the loop.
5. `κ_p = ‖X̃‖_p · ‖Ỹ‖_p /(1 − ‖R₂‖_p)` (certified), using `‖X̃⁻¹‖_p ≤
   ‖Ỹ‖_p/(1−‖R₂‖_p)`.

### 3.1 Escalation ladder and setup-time gating (master control flow)

The floors form a ladder of increasing robustness-and-cost, descended only as far
as the matrix forces. **All gates are setup-time scalars you already have**, so
the descent is decided before sweeping, and the expensive factorizations are
computed lazily (only when a gate or a region demands them).

```
eigs  →  VBD/block  →  Schur  →  SVD
(rung 1)  (rung 2)     (rung 3)   (rung 4)
```

- **Rung 1 — eigs (diagonal residue, §4c/§4d).** Default. Cheapest per-`z`
  (`O(n)`, fixed radii, `z`-invariant residuals — Remark transeig of the SVD
  note), tightest gap (`O(‖R₁‖)` radii). **Gate:** certified `‖R₂‖ < 1`
  (validity — proves `X̃` nonsingular) AND `κ(X̃)` below tolerance (tightness —
  the bound carries this prefactor). Both known at setup.
- **Rung 2 — VBD/block (§4c, §4d).** When `κ(X̃)` is too large because a *few*
  eigenvectors are near-parallel (a cluster), don't abandon the eigenbasis —
  block the offending eigenvalues (Rump `conncomp`) and treat the cluster's
  invariant subspace as a unit with `σ_min(Pⱼ − zI)`. Still a similarity, still
  `z`-invariant (shift passes through blocks). Where separation localizes the
  conditioning blowup.
- **Rung 3 — Schur (§4b).** When even the block eigenbasis can't be conditioned
  (strongly non-normal, or genuinely defective), fall to the unitary Schur form:
  always exists, stable, `κ=1`, no eigenbasis needed. Lazily computed — only on a
  global gate failure (`‖R₂‖≥1` / huge `κ`) or the first inconclusive
  non-spectrum region. Once computed, it serves *both* the block (VBD-style
  `σ_min` on a reordered cluster) and the far (triangular gap) roles from one
  factorization.
- **Rung 4 — SVD frame.** Last resort, where even the Schur triangular gap is too
  loose (large tails). Targets `σ_min²` directly in orthonormal bases (`κ=1`,
  tight) but is **not** a similarity, so `z`-quadratic and anchor-local (Remark
  transsvd) — hence last, and applied locally on the region that needs it.

**The descent is monotone in two coupled axes:** robustness to ill-conditioning
↑ and `z`-awkwardness ↑ (rungs 1–3 are shift-invariant similarities; rung 4 is
the non-similarity tight-but-`z`-local option). For a well-conditioned eigenbasis
— plausibly the transfer-operator coarse-matrix case — you stop at rung 1 for
almost the whole sweep, dropping to rung 2 only around resonance clusters and to
3/4 essentially never. **The routing histogram (which rung binds where) is the
headline benchmark** (§9) and measures directly which regime a matrix sits in.
NJD never appears (§4c): `σ_min` on the block replaces it.

Each returns a **certified lower bound** on `σ_min(A − zI)` (or its norm-specific
surrogate). All take `z` and the `CertSetup`. Order them by cost.

### 4.1 Free triangular gap — Lemma 2.1, note §3.3  (cost `O(n)`)
```
g_tri_inf(z, S) = minimum( abs(z − d̃_i) − tails_inf[i]  for i in 1:n )
g_tri_1(z, S)   = minimum( abs(z − d̃_j) − tails_1[j]    for j in 1:n )
```
Uses only `T̃` diagonal and fixed tails — **no residual data**, nothing per-`z`
but the centres. This is the free first-plot layer. Returns may be negative
(inconclusive); that is the refinement trigger.

### 4.2 Uniform Miyajima gap — Thm 1.1  (cost `O(n²)`)
```
g_unif_p(z) = min_i|z − d̃_i| − ‖S(z)‖_p        # diagonalizable case
```
with `S(z) = S_C − z S_B` (interval), `‖·‖_p` certified.

### 4.3 Recentred per-row gap — Thm 1.5  (cost `O(n²)`)
```
# row i:  |z − (d̃_i − S(z)_ii)| − Σ_{j≠i}|S(z)_ij|   − β_p
g_rec_inf(z) = min_i ( abs(z − (d̃_i − S(z)[i,i]))
                       − (sum_abs_row(S(z), i) − abs(S(z)[i,i])) ) − β_inf
```
`ℓ¹` version transposes rows↔cols. This is the sharpest scalar layer.

### 4.4 Interpolated `ℓ²` gap — Cor (Riesz–Thorin)
```
resolvent_2(z) ≤ sqrt( (κ_1/g_rec_1(z)) * (κ_inf/g_rec_inf(z)) )
# i.e. certified σ_min lower bound:
g_2(z) = sqrt( g_rec_1(z) * g_rec_inf(z) ) / sqrt(κ_1 * κ_inf)   # see note
```
(Take the `inf` over the geometric-mean and the two endpoints; never let a
negative endpoint into the sqrt — guard with `max(0,·)`.)

### 4.5 Triangular/Schur gap on `T̃ − S` — note §3.2–3.3  (cost `O(n²)`)
For the Schur (non-diagonal) case, the gap subtracts both the genuine `T̃`
off-diagonal tails and the `S` off-diagonal:
```
g_schur_inf(z) = min_i ( abs(z − (d̃_i − S(z)[i,i]))
                         − Σ_{j≠i}|(T̃ − S(z))[i,j]| ) − β_inf
```

---

## 4b. Clustering by the coupling graph — `coupling.jl`

> **Frame note.** §4b describes clustering in the *Schur* frame (coupling graph on
> the triangular factor, conjugate-pair edges for `ordschur`). The project has
> since committed to the **diagonal-residue (eigendecomposition) frame** — see
> §4c, which is the route to implement. §4b is retained as the orthonormal-frame
> alternative (defective-safe, `κ=1`, looser gap) to fall back to only if the
> eigenvector frame proves ill-conditioned globally. Read §4c first.

How the eigenvalues are grouped into blocks is decided **once**, from the first
Schur factor, by the structure of the off-diagonal mass — *not* by an annulus /
modulus heuristic. The off-diagonal entries `T̃_ij` (i<j) are exactly what a
per-eigenvalue (scalar) bound cannot absorb, so eigenvalues linked by
above-threshold coupling must share a block. This clusters by the quantity that
actually drives resolvent looseness, rather than by modulus (a proxy that only
correlates with it).

### Construction
```
# G: boolean adjacency on the strict upper triangle of the Schur factor T̃
G[i,j] = ( abs(T̃[i,j]) > θ_ij )           for i < j
```
Clusters = connected components of `G` (treating it as undirected). Connected
components automatically capture **transitive** coupling: a chain
`i → k → j` of moderate couplings groups `i` and `j` even with no direct large
`T̃_ij`, which is correct because the resolvent sees the composite path.

### Threshold `θ_ij` — tie it to the question, make it gap-relative
A bare absolute threshold is scale-dependent. The coupling that matters is the
off-diagonal mass *relative to the gap it bridges* and *to the pseudospectral
level being certified*:
```
θ_ij = θ * min( abs(t̃_i − t̃_j), ε_target )      # θ ∈ (0,1), e.g. 0.1–0.5
```
- gap-relative (`|t̃_i − t̃_j|`): `T̃_ij` is dangerous when comparable to the
  eigenvalue separation it sits above;
- ε-tied (`ε_target`): couplings below the resolution you are enclosing need not
  be blocked, so a coarse pseudospectrum yields fewer/larger thresholds, a fine
  one resolves more couplings. The clustering thus adapts to the question.

### Percolation caveat
The threshold has a percolation character. Too low → the graph connects into one
giant component (a dense n×n block; no win). Too high → genuine couplings are
missed. Detect the sweet spot: sweep `θ` downward and watch component sizes;
stop **before** the giant-component transition (where the largest component
suddenly jumps). Report the chosen `θ` and the resulting component-size
histogram — this is diagnostic of the matrix's non-normality structure.

### Real Schur: force conjugate-pair edges
Before taking components, add edges linking the two indices of each 2×2
quasi-diagonal block (conjugate pairs are atomic for `ordschur`). This guarantees
a pair is never split across a block boundary.

### Graph gives membership; modulus (optional) gives traversal order
The components give cluster **membership**. To realize a cluster as a contiguous
leading block, feed its indices as the `select` mask to `ordschur` (§5). Modulus
is *not* used for clustering, but may still order the **sweep traversal** (e.g.
process the leading-eigenvalue region first for a transfer operator) — the two
roles are independent: graph for membership, geometry for traversal.

### Adaptive threshold (ties clustering to certificate outcome)
Start with a high `θ` (few, small clusters). If a cluster's block certificate
comes back inconclusive — the block σ_min is still small relative to its
coupling to the nominal far set — **lower `θ` for that region** to absorb the
offending neighbours into the block, and recertify. This is the graph-level
analogue of the bisection's escalate-vs-subdivide: the threshold is driven by
the certificate result, not guessed up front.

```
function cluster_components(T̃, t̃; θ, ε_target)
    n = size(T̃,1); G = falses(n,n)
    for j in 1:n, i in 1:j-1
        θij = θ * min(abs(t̃[i] - t̃[j]), ε_target)
        if abs(T̃[i,j]) > θij; G[i,j] = G[j,i] = true; end
    end
    force_conjugate_pairs!(G, T̃)          # real Schur only
    return connected_components(G)          # Vector of index-sets
end
```

---

## 4c. Diagonal-residue frame and Rump-style clustering — `coupling.jl`, `cluster.jl`

This is the committed route. We work in the **eigenvector (diagonal-residue)
frame**, not the Schur frame: accept the prefactor `κ(X̃)` in exchange for the
*tight* gap (residue radii `O(‖R₁‖)` instead of the loose `O(1)` Schur tails).
The clustering, scalar-vs-block gate, and block treatment all follow Rump's
`verifyeigall` (Ru22a) almost directly, with the per-`z` block evaluation being
a certified `σ_min(Pⱼ − zI)` in place of his fixed eigenvalue inclusion.

### Why the eigenvector frame is viable (the `u^{1/k}` regularization)
Rump's key empirical fact: even for a true `k×k` Jordan block, `eig` returns
columns linearly dependent only up to the cluster sensitivity `u^{1/k}`, so
`cond(W) ~ u^{−1/k}` is **large but finite**, not singular. Floating point
regularizes the ill-posed Jordan structure into a finite-conditioned cluster.
Hence `W⁻¹BW` is computable and almost diagonal, and the diagonal-residue floor
is usable on essentially all matrices — `κ(X̃)` blows up only in special
constructed cases (integer Jordan/Frank matrices, Ru22a Table 12).

### Frame (Rump `transform`, our setup §3 already builds this)
`A = W⁻¹BW`, almost diagonal: `A = D + E`, `D` the diagonal, `E` the small
off-diagonal (this is our `S`/`S̃` residue). One Newton step on the eigenvectors
before certifying tightens `E`; `verifylss(W, intval(B)*W)` simultaneously proves
`W` nonsingular and encloses `A`. **One `O(n³)` step; everything below is `O(n²)`.**

### Separation-driven clustering (Rump step 2 — = `conncomp(graph(dist))`)
```
dist  = mig(d .- d') .<= tol * norm(A,Inf)   # tol ≈ machine-precision-relative
mu    = connected_components(graph(dist))     # clusters = components
```
Clusters are connected components of the close-eigenvalue graph: eigenvalues too
close to separate at the working precision must be one block. Transitive (a chain
of close eigenvalues merges). This *is* the coupling-graph idea of §4b, but the
threshold is intrinsic — `mig(dᵢ−dⱼ) ≤ tol·‖A‖`, "indistinguishable at this
arithmetic" — rather than a chosen `θ`. Parameter-free.

### The reciprocal-distance preconditioner R̃ (Rump step 4 — separation, explicit)
```
R̃ = 1 ./ (D .- D')        # off-diagonal: 1/(dᵢ − dⱼ)
R̃[diagonal]      = -1      # neutralize diagonal
R̃[within-cluster] = -1     # neutralize within each cluster block
```
`R̃` is the entrywise inverse of the (almost-diagonal) Newton Jacobian — the
classic Sylvester-operator inverse `1/(λᵢ−λⱼ)`. This is why the verification is
`O(n²)`: a Hadamard product, not a linear solve. **Separation enters here
quantitatively**: well-separated pairs → small `R̃ᵢⱼ`; close pairs → large
`R̃ᵢⱼ`. Setting within-cluster entries to `−1` is the explicit *refusal to
separate within a cluster* (the ill-posed part) — you block it instead.

### The O(n²) self-map test (Rump Thm 2.2)
With `Y = X_O X_D − E − E X_O` (split by clusters), `Z = R̃ ∘ Y`, check the
Krawczyk self-mapping `Z Vᵢ ⊆ int(X Vᵢ)` per cluster `i` (epsilon-inflation to
give interior room). Success certifies a genuine Jordan block `M̂ᵢ` with
invariant subspace. Inter-cluster distinctness is certified by
`max ρ(Z) < 1` (cheap via `ρ(mag Z) < 1` or `max(‖Z‖₁,‖Z‖∞) < 1`).

### Guess–verify–merge–recurse (Rump step 6 — the adaptive refinement)
The clustering guess is **not** fixed. Loop: epsilon-inflate, count columns
satisfying the self-map; if not all succeed and the count stalls, **recurse on
the failed columns** — merge the eigenvalues that couldn't be separated into
larger clusters and retry. Terminates when merging stops increasing the verified
count. This is the rigorous form of the "adaptive threshold" of §4b: separation
guesses the clusters, the verification outcome refines them.

### Scalar-vs-block gate = where κ would blow up
The gate between treatments is exactly the separation test:
- **Separated eigenvalue** (singleton component): scalar **recentred-disc floor**
  `gₑᵢ𝓰(z) = minᵢ(|z − (d̃ᵢ − Sᵢᵢ)| − Σⱼ≠ᵢ|Sᵢⱼ|)`, `O(n)` per `z`, `κ` benign.
- **Cluster** (`|μᵢ| > 1`): **block treatment**, certified `σ_min(Pⱼ − zI)`.
Clustering blocks precisely the eigenvalues whose eigenvectors are near-parallel
— i.e. where scalar `κ(X̃)` would otherwise inflate. Blocking the invariant
subspace (not separating it) is what keeps the prefactor controlled.

### What replaces Rump's eigenvalue inclusion: the resolvent σ_min
Rump computes a fixed eigenvalue/subspace inclusion. We reuse his frame,
clustering, and `R̃`/self-map machinery, but the per-`z` block deliverable is the
certified `σ_min(Pⱼ − zI)` over the sweep — the resolvent surface, not a fixed
inclusion. The blocks `Pⱼ` are the invariant-subspace blocks in the eigenvector
frame (no Schur, no `ordschur`).

### NO numerical Jordan decomposition
Avoided entirely, and the avoidance is empirically vindicated by Ru22a §3.3:
Miyajima's NJD variant (`VAE_NJD`) hits infinite loops and Matlab errors and runs
up to 356× slower, because the Jordan decomposition is ill-posed. We never need
it: `σ_min(Pⱼ − zI)` on the coupled block handles defective clusters without any
Jordan structure (`σ_min ~ |z−λ|^m` falls out of the SVD of the block).

---

## 4d. The splittable / non-splittable dichotomy — the determinant trap

The single most important correctness point for the block path.

**Splittable cluster (well-separated, diagonalizable block).** `σ_min(Pⱼ − zI)`
is explicit:
```
σ_min(Pⱼ − zI) = minᵢ |z − λᵢ|        # the MINIMUM distance
```
the recentred scalar floor. Cheap, no SVD.

**Non-splittable cluster (coupled / near-defective block).** The block is **NOT
diagonal** — that is *why* it could not be split. It carries irreducible coupling
`Pⱼ = λ̂ⱼ I + Nⱼ`. Here `σ_min` is **neither** the min distance **nor** the
product of distances; it requires a **verified SVD on the m×m block**.

> **Determinant trap (do not do this).** `∏ᵢ|z − λᵢ| = |det(Pⱼ − zI)|` controls
> the **product** of all singular values, not the **smallest**. The resolvent
> norm is `1/σ_min`, NOT `1/|det|`. For `[[λ,t],[0,λ]]`: `det = (λ−z)²` but
> `σ_min ≈ |z−λ|²/t` (smaller by `1/t`). Using `∏ᵢ|z−λᵢ|` would **underestimate**
> the resolvent norm and **void the certificate** — exactly in the coupled case
> where it matters most. This is the Jordan lesson: eigenvalue-derived
> quantities (including products) miss the non-normal sensitivity; only `σ_min`
> sees it.

So: splittable → explicit `min`-distance; non-splittable → verified
`σ_min(Pⱼ − zI)`, never the determinant/product.

### The huge-Jordan-block limit (honest degradation)
`σ_min` is the right object but "good enough" has a precision ceiling. For an
`m×m` Jordan block the sensitivity is `u^{1/m}`:
```
m=2: u^{1/2} ≈ 1e-8     m=5: u^{1/5} ≈ 1e-3     m=10: u^{1/10} ≈ 0.025
```
Near the cluster `σ_min ~ |z−λ|^m` collapses, and the verified `σ_min` enclosure
becomes as wide as the `u^{1/m}` cluster radius. The certificate stays **rigorous
but uninformative** — it goes *loose*, not *wrong* (the determinant product would
have gone confidently wrong). Handling:
1. **Detect and report.** `m` and `u^{1/m}` are known at clustering time; if
   `u^{1/m} > ε_target`, flag the region "unresolvable at this precision" rather
   than returning a vacuous bound. (Rump's "fail" is honest, not a bug.)
2. **Extended precision on the block.** The bad region is localized to the small
   `m×m` cluster: recompute *just that block* in BigFloat (GenericSchur.jl) to
   buy `u^{1/m}` back down; the rest of the sweep stays Float64. Affordable
   because the block is tiny even when `m` is large relative to precision.
3. Otherwise accept the wide-but-valid enclosure with the cluster flagged.

The virtue worth keeping: at its limits the method degrades *gracefully and
honestly* (loose, valid, flagged) tied to the genuine `u^{1/m}` sensitivity, with
per-block extended precision as the escape hatch.

---

## 4e. The solve: float Newton + merge detection + single certificate — `solve.jl`

The eigenvalue/cluster solve and the pseudospectrum certificate are **one
artifact**. Rump's interval Krawczyk loop is split into (i) a *float* Newton
iteration (Krawczyk = interval Newton, so run it in floating point as the
candidate producer) with separation preconditioner and conservative merge, then
(ii) a *single* Miyajima two-residual certification at the end. No per-step
certification, no per-step inverse, no per-step recompute of `W⁻¹BW`.

### Why deferring certification is safe (the asymmetry)
The merge-recurse needs a "did this cluster separate?" signal. Rump's is the
certified self-map failure; we use **float surrogates** (below). Correctness does
NOT rest on them, because of an asymmetry:
- **under-clustering** (declared two things separate that aren't) → caught by the
  *end* certificate as a recentred-disc **overlap** → re-merge and recompute;
- **over-clustering** (merged two separable things) → **harmless**: a slightly
  larger block, looser bound, more `σ_min` work, but always valid.
So the float merge only needs to be *conservative* (merge when in doubt); the
single end certificate guarantees correctness.

### In-place Newton step (no recompute, no inverse)
`A` almost-diagonal, `d = diag(A)`. Per step: entrywise reciprocal-distance
correction `X` (the Sylvester/commutator inverse, within-cluster neutralized),
then a **near-identity similarity** update of `A` (and optionally `W`). `X` small
⟹ `(I+X)⁻¹ ≈ I − X`; never form an inverse.
```
function newton_step!(A, d, clusters; cluster_of)
    n = size(A,1); X = zeros(eltype(A), n, n)
    for j in 1:n, i in 1:n
        if i != j && cluster_of[i] != cluster_of[j]
            X[i,j] = A[i,j] / (d[i] - d[j])     # entrywise; O(n²)
        end                                      # diag & within-cluster → 0
    end
    A .= (I - X) * A * (I + X)                    # near-identity similarity
    # optional, only if eigenvectors wanted live:  W .= W * (I + X)
    d .= diag(A)                                  # refresh (cheap)
    return norm_offdiag(A)                        # convergence monitor
end
```
Cost: `O(n²)` to form `X`; the similarity is a near-identity update (dense
`O(n³)` if naive, but one or two steps suffice à la Rump, so negligible; exploit
small-`X`/almost-diagonal for strict `O(n²)`).

### Merge detection (float surrogates — three signals)
Conservative: any signal fires ⟹ merge the offending pair/columns into one
cluster. All three are pure float, no certification.
```
function should_merge(i, j, A, d, W; u = eps())
    # 1. Separation below the sensitivity floor for the (tentative) block size.
    #    A guess of distinct eigenvalues at separation δ is meaningless if
    #    δ ≲ u^{1/k}‖A‖ (k = size of the cluster they'd form).
    k   = tentative_cluster_size(i, j)
    sep_floor = u^(1/k) * norm(A, Inf)
    if abs(d[i] - d[j]) <= sep_floor;            return true; end

    # 2. Newton preconditioner blowup: the within-step factor 1/(dᵢ−dⱼ) is
    #    large relative to the off-diagonal it multiplies → step not contracting.
    if abs(A[i,j]) / abs(d[i] - d[j]) >= 1;      return true; end

    # 3. Eigenvector-subblock conditioning spike: columns i,j of W becoming
    #    near-dependent (the σ_min of the 2-col block collapses toward u^{1/k}).
    if smallest_sv(W[:, [i,j]]) <= sqrt(u) * norm(W[:, [i,j]]); return true; end

    return false
end
```
Signal 2 is exactly *why* Rump neutralizes within-cluster `R̃` to `−1`: the need
to neutralize **is** the blowup signal. Signal 1 is `mig(d−d') ≤ tol` made
adaptive to cluster size `k` (tol = `u^{1/k}‖A‖`, not a fixed `1e-14`).

### The solve loop (guess → Newton → merge → recurse)
```
function solve_clusters(B; maxsteps = 3)
    W, D = eig(B)                                  # numerical eigendecomposition
    A    = transform_to_almost_diag(B, W)          # one O(n³): A ≈ W⁻¹BW
    d    = diag(A)
    clusters = conncomp_close(d, A)                # initial guess (§4c)

    for step in 1:maxsteps
        cluster_of = membership(clusters, length(d))
        off = newton_step!(A, d, clusters; cluster_of)

        # conservative merge sweep on current off-diagonal mass
        merged = false
        for j in 1:length(d), i in 1:j-1
            if cluster_of[i] != cluster_of[j] && should_merge(i, j, A, d, W)
                clusters = merge_clusters(clusters, i, j); merged = true
            end
        end
        (off < tol && !merged) && break            # converged & stable
    end
    return (A = A, W = W, d = diag(A), clusters = clusters)
end
```

### Single end certification (freeze, then certify once — §3, §4c)
`W` and `A` are now **frozen**. Compute the two residuals ONCE; this certificate
*is* the certified resolvent data.
```
function certify!(sol, B; Y = approx_inv(sol.W))
    W, A = sol.W, sol.A
    R₂ = I - Y*W                                   # interval: nonsingularity + ‖W⁻¹‖
    @assert opnorm(R₂) < 1                          # else escalate precision (§4d)
    invW_bound = opnorm(Y) / (1 - opnorm(R₂))       # ‖W⁻¹‖  → resolvent prefactor κ
    R₁ = Y*(intval(B)*W - W*A)                      # interval: accuracy residual
    S̃  = solve_approx(I + R₂, R₁)                    # candidate residue (= (I+R₂)⁻¹R₁)
    β  = opnorm((I + R₂)*S̃ - R₁) / (1 - opnorm(R₂))  # slack: ‖S − S̃‖ ≤ β  (note Lem 4.1)

    # CORRECTNESS CHECK: declared-separate clusters must be certifiably separate.
    centres = sol.d .- diag(S̃)                      # recentred d̃ᵢ − S̃ᵢᵢ
    radii   = offdiag_rowsums(S̃) .+ β               # Σⱼ≠ᵢ|S̃ᵢⱼ| + β
    if discs_overlap_across_clusters(centres, radii, sol.clusters)
        # under-clustering detected → merge overlapping clusters, re-solve
        return :remerge
    end
    return (W=W, A=A, S̃=S̃, β=β, invW_bound=invW_bound,
            centres=centres, radii=radii, clusters=sol.clusters)   # the certificate
end
```
On `:remerge`, merge the overlapping clusters and re-run `solve_clusters` /
`certify!` (rare; well-separated spectra pass first time). The returned
certificate feeds directly into the resolvent layer: recentred discs
(`centres`/`radii`) for singletons, `σ_min(Pⱼ − zI)` for `|cluster|>1`, scaled by
`invW_bound` — the certified pseudospectrum surface (§4f, §6).

---

## 4f. Per-block far-field floor — `cluster.jl`

Each cluster contributes a cheap, `z`-uniform certified lower bound on its
`σ_min(Pⱼ − zI)` valid on the whole far field, computed **once** per block — so
only a small near-field disc needs the per-`z` SVD.

Write `Pⱼ = cⱼ I + Nⱼ` (`cⱼ` the cluster centre, `Nⱼ = Pⱼ − cⱼI` the deviation
from scalar — the block's internal coupling / non-normality). Precompute the
operator norm `‖Nⱼ‖` once. Then by a Neumann bound on `(Pⱼ − zI)⁻¹ =
(cⱼ−z)⁻¹(I − (cⱼ−z)⁻¹Nⱼ)⁻¹`, for `|z − cⱼ| > ‖Nⱼ‖`:
```
σ_min(Pⱼ − zI)  ≥  |z − cⱼ| − ‖Nⱼ‖          (far field, growing, no per-z SVD)
```
- **Far field** `|z − cⱼ| > ‖Nⱼ‖`: use the formula above — safe, grows with
  distance, one `‖Nⱼ‖` computation certifies it everywhere.
- **Near field** `|z − cⱼ| ≤ ‖Nⱼ‖`: Neumann fails; fall back to the verified
  `σ_min(Pⱼ − zI)` per `z` (or per cell with 1-Lipschitz padding, §6).

**Warning (determinant trap, again).** The far bound uses the *operator norm*
`‖Nⱼ‖`, NOT eigenvalue distances or `∏|z−λᵢ|`. For a fat Jordan block `‖Nⱼ‖` is
order 1, so the "transparent" radius is honestly large — the bound degrades into
a *bigger excluded disc*, not a wrong number.

**Tie-in to the maximum principle (Approach 1′, §6).** Take the excluded discs of
the global-bound construction to be `{|z − cⱼ| ≤ ‖Nⱼ‖}`. Outside them every block
contributes its cheap far-field floor; the small `‖Nⱼ‖`-disc is the honest
expensive zone (per-`z` SVD), and is small exactly when the block is mildly
non-normal.

`σ_min(Pⱼ)` itself (or at any anchor `z₀`) gives instead a *near*-field bound via
1-Lipschitz, `σ_min(Pⱼ − zI) ≥ σ_min(Pⱼ − z₀I) − |z − z₀|` — good near `z₀`,
useless far. The two are complementary: `‖Nⱼ‖` for the far field, anchored
`σ_min` for the near.

--- 

## 5. Cluster / block path — `cluster.jl`, `assembly.jl`, `twobytwo.jl`

Block membership comes from the separation-driven components of §4c/§4e (Rump
`conncomp` + merge); this section certifies a given block. Triggered when the
scalar recentred-disc floor (§4c) is marginal/negative on a component with more
than one eigenvalue.

> **Frame note.** In the committed diagonal-residue frame (§4c) the block `Pⱼ` is
> the invariant-subspace block in the eigenvector frame and is certified directly
> by `σ_min(Pⱼ − zI)` (§4d) — no `ordschur`. The `reorder_cluster`/`far_gap`
> entries below are the **Schur-frame fallback** (§4b), to use only if the
> eigenvector frame is globally ill-conditioned. Both share `sigma_min_block`.

- `reorder_cluster(setup, idx)`: `idx` is a coupling-graph component (§4b); call
  `ordschur` (Float64) with the `select` mask for `idx` to bring those
  eigenvalues to the leading block. The reorder itself is `O(n²m)` (adjacent
  swaps), not a fresh `O(n³)`. **Re-certify by residuals**: form the two
  interval quantities `‖Z_reord^H Z_reord − I‖` and `‖A Z_reord − Z_reord
  T_reord‖` once (the reorder is a black-box producer of a candidate
  factorization — certify the product, never the individual Givens rotations);
  feed these as slacks to the assembly. Grab LAPACK `trsen`'s `sep`/`s`
  condition estimates as a free a-priori cluster-separation sanity check (via
  GenericSchur.jl for extended precision).
- `far_gap(z, 𝓕)`: Lemma 3.1, triangular gap on the far block `T̃_𝓕𝓕`.
- `localize(z)`: check `‖S(z)‖_∞ / far_gap(z) < 1` (computed form uses
  `(‖S̃‖+β)/g_𝓕`). Certifies the dominant component sits in the cluster.
- `sigma_min_block(P)`: certified **lower** bound on `σ_min` of the `m×m`
  cluster block `P = T̃_𝓒𝓒 − S_𝓒𝓒 − zI`.
  - `m == 1`: `abs(P[1,1])`.
  - `m == 2`: closed-form Kahan, `twobytwo.jl` (below).
  - `m ≥ 3`: verified SVD on the small block (interval Jacobi, or certify an
    approximate SVD via a residual bound). Only here is the genuinely hard
    `ℓ²` object computed, and only on an `m×m` matrix.
- `assemble(z)`: Thm 3.x block-triangular bound
  ```
  ‖(A−zI)⁻¹‖₂ ≤ κ₂ * [ (1 + ‖Q‖₂‖W⁻¹‖₂) ‖P⁻¹‖₂ + ‖W⁻¹‖₂ ]
  ```
  with `‖P⁻¹‖₂ = 1/sigma_min_block(P)`, `‖W⁻¹‖₂` from interpolated far gaps,
  `‖Q‖₂ = opnorm(T̃_𝓒𝓕 − S_𝓒𝓕, 2)` (certified 2-norm of the small coupling).

### `twobytwo.jl` — closed-form certified `σ_min` (note Rem 3.8, Kahan form)
```
# M = [a b; c d] interval entries
p = sqrt( (a+d)^2 + (b−c)^2 )       # sum of squares: no cancellation
q = sqrt( (a−d)^2 + (b+c)^2 )
σ_max = (p + q)/2
σ_min = abs(p − q)/2                 # only subtraction is here; interval-safe
```
For real `z`, only `a+d = (α+δ) − 2z` depends on `z`. For complex `z`, take
`σ_min² = λ_min((M)^H M)`, again 2×2 closed form. **Use the Kahan form**: the
naive `√((F−√(F²−4D²))/2)` cancels exactly where `σ_min→0`.

---

## 6. Sweep + adaptive refinement — `sweep.jl` (note §2.4, §3.4)

**Box bisection, not uniform grid.** The map `z ↦ σ_min(A−zI)` is 1-Lipschitz
(note Lemma 3.x), so one certified centre evaluation settles a whole cell:
on a square of half-diagonal `r` centred at `z₀`, `σ_min(A−zI) ≥ σ_lo(z₀) − r`
for all `z` in the cell. This makes the sweep a *rigorous enclosure of the
continuum*, with refinement concentrated on the ε-contour (a 1-D set), so cell
count scales with contour-length/resolution, not area/resolution².

```
function bisect(setup; box, ε, min_halfdiag)
    stack = [box]; excluded = []; included = []; ambiguous = []
    while !isempty(stack)
        cell = pop!(stack); z0 = center(cell); r = halfdiag(cell)
        σlo = g_tri_inf(z0, setup)                 # cheapest, O(n)
        if σlo − r > ε
            push!(excluded, cell); continue
        end
        σhi = inclusion_upper(z0, setup)           # cluster σ_min upper bound
        if σhi + r < ε
            push!(included, cell); continue
        end
        # ambiguous: straddles ε
        τ = active_tail(z0, setup)                 # Lemma 3.x local sandwich
        if τ > ESCALATE_THRESH * ε                 # free value is loose here
            σlo = best_refined_lower(z0, setup)     # recentred → interp2 → cluster
            (σlo − r > ε) && (push!(excluded, cell); continue)
        end
        if r < min_halfdiag
            push!(ambiguous, cell)                  # at target resolution
        else
            append!(stack, quadrisect(cell))
        end
    end
    (excluded, included, ambiguous)
end
```

`active_tail(z0)` returns `τ_{i*(z0)}` from the free-gap argmin — free, and by
the local sandwich (note Lemma 3.x) it bounds the free overestimate
`0 ≤ σ_min − g_tri ≤ τ_{i*}`. Use it to decide *escalate vs. subdivide*: large
tail ⇒ free value is loose ⇒ escalating the certificate at `z0` may resolve the
cell without subdividing; small tail ⇒ ambiguity is real ⇒ subdivide.

Record, per terminal cell, which layer/decision settled it — this is the data
for the routing histogram (§9), the central benchmark.

A uniform-grid `sweep` (as in earlier drafts) may be kept only for
apples-to-apples timing against grid inverse-Lanczos; the box bisection is the
method's intended driver.

---

## 7. Baseline — `baseline_lanczos.jl` (NON-rigorous, for comparison only)

Classical grid / inverse-Lanczos (Trefethen `psa`, Wright–Trefethen EigTool):
1. `F = schur(A)` once.
2. Per `z`: smallest singular value of `T − zI` via inverse Lanczos —
   Lanczos on `(T−zI)⁻ᴴ(T−zI)⁻¹`, each step two triangular solves (`O(n²)`),
   `k` steps → `O(k n²)` per point.
Implement with Float64 only; this is the *unverified* reference whose **cost**
(wall-clock, `k` per point) and **value** (`σ_min`, treated as ground truth)
the certified method is compared against.

Also include a brute-force `svdvals(A − zI)[end]` per point (`O(n³)`) as the
exact-`σ_min` reference for tightness plots on small `n`.

---

## 8. Test matrices and examples — `bench/matrices.jl`

Operator-free gallery (the Paper A audience) plus the Gauss witness. Each
example is chosen to stress a *specific* feature of the method; the table notes
what it probes and what behaviour to expect. Where a matrix is in the
Trefethen–Embree gallery or EigTool, reproduce their pseudospectrum as the
visual ground truth.

### 8.1 Normal / near-normal — sanity and tightness floor
- **Diagonal / normal random** (`Q diag(λ) Q*`, `Q` unitary). Free gap is
  *exact* here (`τ_i = 0` after a perfect Schur form); confirms the method
  recovers `σ_min = dist(z, σ)` with zero overestimate. Regression baseline.
- **Symmetric/Hermitian random.** `κ₂ = 1`, pseudospectrum is `ε`-neighbourhood
  of spectrum; confirms the conditioning prefactor is benign and the interpolated
  ℓ² bound is tight.

### 8.2 Non-normal Toeplitz — the classical pseudospectrum benchmarks
- **Grcar(n)** (`n = 64, 256`). The canonical non-normal example; large
  pseudospectra bulging far from the spectrum. *The headline figure*: free
  triangular contour overlaid on true pseudospectrum, then the refinement
  annulus. Probes free-layer overshoot and box-bisection contour concentration.
- **Limaçon / "Toeplitz with symbol" examples** (banded Toeplitz, various
  symbols). Pseudospectra predicted by the symbol curve; tests behaviour for a
  family with controllable non-normality.
- **Demmel matrix** (upper triangular, parametrised by a large off-diagonal).
  Tunable non-normality via one parameter — sweep it to plot free-layer
  overestimate (and the `τ_{i*}` sandwich bound) vs. departure from normality.

### 8.3 Defective / near-defective — why σ_min and not eigenvalue distance
- **Single Jordan block** `J_m(λ)` and **perturbed Jordan** `J_m(λ) + δ·rand`.
  σ_min(J_m − zI) ∼ |z−λ|^m; verifies the method tracks the true (tiny) σ_min
  where eigenvalue distance would report |z−λ|. Direct illustration of note
  Remark 2.3 (no diagonalizability assumed).
- **Coalescing pair** `[[λ, t],[0, λ+s]]` scaled up, `s→0` with `t` fixed
  (2×2 → small block). Drives the cluster path and the closed-form Kahan σ_min;
  tests the localization condition and the block-triangular assembly as the
  two eigenvalues merge.
- **Clustered spectrum** (several eigenvalues within `O(ε)`, non-trivial Jordan
  structure). Forces `ordschur` grouping + the m×m σ_min certificate; the case
  that defeats per-eigenvalue (scalar) bounds.

### 8.4 Graded / ill-conditioned eigenvectors — the κ(X̃) stress
- **Kahan matrix** (graded upper triangular). Near-defective with
  ill-conditioned eigenvectors; stresses the gap between eigenvalue distance and
  σ_min, and makes `κ_p(X̃)` large — tests that conditioning enters only the
  resolvent *magnitude*, not the enclosure, and that the ℓ¹/ℓ∞ interpolation
  beats a naive ℓ² route.
- **Frank matrix / Wilkinson-type** (notoriously sensitive eigenvalues). Tests
  robustness of the certified setup (`‖R₂‖<1` verification) under bad
  conditioning; expect some `ε` regimes where the certificate reports failure
  (the honest Miyajima Step-4 outcome) rather than a wrong answer.

### 8.5 Structured spectra — bisection efficiency
- **Companion matrix** of a polynomial with known roots. Spectrum = roots;
  good for box-bisection cell-count scaling (contour length vs. resolution) and
  for checking inclusion certificates at known eigenvalues.
- **Convection–diffusion / discretised non-self-adjoint operator** (1-D
  upwind/centred difference, a standard EigTool example). A non-normal operator
  discretisation that is *not* a transfer operator — keeps Paper A operator-free
  while showing applicability to PDE-type spectra.

### 8.6 The Gauss witness — before/after
- **Gauss coarse matrix.** The finite Galerkin/Hardy truncation of the
  Gauss–Kuzmin–Wirsing operator, loaded from a data file as a dense non-normal
  matrix (no operator machinery imported). Reproduce the certified resolvent
  bound of the Gauss paper via the generic Miyajima route; report the
  effort/cost reduction ("almost trivial"). This is the demonstrating example
  that motivates the abstraction, presented as application, not motivation.

### 8.7 Scaling sweeps
For the cost/cell-count tables, run a size sweep `n ∈ {32, 64, 128, 256, 512}`
on Grcar and the convection–diffusion example, recording setup time, per-cell
time per layer, terminal-cell counts, and the routing histogram, against
inverse-Lanczos and full-SVD references.

### Implementation note
Provide each as a function returning a `Matrix{ComplexF64}` (or real), plus a
metadata `NamedTuple` (`name`, `true_spectrum` if known, `suggested_box`,
`suggested_ε`, `expects_failure::Bool`). Keep a `gallery()` returning the full
list so `run_benchmarks.jl` can iterate.

---

## 9. Benchmarks — `bench/run_benchmarks.jl`

The central claim is **not** per-point cost parity but that the method *routes
around* the expensive code: almost the entire plane is settled by the `O(n)`
free layer on large bisection cells, and the expensive cluster certificate runs
at very few points (a thin annulus around the ε-contour). Measure the
distribution, not just the worst case.

**Primary metric — routing / cell histogram (the rung ladder of §3.1).** Per
matrix and ε, record which **rung** binds per terminal cell (and per area):
- rung 1 eigs (scalar recentred-disc floor) / rung 2 block `σ_min` / rung 3 Schur
  triangular / rung 4 SVD frame / escalated (none);
- total certified evaluations at each rung.
Expected eigs ≫ block ≫ Schur/SVD (e.g. 90+ : few : ~0) for a well-conditioned
eigenbasis. This histogram *is* the contribution made visible, and it measures
which non-normality regime the matrix sits in (separation / eigenvector
conditioning / Schur tails). For the transfer-operator coarse matrices,
**measure `κ(X̃)` and `‖R₂‖` first** — they predict whether rung 1 dominates.

**Cumulative cost decomposition.** Total flops/time split as free / refined /
cluster, showing the cluster term is a small slice of the total despite being
the most expensive per call. Contrast with inverse-Lanczos, whose `O(kn²)` cost
is flat across the plane and whose `k` *spikes* in the non-normal/cluster region
— i.e. it pays most exactly where it struggles.

**Cell count vs. grid count.** Box bisection: number of terminal cells to
resolve the ε-contour at resolution `h`, scaling like contour-length/`h`.
Uniform grid / inverse-Lanczos: `O(1/h²)` points for the same resolution. Report
the ratio — this is the spatial half of the speed advantage.

**Tightness (price of rigor).** Certified σ_min lower bound vs. true σ_min
(full SVD) along a line crossing the spectrum, per layer; show the ladder
contracting the overshoot. Overlay the local-sandwich bound `τ_{i*(z)}` to
confirm it predicts the free-layer error.

**Cost vs. baseline (secondary).** Setup + total time for free / refined /
cluster vs. inverse-Lanczos (report `k`) vs. full-SVD grid, on a fixed uniform
grid for apples-to-apples per-point timing. Confirms refined-vs-Lanczos per-`z`
parity; the bisection results above are where the real win shows.

---

## 10. Tests — `test/runtests.jl`

Rigor regressions (these must hold or the certificate is wrong):
- **Containment:** for random `z`, the certified `σ_min` lower bound `≤`
  `svdvals(A−zI)[end]` (true value). Never exceed truth.
- **Eigenvalue sanity:** at a true eigenvalue `λ` of a constructed matrix, the
  certified gap `≤ 0` (cannot certify exclusion at the spectrum).
- **Kahan 2×2:** certified `σ_min` brackets the analytic `σ_min` of random 2×2
  blocks, including near-singular ones (cancellation test).
- **Neumann guard:** routines refuse to proceed when `‖R₂‖ ≥ 1`.
- **Affine consistency:** `S_C − z S_B` matches a direct solve of
  `(I+R₂)S=R₁(z)` within the certified `β`.
- **Defective case:** Jordan block of size `m`, verify the certified bound
  tracks `|z−λ|^m`, not `|z−λ|`.

---

## 11. Suggested first milestone

1. `setup.jl` + `triangular.jl` + a bare `sweep` with only the free layer.
2. `baseline_lanczos.jl` + full-SVD reference.
3. Free-plot figure on Grcar(64) with the `g_tri = ε` contour overlaid on the
   true pseudospectrum — the "rigorous, almost free" first picture.
4. Then add `recentred.jl` and the cost table; cluster path last.

This already produces the paper's central figure and the headline cost claim
(free layer `O(Gn)` vs. Lanczos `O(Gkn²)`), before any of the harder cluster
machinery is in place.
