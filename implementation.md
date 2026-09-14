# Implementation map: `draft_lean.tex` label by label

The package implements `~/Write/KrylovPseudospectra/draft_lean.tex` (Nisoli, *Certified pseudospectra
and faithful Krylov reductions for implicitly defined linear operators on Hilbert spaces*). This
file maps every label of §2 to §5 to the code, and records every place where the code does
something the paper does not say, or does not do something the paper says. Section numbers shift
between drafts; labels do not.

Conventions of the code:

- A **certified** number is either the endpoint of an outward-rounded ball quantity (BallArithmetic
  ball products and sums, `svdbox`) or an a priori bound evaluated with directed rounding
  (`src/rounding.jl`: round-to-nearest followed by one `nextfloat`/`prevfloat`, which is a bound
  for every correctly rounded operation and is safe inside threads). No Float64 quantity enters a
  certificate without being enclosed first.
- **Lower bounds** (σ11, σ22, floors, δ) and **upper bounds** (γ1, γ2, ḡ, ā, α, norms, radii) are
  kept in their direction; every formula of §2 is monotone in its arguments in the direction
  that keeps the output a bound, and that is used, not assumed (comments at each function).
- A function that cannot certify returns `ok = false` and a zero bound; none returns a positive
  number without its hypothesis checked.
- From BallArithmetic only the ball types, ball matrix/vector products and sums, and `svdbox`
  (with `apply_vbd = false`) are used. Norm bounds for ball matrices are the package's own
  (`src/normbounds.jl`). BallArithmetic's VBD, block Schur and projector routines are not called.

## Files

| file | paper |
|---|---|
| `src/rounding.jl` | directed rounding for scalars; `γ_ℓ` (Higham (3.8); complex `γ_{ℓ+2}`) |
| `src/normbounds.jl` | upper bounds on `‖X‖₂` over a ball: Frobenius, `√(‖·‖₁‖·‖∞)`, upper end of `svdbox` |
| `src/svdenclosure.jl` | §3.1 `thm:svdbox` |
| `src/twoblock.jl` | §2 `thm:assembly`, `prop:blockdom`, `eq:farshort`, `eq:farpower`, `prop:complete`, `prop:count` (circle and union covers), `cor:selfadj`, `cor:weyl` |
| `src/compression.jl` | §3.2 `sec:candidate`, `sec:certification` (`eq:residuals`, `lem:neumann`, `lem:deviation`, `thm:enclosure`, `cor:clusters`, `prop:blockgersg`), `sec:transport` (`eq:blockres`, exact-SVD escape); `miyajima_rump`; the split by modulus (`split_compression`, `thm:assembly` re-applied to the compression) |
| `src/operator.jl` | `eq:roundA`: the operator and its rounding model |
| `src/couplings.jl` | §3.3 `eq:gram`; `eq:alpha`; the factors of `prop:almostortho` |
| `src/farblock.jl` | §4 `eq:mincert`, `eq:interp`, `prop:ladder`, `eq:sweep`, §4.3 `sec:fpsweep` (`eq:epsP`, `lem:onestep`, `eq:FI`, `prop:fpchecks`, `rem:impl`, `alg:fpsweep`) |
| `src/frame.jl` | §5.1 `def:Q`, `prop:frame` (frames only), `rem:lanczos`; §5.3 `eq:depthrule` (floating point) |
| `src/splitting.jl` | §5 `sec:exterior`, `cor:gap`, `sec:almostortho`, `sec:sound`; entry points |
| `src/deepsplit.jl` | §3.2 the deep tier: the reduced frame `Q·U`, the merged far block through the monotone 2×2 bound, and the derived couplings (none of it in the paper yet) |
| `src/poly_range.jl` | certified range of a polynomial; not part of the paper, kept unchanged |

## §2 The two-block bound (`src/twoblock.jl`, `src/splitting.jl`)

| label | code | notes |
|---|---|---|
| `thm:assembly`, `eq:assembly` | `assembly_floor(σ11, σ22, γ1, γ2)` | returns `(0, false)` unless `σ11σ22 − γ1γ2 > 0` is certified (product of lower bounds rounded down, product of upper bounds rounded up); γ1, γ2 separate |
| `rem:counterexample` | test `rem:counterexample` | `[1 1; 1 1]`, `e1`, `z = 0` and `[1 1; 1 0]`, `e1`, `z = φ` give no bound |
| `prop:blockdom`, `eq:blockdom` | `blockdom_floor` | `τ = max(γ1/σ11, γ2/σ22)`; `resolvent_floor` reports it separately and takes the larger of the two valid floors |
| `eq:farshort` | `farshort_lo(z, ḡ)` | |
| `prop:power`, `eq:farpower` | `farpower_lo(z, ḡs)` | consumes `ḡ_j ≥ ‖G22^j‖` supplied by the caller (`certify_split(...; gbars_power)`); the package does not certify `‖G22^j‖` for `j > 1` (see "Not done") |
| `lem:rungs` | not code | |
| `prop:complete`, `eq:incl` | `inclusion_delta`, `pseudospectrum_outside(sc, ε, R)`; pointwise `resolvent_floor`, `pseudospectrum_outside(sc, z)` | δ uses a lower bound of `R − ḡ`, δ being increasing in it |
| `prop:count` | `check_circle`, `count_disc`, `count_exterior`, `count_union`, `union_boundary_arcs` | the cover uses that the true σ11 and σ22 are 1-Lipschitz; arcs between samples get `h = rΔθ/2` plus a bound on the error of the computed sample point; failing arcs are bisected; `D ∪ Γ ⊆ {|z| > ḡ}` is checked (`region_ok`); the count of G11 is read off the certified enclosure of the compression (each component inside the region or disjoint from it, else no count). For a union of discs only the arcs of each circle that can lie on the boundary of the union are checked; skipped arcs are strictly inside another disc with a margin far above the rounding of the geometry |
| `rem:bounded` | not code | |
| `cor:agree` | `certify_exterior(sc, R)` | isolation discs: for each component of the enclosure of G11 outside R, the smallest radius in a geometric sequence at which `count_disc` succeeds, discs kept pairwise disjoint (rechecked) and inside `{|z| > R}` |
| `prop:projector`, `eq:projector` | `certify_projector(sc, c, r; region)`, `projector_bound`, `circle_min` | `bound = r M M₀ max(γ1, γ2)` since `|Γ|/2π = r`, with `M = 1/floor` and `M₀ = 1/min(minσ11, minσ22)` read off the same arc cover as the counts (`check.floor`, `check.minσ11`, `check.minσ22`), so both hold on the whole contour. `count` and the basis `Q·W_S` of the candidate for `ran P₀` come from the block diagonalisation; `subspace_gap` bounds the gap between `ran P₀` and that basis, through `certify_block_projector` and the factor `κ(QW) ≤ √((1+α)/(1−α))(1+α)κ`. For the exterior region `P = I − P_in` and `‖P‖ ≤ 1 + rM`. The contour should be taken as large as prop:count allows: `bench/projector_sweep.jl` and the tests sweep the radius |
| `prop:projector` inside §3.2 | `certify_block_projector(cc, c, r)` | the same with `A₀ = Λ` and `E = M_c − Λ`: `min_i σ_min(P_i − zI)` is covered on the contour (`circle_min`, the function being 1-Lipschitz), giving `M₀`; subtracting `β_Λ` gives `M`; `‖E‖ ≤ β_Λ` (lem:deviation), so `‖P_{M_c,S} − P_{Λ,S}‖ ≤ r M M₀ β_Λ`. Blocks are classified inside or outside the circle by their outer discs |
| `cor:selfadj` | `selfadj_enclosure`, `selfadj_count`, `eig_intervals`, `certify_selfadjoint` | (i) as a union of intervals with the quadratic radii `ρ±`; (ii) counts on `[x1, x2]`, negative side through `−A`; γ̄ = min(γ1, γ2) (both bound the same number for `A = A*`) |
| `cor:weyl` | `weyl_pairing` | ordered eigenvalues of G11 located by cumulative counts of disjoint intervals |
| `rem:lanczos` | `lanczos_frame` | the frame is not re-orthonormalised; γ̄ comes from the Gram bound on the frame computed |
| `sec:reach` (`prop:essfloor`, `prop:essreach`) | not code | |

## §3 The small blocks

### §3.1 `thm:svdbox` (`src/svdenclosure.jl`)

`svd_enclosure(B)` calls BallArithmetic's `svdbox(B; apply_vbd = false)`; default method
`MiyajimaM1()` is Miyajima (2014) Thm 7 in economy form (checked by reading
`BallArithmetic/src/decompositions/svd/svd.jl` at v0.3.0: residuals `Ê, F̂, Ĝ` in ball arithmetic,
bounds `σ̂_i√((1∓‖F̂‖)(1∓‖Ĝ‖)) ∓ ‖Ê‖`, infinite radii if `‖F̂‖ ≥ 1` or `‖Ĝ‖ ≥ 1`). The default
`apply_vbd = true` also runs `miyajima_vbd` on `Σ*Σ` and discards the result; we switch it off.
Endpoints are taken with directed rounding (`sigmin_lo`, `sigmax_up`).

### §3.2 The compression (`src/compression.jl`)

Everything is written for a pencil `(G̃, M)`, `G̃ ⊇ Q*AQ`, `M ⊇ Q*Q`, because the frame is only almost
orthonormal (`prop:almostortho`); `M = I` is the paper's case. The eigenvalues of
`G11 = Z*AZ = S⁻¹G̃S⁻¹` are those of `M⁻¹G̃`; the residuals are `R2 = Y(MW) − I`,
`R1 = Y(G̃W − MWΛ)`, `M_c = (MW)⁻¹G̃W` and `M_c − Λ = (I + R2)⁻¹R1`, and
`G̃ − zM = MW(M_c − zI)W⁻¹`. The paper states the pencil only in one sentence of §5.5; this
generalisation is derived in the header of the file.

| label | code | notes |
|---|---|---|
| `sec:candidate` 1–4 | `candidate_sylvester` (default), `candidate_newton` | see below |
| `eq:residuals` | `certify_compression` | ball products |
| `lem:neumann` | `certify_compression` (`nrmR2 < 1` or refuse), `kappa = ‖Y‖‖W‖/(1 − ‖R2‖)` | |
| `lem:deviation`, `eq:blockdev` | `betaΛ = ‖R1‖/(1 − ‖R2‖)` | |
| `thm:enclosure` | `outer_radius(P_i, c_i, β)`, `ρglobal` | outer disc of `Ω_i`: the least over `m ≤ 8` of the radius from `prop:power` applied to `N = P_i − c_iI` with `g_j ≥ ‖N^j‖` from ball powers; `m = 1` is `‖P_i − c_iI‖ + β_Λ`, the disc of the brief; the higher `m` is proved in the docstring (the lower bound `f_m(r)` increases for `r > g_m^{1/m}`) |
| `cor:clusters` | `components_global`, `components_local` (union-find on outer discs not certified disjoint) | the count needs the union over `S` to be disjoint from the others; connectedness is not used |
| `prop:blockgersg` | `rloc`, `ρlocal` | `r_i = Σ_j ‖Ñ_{C_iC_j}‖ + qβ₂`, `β₂ = ‖R2‖‖M̃‖/(1 − ‖R2‖)`, `M̃ = YG̃W`; the sum includes `j = i` because `Λ` is a floating-point matrix and `M̃` a ball, so `Ñ_{C_iC_i}` is a ball about a rounding-level midpoint (in the paper it vanishes). The two families are never mixed block by block; `miyajima_rump` picks one family as a whole |
| `eq:blockres` | `sigma_blocks_lo(cc, z)`; `sigma11_lo(sc, z; method = :blocks)` | 1×1 blocks in closed form (`|p − z|` with the rounding of the subtraction); blocks of size ≥ 2 by `svdbox` (the paper's closed form for size 2 is not implemented) |
| exact-SVD escape | `sigma11_lo(sc, z)` (default `method = :svd`) | one `svdbox` of the ball `G̃ − zM` per `z`, divided by `1 + α` |
| `rem:rump` | not code | |

Candidates (floating point, nothing certified):

- `candidate_sylvester(G; xmax = 1e4)`: complex Schur form; clusters = connected components of
  `|t_ii − t_jj| ≤ √u‖G‖_F`; clusters made contiguous by LAPACK `trsen`; then, cluster by cluster
  along the diagonal, the triangular Sylvester equation `T_cc X − X T_rr = −T_cr` (LAPACK `trsyl`,
  Bavely and Stewart, SIAM J. Numer. Anal. 16 (1979) 359–367), `W ← W[I X; 0 I]`,
  `Y ← [I −X; 0 I]Y` (the exact inverse; no `inv`); when `‖X‖_F > xmax` the trailing cluster
  nearest in eigenvalue is moved next to the current one (`trsen` on the trailing block) and
  merged. Then QR inside each block (`W_C ← Q_C`, `Y_C ← R_C Y_C`), and `Λ` = block diagonal of
  `YG̃W`. The merge on `‖X‖` was asked for in the brief; sec:candidate says the opposite ("blocks
  are not merged on the size of a decoupling transform").
- `candidate_newton(G; maxiter)`: the paper's step 2, `X_ij = −G̃_ij/(d_i − d_j)` across clusters,
  iterated; the off-block norm is not required to decrease at each step (in 9 of the 13 runs of
  `bench/candidates_compare.jl` it grew before decreasing or before the step limit); the iterate
  with the smallest off-block norm is kept.

`miyajima_rump(G; method = :sylvester | :newton | :best, ...)` wraps candidate and certificate
and returns `clusters` (`idx`, `count`, `hull_center`, `hull_radius`) in the shape
`src/poly_range.jl` reads.

#### The split by modulus (`split_compression`, `ModulusSplit`)

A second layer for §3.2, added 2026-09-12 (report `20260912_compression_split.md`). The candidate
is stopped after one Sylvester solve: the leading block `E` carries the `k₊` eigenvalues of largest
modulus and the trailing block `J` all the rest, `Λ = E ⊕ J`. `k₊` is chosen adaptively under two
conditions, `k₊ ≤ kplus_max` (20) and `ḡ_J·margin < R` with `ḡ_J = min_{j≤m}‖J^j‖^{1/j}` certified;
when they conflict the call declines (`ok = false`, `reason`, `kplus_needed`, `split_reason`).

| label | code | notes |
|---|---|---|
| `sec:candidate` 1–4 | `split_compression` (`_split_candidate`, `_split_candidates`) | one threshold on `|λ|` in place of the clustering by diagonal distance, one `trsyl`, QR inside the two blocks; no merging on `‖X‖` |
| `eq:residuals`, `lem:neumann`, `lem:deviation` | `split_compression` | as in `certify_compression`, plus the block-row deviations `β_i = ‖R1[rows i, :]‖ + ‖R2‖β_Λ`, which follow from `(I+R2)⁻¹ = I + S`, `‖S‖ ≤ ‖R2‖/(1−‖R2‖)`; NOT in the paper |
| `thm:assembly` re-applied | `sigma_split_lo` | `eq:assembly` (and `eq:blockdom`, the larger kept) applied to `M_c` with the splitting `ℂ^{k₊} ⊕ ℂ^{k−k₊}`, `σ11 ≥ σ_min(E−zI) − β₁`, `σ22 ≥ σ_J(z) − β₂`, `γ_i ≤ β_i`; divided by `κ` (`eq:blockres`). §2 states the theorem for an isometry into `H`; nothing in the paper re-applies it inside §3.2 |
| `prop:power`, `eq:farpower` | `sigma_J_lo`, `_power_bounds` | applied to the trailing block `J`; `ḡ_j ≥ ‖(cJ)^j‖` with `c` a power of two (`rem:impl`, there stated only for the §4.3 sweep), and `σ_J(z) = farpower_lo(cz, ḡ)/c` |
| `thm:enclosure`, `cor:clusters`, `prop:blockgersg` | `_nested_candidate` + `certify_compression`; `split_components`, `split_block_discs` | the nested candidate block diagonalises `E` alone and leaves `J` as one block; the trailing block's disc is the origin-centred `D(0, ρ_J)` from `_rho_J` when that reaches less far than `outer_radius` about the mean of its diagonal |
| `prop:projector` inside §3.2 | `certify_block_projector(...; blockfun, discs)` | the trailing block's `σ_min(J − zI)` on the contour comes from `prop:power` (`split_sigma_blocks_lo`), not from a verified SVD |
| entry points | `certify_split(...; compression = :auto/:split/:full/:none, target, split_kplus_max)`, `sigma11_lo(...; method = :auto/:split/:svd/:blocks)` | `:auto` takes the split when `target` is given and `dim Q > split_size` (64); the exact-SVD escape is unchanged under `:svd` |

`_radius_from_powers(g, β)` is the radius search of `outer_radius` factored out, so that the
trailing block's powers are computed once and reused.

#### The deep tier (`src/deepsplit.jl`, `deep_split`, `deep_certificate`)

A third tier, added 2026-09-14 (report `20260914_compression_deepsplit.md`). The compression is cut
twice by modulus into `E` (above `0.9R`), `N` and `D`; `E ∪ N` stays in the frame and `D` leaves it. The new frame is `Q' = Q·U` with `U` the leading `k_EN` Schur vectors, and the
reduced certificate applies the operator not at all.

| label | code | notes |
|---|---|---|
| not in the paper | `deep_split`, the header of the file | the orthogonal decomposition `ran P'⊥ = ran P⊥ ⊕ (ran Q ⊖ ran Q')` and the blocks of `A` in it; every block meeting `P⊥` is an old coupling composed with a contraction, so it inherits γ1 or γ2 |
| not in the paper | `merged_far_bounds` | `‖(G'22)^m‖ ≤ ‖N^m‖`, `N = [ḡ_D γ1; γ2 ḡ]` nonnegative 2×2, by the monotonicity of the norm in the block norms; gives every order of `eq:farpower` for the merged far block in closed form. `prop:fpchecks` (ii) uses the `m = 1` half already (`norm2x2_up`) |
| `prop:almostortho` one level down | `_deep_blocks` | `X' = S_pUK`, `X_D = S_p⁻¹S̃L`, `K = (U*MU)^{-1/2}`, `L = (S̃*M⁻¹S̃)^{-1/2}`; `ḡ_D ≤ ‖L‖²(‖S̃*G̃S̃‖ + ‖S̃‖²‖G̃‖ε_α(1/(1−α)+1))`, `ε_α = α/(1−α)`, and the two off-diagonal norms; all direct ball products on `G̃` and `M` |
| `thm:assembly` at `\|z\| = R` | the gate in `deep_split` | `γ1γ2 < R(R − ḡ)` is necessary for any nonempty cut, being the spectral radius of `[0 γ1; γ2 ḡ]` below R |
| `sec:exterior` (`far_bound`) | `deep_certificate` | the reduced certificate takes `ḡ'` and every `ḡ'_j` from the 2×2, the couplings from the derivation, and `G̃' = U*G̃U`, `M' = U*MU` from the old ball |
| `sec:depth` (floating point) | `criterion = :assembly`, `amargin` | the cut is chosen by bisection on `k_EN` with an uncertified estimate of the hypothesis; the arc cover afterwards is the certificate |
| `rem:limits` | `deep_reason` | the obstructions: `:farblock` (the gate), `:nocut` (no cut passes the conditions), `:cost` (the conditions pass past the stopping rule; `inner.kwanted` and `inner.tz` say which `k_EN` and at what measured seconds per z). There is no cap on `k_EN`: `E` is fixed by the circle, `N` by the conditions, and the size of `E ∪ N` is an outcome whose only consequence is the per-z cost |

### §3.3 The couplings (`src/couplings.jl`)

`frame_data(op, Q)`: `α ≥ ‖I − Q*Q‖` (ball Gram, `opnorm_up` with `svdbox`); `G̃`, `M`;
`γ2_Q ≥ ‖P⊥AQ‖`, `γ1_Q ≥ ‖Q*AP⊥‖`, `s_Q ≥ ‖Q*AQ‖`. `P⊥` is the orthogonal projector onto
`(ran Q)^⊥`; since `P⊥ = P⊥(I − QQ*)` and `‖P⊥‖ ≤ 1`, `‖P⊥AQ‖ ≤ ‖(I − QQ*)AQ‖`, which is formed in
ball arithmetic without the inverse of `Q*Q`. `eq:gram`: `γ² = ‖C*C‖` with the Gram as a ball
product and its norm by `opnorm_up` (Frobenius, interpolation, `svdbox`), square root rounded
upward (`coupling = :gram`, default); `coupling = :direct` uses `svdbox(C)`.

- stored matrix (`LinOp(A)`, dense or `SparseMatrixCSC`): `AQ`, `A*Q` as ball products;
- matrix-free (`LinOp(Amul, Aadj, n; epsA)`): `X = fl(AQ)`; each column of `X − AQ` has norm
  `≤ ε_A‖A‖‖q_i‖ + η_A`, so every bound is returned as `c0 + c1·a` (valid when `‖A‖ ≤ a`), as
  `prop:fpchecks` needs; after the far block certifies `ā`, the bounds are evaluated at `ā` and the
  entries of `G̃` get the extra radius `√(1+α)(ε_A ā‖Q‖_F + √k η_A)`.

The isometry values are `γ_i = γ_{i,Q}/√(1 − α)` and `‖G11‖ ≤ s_Q/(1 − α)` (`isometry_bounds`).

## §4 The far block (`src/farblock.jl`, `src/operator.jl`)

| label | code | notes |
|---|---|---|
| `eq:roundA` | `LinOp` fields `epsA`, `etaA` | stored: `ε_A = √w γ_w`, `w` = largest number of nonzeros in a row or column (`n` if dense); matrix-free: `epsA` supplied by the caller (required keyword) |
| `eq:epsP` | `epsP_up(α, n, k, T)` | the paper's formula with `u_ℓ = γ_ℓ` (complex `γ_{ℓ+2}`) |
| `lem:onestep` | `onestep_up` | `ε_B` as in the paper, the larger of the constants for `B` and `B*` |
| `eq:sweep`, `prop:ladder` | `trace_sweep` | canonical basis; chain `x_0 = e_j`, `x_1 = fl(Bx_0)`, `x_2 = fl(B*x_1)`, …; for every order `t ≤ m` `Σ_j ‖x_t‖²` is the value of `tr H^t`, so one run gives all orders `t = 1..m` (even and odd) |
| `eq:FI` | `F_up`, `I_up` | from the sums `Γ[s,s'] = Σ_j ν_{j,s}ν_{j,s'}` and `Σ_j ν_{j,s}`, `s, s' ≤ m` (O(m²) scalars) accumulated with upward rounding |
| `prop:fpchecks` | `check_ii`, `check_iF`, `check_iI`, `least_a`, `least_g` | (ii) by the closed form of the norm of a nonnegative 2×2 matrix, rounded upward; bisection in `ā` for each `ḡ`, then in `ḡ`; every order `t ≤ m` and the interpolation half are tried, the least passing `ḡ` is returned (`eq:mincert`) |
| `rem:impl` | `certify_far_block` (`scale`) | one power of two `c` for the whole sweep (the operator `cA`), chosen so that the least stochastic estimate lies in `[1, 2)`; `ḡ/c`, `ā/c` at the end |
| `alg:fpsweep` | `certify_far_block` | order `m` from `estimate_orders` (`s` Gaussian vectors, `ms` applications, no certificate), or given |
| `prop:floor`, `rem:positive`, `rem:sci`, `prop:gelfand` | not code | |

Error analysis for a general order `t` (the paper writes `m = 2p`): with `g ≥ ‖B‖`,
`err_{s+1} ≤ g·err_s + ε_B‖x_s‖ + η_B`, so `‖x_t − exact‖ ≤ ε_B Σ_{s<t} g^{t−1−s}‖x_s‖ + η_B Σ_{s<t} g^s`,
which at `t = 2p`, `η_B = 0` is the paper's `E_j(g)`. The scaling argument of `prop:fpchecks`
(`F_{λa}(λg) ≤ λ^{2t}F_a(g)` for `λ ≥ 1`) holds with the extra terms, which have degree at most
one in `(a, g, γ)` per factor.

Absolute (underflow) terms, not in the paper: Higham's inner-product bound, on which `eq:roundA`
and `eq:epsP` rest, assumes no underflow. We carry `‖fl(Ay) − Ay‖ ≤ ε_A‖A‖‖y‖ + η_A`,
`‖fl(P⊥y) − P⊥y‖ ≤ ε_P‖y‖ + η_P` (`etaP_up`) and propagate them through `lem:onestep`
(`η_B`), `eq:FI` and the norm bounds (`+2nη` on sums of squares). With `η = 2^-1074` these terms
are below `1e-300` at the sizes used and do not change any number printed; they make the
certificate hold when iterates underflow, which the power-of-two scaling makes rare but does not
exclude. For a matrix-free operator `etaA` defaults to 0, the paper's model.

Threads: blocks of columns are tasks of `Threads.@threads`, each with its own accumulators
(declared `local`; a shared accumulator was a bug found during development and fixed), combined
with upward rounding at the end; BLAS is set to one thread during the sweep. The user's
`Amul`/`Aadj` must be safe to call from several threads.

## §5 The certified splitting (`src/frame.jl`, `src/splitting.jl`)

| label | code | notes |
|---|---|---|
| `eq:2block` | `SplitCertificate` | |
| `def:Q`, `sec:frame` | `two_sided_frame`, `eigendirections`, `gkl`, `rrqr_frame` | Arnoldi eigendirections with `|θ| > R` (for a real operator, real and imaginary parts), right and left Ritz singular vectors of Golub–Kahan–Lanczos, column-pivoted QR with columns `|R_ii| < 1e-8|R_11|` dropped; `sides = :right/:left` for the one-sided rows of the table |
| `prop:frame` | not code (tests of the regime in `bench/regime_figure.jl`) | |
| `sec:exterior`, `cor:gap` | `certify_exterior`, `certify_gap(sc, R)`, `certify_gap(Amul, Aadj, n; R, epsA)` | |
| `sec:depth`, `eq:depthrule` | `choose_frame`, `float_split_estimates` | floating point: ks grows by a factor 1.5 until `γ1γ2 < σ11(R − ḡ)` with estimates (stochastic trace estimate for ḡ, SVDs for the couplings, SVDs on 32 points of `|z| = R` for σ11); after a failure a 30-step Arnoldi on G22 adds Ritz vectors with `|θ| > R` and residual `< 1e-6|θ|`; ceiling `kmax` (`rem:limits`) |
| `sec:almostortho`, `prop:almostortho`, `eq:alpha`, `eq:alphacond` | `frame_alpha`, `isometry_bounds`, `sigma11_lo` | `σ11 = σ̃11/(1 + α)`, `γ_i/√(1 − α)`; the product of the two factors is the paper's `(1+α)/(1−α)` |
| `sec:sound` | the whole pipeline | |

Entry points (matrix-free signature `(Amul, Aadj, n; kwargs...)` and dense wrappers):
`certify_split`, `krylov_resolvent_floor`, `certify_leading_eigs`, `certify_gap`,
`certify_selfadjoint`, `pseudospectrum_outside`, `miyajima_rump`.

## Not done

- The certification of `‖G22^m‖` for `m > 1` by a sweep of `G22^m` (sec:exterior mentions it, but
  `prop:fpchecks` is stated for `B = G22` only; the one-step error of `m` applications and the
  corresponding check are not in the paper). `eq:farpower` is implemented as a formula consuming
  caller-supplied `ḡ_j`.
- The closed form of `σ_min(P_i − zI)` for 2×2 blocks (`svdbox` is used).
- BigFloat: the old code offered `prec = BigFloat`; the new code is Float64/ComplexF64 only
  (the rounding helpers are generic, the constants `η`, `u` and the sweep are not).
- The Collatz bound (`rem:positive`) and the sweep fallback for the couplings of an implicit `Q`
  (sec:coupling): not implemented.
- A certified basis of `ran P₀` itself: the basis returned by `certify_projector` is the
  floating-point `Q·W_S`, and what is certified about it is `subspace_gap`, its distance to
  `ran P₀`.
- A fallback from the split to the exact-SVD escape when the split's arc cover fails: the caller
  has to ask for `method = :svd`. Taking the larger of the two bounds at each `z` is sound and
  costs the `svdbox` the split exists to avoid.
