# RigPseudospectra experiment log — 2026-07-28

## Why certify_gap_locked "beat" certify_gap on the Hénon gap — it was a loose opnorm

Investigation triggered by: "how can locked be better? the outside eigenvalues are already
enclosed… there must be a bug." Chased it with measurements (Hénon depth 6, n=4096, |z|=0.834,
nsep=1). Diagnostic scripts: RigorousUlam2D/examples/14h–14l.

### What was NOT the cause (measured, hypotheses killed)
- **svdbox looseness on the big block:** NO. `svdbox` σ_min of the full 256×256 Hessenberg
  block equals the true σ_min to every digit (14h). Verified SVD is tight here.
- **Arnoldi ghost dragging σ_min down:** NO. Materialized the full 4096² operator (14i):
  true σ_min(A−zI) on the circle is genuinely small (min 0.0345 at z=−0.834), even smaller
  than the block's 0.0497. A is strongly non-normal there — nearest true eigenvalue is 0.159
  away while σ_min=0.034 (4.6× pseudospectral bulge). Both certs are SOUND lower bounds on it.
- **Wrong deflation projector / invariant subspace:** partially. certify_gap deflates the RAW
  Krylov basis V, not the RM frame — but span(V)=span(V·mr.W), same projector. Measured (14k):
  the certified leading invariant frame drives the FORWARD coupling to ~1e-15 but the ADJOINT
  coupling stays ~1.3 for every frame (rank 4…160) — a right-invariant subspace is not
  A*-invariant; intrinsic non-normality, unfixable by a bigger/better right frame.

### The actual cause (measured, 14l)
`_coupling_cert` bounded γ with `upper_bound_L2_opnorm` = min(Collatz, √(‖·‖₁‖·‖∞)). On the
WIDE n×256 whole-Krylov deflation this overestimates the spectral norm by ~√rank:
**cheap 5.34 vs svd_bound 1.325 vs true 1.325 (4×).** That inflated γ alone sank certify_gap's
assembly floor: `0.0497·0.406/(0.0497+0.406+γ)` − arc = +6.2e-3 (γ=1.33) vs −1.6e-3 (γ=5.34).
certify_gap_locked only "won" because its narrow m=4 frame is where the cheap bound is already
tight (2.21≈true).

### Fix (committed, RPS f4760fb, branch feat/hkk-locking)
`_coupling_cert` → `svd_bound_L2_opnorm` (user pointed to it). γ is z-independent (computed once),
so the O(n·m²) SVD is negligible. End-to-end re-run (14j), |z|=0.834:
- **certify_gap: nθ=256 → floor +1.06e-3 ✓** (γ 5.33→1.32). Was −6.7e-3 ✗.
- certify_gap_locked: nθ=256 → −3.0e-4 ✗ (γ 2.21→1.26; also benefits).
So with an honest γ, **certify_gap (three-block good+bad-in-block) now certifies with FEWER
angles and beats locked** — its σ11·σ22 numerator is bigger (small dir in captured block paired
with the large far σ22=0.40, vs locked splitting it to σ22=0.095). Vindicates the "info between
good & bad, ugly far away" structure.

### Also this session (unrelated legs)
- Tighter Hénon mixing-rate circles certify: r=0.750 (nθ=2048) σ≥+1.04e-3 ✓, r=0.720 (nθ=3072)
  σ≥+3.92e-4 ✓ → certified L¹ rate ≤ 0.720 (was ≤0.834). Pointwise floors were always positive;
  only the arc-cover needed finer nθ.
- draft.tex §matfree: spelled out Arnoldi/Hessenberg (captured block) vs tridiagonal Lanczos on
  the Hermitian M_z (far block) — answers Poloni's "A non-symmetric but tridiagonal Lanczos?".

### Open / next
- Switch example 14 back to certify_gap (now the better + cleaner one)? It matches the email's
  three-block story and certifies at nθ=256.
- Bigger direction (Nisoli's suggestion): follow the σ_min=ε pseudospectral level curve instead
  of a circle (the dormant src/pseudospectra/ engine) — hugs the 4.6× non-normal bulge.

## Consolidated matvec isolation pipeline (certify_leading_eigs_sing) — evening

New function `certify_leading_eigs_sing` (RPS commits 50d82ed, e41fcfb): good + deep-far
architecture. Arnoldi (small eigen block) + A*A Lanczos (singular block, `_singular_block`,
stopping rule "peel σ until < |λ_g|", plateau-count detection robust to A*A's large nullspace)
→ Q = orth(V∪U) a FAITHFUL compression (σ_min(Q*AQ−zI) ≈ σ_min(A−zI) to ~3%, measured 14t) →
VBD on G11 + deep far block (‖G22‖<|λ_g|, matvec far certifier converges easily).

Key measurements (Hénon depth 6, examples 14h–14w):
- svdbox on the FULL 256-block is TIGHT (14h); no Arnoldi ghost — true σ_min(A−zI) genuinely
  small on the gap circle, 4.6× non-normal bulge (14i).
- Eigenvector deflation is 3.7× INEFFICIENT at the operator norm: ‖P_V A P_V‖=0.43 vs σ_257=0.12,
  because σ₁(H)=1.48 < σ₁(A)=1.71 — the top singular directions leak outside the Krylov space
  (14k, 14o, 14p). Fix = deflate A's SINGULAR vectors (A*A Lanczos), not eigenvectors.
- The block's own singular vectors span span(V) ⇒ deflating them = deflating V (0.43, identical);
  only A*A's singular vectors reach σ_{k+1} (14p). Bit-exact confirmation.
- adaptive pearl cover (point svdbox + 1-Lipschitz, ρ=safety·σ_min(z)) replaces uniform nθ:
  Hénon 5 leading all ✓ in 34s (was 175s at nθ=384). NOTE M−Ball(z,ρ)I perturbs each diagonal
  independently (far looser) — common-shift Lipschitz is the right pearl.

Results:
- **Hénon, 5 leading eigenvalues: ALL certified** σ_min ≥ 1.02e-2 … 1.78e-4 (matvec-only, 34s).
- **Lorenz, Float64: 3/5** (0.999→2.18e-2, 0.315 pair→4.8e-5); deepest pair (|λ|=0.269=|λ_g|)
  σ11 slightly negative — κ₂=4.7e5, svdbox hits Float64 floor; needs BigFloat (a `prec` knob TODO).
- **Rigorous pseudospectra plot** outside the far circle (`pseudospectrum_outside`, svdbox(G11) —
  κ-free/faithful; the block resolvent would pay κ₂=5036, useless): examples/14_henon_pseudo.png.

draft.tex §matfree updated with the deep-far-block-by-singular-deflation subsection.
Old certify_leading_eigs / certify_gap / certify_gap_locked kept intact.
