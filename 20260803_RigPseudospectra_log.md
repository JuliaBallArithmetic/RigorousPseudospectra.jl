
## 2026-08-03 — gallery vs the updated BallArithmetic VBD; noise sweep

**BallArithmetic updated by Nisoli** (e8cb587 = my drop-column-norm-merge + gated/monotone sweeps;
e8afacd = "race three candidate bases, keep the best certified one"). The racing RESOLVES the
separation failure:
  Kahan60    1 cluster r≤15.0   →  60 singletons r≤1.36e-8   (9 orders)
  convdiff80 1 cluster r≤156    →  80 singletons r≤1.70e-2   (4 orders)
  Grcar100   1 cluster r≤3.83   →  unchanged (correct: cond(V)=5.1e17 > 1/u)
  Jordan20 / Demmel10           →  1 cluster (correct: defective)
Containment 100% in every case. bench/nonnormal_gallery.jl rewritten onto BallArithmetic's
schur_newton_vbd (the kernel owns the enclosure; this package layers the method).

**Pipeline re-verified against the updated kernel — both improved:**
  Lorenz R=0.6: ‖G22‖=0.2337, #out=1, σ_min≥2.271e-2, r_heur 5.2e-7 → 2.0e-15, enclosure 1.6e-6 → 1.0e-7
  Hénon  R=0.6: ‖G22‖=0.4446, #out=4, all ✓, enclosure 1.0e-7

**NOISE SWEEP (Hénon, depth 6, R=0.6) — the regime boundary, measured:**
  noise 0.20 (rx=3): nsing=18 dimQ=98  ‖G22‖=0.257  #out=4  207s
  noise 0.10 (rx=2): nsing=54 dimQ=134 ‖G22‖=0.442  #out=4  211s
  noise 0.05 (rx=1): nsing=97 dimQ=177 ‖G22‖=0.472  #out=7  266s
Deflation grows 18→97 as noise halves twice; ‖G22‖ rises toward R; ALL still certify (~20% margin).
Second effect: the subdominant spectrum moves OUT as noise falls (|λ2| = 0.6425 → 0.6955 → 0.7457) —
less smoothing ⇒ less contraction ⇒ closer to the deterministic operator. Both sides squeeze.
LIMIT: depth 6 floors at rx=1; depth 7 is barred by the FULL-BASIS memory O(n·N) ≈ 4 TB at n=16384
(260 MB at n=4096). That is a limit of the certificate, not the operator.

**Fixes:** Jordan BoundsError (thin QR gives only n columns when dimVg+|keep|>n — 30x30 asked for 45)
+ degenerate empty far block (Ng≤0 ⇒ ‖G22‖=0, no Lanczos). κ₂ NaN → Inf (defective input overflows
cond(V); Thm 1 doesn't use κ₂ but the resolvent transport divides by it and NaN passes every
comparison silently).

**Paper (20 pp):** Table 1 rebuilt on the measured gallery. Demmel written up properly per Nisoli's
question: Thm 1 does NOT use κ₂, but it DOES use R₂ via β_Λ = ‖N‖ + ‖R₂‖‖Ã‖/(1−‖R₂‖); for Demmel
N=0 and ‖R₂‖=4.4e-16 yet ‖Ã‖=1e36 ⇒ β_Λ=4.4e20 ⇒ radius 1e36 = ‖A‖. SOUND BUT VACUOUS — the bound is
relative to ‖A‖, and a 36-decade graded spectrum carries no information about small eigenvalues at
working precision. Earlier wording ("the enclosure is unaffected") was misleading and is corrected.

**CAVEAT:** the noise sweep ran while BallArithmetic was being edited (a recompile happened mid-run);
the trend dwarfs any plausible perturbation but the numbers should be regenerated before publication.

### Demmel resolved (Nisoli: "our Krylov approach may work with demmel") — it does, in precision

Demmel's singular values DO collapse (σ₂/σ₁ = 2.2e-24, numerically rank 1) — genuinely the Krylov
regime. Two separate effects, measured:
 (1) exact-SVD escape avoids the ‖Ã‖ factor of β_Λ ⇒ localization 1e36 → 1e30 (6 orders) in Float64;
 (2) but a verified SVD has an ABSOLUTE floor u‖A‖ = 1e-16·1e36 = 1e20, so Float64 can never resolve
     an eigenvalue at −1. Obstruction is PRECISION, not block structure.
Extended precision settles it (pipeline is generic in the float type):
  256 bits: certifies r=1.0 (σ_min ≥ 3.8e-39); fails r=0.1
  512 bits: certifies r=1.0, 0.1, 0.01 → σ_min ≥ 3.9e-39, 4.7e-47, 9.2e-57
The three values drop ~10 decades per decade of r: the |z−λ|^m law of Thm 1 for the m=10 Jordan
block, OBSERVED. Demmel is now an illustration of the block region, not a failure. Paper updated.

### Matrix Market NEP (bench/matrixmarket.jl) — real sparse non-normal operators

R = √(|λ1||λ2|), so a successful run certifies a spectral gap. Tolosa is NOT served at any
pub/MatrixMarket2/NEP/tolosa/... path (all 404); olmstead, brussel (rdb), dwave, airfoil are.

  rdb200   n=200  σ50/σ1=0.59  ‖A‖/ρ=1.04  ‖G22‖=28.7 < 32.9 ✓  → CERTIFIED GAP |λ2/λ1| ≤ 0.986
  olm500   n=500  σ50/σ1=0.91  ‖A‖/ρ=9.09  ‖G22‖=2190 < 2544 ✓  → 5 certified (dimQ=279 of 500!)
  rdb800l  n=800  σ50/σ1=0.85  ‖A‖/ρ=1.05  ‖G22‖=27.7 < 30.9 ✓  → 3 certified
  olm1000  n=1000 σ50/σ1=0.98  ‖A‖/ρ=9.06  ‖G22‖=91955 > 10163 ✗ → DECLINES (hypothesis fails)

SHARPENS THE REGIME CLAIM: rdb200 certifies with σ50/σ1=0.59 (almost no decay), olm1000 fails with
a similar profile. The predictor is not raw singular decay but ‖A‖/ρ(A) — the departure from
normality. rdb: ratio ≈1.05, small dimQ (93/200, 127/800). Olmstead: ratio ≈9; olm500 only succeeds
by capturing 279 of 500 dims (brute force), olm1000 cannot. Honest boundary = "‖G22‖ < R after an
affordable deflation", not "singular values decay".

### Test suite GREEN

test/runtests.jl now passes fully (was 5 pass / 8 fail all session). The failing test asserted that
certify_leading_eigs isolates 4 eigenvalues of a matrix whose spectrum is 1.000, 0.794, 0.757, 0.753,
0.734 — a tight cloud with ~4e-3 gaps that CANNOT be isolated. Rewritten to assert what is actually
guaranteed: the dominant (separated) eigenvalue certifies, and SOUNDNESS (floor ≤ true σ_min) holds
for every result whether or not it certifies. nblock 16 → 80 (at 16 the certified ‖G22‖=0.94 > |λ2|).

### CORRECTION (Nisoli): Hénon DOES certify a spectral gap — sharply

My sweep only printed a gap when #outside==1. BACKWARDS. By the exact-count proposition the certified
list outside R is COMPLETE, so with ≥2 eigenvalues outside R the second-largest modulus is KNOWN
(the runner-up of the certified list, enclosed to 1e-7) and the gap is SHARP — a BETTER statement
than the |λ2| ≤ R bound available when only one is outside.

HENON (R=0.6): 0.30 → ≥0.4000 (1 out); 0.20 → 0.3575 SHARP (|λ2|=0.6425); 0.10 → 0.3045 SHARP
(0.6955); 0.05 → 0.2543 SHARP (0.7457). Gap NARROWS as noise falls (less smoothing ⇒ less
contraction ⇒ subdominant spectrum moves out).
LORENZ (R=0.6): only the Perron eigenvalue outside at every level ⇒ lower bounds only:
≥0.2477 / 0.3754 / 0.3988 / 0.3996. Sharpening needs R lowered toward its true |λ2| ≈ 0.32.
Fixed in 15_noise_sweep.jl; bench/results/certified_gaps.txt; paper Table tab:gaps.
Hénon 0.2/0.1/0.05 were run twice (two independent runs) and agree to 3 digits — reproducibility check.

### Pseudospectral obstruction (Nisoli's observation, now a Remark)
Eckart–Young forces ‖G22‖ ≥ σ_{k+1}(A), so the hypothesis needs σ_{k+1}(A) < R ≈ ρ(A). For a NORMAL
operator σ_j = |λ_j| and this is free. The EXCESS of σ_j over |λ_j| is non-normality, whose resolvent
form IS the pseudospectrum. So the regime condition is not an algorithmic artifact but the
pseudospectral geometry: the method certifies exactly when the PSEUDOSPECTRUM (not the spectrum) fits
inside R after an affordable deflation. ‖A‖/ρ(A) is the crude proxy and already predicts the Matrix
Market outcomes (rdb ≈1.05 certifies, Olmstead ≈9 struggles/fails).

### Lorenz recomputed with PERIODIC boundary (the earlier runs were ABSORBING — wrong)

Nisoli: "the certificate for Lorenz was with absorbing boundary condition, we need periodic."
Symptom of the old bug, visible in our own output: λ1 = 0.9988 not 1, and ‖A*1−1‖/‖1‖ = 1.27e-2
(mass leaking out of the absorbing boundary). Also fixed: rx = round(w/cellwidth), since
uniform_noise_box(w,2) has HALF-WIDTH w (so the earlier rx labels were half the true value).

LORENZ, periodic, R=0.6, depth 6 — λ1 = 1.0000 EXACTLY at every level (mass conservation):
  noise 0.30 rx=19 nsing=2  dimQ=82  ‖G22‖=0.0689  #out=1  gap ≥ 0.4000  227s
  noise 0.20 rx=13 nsing=6  dimQ=86  ‖G22‖=0.1108  #out=1  gap ≥ 0.4000  176s
  noise 0.10 rx=6  nsing=12 dimQ=92  ‖G22‖=0.2351  #out=1  gap ≥ 0.4000  157s
  noise 0.05 rx=3  nsing=32 dimQ=112 ‖G22‖=0.3876  #out=1  gap ≥ 0.4000  152s
All σ_min ≥ 6.7e-2 / 5.3e-2 / 2.3e-2 / 8.5e-3, enclosures 1e-7. Cleaner than the absorbing run:
uniform gap ≥ 0.4 and an exact Perron eigenvalue. Paper §6.1 + Table tab:gaps updated;
bench/results/ and RigorousUlam2D/results/ regenerated.

### SESSION CLOSE (Nisoli AFK ~6 months from 2026-08-03)
Memory written: krylov-paper-state-2026-08.md (state + open items),
certified-lanczos-needs-full-basis.md (the Prop-5 lesson).
NOTE: RigPseudospectra (feat/hkk-locking) and RigorousUlam2D (main) are AHEAD OF UPSTREAM and
NOT PUSHED; BallArithmetic (gram-weighted-certification) has no upstream configured.
