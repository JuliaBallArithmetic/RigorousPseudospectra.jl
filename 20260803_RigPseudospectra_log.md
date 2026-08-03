
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
