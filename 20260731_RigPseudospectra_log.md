
## 2026-07-31 — Block-Geršgorin sharpening of the outer Krylov resolvent floor

**What:** Added the Feingold–Varga block-dominance floor to the outer 2-block Krylov
assembly (`_assemble2` = max(Schur-complement ★, min(σ11,σ22)(1−τ)), κ-free since the
captured/far outer basis is unitary). Asymmetric coupling pair (‖G12‖,‖G21‖) threaded
through all call sites (`_coupling_pair`; `_coupling_cert` = max of pair). All sites
back-compat via scalar wrapper.

**Verification:** primitive never-worse than pure Schur over 1e5 random points; near-circle
(σ11~1e-2, γ~1.3) collapses to pure Schur (τ≫1, block-dominance dead); far field weak
coupling → ~2×.

**Demo (bench/blockdom_farfield.jl):** transfer-like smoothing operator n=300 (σ50/σ1=1e-8),
dimQ=52, ‖G22‖≈0, γ≈0 (pure decoupling limit). Over 19,880 far-field grid points the
block-dominance floor improves the pure-Schur floor by median 1.99×, max 2.00× (>1.05 at
99%). Figure figures/blockdom_farfield.png: block-dominance ε-contours strictly inside
Schur-only. Gain confined to far field (as predicted); vanishes on the tight isolation circles.

**Tests:** `test/runtests.jl` — the 8 Krylov-decoupling failures are PRE-EXISTING (identical
5-pass/8-fail on the clean baseline, nblock=16 too small to certify; soundness check passes).
No regression from the change.

**Paper (draft_krylov.tex → 14 pp):** restructured around Nisoli's three-pillar narrative
— (i) Miyajima–Rump enclosure, (ii) standalone **certified Lanczos decomposition** (promoted
from §5.1: partial-reorth tridiagonal + Paige-streamed R1/R2, Prop lanczosfloor, framed as
independently useful for λ_min of any Hermitian matvec op), (iii) the **certified Krylov
method** = the join. Added Prop outerbdom (block-dominance) + fig:farfield; Remark rem:schur
(one Schur-complement thread: block floor / localized / outer-dominance / assembly all Schur
complements; Lanczos = matrix-free eval of the far factor); Remark rem:rump (checked Rump2022a
PDF: discrete all-eigenpairs enclosure ONLY — no resolvent/σ_min/pseudospectrum/Schur/Krylov;
all of those are ours).

## 2026-07-31 (later) — Proposition 5 was UNSOUND; fixed by the full-basis similarity

**The gap (Nisoli spotted it).** Prop 5 claimed a σ_min floor "sound for any k". False. T_k = Q̄*MQ̄ is
a COMPRESSION, so min–max gives θ_min ≥ λ_min(M) ALWAYS — Ritz converges from ABOVE. A floor needs the
COMPLETENESS direction spec(M) ⊆ spec(T_k)+[−β,β], which residuals cannot supply: the deficit is the
angle ∠(u_min, Krylov), invisible to any residual. Worse, the code used R1 = Q̄*S — the residual
PROJECTED BACK onto the Krylov space, where S = (I−Q̄Q̄*)MQ̄ is orthogonal ⇒ ‖R1‖ ~ √u·β_k for EVERY k.
That projection annihilated exactly the evidence of incompleteness and made the floor look k-independent.

**The fix (Nisoli's call: full Lanczos).** At k = N the basis is SQUARE ⇒ the residual identity
MQ̄ = Q̄T + S becomes a SIMILARITY Q̄⁻¹MQ̄ = T + E, E = Q̄⁻¹S ⇒ spec(M) = spec(T+E) EXACTLY; Bauer–Fike
for the normal (symmetric) T gives spec(M) ⊆ spec(T) ± ‖E‖, ‖E‖ ≤ ‖S‖/√(1−ρ), σ_min(Q̄)² ≥ 1−ρ.
Residual enters BARE: S = −F + β_N q_{N+1}e_N* ⇒ ‖S‖ ≤ β_N + ‖F‖_F. All streamed. Self-policing:
at k<N the bare β_k = O(‖T‖) swamps θ_min ⇒ floor ≤ 0 (vacuous but SOUND).

**Breakdown is NOT completeness.** β_k≈0 at k<N ⇒ ran(Q̄_k) merely INVARIANT ⇒ spec(T_k) a near-SUBSET;
λ_min may live entirely on the untouched complement. "Happy breakdown" needs an uncertifiable genericity
assumption on the start vector (1e-18 component ≢ none). Fix: RESTART on breakdown (β_k:=0 block
boundary + fresh orthogonal vector) so the basis still reaches N; T becomes block-tridiagonal.

**Code:** far_floor(L, N) with bare-β + √(1−ρ) and a strict k≥N gate; lanczos_deviation;
lanczos_lambda_max (the ‖G22‖ UPPER bound needs the SAME completeness — a missed LARGE eigenvalue is
equally invisible; it was being taken from a 40-step run!). Default kmax → full N = n−m (was n−m−1).
Restart-on-breakdown in lanczos_pro.

**Verified:** partial k → refuses (0/Inf); full k=N → SOUND + TIGHT (λfloor 9.91e-7 vs true 1.00e-6;
low-rank/breakdown case exactly 0.25; λmax exact 3.000000). Corrected bound is TIGHTER than the old
projected-R1 one at full basis. Honest note: I could NOT numerically break the old bound even with a
start ⟂ u_min (|⟨u1,v⟩|=5e-18) — rounding reinjects the direction and Lanczos converges anyway (Paige
ghosts). The gap was in the PROOF, not in observed behaviour; that is why production runs looked fine.

**Impact:** bench/blockdom_farfield.jl reproduces IDENTICAL numbers with full Lanczos (median 1.987,
max 2.000), now with a certified ‖G22‖. test/runtests.jl 5-pass/8-fail = unchanged pre-existing baseline.

**Paper (16 pp):** Prop 5 restated (full basis, two-sided, η/√(1−ρ)) with a 5-step proof (invertibility
via Weyl; columnwise S; squareness⇒similarity; Bauer–Fike for normal T; extremal bounds). Added
"why a partial basis cannot give a floor" (min–max + the projection argument), rem:breakdown,
rem:cost ("a certified method is necessarily more expensive"), and rem:sturm — Nisoli's point that
λ_min(T_N) is certified by MIYAJIMA–RUMP itself (κ₂=1 since T_N symmetric): pillar (i) discharges
pillar (ii)'s hypothesis, so the chain closes with no external verified eigensolver (O(N) Sturm
inertia bisection noted as the cheap structured alternative).

### Certifying θ_min(T_N) without densifying — Rump's Cholesky trick

**Nisoli's two points:** (a) λ_min(T_N) can be certified by Miyajima–Rump itself; (b) better, by
Rump's positive-definiteness trick — if the LIBRARY-optimized float64 Cholesky of T_N − αI succeeds,
then λ_min > α. Concern raised: *does the eigensolver/Cholesky densify?*

**Measured (this is the answer):** `ldlt(::SymTridiagonal)` stays STRUCTURED (`LDLt{…,SymTridiagonal}`),
O(N) time and memory: N=10^4/10^5/10^6 → 0.15/1.53/15.3 MB, 0.0002/0.001/0.015 s. Dense at N=10^6
would be 7,450 GB. So the library route is viable WITHOUT densification. Full certificate
(`cert_lambda_min_chol` structured overload) at N=10^6: 0.030 s, 38 MB.
M–R on T_N would work (κ₂=1, T_N symmetric) but is O(N³)/O(N²) dense ⇒ small-N only, NOT the default.

**Implemented:** `cert_lambda_min_chol(C, α)` — Rump/Demmel: float Cholesky of C−αI succeeds ⇒
C−αI+ΔC = RᵀR ⪰ 0, |ΔC| ≤ γ_{N+1}|Rᵀ||R| ⇒ λ_min(C) ≥ α − γ_{N+1}‖|Rᵀ||R|‖₁. Dense method +
SymTridiagonal overload (ldlt, O(N); |Rᵀ||R| tridiagonal ⇒ O(N) 1-norm; fixed an initial WRONG
row-sum formula that under-estimated). NOTE the shift is essential: unshifted Cholesky gives only
λ_min ≥ −‖ΔC‖ (vacuous). Also `certified_tridiag_extremes` (ball-arithmetic Sturm/inertia bisection,
O(N)/O(1)) wired into far_floor + lanczos_lambda_max, replacing the Float64 `eigvals`.

**Implementation validation (debugging, NOT paper evidence):** Sturm 1000 random tridiagonals, slack
≤4.5e-15 rel; Cholesky 500 cases; both agree with LAPACK λ_min to all printed digits; N=64000 Sturm
0.047 s. Tests unchanged (5 pass / 8 fail = pre-existing baseline).

**FEEDBACK (Nisoli, important):** "what does it mean empirically? this is a math paper, either it is
rigorous or it isn't." Removed all "sound in every case / 0 unsound" phrasing from the paper —
soundness is a THEOREM, only SHARPNESS is measured. Also fixed the Faithfulness paragraph, where a
"Hence" made rigor appear to follow from a measurement: svdbox(G11−zI) is κ-free rigorous for ANY Q;
faithfulness governs only tightness ("a poor choice of Q costs accuracy, never correctness").
Saved as memory rigor-is-proved-not-measured.md. Paper now 17 pp.
