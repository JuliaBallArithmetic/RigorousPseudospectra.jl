
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
