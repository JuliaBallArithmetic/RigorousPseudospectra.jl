# Free-estimate field report

**Left panel** — the cheap certified lower bound `g(z) = max(gap_tri, gap_svd, gap_recentred)`. At each `z`, `g(z)` is the **best ε we can certify exclusion for from the formulas alone**: `z` is provably outside `σ_ε` for every `ε < g(z)`. The **red curve `g(z)=0`** is the failure boundary — inside it the free formulas certify nothing; moving outward, `g` grows toward the far-field value. **Right panel** — the true `σ_min` for comparison (same colour scale). × = eigenvalues.

| matrix | box-fraction where free works (`g>0`) | max free `ε` | max true `σ_min` |
|---|---|---|---|
| normal16 | 100% | 3.25 | 3.25 |
| grcar32 | 5% | 0.90 | 2.56 |
| jordan8 | 36% | 0.56 | 0.67 |
| two_clusters | 40% | 1.27 | 1.57 |
| chain | 67% | 1.80 | 1.67 |

## normal16

![free_normal16](figures/free_normal16.png)

## grcar32

![free_grcar32](figures/free_grcar32.png)

## jordan8

![free_jordan8](figures/free_jordan8.png)

## two_clusters

![free_two_clusters](figures/free_two_clusters.png)

## chain

![free_chain](figures/free_chain.png)

