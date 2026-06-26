# Free estimate with the worst block removed

Left: the free triangular gap over **all** eigenvalues — dragged negative (failing, dark; red = `g=0`) by the high-tail eigenvalues. Middle: the same gap with the **worst block removed** (eigenvalues whose off-diagonal tail exceeds the threshold — red diamonds), recomputed over the kept eigenvalues (white ×). Right: true `σ_min`. The middle panel is a diagnostic (it ignores the dropped eigenvalues, so it is not a certified bound for `A`) — it shows how much of the plane the free formula recovers once the worst block is set aside for the block SVD.

| matrix | eigenvalues dropped | tail threshold | free works (all) | free works (worst removed) |
|---|---|---|---|---|
| normal16 | 4 / 16 | 0.00 | 100% | 100% |
| grcar32 | 8 / 32 | 3.05 | 5% | 17% |
| jordan8 | 0 / 8 | 1.00 | 36% | 36% |
| two_clusters | 1 / 6 | 2.00 | 40% | 51% |
| chain | 0 / 12 | 0.70 | 67% | 67% |

## normal16

![free_reduced_normal16](figures/free_reduced_normal16.png)

## grcar32

![free_reduced_grcar32](figures/free_reduced_grcar32.png)

## jordan8

![free_reduced_jordan8](figures/free_reduced_jordan8.png)

## two_clusters

![free_reduced_two_clusters](figures/free_reduced_two_clusters.png)

## chain

![free_reduced_chain](figures/free_reduced_chain.png)

