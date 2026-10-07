#!/bin/bash
# Gallery batches with dense baselines (n ≤ 6000). Reproduces gallery_*.csv.
cd "$(dirname "$0")"
L=~/ResearchData/KrylovPseudospectra_Data/fpsweep/results/logs; mkdir -p $L
J="julia --project=. -t 128"
$J run_gallery.jl gallery_eigtool.csv grcar2000 kahan500 hatano2000 airy200 davies200 landau400 orrsom200 basor500 convdiff_cheb200 chebdiff300 > $L/gallery_eigtool.log 2>&1
$J run_gallery.jl gallery_controls.csv gauss1000 gauss2000 jordan2000 chain2000 > $L/gallery_controls.log 2>&1
$J run_gallery.jl gallery_nep.csv olm2000 rdb800l rdb2048 dw2048 tols2000 pde2961 bwm2000 ck656 qh1484 tols4000 olm5000 rdb5000 rw5151 > $L/gallery_nep.log 2>&1
