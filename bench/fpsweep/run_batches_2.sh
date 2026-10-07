#!/bin/bash
# Hermitian set (converged-Ritz frame), κ scans, and large operators (no dense baseline).
cd "$(dirname "$0")"
L=~/ResearchData/KrylovPseudospectra_Data/fpsweep/results/logs; mkdir -p $L
J="julia --project=. -t 128"
$J run_gallery.jl gallery_hermitian.csv nos3 nos5 nos6 nos7 bcsstk08 bcsstk13 bcsstk14 bcsstk15 bcsstk16 Si2 SiH4 USpowerGrid_lap laplace2d_N60 anderson3d_L16 > $L/gallery_hermitian.log 2>&1
KS=0,10,20,40,60,80,100,120,140,160,200,240,280,320 ITERS=200 $J run_gallery.jl kappa_graded.csv graded1000 > $L/kappa_graded.log 2>&1
KS=0,20,40,60,70,80,90,100,120,150,200,250 ITERS=200 $J run_gallery.jl kappa_landau.csv landau400 > $L/kappa_landau.log 2>&1
KS=0,5,20 ITERS=100 $J run_gallery.jl gallery_large.csv dw8192 benzene fv1 af23560 Andrews convdiff_fd_N125 skewlap3d_N41 laplace2d_N300 anderson3d_L50 > $L/gallery_large.log 2>&1
