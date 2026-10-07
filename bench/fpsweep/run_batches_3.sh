#!/bin/bash
# Ball-arithmetic sweeps, thread scaling, and the n = 250 000 stencil (calibration for n = 10^6).
cd "$(dirname "$0")"
L=~/ResearchData/KrylovPseudospectra_Data/fpsweep/results/logs; mkdir -p $L
J="julia --project=. -t 128"
julia --project=. -t 16 run_constants.jl constants.csv grcar500:10 graded400:40 graded1000:100 landau150:10 nos5:10 convdiff_fd_N30:10 tols2000:10 qh1484:40 convdiff_cheb200:20 chebdiff300:20 anderson3d_L16:10 > $L/constants.log 2>&1
$J run_threads.jl threads.csv convdiff_fd_N60 10 1 > $L/threads_convdiff60.log 2>&1
$J run_threads.jl threads.csv af23560 20 1 > $L/threads_af23560.log 2>&1
$J run_threads.jl threads.csv convdiff_fd_N125 10 1 8,16,32,64,128 > $L/threads_convdiff125.log 2>&1
$J run_ballsweep.jl ballsweep.csv grcar2000:0:0 grcar2000:20:0 olm2000:20:0 \
   convdiff_cheb200:0:0 convdiff_cheb200:10:0 convdiff_cheb200:20:0 convdiff_cheb200:40:0 \
   chebdiff300:20:0 qh1484:40:0 nos5:20:0 anderson3d_L16:20:0 \
   graded1000:20:0 graded1000:60:0 graded1000:100:0 graded1000:140:0 \
   convdiff_fd_N125:10:512 af23560:20:256 > $L/ballsweep.log 2>&1
KS=0,5 PMAX=2 BLOCK=32 ITERS=50 $J run_gallery.jl gallery_huge.csv convdiff_fd_N250 > $L/gallery_huge_N250.log 2>&1
