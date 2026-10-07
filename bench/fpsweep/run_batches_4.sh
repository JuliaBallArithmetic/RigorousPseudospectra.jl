#!/bin/bash
# Block size b at 32 and 128 threads on the n = 62 500 stencil (threads_block.csv).
cd "$(dirname "$0")"
L=~/ResearchData/KrylovPseudospectra_Data/fpsweep/results/logs
for b in 4 8 16 64 32 4; do
  BLOCK=$b julia --project=. -t 128 run_threads.jl threads_block.csv convdiff_fd_N125 10 1 32,128 2>&1 | grep "T="
done > $L/threads_block.log
