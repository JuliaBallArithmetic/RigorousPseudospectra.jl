#!/bin/bash
# Repeats of the thread-scaling series on the n = 62 500 stencil, block sizes 32 and 4.
cd "$(dirname "$0")"
L=~/ResearchData/KrylovPseudospectra_Data/fpsweep/results/logs
for rep in 1 2; do for b in 32 4; do
  BLOCK=$b julia --project=. -t 128 run_threads.jl threads_repeat.csv convdiff_fd_N125 10 1 1,8,16,32,64,128 2>&1 | grep "T=" | sed "s/^/rep=$rep b=$b /"
done; done > $L/threads_repeat.log
