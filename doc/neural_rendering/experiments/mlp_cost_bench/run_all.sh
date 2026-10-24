#!/usr/bin/env bash
# Runs the full benchmark matrix sequentially (benchmarks must not overlap: they
# are single-thread and sensitive to memory bandwidth / turbo). ~35-40 minutes.
# Logs go to results/<build>.log (CSV-ish "RESULT," lines, see make_table.py).
#   ./run_all.sh            full matrix
#   QUICK=1 ./run_all.sh    N=1M, 1 rep (smoke test, ~3 minutes)
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results
if [[ "${QUICK:-0}" == 1 ]]; then
  N=1e6; N_CYC=1e6; N64=5e5; REPS=1
else
  N=1e7; N_CYC=5e6; N64=3e6; REPS=3
fi

run() { # build, extra args...
  local b=$1; shift
  echo "### $b: $*" | tee -a results/progress.txt
  ./bin/bench_$b --reps $REPS "$@" > results/$b.log 2>&1
  echo "### $b done $(date +%T)" | tee -a results/progress.txt
}

: > results/progress.txt
# The builds requested in the task (N = 10M evals for everything but 64x3).
run gcc_fast       --n $N --n64 $N                                  # everything, incl. IN=12 (no latent)
run gcc_strict     --n $N --n64 $N64 --no-in12 --skip h64x3_in28_b8t
run clang_fast     --n $N --n64 $N64 --no-in12 --skip h64x3_in28_b8t
run gcc_fast_novec --n $N --no-in12 --only an_,h32x2_in28,stream
# Builds with Cycles' real AVX2 + math flags (no -ffast-math), 5M evals.
run gcc_cycles     --n $N_CYC --n64 $N64 --no-in12 --skip h64x3_in28_b8t
run clang_cycles   --n $N_CYC --n64 $N64 --no-in12 --skip h64x3_in28_b8t
echo "ALL DONE $(date +%T)" | tee -a results/progress.txt

# Second pass for the cheap analytic/part variants with 7 reps each, to filter out
# transient noise from the shared VM; make_table.py keeps the min of both passes.
if [[ "${QUICK:-0}" != 1 ]]; then
  for b in gcc_fast gcc_strict clang_fast gcc_fast_novec gcc_cycles clang_cycles; do
    ./bin/bench_$b --n 5e6 --reps 7 --no-peak --no-in12 --only an_,part_,stream > results/${b}_extra.log 2>&1
  done
  echo "EXTRA DONE $(date +%T)" | tee -a results/progress.txt
fi
