#!/usr/bin/env bash
# Builds bench.cpp with every compiler/flag combination used in the report.
# Usage: ./build.sh            (binaries land in ./bin/)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin

STD="-std=c++17"
# Flags Cycles itself uses (intern/cycles/CMakeLists.txt): AVX2 kernel flags +
# CYCLES_MATH_FLAGS. Note: NO -ffast-math, so no reassociation (reductions stay serial).
CYCLES_AVX2="-mavx2 -mfma -mlzcnt -mbmi -mbmi2 -mf16c"
CYCLES_MATH="-fno-trapping-math -fno-math-errno -fno-signed-zeros -ffp-contract=on -freciprocal-math"
CYCLES_MATH_GCC="-fno-signaling-nans -fno-rounding-math -mfpmath=sse"

build() {  # name compiler flags...
  local name=$1 cxx=$2; shift 2
  echo "== $name: $cxx $*"
  "$cxx" $STD -O3 "$@" -DBUILD_NAME="\"$name\"" -DBUILD_FLAGS="\"$cxx -O3 $*\"" bench.cpp -o "bin/bench_$name" &
}

# The four builds asked for in the task:
build gcc_fast       g++     -march=native -ffast-math
build gcc_strict     g++     -march=native
build clang_fast     clang++ -march=native -ffast-math
build gcc_fast_novec g++     -march=native -ffast-math -fno-tree-vectorize
wait
# Extra builds that mimic how Cycles' AVX2 kernel is really compiled (no fast-math):
build gcc_cycles     g++     $CYCLES_AVX2 $CYCLES_MATH $CYCLES_MATH_GCC
build clang_cycles   clang++ $CYCLES_AVX2 $CYCLES_MATH
wait
echo "built:"; ls -la bin/
