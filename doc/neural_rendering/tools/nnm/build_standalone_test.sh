#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Proves that the Cycles kernel MLP decoder (intern/cycles/kernel/util/mlp.h) and the numpy
# reference decoder agree:
#   1. python3 -m nnm.selftest writes 27 .nnm models (3 widths x 3 activations x 3 output
#      transforms, with embedded test vectors made by the numpy reference decoder),
#   2. standalone_test.cpp is compiled against the real Cycles util/ headers with every available
#      compiler (g++, clang++) in two configurations (AVX2+FMA+F16C like the Cycles AVX2 kernel,
#      and the generic x86-64 baseline), and run on all models.
#
# Usage: build_standalone_test.sh [work_dir]     (default: a temp directory)
# Env:   CXXS="g++ clang++"  compilers to use
#
# Notes on the build:
# - Cycles util/ headers need C++20 (std::popcount), so -std=c++20, not 17.
# - The only compat defines needed are CCL_NAMESPACE_BEGIN/END (normally passed by
#   intern/cycles/CMakeLists.txt). No stub headers are required: util/half.h, util/math.h and
#   util/types.h compile standalone.
# - No -ffast-math, like the Cycles build.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tools="$(cd "$here/.." && pwd)"
repo="$(cd "$here/../../../.." && pwd)"
cycles="$repo/intern/cycles"
work="${1:-$(mktemp -d "${TMPDIR:-/tmp}/nnm_standalone.XXXXXX")}"
mkdir -p "$work"

echo "== generating models into $work/models"
(cd "$tools" && python3 -m nnm.selftest --out "$work/models")

defs=("-DCCL_NAMESPACE_BEGIN=namespace ccl {" "-DCCL_NAMESPACE_END=}")
incs=("-I$cycles" "-I$here")
status=0

for cxx in ${CXXS:-g++ clang++}; do
  if ! command -v "$cxx" >/dev/null 2>&1; then
    echo "== skipping $cxx (not found)"
    continue
  fi
  for cfg in avx2 generic; do
    case "$cfg" in
      avx2)    flags=(-mavx2 -mfma -mf16c) ;;
      generic) flags=() ;;
    esac
    exe="$work/standalone_${cxx}_${cfg}"
    echo "== $cxx -O2 -std=c++20 ${flags[*]:-} ($cfg)"
    "$cxx" -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter "${flags[@]}" "${defs[@]}" "${incs[@]}" \
      "$here/standalone_test.cpp" -o "$exe"
    if "$exe" "$work"/models/*.nnm > "$work/log_${cxx}_${cfg}.txt"; then
      grep -E '^(PASS|FAIL) ' "$work/log_${cxx}_${cfg}.txt" | grep -v '^PASS .*(width' || true
      tail -n 1 "$work/log_${cxx}_${cfg}.txt"
      head -n 1 "$work/log_${cxx}_${cfg}.txt"
    else
      cat "$work/log_${cxx}_${cfg}.txt"
      status=1
    fi
  done
done

if [ "$status" -eq 0 ]; then
  echo "ALL PASS (logs in $work)"
else
  echo "SOME FAILED (logs in $work)"
fi
exit "$status"
