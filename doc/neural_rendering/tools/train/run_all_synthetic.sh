#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Torch-free end-to-end smoke test: unit tests, synthetic dump, numpy training (a few hundred steps),
# .nnm export + verification, evaluation with sphere render. Needs python3 + numpy only.
#
#   ./run_all_synthetic.sh [OUTDIR]        (default: a fresh temp dir; kept and printed)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$(mktemp -d)}"
mkdir -p "$OUT"
cd "$HERE"
PY="${PYTHON:-python3}"

echo "== 1. torch-free unit tests"
"$PY" test_torchfree.py

echo "== 2. synthetic dump (analytic layered BRDF, material.py) + sphere queries"
"$PY" make_synthetic_dump.py "$OUT/synthetic.npz" --n 200000 --sphere "$OUT/synthetic_sphere.npz"

echo "== 3. numpy training (400 steps) and a random-weights model"
"$PY" numpy_train_small.py "$OUT/synthetic.npz" "$OUT/trained.npz" --steps 400 --batch 4096
"$PY" numpy_train_small.py "$OUT/synthetic.npz" "$OUT/random.npz" --steps 0

echo "== 4. export to .nnm (re-read, test vectors, float64 reference, fp16 check)"
"$PY" export_nnm.py "$OUT/trained.npz" "$OUT/trained.nnm"
"$PY" export_nnm.py "$OUT/random.npz" "$OUT/random.nnm"

echo "== 5. evaluate against the held-out records and the sphere ground truth"
"$PY" evaluate.py "$OUT/trained.nnm" --dump "$OUT/synthetic.npz" --sphere-dump "$OUT/synthetic_sphere.npz" \
    --out "$OUT/eval_trained" --assert-rel-l1 0.8
"$PY" evaluate.py "$OUT/random.nnm" --dump "$OUT/synthetic.npz" --sphere-dump "$OUT/synthetic_sphere.npz" \
    --out "$OUT/eval_random"

"$PY" - "$OUT" <<'PYEOF'
import json, sys
o = sys.argv[1]
t = json.load(open(f"{o}/eval_trained/metrics.json"))
r = json.load(open(f"{o}/eval_random/metrics.json"))
assert t["rel_l1"] < 0.8 * r["rel_l1"], (t["rel_l1"], r["rel_l1"])
assert t["sphere_rel_l1"] < 0.5 * r["sphere_rel_l1"], (t["sphere_rel_l1"], r["sphere_rel_l1"])
print(f"trained rel_l1 {t['rel_l1']:.3f} (random {r['rel_l1']:.3f}); sphere {t['sphere_rel_l1']:.3f} (random {r['sphere_rel_l1']:.3f})")
PYEOF

echo
echo "ALL OK. Artifacts in $OUT (trained.nnm, eval_trained/sphere_compare.png, eval_trained/error_map.npy)"
