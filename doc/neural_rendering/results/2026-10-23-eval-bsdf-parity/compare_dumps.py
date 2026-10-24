"""Compare eval_bsdf dumps from different devices record by record (same seed => same queries)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
COLS = ["f_cos_R", "f_cos_G", "f_cos_B", "pdf", "albedo_R", "albedo_G", "albedo_B"]


def load(dev):
    d = np.load(HERE / f"dump_{dev}.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def compare(ref, test):
    for k in ("prims", "uv", "wi", "wo"):
        assert np.array_equal(ref[k], test[k]), f"query array {k} differs: not the same records"
    a, b = ref["out"].astype(np.float64), test["out"].astype(np.float64)
    flags_equal = bool(np.array_equal(a[:, 7], b[:, 7]))
    ok = (a[:, 7] == 0) & (b[:, 7] == 0)
    rows = {}
    for i, name in enumerate(COLS):
        x, y = a[ok, i], b[ok, i]
        assert np.isfinite(x).all() and np.isfinite(y).all(), f"non-finite values in {name}"
        absd = np.abs(x - y)
        rel = absd / np.maximum(np.abs(x), 1e-6)
        rows[name] = {"max_abs": float(absd.max()), "max_rel": float(rel.max()),
                      "p99_rel": float(np.quantile(rel, 0.99)), "median_rel": float(np.median(rel)),
                      "ref_mean": float(x.mean()), "test_mean": float(y.mean())}
    return {"records": int(len(a)), "valid_both": int(ok.sum()), "flags_equal": flags_equal,
            "flagged_ref": int((a[:, 7] != 0).sum()), "flagged_test": int((b[:, 7] != 0).sum()),
            "columns": rows}


def selftest():
    rng = np.random.default_rng(0)
    q = {"prims": np.arange(4), "uv": rng.random((4, 2)), "wi": rng.random((4, 3)), "wo": rng.random((4, 3))}
    out = np.ones((4, 8), np.float32)
    out[:, 7] = 0
    r = compare(dict(q, out=out), dict(q, out=out * np.float32(1.001)))
    assert r["valid_both"] == 4 and abs(r["columns"]["pdf"]["max_rel"] - 0.001) < 1e-5
    try:
        compare(dict(q, out=out), dict(q, prims=q["prims"] + 1, out=out))
        raise AssertionError("mismatched queries must be rejected")
    except AssertionError as e:
        assert "query array" in str(e)
    print("compare selftest PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    ref = load("CPU")
    result = {dev: compare(ref, load(dev)) for dev in ("CUDA", "OPTIX")}
    (HERE / "parity.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
