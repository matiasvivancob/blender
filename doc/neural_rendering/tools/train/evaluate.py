#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""numpy-only evaluation of an `.nnm` against a dump (no torch, no Blender).

    python3 evaluate.py material.nnm --dump dump.npz [--sphere-dump sphere.npz] --out eval/

Decodes the model with nnm.reference (the same arithmetic as the kernel decoder, fp16 weights) on the
validation records of the dump and writes into --out:
  metrics.json        relative L1 overall / brightest 1% / grazing > 75 deg, pow-loss, peak energy
  error_map.npy       float32 (map_res, map_res): per-texel-bin relative L1 = sum|pred-gt| / sum(gt), NaN where no
                      records; row index = v, column index = u (same as the latent grid)
  sphere_compare.png  (with --sphere-dump) top: ground truth | model, bottom: 4x amplified |error|
By default only the validation split is used (same --split-seed / --val-frac / --val-max as train.py, so the
records were never seen in training); --split all uses every valid record.

The sphere dump holds the inputs of sphere.make_records() with `out` filled by the reference renderer for
the same material: make_synthetic_dump.py --sphere for the synthetic material, or the Cycles eval_bsdf
wrapper (blender_dump.py --fill) for a real one. Ground truth = its out[:, 0:3].
"""
import argparse
import json
import os
import sys

import numpy as np

import _paths  # noqa: F401
import dataset
import metrics
import png_io
import sphere
from nnm.format import ENC_BSDF_DIRS, OUT_EXP, read_nnm
from nnm.reference import decode


def decode_records(model, uv, wi, wo, chunk=32768):
    caller = np.concatenate([wi, wo], 1).astype(np.float32)
    uv = uv.astype(np.float32)
    return np.concatenate([decode(model, caller[s:s + chunk], uv[s:s + chunk]) for s in range(0, len(uv), chunk)])


def check_model(m):
    if m.encoding != ENC_BSDF_DIRS or m.latent_channels == 0 or m.output_transform != OUT_EXP or m.output_dim != 3:
        raise SystemExit("evaluate: expected a bsdf_dirs model with a latent grid, exp output and 3 outputs")


def error_map(tex_uv, pred, tgt, res):
    """Relative L1 per (res x res) bin of the texture domain, NaN where empty."""
    ix = np.clip((tex_uv[:, 0] * res).astype(int), 0, res - 1)
    iy = np.clip((tex_uv[:, 1] * res).astype(int), 0, res - 1)
    flat = iy * res + ix
    err = np.bincount(flat, weights=np.abs(pred - tgt).sum(1), minlength=res * res)
    den = np.bincount(flat, weights=tgt.sum(1), minlength=res * res)
    cnt = np.bincount(flat, minlength=res * res)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where((cnt > 0) & (den > 0), err / den, np.nan)
    return m.reshape(res, res).astype(np.float32), cnt.reshape(res, res)


def evaluate_records(model, rs, map_res=32):
    pred = decode_records(model, rs.tex_uv, rs.wi, rs.wo)
    res = metrics.compute_metrics(pred, rs.target, rs.wi, rs.wo)
    emap, cnt = error_map(rs.tex_uv, pred, rs.target, map_res)
    res["error_map_res"] = map_res
    res["error_map_max"] = float(np.nanmax(emap)) if np.isfinite(emap).any() else float("nan")
    return res, emap, pred


def render_sphere(model, sphere_path, out_dir):
    d = dataset.load_dump(sphere_path)
    rec = sphere.make_records(d.meta.get("sphere_size", sphere.SIZE))
    problem = sphere.check_dump_matches(rec, d)
    if problem:
        raise SystemExit(f"evaluate: sphere dump mismatch: {problem}")
    gt_rgb = np.where((d.out[:, 7] == 0)[:, None], np.maximum(d.out[:, 0:3], 0), 0)
    pr_rgb = np.where((d.out[:, 7] == 0)[:, None], decode_records(model, rec["tex_uv"], rec["wi"], rec["wo"]), 0)
    gt, pr = sphere.shade(rec, gt_rgb), sphere.shade(rec, pr_rgb)
    path = os.path.join(out_dir, "sphere_compare.png")
    png_io.write_png(path, sphere.compare_image(rec["mask"], gt, pr))
    return {"sphere_rel_l1": sphere.sphere_rel_l1(gt, pr), "sphere_png": path}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nnm")
    ap.add_argument("--dump", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--split", choices=("val", "all"), default="val")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-max", type=int, default=262144)
    ap.add_argument("--train-max-records", type=int, default=0,
                    help="the --max-records value given to train.py (it subsamples before the split)")
    ap.add_argument("--max-records", type=int, default=0, help="subsample the evaluated records (0 = all)")
    ap.add_argument("--map-res", type=int, default=32)
    ap.add_argument("--sphere-dump", default=None)
    ap.add_argument("--assert-rel-l1", type=float, default=None, help="exit 1 if overall relative L1 exceeds this")
    a = ap.parse_args(argv)

    model = read_nnm(a.nnm)
    check_model(model)
    rs, reports, _ = dataset.load_records(a.dump)
    print(f"{rs.n} valid records ({sum(r['dropped'] for r in reports)} dropped)")
    if a.train_max_records and rs.n > a.train_max_records:
        rs = rs.take(np.random.default_rng(0).choice(rs.n, a.train_max_records, replace=False))
    if a.split == "val":
        _, idx = dataset.split_indices(rs.n, a.val_frac, a.val_max, a.split_seed)
        rs = rs.take(idx)
    if a.max_records and rs.n > a.max_records:
        rs = rs.take(np.random.default_rng(0).choice(rs.n, a.max_records, replace=False))
    os.makedirs(a.out, exist_ok=True)
    res, emap, _ = evaluate_records(model, rs, a.map_res)
    res["nnm"] = a.nnm
    res["split"] = a.split
    np.save(os.path.join(a.out, "error_map.npy"), emap)
    if a.sphere_dump:
        res.update(render_sphere(model, a.sphere_dump, a.out))
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    keys = ("n", "pow_loss", "rel_l1", "rel_l1_top1", "rel_l1_grazing", "peak_energy_top01", "sphere_rel_l1")
    print("  ".join(f"{k}={res[k]:.4g}" if isinstance(res.get(k), float) else f"{k}={res.get(k)}" for k in keys if k in res))
    if a.assert_rel_l1 is not None and not res["rel_l1"] <= a.assert_rel_l1:
        print(f"FAIL: rel_l1 {res['rel_l1']:.4g} > {a.assert_rel_l1}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
