#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end check of the Cycles neural closure against the Python references (numpy, no GPU).

    python neural_closure_check.py --blender <blender.exe> --model <model_with_proxy.nnm> \
        --quad-scene <probe_quad.blend> --work <abs dir> [--device CPU|CUDA|OPTIX --gpu 3090] [--n 200000]

It puts a Neural BSDF with that model on the one-quad probe scene, dumps `n` random (wi, wo, uv) records with
SHADER_EVAL_BSDF (blender_dump.py), which evaluates the real kernel closure, and compares, record by record:

  out[0:3]  f * cos(theta_o)   with  nnm.reference.decode(model, [wi, wo], uv)     (frame, UV, decoder, transform)
  out[3]    pdf                with  proxy_fit.proxy_pdf(wi, wo, alpha, p_diffuse)  (the pdf the sampler reports)
  out[4:7]  albedo             with  the proxy block albedo                         (bsdf_albedo)

Prints NEURAL CLOSURE PASS or FAIL and a table of errors. A record is skipped only if the dump flagged it.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(TOOLS, "train"), TOOLS]

import dataset  # noqa: E402
import proxy_fit  # noqa: E402
from nnm.format import read_nnm  # noqa: E402
from nnm.reference import decode  # noqa: E402


def run(cmd, env, log):
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"command failed ({r.returncode}), see {log}")
    return open(log, encoding="utf-8", errors="replace").read()


def rel_stats(got, want, floor):
    err = np.abs(got - want)
    rel = err / np.maximum(np.abs(want), floor)
    return float(np.median(rel)), float(np.quantile(rel, 0.99)), float(rel.max()), float(err.max())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blender", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quad-scene", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--device", default="CPU", choices=["CPU", "CUDA", "OPTIX"])
    ap.add_argument("--gpu", default="")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--tol-f", type=float, default=2e-3, help="max p99 relative error of f*cos")
    ap.add_argument("--tol-pdf", type=float, default=2e-3, help="max p99 relative error of pdf")
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)

    env = dict(os.environ, NR_MODEL=a.model, NR_DEVICE=a.device, NR_GPU=a.gpu)
    scene = os.path.join(a.work, "neural_quad.blend")
    run([a.blender, "-b", a.quad_scene, "--python-exit-code", "1", "--python",
         os.path.join(HERE, "make_neural_quad_scene.py"), "--", scene], env, os.path.join(a.work, "make_scene.log"))

    dump = os.path.join(a.work, "dump.npz")
    log = run([a.blender, "-b", scene, "--python-exit-code", "1",
               "--python", os.path.join(HERE, "pin_device.py"),
               "--python", os.path.join(TOOLS, "train", "blender_dump.py"),
               "-o", os.path.join(a.work, "render_"), "-E", "BSDF_DUMP", "-f", "1", "--",
               "--object", "ProbeQuad", "--n", str(a.n), "--seed", str(a.seed), "--out", dump],
              env, os.path.join(a.work, "dump.log"))
    if "wrote" not in log:  # Blender exits 0 even if the engine raised
        raise SystemExit("dump did not complete, see dump.log")

    d = dataset.load_dump(dump)
    out = d.out.astype(np.float64)
    ok = out[:, 7] == 0
    print(f"records {len(out)}, flagged {int((~ok).sum())}")
    if (~ok).mean() > 0.001:
        raise SystemExit("too many flagged records: the closure did not evaluate")
    wi, wo, uv = d.wi[ok].astype(np.float64), d.wo[ok].astype(np.float64), d.texture_uv()[ok].astype(np.float64)
    out = out[ok]

    model = read_nnm(a.model)
    assert model.proxy is not None, "model has no proxy block"
    alpha, _, p_d, *albedo = model.proxy
    ref_f = decode(model, np.concatenate([wi, wo], 1).astype(np.float32), uv.astype(np.float32)).astype(np.float64)
    ref_pdf = proxy_fit.proxy_pdf(wi, wo, alpha, p_d)

    rows = [("f*cos (RGB)", *rel_stats(out[:, 0:3], ref_f, 1e-3)),
            ("pdf", *rel_stats(out[:, 3], ref_pdf, 1e-6)),
            ("albedo", *rel_stats(out[:, 4:7], np.tile(albedo, (len(out), 1)), 1e-3))]
    print(f"{'quantity':<14}{'median rel':>12}{'p99 rel':>12}{'max rel':>12}{'max abs':>12}")
    for name, med, p99, mx, mxa in rows:
        print(f"{name:<14}{med:>12.2e}{p99:>12.2e}{mx:>12.2e}{mxa:>12.2e}")
    print(f"energy: kernel sum(f*cos) / reference = {out[:, 0:3].sum() / ref_f.sum():.6f}; "
          f"mean pdf kernel / reference = {out[:, 3].mean() / ref_pdf.mean():.6f}")

    passed = rows[0][2] < a.tol_f and rows[1][2] < a.tol_pdf and rows[2][3] < 1e-4
    print("NEURAL CLOSURE", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
