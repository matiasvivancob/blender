#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a SHADER_EVAL_BSDF-style dump from the analytic layered BRDF of
experiments/neural_material_train/material.py, so the whole pipeline can be exercised without Blender.

    python3 make_synthetic_dump.py synthetic.npz --n 200000 --sphere synthetic_sphere.npz

Convention of the real dump (dump_format.md): `wi` = direction to the viewer, `wo` = direction to the light,
out[:, 0:3] = f(wi, wo) * cos(wo). The analytic BRDF is reciprocal, so brdf(wi, wo) is used as is and the
cosine is taken of `wo`. pdf and albedo are placeholders (cosine pdf of wo; base colour), the trainer ignores
them. A small fraction of records is made invalid (flags = 2, wi below the horizon) to exercise the filter.
"""
import argparse
import time

import numpy as np

import _paths  # noqa: F401
import dataset
import sampling
import sphere

M = _paths.load_material()


def synth_out(tex, tex_uv, wi, wo):
    """Analytic ground truth, float64 -> (n, 8) float32 [f*cos(wo) rgb, pdf, albedo rgb, flags=0]."""
    p = M.lookup_params(tex, np.asarray(tex_uv, np.float64))
    f = M.brdf(p, wi, wo)
    out = np.zeros((len(wi), 8), np.float32)
    out[:, 0:3] = f * np.asarray(wo, np.float64)[:, 2:3]
    out[:, 3] = np.asarray(wo, np.float64)[:, 2] / np.pi
    out[:, 4:7] = p[:, 0:3]
    return out


def make(n, seed=0, peak_frac=0.5, grazing_frac=0.1, invalid_frac=0.01, tex_seed=7, res=M.TEX_RES):
    rng = np.random.default_rng(seed)
    tex = M.make_texture(seed=tex_seed, res=res)
    tex_uv = rng.random((n, 2))
    wi, wo = sampling.sample_directions(rng, n, peak_frac, grazing_frac)
    out = np.empty((n, 8), np.float32)
    for s in range(0, n, 131072):                         # chunked: brdf() allocates several (n,3) float64 arrays
        sl = slice(s, s + 131072)
        out[sl] = synth_out(tex, tex_uv[sl], wi[sl], wo[sl])
    bad = rng.random(n) < invalid_frac
    out[bad] = 0
    out[bad, 7] = 2                                       # "wi on the backfacing side"
    flip = bad & (rng.random(n) < 0.5)
    wi[flip, 2] *= -1
    prims = np.zeros(n, np.int32)
    meta = {"object_name": "synthetic_plane", "frame": "uv_tangent", "material_name": "analytic_layered_brdf (material.py)",
            "blender_version": "synthetic (no Blender)", "sample_count": int(n), "uv_kind": "texture",
            "texture_res": int(res), "texture_seed": int(tex_seed), "seed": int(seed),
            "sampler": {"peak_frac": peak_frac, "grazing_frac": grazing_frac, "invalid_frac": invalid_frac},
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return dict(prims=prims, uv=tex_uv.astype(np.float32), tex_uv=tex_uv.astype(np.float32), wi=wi.astype(np.float32),
                wo=wo.astype(np.float32), out=out, frame_local=True, meta=meta), tex


def make_sphere_dump(tex, size=sphere.SIZE):
    rec = sphere.make_records(size)
    out = synth_out(tex, rec["tex_uv"], rec["wi"], rec["wo"])
    n = len(out)
    meta = {"object_name": "synthetic_plane", "frame": "uv_tangent", "material_name": "analytic_layered_brdf (material.py)",
            "blender_version": "synthetic (no Blender)", "sample_count": n, "uv_kind": "texture",
            "kind": "sphere_queries", "sphere_size": size}
    return dict(prims=np.zeros(n, np.int32), uv=rec["tex_uv"], tex_uv=rec["tex_uv"], wi=rec["wi"], wo=rec["wo"],
                out=out, frame_local=True, meta=meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peak-frac", type=float, default=0.5)
    ap.add_argument("--grazing-frac", type=float, default=0.1)
    ap.add_argument("--invalid-frac", type=float, default=0.01)
    ap.add_argument("--sphere", default=None, help="also write the sphere-render queries with ground truth here")
    a = ap.parse_args(argv)
    d, tex = make(a.n, a.seed, a.peak_frac, a.grazing_frac, a.invalid_frac)
    dataset.save_dump(a.out, **d)
    print(f"wrote {a.out}: {a.n} records")
    if a.sphere:
        dataset.save_dump(a.sphere, **make_sphere_dump(tex))
        print(f"wrote {a.sphere}: sphere queries")


if __name__ == "__main__":
    main()
