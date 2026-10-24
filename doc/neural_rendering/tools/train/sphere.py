# SPDX-License-Identifier: Apache-2.0
"""Sphere-under-point-lights render records (numpy), logic of experiments/neural_material_train/render.py.

A virtual sphere of S x S pixels is viewed head-on (view direction +z in world space) and lit by point
lights. Every (light, pixel) pair is one dump record: planar texture coordinates tex_uv = ((x+1)/2, (y+1)/2),
`wi` = view direction and `wo` = light direction in the local shading frame of the sphere normal. Evaluating
those records with Cycles' eval_bsdf on a flat quad whose UV layout is the unit square (see README.md) gives
the ground truth image of the real material; for the synthetic material the analytic BRDF gives it.

The records are generated deterministically from (size, lights), so evaluate.py regenerates them and checks
that a supplied dump has exactly these inputs before using its `out` as ground truth.
"""
import numpy as np

import png_io

SIZE = 96
# (direction to the light in world space, intensity), as in render.py
LIGHTS = [((-0.45, 0.55, 0.70), 3.0), ((0.60, -0.30, 0.74), 0.8)]


def frames(n_pix=SIZE):
    ys, xs = np.mgrid[0:n_pix, 0:n_pix]
    x = (xs + 0.5) / n_pix * 2 - 1
    y = 1 - (ys + 0.5) / n_pix * 2
    r2 = x * x + y * y
    mask = r2 <= 1.0
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    n = np.stack([x, y, z], -1)[mask]
    uv = np.clip(np.stack([(x + 1) / 2, (y + 1) / 2], -1)[mask], 0, 1)
    ref = np.where((np.abs(n[:, 1]) < 0.9)[:, None], np.array([0, 1.0, 0]), np.array([1.0, 0, 0]))
    t = ref - np.sum(ref * n, 1, keepdims=True) * n
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    return mask, n, t, b, uv


def to_local(v, n, t, b):
    return np.stack([np.sum(v * t, 1), np.sum(v * b, 1), np.sum(v * n, 1)], 1)


def make_records(size=SIZE, lights=LIGHTS, min_cos=0.005):
    """-> dict(mask, tex_uv (R,2), wi (R,3), wo (R,3), light (R,) int, pixel (R,) int [index in mask order],
    intensity (R,)). Only records where both directions are above the horizon are produced."""
    mask, n, t, b, uv = frames(size)
    wi_l = to_local(np.tile(np.array([0, 0, 1.0]), (len(n), 1)), n, t, b)
    recs = {k: [] for k in ("tex_uv", "wi", "wo", "light", "pixel", "intensity")}
    for li, (ldir, inten) in enumerate(lights):
        ld = np.asarray(ldir, np.float64)
        ld = ld / np.linalg.norm(ld)
        wo_l = to_local(np.tile(ld, (len(n), 1)), n, t, b)
        ok = (wo_l[:, 2] > min_cos) & (wi_l[:, 2] > min_cos)
        pix = np.nonzero(ok)[0]
        recs["tex_uv"].append(uv[pix])
        recs["wi"].append(wi_l[pix])
        recs["wo"].append(wo_l[pix])
        recs["light"].append(np.full(len(pix), li))
        recs["pixel"].append(pix)
        recs["intensity"].append(np.full(len(pix), inten))
    out = {k: np.concatenate(v) for k, v in recs.items()}
    for k in ("tex_uv", "wi", "wo", "intensity"):
        out[k] = out[k].astype(np.float32)
    out["mask"] = mask
    out["size"] = size
    return out


def shade(rec, rgb):
    """Accumulate intensity * rgb (R,3) per pixel -> image buffer (n_mask_pixels, 3) linear."""
    n_pix = int(rec["mask"].sum())
    img = np.zeros((n_pix, 3))
    np.add.at(img, rec["pixel"], rec["intensity"][:, None].astype(np.float64) * np.asarray(rgb, np.float64))
    return img


def tonemap(x):
    x = np.maximum(x, 0)
    x = x / (1 + x)
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def paint(mask, vals8):
    im = np.zeros(mask.shape + (3,), np.uint8)
    im[mask] = vals8
    return im


def compare_image(mask, gt, pred):
    """Top row: ground truth | model. Bottom row: blank | 4x amplified |error| (same tonemap). 3x upscaled."""
    a, b = paint(mask, tonemap(gt)), paint(mask, tonemap(pred))
    e = paint(mask, tonemap(4 * np.abs(pred - gt)))
    top = np.concatenate([a, b], 1)
    bot = np.concatenate([np.zeros_like(a), e], 1)
    return png_io.upscale(np.concatenate([top, bot], 0), 3)


def sphere_rel_l1(gt, pred):
    s = gt.sum()
    return float(np.abs(pred - gt).sum() / s) if s > 0 else float("nan")


def check_dump_matches(rec, dump, tol=1e-4):
    """The dump's inputs must equal the regenerated records (so out[:, :3] is the ground truth for them)."""
    if dump.n != len(rec["wi"]):
        return f"dump has {dump.n} records, expected {len(rec['wi'])}"
    for name, mine, theirs in (("wi", rec["wi"], dump.wi), ("wo", rec["wo"], dump.wo),
                               ("tex_uv", rec["tex_uv"], dump.texture_uv())):
        if np.abs(mine - theirs).max() > tol:
            return f"dump '{name}' differs from the regenerated sphere records (max {np.abs(mine - theirs).max():.2e})"
    return None


if __name__ == "__main__":
    import argparse

    import dataset

    ap = argparse.ArgumentParser(description="Write the sphere-render query records (out = 0) for Cycles to fill.")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=SIZE)
    a = ap.parse_args()
    r = make_records(a.size)
    n = len(r["wi"])
    dataset.save_dump(a.out, prims=np.zeros(n, np.int32), uv=r["tex_uv"], tex_uv=r["tex_uv"], wi=r["wi"], wo=r["wo"],
                      out=np.zeros((n, 8), np.float32), frame_local=True,
                      meta={"kind": "sphere_queries", "sphere_size": a.size, "uv_kind": "texture", "sample_count": n})
    print(f"wrote {a.out}: {n} sphere queries; fill `out` with blender_dump.py --fill, then evaluate.py --sphere-dump")
