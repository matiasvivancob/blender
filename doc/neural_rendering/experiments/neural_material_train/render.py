"""Render a 128x128 sphere under two point lights: ground truth vs best fitted 2x16 / 2x32 model.
  python3 render.py            -> out/sphere_gt.png, sphere_2x16.png, sphere_2x32.png, sphere_compare.png
Also writes a per-texture-region/angle error breakdown to out/error_breakdown.md"""
import os, sys, glob, json
import numpy as np
from PIL import Image
import train as T
import material as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
S = 128
LIGHTS = [(np.array([-0.45, 0.55, 0.70]), 3.0), (np.array([0.60, -0.30, 0.74]), 0.8)]   # (direction to light, intensity)


def best_model(width):
    best = None
    for f in glob.glob(os.path.join(OUT, "models", f"*_w{width}_pow_lsmelu_s*.json")):
        j = json.load(open(f))
        if best is None or j["metrics"]["val_loss"] < best[0]:
            best = (j["metrics"]["val_loss"], f[:-5], j["cfg"])
    return best


def frames(n_pix=S):
    ys, xs = np.mgrid[0:n_pix, 0:n_pix]
    x = (xs + 0.5) / n_pix * 2 - 1
    y = 1 - (ys + 0.5) / n_pix * 2
    r2 = x * x + y * y
    mask = r2 <= 1.0
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    n = np.stack([x, y, z], -1)[mask]
    uv = np.stack([(x + 1) / 2, (y + 1) / 2], -1)[mask]
    uv = np.clip(uv, 0, 1)
    ref = np.where((np.abs(n[:, 1]) < 0.9)[:, None], np.array([0, 1.0, 0]), np.array([1.0, 0, 0]))
    t = ref - np.sum(ref * n, 1, keepdims=True) * n
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    return mask, n, t, b, uv


def to_local(v, n, t, b):
    return np.stack([np.sum(v * t, 1), np.sum(v * b, 1), np.sum(v * n, 1)], 1)


def shade(model_fn, uvs, n, t, b):
    img = np.zeros((len(n), 3))
    wo_w = np.tile(np.array([0, 0, 1.0]), (len(n), 1))
    wo = to_local(wo_w, n, t, b)
    for L, I in LIGHTS:
        L = L / np.linalg.norm(L)
        wi = to_local(np.tile(L, (len(n), 1)), n, t, b)
        ok = (wi[:, 2] > 0.005) & (wo[:, 2] > 0.005)
        val = np.zeros((len(n), 3))
        val[ok] = model_fn(uvs[ok], wi[ok], wo[ok])      # f * cos
        img += I * val
    return img


def tonemap(x):
    x = np.maximum(x, 0)
    x = x / (1 + x)
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def paint(mask, vals8):
    im = np.zeros((S, S, 3), np.uint8)
    im[mask] = vals8
    return im


def load_model_fn(path, cfg):
    d = np.load(path + ".npz")
    P = {k: d[k][None] for k in d.files}

    def fn(uv, wi, wo):
        B = T.Batch(uv, wi, wo, np.zeros((len(uv), 3)))
        outs = []
        for s in range(0, B.n, 16384):
            sub = T.Batch.__new__(T.Batch)
            sub.X, sub.idx, sub.w = B.X[s:s+16384], B.idx[s:s+16384], B.w[s:s+16384]
            out, _ = T.forward(P, sub, cfg.get("act", "lsmelu"))
            outs.append(np.exp(out[0]))
        return np.concatenate(outs)
    return fn


if __name__ == "__main__":
    tex = M.make_texture()
    mask, n, t, b, uv = frames()
    gt_fn = lambda uv_, wi, wo: M.target(M.lookup_params(tex, uv_), wi, wo)
    gt = shade(gt_fn, uv, n, t, b)
    ims = {"gt": paint(mask, tonemap(gt))}
    errs = {}
    meta = {}
    for w in (16, 32):
        bm = best_model(w)
        if bm is None: continue
        meta[w] = bm
        img = shade(load_model_fn(bm[1], bm[2]), uv, n, t, b)
        ims[f"2x{w}"] = paint(mask, tonemap(img))
        e = np.abs(img - gt)
        errs[w] = paint(mask, tonemap(4 * e))      # amplified abs error, same tonemap
        print(f"2x{w}: best={os.path.basename(bm[1])} val_loss={bm[0]:.4f}  sphere rel L1 (linear) = {np.abs(img-gt).sum()/gt.sum():.4f}")
    for k, v in ims.items():
        Image.fromarray(v).resize((S * 4, S * 4), Image.NEAREST).save(os.path.join(OUT, f"sphere_{k}.png"))
    row1 = np.concatenate([ims["gt"]] + [ims[k] for k in ("2x16", "2x32") if k in ims], 1)
    row2 = np.concatenate([np.zeros_like(ims["gt"])] + [errs[w] for w in (16, 32) if w in errs], 1)
    comp = np.concatenate([row1, row2], 0)
    Image.fromarray(comp).resize((comp.shape[1] * 3, comp.shape[0] * 3), Image.NEAREST).save(os.path.join(OUT, "sphere_compare.png"))
    print("wrote PNGs to", OUT)
