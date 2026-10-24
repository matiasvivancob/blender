"""Neural-material fitting in pure numpy (no torch available): latent grid + tiny MLP decoder,
manual backprop, Adam.  Supports K independent instances trained in lock-step on a shared batch,
with the Bitterli-et-al.-style phase/cull/batch-growth schedule.

Single run example:
  python3 train.py --mode bitterli --width 16 --seed 0 --steps 4000 --out out/test
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, time
import numpy as np
import material as M

F32 = np.float32
LAT = 8            # latent size per texel
RES = M.TEX_RES
NIN_DIR = 12       # wi(3) wo(3) h(3) wd'(3)
BETA, LEAK = 1.0, 0.05      # LeakySmeLU parameters (my reconstruction, see README)
POW_N = 3.0

# ----------------------------------------------------------------------------- data
def _hemi(rng, n):
    ct = rng.random(n) * 0.995 + 0.005          # uniform in cos(theta) -> uniform solid angle
    st = np.sqrt(1 - ct * ct)
    ph = rng.random(n) * 2 * np.pi
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1)


def sample_dirs(rng, n, peak_frac=0.5):
    """Mixture: uniform wi/wo, plus (peak_frac) wo=reflect(wi, h) with h ~ GGX(alpha~logU[.01,.5])."""
    wi = _hemi(rng, n)
    wo = _hemi(rng, n)
    npk = int(n * peak_frac)
    if npk:
        a = np.exp(rng.uniform(np.log(0.01), np.log(0.5), npk))
        u = rng.random(npk)
        tan2 = a * a * u / (1 - u)
        ct = 1 / np.sqrt(1 + tan2)
        st = np.sqrt(1 - ct * ct)
        ph = rng.random(npk) * 2 * np.pi
        h = np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1)
        wi_p = wi[:npk]
        wo_p = 2 * np.sum(wi_p * h, 1, keepdims=True) * h - wi_p       # reflect wi about h
        ok = wo_p[:, 2] > 0.005
        wo[:npk][ok] = wo_p[ok]
    return wi, wo


def sample_batch(rng, tex, n):
    uv = rng.random((n, 2))
    wi, wo = sample_dirs(rng, n)
    p = M.lookup_params(tex, uv)
    t = M.target(p, wi, wo)
    return uv, wi, wo, t


def dir_features(wi, wo):
    """(wi, wo, h, wd') with wd' = shortest-arc rotation of wi taking h to +z."""
    wi = wi.astype(np.float64); wo = wo.astype(np.float64)
    h = wi + wo
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    hx, hy, hz = h[:, 0], h[:, 1], h[:, 2]
    v = np.stack([hy, -hx, np.zeros_like(hx)], 1)
    wd = (wi * hz[:, None] + np.cross(v, wi)
          + (np.sum(v * wi, 1) / (1 + hz))[:, None] * v)
    return np.concatenate([wi, wo, h, wd], 1).astype(F32)


class Batch:
    """Everything the network needs for a batch of samples."""
    def __init__(self, uv, wi, wo, t):
        self.X = dir_features(wi, wo)
        idx, w = M.bilinear_corners(uv)
        self.idx, self.w = idx, w.astype(F32)
        self.t = t.astype(F32)
        self.wi, self.wo, self.uv = wi, wo, uv
        self.n = len(uv)


# ----------------------------------------------------------------------------- activations
def act_fwd(x, kind):
    if kind == "relu":
        return np.maximum(x, 0)
    s = np.clip(x + BETA, 0.0, 2 * BETA)
    return LEAK * x + (1 - LEAK) * (s * s * (0.25 / BETA) + np.maximum(x - BETA, 0))


def act_bwd(x, kind):
    if kind == "relu":
        return (x > 0).astype(x.dtype)
    return LEAK + (1 - LEAK) * np.clip((x + BETA) * (0.5 / BETA), 0.0, 1.0)


# ----------------------------------------------------------------------------- model
def init_params(rng, K, W):
    def U(shape, fan_in, gain=1.0):
        b = gain * np.sqrt(6.0 / fan_in)
        return rng.uniform(-b, b, shape).astype(F32)
    nin = NIN_DIR + LAT
    P = {
        "grid": rng.uniform(-0.5, 0.5, (K, RES * RES, LAT)).astype(F32),
        "W1d": U((K, NIN_DIR, W), nin), "W1z": U((K, LAT, W), nin),
        "b1": np.zeros((K, 1, W), F32),
        "W2": U((K, W, W), W), "b2": np.zeros((K, 1, W), F32),
        "W3": U((K, W, 3), W, 0.3), "b3": np.full((K, 1, 3), np.log(0.2), F32),
    }
    return P


def subset(P, keep):
    return {k: v[keep] for k, v in P.items()}


def forward(P, B, act, keep_cache=False):
    """Returns out (K,Bn,3) [pre-exp] and cache."""
    G = P["grid"][:, B.idx]                                    # (K,Bn,4,8)
    z = (G * B.w[None, :, :, None]).sum(2)                     # (K,Bn,8)
    h1 = np.matmul(B.X[None], P["W1d"]) + np.matmul(z, P["W1z"]) + P["b1"]
    a1 = act_fwd(h1, act)
    h2 = np.matmul(a1, P["W2"]) + P["b2"]
    a2 = act_fwd(h2, act)
    out = np.matmul(a2, P["W3"]) + P["b3"]
    cache = (z, h1, a1, h2, a2) if keep_cache else None
    return np.clip(out, -20.0, 14.0), cache


def loss_and_gout(out, t, loss):
    """Per-instance mean L1 loss in transformed space and d loss / d out (summed objective)."""
    K, Bn, _ = out.shape
    if loss == "pow":
        ep = np.exp(out * (1.0 / POW_N))                       # pred^(1/n)
        Mp = POW_N * (ep - 1.0)
        Mt = POW_N * (np.cbrt(t) - 1.0) if POW_N == 3.0 else POW_N * (t ** (1 / POW_N) - 1.0)
        d = Mp - Mt
        g = np.sign(d) * ep
    else:  # log(1+x)
        e = np.exp(out)
        d = np.log1p(e) - np.log1p(t)
        g = np.sign(d) * (e / (1.0 + e))
    L = np.abs(d).mean((1, 2))
    return L, g / (Bn * 3.0)


def backward(P, B, cache, gout, act):
    z, h1, a1, h2, a2 = cache
    G = {}
    mm = np.matmul
    G["W3"] = mm(a2.transpose(0, 2, 1), gout); G["b3"] = gout.sum(1, keepdims=True)
    gh2 = mm(gout, P["W3"].transpose(0, 2, 1)) * act_bwd(h2, act)
    G["W2"] = mm(a1.transpose(0, 2, 1), gh2); G["b2"] = gh2.sum(1, keepdims=True)
    gh1 = mm(gh2, P["W2"].transpose(0, 2, 1)) * act_bwd(h1, act)
    G["b1"] = gh1.sum(1, keepdims=True)
    G["W1d"] = mm(B.X.T[None], gh1)
    G["W1z"] = mm(z.transpose(0, 2, 1), gh1)
    gz = mm(gh1, P["W1z"].transpose(0, 2, 1))                   # (K,Bn,8)
    K, Bn, _ = gz.shape
    # scatter into latent grid
    vals = (gz[:, :, None, :] * B.w[None, :, :, None]).reshape(-1)
    flat = (np.arange(K)[:, None, None, None] * (RES * RES * LAT)
            + B.idx[None, :, :, None] * LAT + np.arange(LAT)[None, None, None, :]).reshape(-1)
    G["grid"] = np.bincount(flat, weights=vals, minlength=K * RES * RES * LAT).astype(F32).reshape(K, RES * RES, LAT)
    return G


class Adam:
    def __init__(self, P, b1=0.9, b2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0; self.b1, self.b2, self.eps = b1, b2, eps

    def step(self, P, G, lr):
        self.t += 1
        c1 = 1 - self.b1 ** self.t; c2 = 1 - self.b2 ** self.t
        for k in P:
            m, v = self.m[k], self.v[k]
            m *= self.b1; m += (1 - self.b1) * G[k]
            v *= self.b2; v += (1 - self.b2) * G[k] * G[k]
            P[k] -= (lr / c1) * m / (np.sqrt(v / c2) + self.eps)

    def subset(self, keep):
        self.m = subset(self.m, keep); self.v = subset(self.v, keep)


# ----------------------------------------------------------------------------- evaluation
def make_val_set(tex, n=65536, seed=12345):
    rng = np.random.default_rng(seed)
    return Batch(*sample_batch(rng, tex, n))


def evaluate(P, V, act, chunk=16384):
    """Held-out metrics per instance (K,). Pred = exp(out)."""
    K = P["grid"].shape[0]
    preds = np.empty((K, V.n, 3), F32)
    for s in range(0, V.n, chunk):
        sl = slice(s, s + chunk)
        sub = Batch.__new__(Batch)
        sub.X, sub.idx, sub.w = V.X[sl], V.idx[sl], V.w[sl]
        out, _ = forward(P, sub, act)
        preds[:, sl] = np.exp(out)
    return preds


def metrics_from_preds(preds, V):
    t = V.t
    lum = t.mean(1)
    top = lum >= np.quantile(lum, 0.99)
    th_i = np.degrees(np.arccos(np.clip(V.wi[:, 2], 0, 1)))
    th_o = np.degrees(np.arccos(np.clip(V.wo[:, 2], 0, 1)))
    graz = (th_i > 75) | (th_o > 75)
    res = []
    for p in preds:
        Mp = POW_N * (np.cbrt(p) - 1); Mt = POW_N * (np.cbrt(t) - 1)
        l1 = np.abs(Mp - Mt)
        err = np.abs(p - t)
        res.append(dict(
            val_loss=float(l1.mean()),                               # power-loss (n=3) L1, held-out
            val_log_loss=float(np.abs(np.log1p(p) - np.log1p(t)).mean()),
            rel_l1=float(err.sum() / t.sum()),
            rel_l1_top1=float(err[top].sum() / t[top].sum()),
            rel_l1_grazing=float(err[graz].sum() / t[graz].sum()),
            pow_loss_top1=float(l1[top].mean()),
            pow_loss_grazing=float(l1[graz].mean()),
            med_rel_err=float(np.median(err.sum(1) / (t.sum(1) + 1e-3))),
        ))
    return res


# ----------------------------------------------------------------------------- training
SCHEDULES = {
    # name: list of (K, batch) phases; each phase gets steps/len(phases) steps, K*batch constant
    "baseline": [(1, 8192)],
    "bitterli": [(64, 128), (16, 512), (4, 2048), (1, 8192)],
}


def lr_at(step, total, lr0, lr_end_frac=0.02):
    c = 0.5 * (1 + np.cos(np.pi * step / total))
    return lr0 * (lr_end_frac + (1 - lr_end_frac) * c)


def run(cfg, tex=None, V=None, verbose=False):
    """cfg keys: mode, width, seed, steps, loss, act, lr, out(optional dir to save weights)"""
    t0 = time.time()
    tex = M.make_texture() if tex is None else tex
    V = make_val_set(tex) if V is None else V
    phases = SCHEDULES[cfg["mode"]]
    W, steps, act, loss = cfg["width"], cfg["steps"], cfg.get("act", "lsmelu"), cfg.get("loss", "pow")
    rng = np.random.default_rng(1000 + cfg["seed"])            # data stream
    prng = np.random.default_rng(cfg["seed"])                  # init
    P = init_params(prng, phases[0][0], W)
    opt = Adam(P)
    per_phase = steps // len(phases)
    total = per_phase * len(phases)
    gstep = 0
    info = {"phases": []}
    for pi, (K, bs) in enumerate(phases):
        assert P["grid"].shape[0] == K
        recent = []
        for s in range(per_phase):
            B = Batch(*sample_batch(rng, tex, bs))
            out, cache = forward(P, B, act, keep_cache=True)
            L, gout = loss_and_gout(out, B.t, loss)
            G = backward(P, B, cache, gout, act)
            opt.step(P, G, lr_at(gstep, total, cfg["lr"]))
            gstep += 1
            if s >= per_phase - max(1, per_phase // 4):
                recent.append(L)                                # fresh-batch loss = unbiased selection score
            if verbose and gstep % 200 == 0:
                print(f"step {gstep} phase {pi} K={K} bs={bs} loss(min/mean)={L.min():.4f}/{L.mean():.4f}", flush=True)
        score = np.mean(recent, 0)
        ph = {"K": K, "batch": bs, "steps": per_phase,
              "sel_score_min": float(score.min()), "sel_score_median": float(np.median(score)),
              "sel_score_max": float(score.max()), "sel_score_std": float(score.std())}
        if pi < len(phases) - 1:
            knext = phases[pi + 1][0]
            keep = np.argsort(score)[:knext]
            ph["kept_scores"] = [float(x) for x in score[keep]]
            P = subset(P, keep); opt.subset(keep)
        info["phases"].append(ph)
    preds = evaluate(P, V, act)
    m = metrics_from_preds(preds, V)[0]
    info.update(cfg=cfg, metrics=m, wall_s=time.time() - t0, n_params_decoder=int(
        sum(P[k][0].size for k in P if k != "grid")))
    if cfg.get("out"):
        os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
        np.savez(cfg["out"] + ".npz", **{k: v[0] for k, v in P.items()})
        with open(cfg["out"] + ".json", "w") as f:
            json.dump(info, f, indent=1)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="baseline", choices=list(SCHEDULES))
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--loss", default="pow", choices=["pow", "log"])
    ap.add_argument("--act", default="lsmelu", choices=["lsmelu", "relu"])
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = run(vars(a), verbose=True)
    print(json.dumps(r["metrics"], indent=1), f"wall {r['wall_s']:.1f}s")
