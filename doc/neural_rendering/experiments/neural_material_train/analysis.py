"""Where do the best 2x16 / 2x32 models fail?  -> out/error_breakdown.md"""
import os, numpy as np
import train as T, material as M, render as R

tex = M.make_texture()
V = T.make_val_set(tex)
p6 = M.lookup_params(tex, V.uv)
t = V.t; lum = t.mean(1)
th = np.degrees(np.arccos(np.clip(np.maximum(0, 1) * np.minimum(V.wi[:, 2], V.wo[:, 2]), 0, 1)))   # max of the two angles
L = []

def rel(mask, p):
    return np.abs(p[mask] - t[mask]).sum() / t[mask].sum()

def bias(mask, p):
    return p[mask].sum() / t[mask].sum()

q = np.quantile(lum, [0.5, 0.9, 0.99])
groups = {
    "angle max(th_i,th_o) 0-30": th < 30, "30-60": (th >= 30) & (th < 60), "60-75": (th >= 60) & (th < 75),
    "75-85": (th >= 75) & (th < 85), "85-90": th >= 85,
    "target <= median": lum <= q[0], "median-90%": (lum > q[0]) & (lum <= q[1]), "90%-99%": (lum > q[1]) & (lum < q[2]), "top 1%": lum >= q[2],
    "coat rough < 0.2": p6[:, 5] < 0.2, "coat rough 0.2-0.4": (p6[:, 5] >= 0.2) & (p6[:, 5] < 0.4), "coat rough >= 0.4": p6[:, 5] >= 0.4,
    "metallic > 0.5 (gold checker)": p6[:, 4] > 0.5, "dielectric base (metallic<0.5)": p6[:, 4] <= 0.5,
}
for w in (16, 32):
    bm = R.best_model(w)
    d = np.load(bm[1] + ".npz"); P = {k: d[k][None] for k in d.files}
    p = T.evaluate(P, V, bm[2].get("act", "lsmelu"))[0]
    L.append(f"\n### best 2x{w} model: {os.path.basename(bm[1])} (val pow-loss {bm[0]:.4f})\n")
    L.append("| subset | share of samples | rel L1 | pred/target energy ratio |\n|---|---|---|---|")
    for g, m in groups.items():
        L.append(f"| {g} | {m.mean()*100:.1f}% | {rel(m, p):.3f} | {bias(m, p):.3f} |")
    # peak accuracy: where target > 100 (very sharp)
    m = lum > 100
    L.append(f"| target lum > 100 (sharp peaks) | {m.mean()*100:.2f}% | {rel(m, p):.3f} | {bias(m, p):.3f} |")
    # peak height ratio on top-0.1%
    m = lum >= np.quantile(lum, 0.999)
    L.append(f"| top 0.1% | {m.mean()*100:.2f}% | {rel(m, p):.3f} | {bias(m, p):.3f} |")
open(os.path.join(R.OUT, "error_breakdown.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
