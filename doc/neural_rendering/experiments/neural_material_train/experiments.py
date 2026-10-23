"""Driver: python3 experiments.py run [steps]   -> out/models/*.json|npz
          python3 experiments.py report         -> out/results.md, out/results.json"""
import os, sys, json, time, glob
import multiprocessing as mp
import numpy as np
import train as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LR = 0.04
SEEDS = list(range(8))
ABL_SEEDS = list(range(4))


def tag(c):
    return f"{c['mode']}_w{c['width']}_{c.get('loss','pow')}_{c.get('act','lsmelu')}_s{c['seed']}"


def job(c):
    c = dict(c); c["out"] = os.path.join(OUT, "models", tag(c))
    r = T.run(c)
    return tag(c), r["metrics"]["val_loss"], r["wall_s"]


def build_jobs(steps):
    jobs = []
    for w in (32, 16):
        for s in SEEDS:
            for m in ("bitterli", "baseline"):
                jobs.append(dict(mode=m, width=w, seed=s, steps=steps, lr=LR))
    for s in ABL_SEEDS:      # ablations: baseline schedule, 2x16, same seeds as main runs
        jobs.append(dict(mode="baseline", width=16, seed=s, steps=steps, lr=LR, loss="log"))
        jobs.append(dict(mode="baseline", width=16, seed=s, steps=steps, lr=LR, act="relu"))
    return jobs


def run_all(steps):
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    jobs = [j for j in build_jobs(steps) if not os.path.exists(os.path.join(OUT, "models", tag(j) + ".json"))]
    t0 = time.time()
    with mp.get_context("spawn").Pool(4) as p, open(os.path.join(OUT, "progress.log"), "a") as lg:
        for i, (t, v, w) in enumerate(p.imap_unordered(job, jobs)):
            msg = f"[{i+1}/{len(jobs)}] {t} val_loss={v:.4f} run={w:.0f}s elapsed={time.time()-t0:.0f}s"
            print(msg, flush=True); lg.write(msg + "\n"); lg.flush()
    with open(os.path.join(OUT, "total_wall.txt"), "w") as f:
        f.write(f"{time.time()-t0:.1f}\n")


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "models", pattern + ".json")))]


def stats(x):
    x = np.asarray(x, float)
    return dict(mean=x.mean(), std=x.std(ddof=1), min=x.min(), max=x.max(), n=len(x))


def boot_ratio(a, b, n=20000, seed=0):
    """bootstrap 90% interval of std(a)/std(b)"""
    rng = np.random.default_rng(seed)
    r = []
    for _ in range(n):
        sa = rng.choice(a, len(a)).std(ddof=1); sb = rng.choice(b, len(b)).std(ddof=1)
        if sb > 0: r.append(sa / sb)
    return np.percentile(r, [5, 50, 95])


def report():
    L = []; res = {}
    def row(name, runs, key="val_loss"):
        v = [r["metrics"][key] for r in runs]; s = stats(v)
        return (f"| {name} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | "
                f"{np.mean([r['wall_s'] for r in runs]):.0f} |")
    L.append("## Variance: final held-out power-loss (n=3, L1) over seeds\n")
    L.append("| method | seeds | mean | std | min | max | mean wall/run (s) |\n|---|---|---|---|---|---|---|")
    for w in (16, 32):
        b = load(f"baseline_w{w}_pow_lsmelu_s*"); c = load(f"bitterli_w{w}_pow_lsmelu_s*")
        if not b or not c: continue
        L.append(row(f"2x{w} baseline (1 inst, bs 8192)", b)); L.append(row(f"2x{w} Bitterli (64>16>4>1)", c))
        vb = [r["metrics"]["val_loss"] for r in b]; vc = [r["metrics"]["val_loss"] for r in c]
        res[f"w{w}"] = dict(baseline=stats(vb), bitterli=stats(vc), std_ratio_bitterli_over_baseline_p5_50_95=list(boot_ratio(vc, vb)),
                            per_seed_baseline=vb, per_seed_bitterli=vc)
    L.append("\nstd ratio (Bitterli/baseline), bootstrap 5/50/95 pct: " + "; ".join(
        f"2x{w}: " + ", ".join(f"{x:.2f}" for x in res[f'w{w}']['std_ratio_bitterli_over_baseline_p5_50_95']) for w in (16, 32) if f"w{w}" in res))
    L.append("\n## Accuracy (mean over the 8 seeds; best seed in brackets) -- held-out 65,536 samples\n")
    L.append("| model | method | pow-loss | rel L1 (all) | rel L1 (top 1% brightest) | rel L1 (grazing >75deg) | median per-sample rel err | #decoder params |\n|---|---|---|---|---|---|---|---|")
    for w in (16, 32):
        for m in ("baseline", "bitterli"):
            rs = load(f"{m}_w{w}_pow_lsmelu_s*")
            if not rs: continue
            def f(k): 
                v = [r["metrics"][k] for r in rs]; return f"{np.mean(v):.4f} ({np.min(v):.4f})"
            L.append(f"| 2x{w} | {m} | {f('val_loss')} | {f('rel_l1')} | {f('rel_l1_top1')} | {f('rel_l1_grazing')} | {f('med_rel_err')} | {rs[0]['n_params_decoder']} |")
    L.append("\n## Ablations (2x16, baseline schedule, seeds 0-3; metrics on the same held-out set)\n")
    L.append("| variant | seeds | pow-loss mean | std | rel L1 | rel L1 top1% |\n|---|---|---|---|---|---|")
    for name, pat in [("power n=3 + LeakySmeLU (default)", "baseline_w16_pow_lsmelu_s[0-3]"),
                      ("log(1+x) loss + LeakySmeLU", "baseline_w16_log_lsmelu_s*"),
                      ("power n=3 + ReLU", "baseline_w16_pow_relu_s*")]:
        rs = load(pat)
        if not rs: continue
        v = [r["metrics"]["val_loss"] for r in rs]
        L.append(f"| {name} | {len(rs)} | {np.mean(v):.4f} | {np.std(v, ddof=1):.4f} | {np.mean([r['metrics']['rel_l1'] for r in rs]):.4f} | {np.mean([r['metrics']['rel_l1_top1'] for r in rs]):.4f} |")
        res.setdefault("ablation", {})[name] = dict(val=v, rel_l1=[r['metrics']['rel_l1'] for r in rs], rel_l1_top1=[r['metrics']['rel_l1_top1'] for r in rs])
    # paired per-seed deltas for ablations
    # phase diagnostics
    L.append("\n## Bitterli phase diagnostics (selection score = mean fresh-batch train loss over last quarter of phase)\n")
    L.append("| model | phase (K@batch) | score min | median | max | (mean over seeds) |\n|---|---|---|---|---|---|")
    for w in (16, 32):
        rs = load(f"bitterli_w{w}_pow_lsmelu_s*")
        if not rs: continue
        for pi in range(4):
            ph = [r["phases"][pi] for r in rs]
            L.append(f"| 2x{w} | {ph[0]['K']}@{ph[0]['batch']} | {np.mean([p['sel_score_min'] for p in ph]):.4f} | {np.mean([p['sel_score_median'] for p in ph]):.4f} | {np.mean([p['sel_score_max'] for p in ph]):.4f} | |")
    tw = os.path.join(OUT, "total_wall.txt")
    if os.path.exists(tw): L.append(f"\nTotal wall time of the full experiment batch (4 parallel single-thread workers): {float(open(tw).read()):.0f} s")
    open(os.path.join(OUT, "results.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2400)
    else:
        report()
