"""Tiny LR sanity sweep (baseline + bitterli, 2x16, short run) to pick one shared Adam LR."""
import sys, json
import multiprocessing as mp
import train as T

def job(c):
    r = T.run(c)
    return c, r["metrics"]["val_loss"], r["wall_s"]

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    jobs = [dict(mode=m, width=16, seed=100, steps=steps, lr=lr) for lr in (float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "0.002,0.005,0.01,0.02".split(","))) for m in ("baseline", "bitterli")]
    with mp.get_context("spawn").Pool(4) as p:
        for c, v, w in p.imap(job, jobs):
            print(c["mode"], "lr", c["lr"], "val_loss %.4f" % v, "wall %.0fs" % w, flush=True)
