#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train a neural material (latent grid + MLP decoder) on a SHADER_EVAL_BSDF dump. PyTorch.

NOT RUNNABLE WHERE THIS WAS WRITTEN (no torch there): reviewed by reading and `py_compile` only.
Run `python3 crosscheck_torch.py` first on the workstation, then `--dump synthetic.npz`, then a real dump.

    python3 train.py --dump dump.npz --out runs/a --steps 100000                  # Bitterli culling, 64@1k->16@4k->4@16k->1@64k
    python3 train.py --dump dump.npz --out runs/b --steps 100000 --no-culling     # baseline: 1 instance, batch 64k
    python3 train.py --dump dump.npz --out runs/a --resume runs/a/ckpt_last.pt    # continue

Recipe (Bitterli et al. 2026 as reconstructed in experiments/neural_material_train/; see 04_experiment_results.md B):
  * target f*cos(wo) (dump out[:, 0:3]); prediction exp(clamp(net, -20, 14)); power loss n = 3 (model.power_loss)
  * Adam(0.9, 0.999, 1e-8), cosine LR from --lr to --lr * --lr-end-frac over all steps (one global schedule)
  * culling: K instances train in lock-step on one shared batch. At the end of each of P equal phases the
    instances are ranked by validation loss and the best K / c survive (c = K^(1/(P-1))); the batch grows by c so
    K * batch stays constant. Default K=64, P=4, K*batch = 65536: 64@1024, 16@4096, 4@16384, 1@65536.
    Deviation from the prototype: the prototype ranks by the loss on fresh batches (its data stream never
    repeats); a dump is finite, so ranking uses a fixed held-out subset (--eval-size records of the validation split).
  * `--no-culling`: one instance, batch --batch-total, same number of network evaluations. Baseline for comparisons.

Outputs in --out: train_log.csv, phases.json, config.json, ckpt_last.pt (resume), model_final.pt and
model_final.npz (arrays for export_nnm.py), final_metrics.json (held-out), model.nnm (unless --no-export).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

import _paths  # noqa: F401
import dataset
import metrics
import model as nm
import schedule
from np_model import ACTIVATIONS, ModelConfig, forward_np, save_arrays_npz


# ----------------------------------------------------------------------------- optimizer
class Adam:
    """Adam with state that can be subset along the instance axis (cull). Math identical to the numpy
    prototype: p -= lr/c1 * m / (sqrt(v/c2) + eps). All parameters carry the instance axis first."""

    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8):
        self.b1, self.b2, self.eps = betas[0], betas[1], eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    @torch.no_grad()
    def step(self, params, lr, lr_mults=None):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for i, (p, m, v) in enumerate(zip(params, self.m, self.v)):
            g = p.grad
            if g is None:
                continue
            m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
            denom = (v / c2).sqrt_().add_(self.eps)
            step_lr = lr * (lr_mults[i] if lr_mults else 1.0)
            p.addcdiv_(m, denom, value=-step_lr / c1)

    @torch.no_grad()
    def select(self, keep):
        keep = torch.as_tensor(keep, dtype=torch.long, device=self.m[0].device)
        self.m = [x[keep].clone() for x in self.m]
        self.v = [x[keep].clone() for x in self.v]

    def state(self):
        return {"t": self.t, "m": [x.cpu() for x in self.m], "v": [x.cpu() for x in self.v]}

    def load(self, st, device):
        self.t = int(st["t"])
        self.m = [x.to(device) for x in st["m"]]
        self.v = [x.to(device) for x in st["v"]]


# ----------------------------------------------------------------------------- data
class Sampler:
    """Batches without replacement per epoch (a fresh random permutation when the previous one is used up)."""

    def __init__(self, n, device, gen):
        self.n, self.device, self.gen, self.perm, self.pos = n, device, gen, None, 0

    def next(self, bs):
        if bs >= self.n:                                     # tiny dataset: sample with replacement
            return torch.randint(0, self.n, (bs,), device=self.device, generator=self.gen)
        if self.perm is None or self.pos + bs > self.n:
            self.perm = torch.randperm(self.n, device=self.device, generator=self.gen)
            self.pos = 0
        idx = self.perm[self.pos:self.pos + bs]
        self.pos += bs
        return idx


class DeviceData:
    def __init__(self, rs: dataset.RecordSet, device):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
        self.uv, self.wi, self.wo, self.tgt = t(rs.tex_uv), t(rs.wi), t(rs.wo), t(rs.target)
        self.n = rs.n

    def sub(self, sl):
        return self.uv[sl], self.wi[sl], self.wo[sl], self.tgt[sl]


@torch.no_grad()
def val_losses(model, data: DeviceData, power, chunk=None):
    """(K,) mean power loss of every instance over `data`."""
    chunk = chunk or max(1024, (1 << 20) // max(model.k, 1))
    acc = torch.zeros(model.k, device=data.uv.device)
    for s in range(0, data.n, chunk):
        uv, wi, wo, tgt = data.sub(slice(s, s + chunk))
        acc += nm.power_loss(model(uv, wi, wo), tgt, power) * uv.shape[0]
    return acc / data.n


@torch.no_grad()
def predict_instance(model, data: DeviceData, i=0, chunk=None):
    """exp(output) of instance i over all of `data` -> numpy (N, 3)."""
    chunk = chunk or max(1024, (1 << 20) // max(model.k, 1))
    outs = []
    for s in range(0, data.n, chunk):
        uv, wi, wo, _ = data.sub(slice(s, s + chunk))
        outs.append(torch.exp(model(uv, wi, wo)[i]).cpu())
    return torch.cat(outs).numpy()


# ----------------------------------------------------------------------------- checkpoints
def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_resume(path, model, opt, cfg, args, sched, g, phase, s_next):
    atomic_save({"format": "nnm_train_resume_v1", "config": cfg.to_dict(), "args": vars(args),
                 "schedule": [list(x) for x in sched], "global_step": g, "phase": phase, "step_in_phase": s_next,
                 "k": model.k, "backend": model.backend,
                 "params": {n: p.detach().cpu() for n, p in zip(model.param_names, model.param_list())},
                 "adam": opt.state()}, path)


def self_check(model, arrays, cfg, data: DeviceData, n=4096, tol=1e-3):
    """The exported arrays must reproduce the torch forward of the surviving instance (numpy float64 forward)."""
    n = min(n, data.n)
    uv, wi, wo, _ = data.sub(slice(0, n))
    with torch.no_grad():
        ref = torch.exp(model(uv, wi, wo)[0]).cpu().numpy().astype(np.float64)
    got = forward_np(arrays, cfg, uv.cpu().numpy(), wi.cpu().numpy(), wo.cpu().numpy())
    rel = np.abs(got - ref) / (np.abs(ref) + 1e-3)
    p99 = float(np.quantile(rel, 0.99))
    if not p99 < tol:
        raise RuntimeError(f"export self-check failed: numpy forward of the extracted arrays differs from the torch "
                           f"forward (p99 relative {p99:.3e}, limit {tol}). Do not use these arrays.")
    return p99


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", nargs="+", required=True, help="one or more dump .npz files (dump_format.md)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="init + batch order; the validation split uses --split-seed")
    ap.add_argument("--backend", choices=("torch", "tcnn"), default="torch", help="tcnn: UNTESTED, relu, no culling")
    # model
    ap.add_argument("--width", type=int, default=32, choices=(16, 32, 64))
    ap.add_argument("--hidden", type=int, default=2, help="hidden layers, 1..4")
    ap.add_argument("--latent-channels", type=int, default=8, help="4..16")
    ap.add_argument("--latent-res", default="64", help="N or HxW latent grid size")
    ap.add_argument("--activation", choices=sorted(ACTIVATIONS), default="leaky_smelu")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--loss-power", type=float, default=3.0)
    # optimisation
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=0.02, help="prototype used 0.04 for a 2400-step run; sweep for long runs")
    ap.add_argument("--lr-end-frac", type=float, default=0.02)
    ap.add_argument("--latent-lr-mult", type=float, default=1.0)
    ap.add_argument("--batch-total", type=int, default=65536, help="K * batch, constant in every phase")
    # culling
    ap.add_argument("--no-culling", action="store_true", help="baseline: 1 instance, batch = --batch-total")
    ap.add_argument("--instances", type=int, default=64, help="K at the start")
    ap.add_argument("--phases", type=int, default=4)
    ap.add_argument("--schedule", default=None, help="explicit 'K:batch,...' (last K must be 1); overrides the above")
    # data / validation
    ap.add_argument("--max-records", type=int, default=0, help="use at most this many valid records (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-max", type=int, default=262144)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--eval-size", type=int, default=65536, help="held-out records used for culling and periodic eval")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-export", action="store_true", help="do not write model.nnm at the end")
    ap.add_argument("--stop-after", type=int, default=0, help="(test hook) save a resume checkpoint and exit after this step")
    return ap.parse_args(argv)


def parse_res(text):
    if "x" in text:
        h, w = text.lower().split("x")
        return int(h), int(w)
    return int(text), int(text)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("train.py: --device cuda requested but CUDA is not available (use --device cpu for a slow test)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False            # keep fp32 matmuls exact; the decoder is tiny
    os.makedirs(args.out, exist_ok=True)

    # ---- resume state or fresh config
    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        assert ck.get("format") == "nnm_train_resume_v1", "not a resume checkpoint"
        cfg = ModelConfig.from_dict(ck["config"])
        args.steps = int(ck["args"]["steps"])
        args.backend = ck["backend"]
        sched = [tuple(x) for x in ck["schedule"]]
    else:
        lh, lw = parse_res(args.latent_res)
        cfg = ModelConfig(width=args.width, num_hidden=args.hidden, latent_channels=args.latent_channels,
                          latent_height=lh, latent_width=lw, activation=args.activation, beta=args.beta,
                          eps=args.eps, loss_power=args.loss_power).validate()
        if args.no_culling:
            sched = [(1, args.batch_total)]
        elif args.schedule:
            sched = schedule.parse_schedule(args.schedule)
        else:
            sched = schedule.make_schedule(args.instances, args.phases, args.batch_total)
    if args.backend == "tcnn" and len(sched) != 1:
        sys.exit("train.py: --backend tcnn needs --no-culling")
    psteps = schedule.phase_steps(args.steps, len(sched))

    # ---- data
    rs, reports, metas = dataset.load_records(args.dump)
    for r in reports:
        print(f"{r['path']}: kept {r['kept']}/{r['records']} (flagged {r['flagged']}, nonfinite {r['nonfinite']}, "
              f"not_unit {r['not_unit']}, below_horizon {r['below_horizon']})")
    if args.max_records and rs.n > args.max_records:
        rs = rs.take(np.random.default_rng(0).choice(rs.n, args.max_records, replace=False))
    tr_idx, va_idx = dataset.split_indices(rs.n, args.val_frac, args.val_max, args.split_seed)
    train_data, val_data = DeviceData(rs.take(tr_idx), device), DeviceData(rs.take(va_idx), device)
    ev_sel = np.random.default_rng(0).permutation(val_data.n)[:args.eval_size]
    ev_data = DeviceData(rs.take(va_idx[np.sort(ev_sel)]), device)
    del rs
    print(f"train {train_data.n} records, validation {val_data.n} (culling/eval subset {ev_data.n})")

    # ---- model
    first_k = ck["k"] if ck else sched[0][0]
    model = nm.make_model(cfg, first_k, args.backend, args.seed, device)
    opt = Adam(model.param_list())
    g, start_phase, start_s = 0, 0, 0
    if ck:
        model.load_state_dict(ck["params"])               # keys = parameter names; resume is torch-backend only
        opt.load(ck["adam"], device)
        g, start_phase, start_s = ck["global_step"], ck["phase"], ck["step_in_phase"]
        print(f"resumed at step {g} (phase {start_phase}, K={model.k})")
    lr_mults = ([args.latent_lr_mult if n == "latent" else 1.0 for n in model.param_names]
                if args.backend == "torch" else None)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"model": cfg.to_dict(), "args": vars(args), "schedule": sched, "phase_steps": psteps,
                   "dump_meta": metas}, f, indent=1, default=str)
    print(f"schedule (K@batch x steps): " + " -> ".join(f"{k}@{b} x{n}" for (k, b), n in zip(sched, psteps)))
    print(f"decoder params {model.num_decoder_params() if args.backend == 'torch' else 'n/a'}, "
          f"latent {cfg.latent_height}x{cfg.latent_width}x{cfg.latent_channels} per instance")

    csv_path = os.path.join(args.out, "train_log.csv")
    fresh = not (ck and os.path.exists(csv_path))
    csv_f = open(csv_path, "w" if fresh else "a", newline="")
    log = csv.writer(csv_f)
    if fresh:
        log.writerow(["step", "phase", "k", "batch", "lr", "train_loss_min", "train_loss_mean",
                      "val_loss_min", "val_loss_mean", "elapsed_s"])
    phases_info = []
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed * 1000003 + g)
    sampler = Sampler(train_data.n, device, gen)
    t0 = time.time()

    for pi in range(start_phase, len(sched)):
        K, bs = sched[pi]
        assert model.k == K, f"phase {pi} expects K={K}, model has {model.k}"
        acc, cnt = torch.zeros(K, device=device), 0
        for s in range(start_s if pi == start_phase else 0, psteps[pi]):
            idx = sampler.next(bs)
            uv, wi, wo, tgt = train_data.uv[idx], train_data.wi[idx], train_data.wo[idx], train_data.tgt[idx]
            loss_k = nm.power_loss(model(uv, wi, wo), tgt, cfg.loss_power)               # (K,)
            for p in model.param_list():
                p.grad = None
            loss_k.sum().backward()                                                       # each instance: its own mean
            lr = schedule.lr_at(g, args.steps, args.lr, args.lr_end_frac)
            opt.step(model.param_list(), lr, lr_mults)
            g += 1
            acc += loss_k.detach()
            cnt += 1
            if g % args.log_every == 0 or g == args.steps:
                tl = (acc / cnt).cpu()
                if not torch.isfinite(tl).all():
                    sys.exit(f"train.py: non-finite loss at step {g}; lower --lr")
                vmin = vmean = ""
                if g % args.eval_every == 0 or s == psteps[pi] - 1:
                    vl = val_losses(model, ev_data, cfg.loss_power).cpu()
                    vmin, vmean = f"{vl.min().item():.5f}", f"{vl.mean().item():.5f}"
                log.writerow([g, pi, K, bs, f"{lr:.3e}", f"{tl.min().item():.5f}", f"{tl.mean().item():.5f}",
                              vmin, vmean, f"{time.time() - t0:.1f}"])
                csv_f.flush()
                print(f"step {g}/{args.steps} phase {pi} K={K} bs={bs} lr={lr:.2e} train {tl.min().item():.4f}/"
                      f"{tl.mean().item():.4f} val {vmin or '-'}/{vmean or '-'}  {time.time() - t0:.0f}s", flush=True)
                acc.zero_()
                cnt = 0
            if g % args.ckpt_every == 0 and args.backend == "torch":
                save_resume(os.path.join(args.out, "ckpt_last.pt"), model, opt, cfg, args, sched, g, pi, s + 1)
            if args.stop_after and g >= args.stop_after:
                save_resume(os.path.join(args.out, "ckpt_last.pt"), model, opt, cfg, args, sched, g, pi, s + 1)
                csv_f.close()
                print(f"stopped after step {g} (test hook); resume with --resume")
                return
        # ---- phase end: rank on held-out records, cull
        score = val_losses(model, ev_data, cfg.loss_power).cpu()
        info = {"phase": pi, "k": K, "batch": bs, "steps": psteps[pi], "val_min": float(score.min()),
                "val_median": float(score.median()), "val_max": float(score.max()), "val_std": float(score.std() if K > 1 else 0.0)}
        if pi < len(sched) - 1:
            keep = torch.argsort(score)[:sched[pi + 1][0]]
            info["kept_scores"] = [float(x) for x in score[keep]]
            model.select(keep)
            opt.select(keep)
        phases_info.append(info)
        print("phase end:", json.dumps(info), flush=True)
    csv_f.close()
    with open(os.path.join(args.out, "phases.json"), "w") as f:
        json.dump(phases_info, f, indent=1)

    # ---- final: held-out metrics, checkpoints, export
    assert model.k == 1
    pred = predict_instance(model, val_data)
    vm = metrics.compute_metrics(pred, val_data.tgt.cpu().numpy(), val_data.wi.cpu().numpy(), val_data.wo.cpu().numpy(), cfg.loss_power)
    with open(os.path.join(args.out, "final_metrics.json"), "w") as f:
        json.dump(vm, f, indent=1)
    print("held-out:", json.dumps(vm))
    arrays = model.to_arrays(0)
    tol = 1e-3 if args.backend == "torch" else 5e-2
    print(f"export self-check p99 rel err {self_check(model, arrays, cfg, val_data, tol=tol):.2e}")
    save_arrays_npz(os.path.join(args.out, "model_final.npz"), arrays, cfg)
    atomic_save({"format": "nnm_train_export_v1", "config": cfg.to_dict(),
                 "arrays": {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()},
                 "val_metrics": vm}, os.path.join(args.out, "model_final.pt"))
    if args.backend == "torch":
        save_resume(os.path.join(args.out, "ckpt_last.pt"), model, opt, cfg, args, sched, g, len(sched) - 1, psteps[-1])
    if not args.no_export:
        import export_nnm
        try:
            export_nnm.export(os.path.join(args.out, "model_final.npz"), os.path.join(args.out, "model.nnm"))
        except Exception as e:                                # keep the training result even if export fails
            print(f"WARNING: export to .nnm failed: {e}. Run export_nnm.py manually.", file=sys.stderr)
    print("done")


if __name__ == "__main__":
    main()
