#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Workstation check of the torch code against the torch-free references. RUN THIS FIRST (needs torch).

    python3 crosscheck_torch.py            # CPU, about a minute
    python3 crosscheck_torch.py --cuda     # also runs the training smoke test on the GPU

Checks (each prints PASS/FAIL, exit status 1 if any fails):
  1. model.encode_bsdf_dirs      == encoding_np == nnm.reference == float64 Rodrigues
  2. model.activation            == nnm.reference.activation for relu / leaky_relu / leaky_smelu
  3. model.bilinear_corners      == nnm.reference.latent_sample_bilinear (incl. the u, v = 0 and 1 edges)
  4. model.power_loss + autograd == the numpy prototype's loss_and_gout (value and gradient w.r.t. the output)
  5. train.Adam                  == the numpy prototype's Adam, also after select()
  6. NeuralMaterial forward      == np_model.forward_np, and its exported .nnm (fp16) decodes to the same values
     through nnm.reference; select() keeps the right instance
  7. train.py end-to-end on a tiny synthetic dump (culling 4 -> 1, a stop/resume cycle, export + verify),
     loss must fall
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

import _paths  # noqa: F401

try:
    import torch
except ImportError:
    sys.exit("crosscheck_torch: torch is not installed; nothing to check (the torch-free parts run in run_all_synthetic.sh)")

import export_nnm
import make_synthetic_dump
import model as nm
import np_model
import train as train_mod
from encoding_np import encode_bsdf_dirs as enc_np
from encoding_np import encode_bsdf_dirs_f64, latent_lookup
from nnm import reference as ref
from nnm.format import read_nnm, write_nnm

T = _paths.load_proto()
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def hemi(rng, n, cmin=0.005):
    ct = rng.uniform(cmin, 1.0, n)
    st = np.sqrt(1 - ct * ct)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1).astype(np.float32)


def t32(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def test_encoding():
    rng = np.random.default_rng(0)
    wi, wo = hemi(rng, 20000), hemi(rng, 20000)
    wo[:50] = wi[:50]                                                  # wi == wo: h = wi
    got = nm.encode_bsdf_dirs(t32(wi), t32(wo)).numpy()
    d_np = np.abs(got - enc_np(wi, wo)).max()
    d_ref = np.abs(got - ref.encode_bsdf_dirs(wi, wo)).max()
    d_64 = np.abs(got - encode_bsdf_dirs_f64(wi, wo)).max()
    check("encoding torch vs numpy / nnm.reference / float64", d_np < 2e-6 and d_ref < 2e-6 and d_64 < 5e-5,
          f"max abs diff {d_np:.1e} {d_ref:.1e} {d_64:.1e}")
    got3 = nm.encode_bsdf_dirs(t32(wi).reshape(2, 10000, 3), t32(wo).reshape(2, 10000, 3)).numpy()
    check("encoding keeps leading dims", got3.shape == (2, 10000, 12) and np.abs(got3.reshape(-1, 12) - got).max() < 1e-7)


def test_activation():
    x = np.linspace(-6, 6, 4001).astype(np.float32)
    for kind, aid, beta, eps in (("relu", 0, 0.0, 0.0), ("leaky_relu", 1, 0.0, 0.1), ("leaky_smelu", 2, 1.0, 0.05),
                                 ("leaky_smelu", 2, 0.5, 0.0)):
        got = nm.activation(t32(x), kind, beta if beta else 1.0, eps).numpy()
        d = np.abs(got - ref.activation(x, aid, beta, eps)).max()
        check(f"activation {kind} beta={beta} eps={eps}", d < 2e-6, f"max abs diff {d:.1e}")


def test_bilinear():
    rng = np.random.default_rng(1)
    H, W, C = 7, 5, 3
    lat16 = rng.uniform(-1, 1, (H, W, C)).astype(np.float16)
    uv = rng.random((5000, 2)).astype(np.float32)
    uv[:8] = [[0, 0], [1, 1], [0, 1], [1, 0], [0.5, 1], [1, 0.5], [0, 0.3], [0.7, 0]]
    idx, wt = nm.bilinear_corners(t32(uv), H, W)
    flat = torch.from_numpy(lat16.astype(np.float32).reshape(H * W, C))
    got = (flat[idx] * wt[..., None]).sum(-2).numpy()
    d = np.abs(got - ref.latent_sample_bilinear(lat16, uv[:, 0], uv[:, 1])).max()
    d2 = np.abs(got - latent_lookup(lat16.astype(np.float32), uv)).max()
    check("bilinear latent lookup vs nnm.reference", d < 2e-6 and d2 < 2e-6, f"max abs diff {d:.1e} {d2:.1e}")
    check("bilinear weights sum to 1", np.abs(wt.sum(-1).numpy() - 1).max() < 1e-6)


def test_loss():
    rng = np.random.default_rng(2)
    K, B = 2, 257
    out = rng.uniform(-3, 3, (K, B, 3)).astype(np.float32)
    tgt = np.abs(rng.standard_normal((B, 3))).astype(np.float32) ** 2
    tgt[:10] = 0
    L_np, g_np = T.loss_and_gout(out.astype(np.float64), tgt, "pow")
    o = torch.tensor(out, requires_grad=True)
    L = nm.power_loss(o, t32(tgt), 3.0)
    L.sum().backward()
    check("power_loss value vs prototype", np.abs(L.detach().numpy() - L_np).max() < 1e-5, f"{L.detach().numpy()} {L_np}")
    check("power_loss gradient vs prototype", np.abs(o.grad.numpy() - g_np).max() < 1e-6,
          f"max abs diff {np.abs(o.grad.numpy() - g_np).max():.1e}")
    o2 = torch.tensor([[[30.0, -30.0, 0.0]]], requires_grad=True)
    nm.power_loss(nm.clamp_straight_through(o2), t32([[1.0, 1.0, 2.0]]), 3.0).sum().backward()
    check("straight-through clamp: forward clamped, gradient passes",
          np.allclose(nm.clamp_straight_through(o2).detach().numpy(), [[[14.0, -20.0, 0.0]]]) and bool((o2.grad != 0).all()))


def test_adam():
    rng = np.random.default_rng(3)
    P = {"a": rng.standard_normal((3, 4, 5)).astype(np.float32), "b": rng.standard_normal((3, 1, 5)).astype(np.float32)}
    Pt = [torch.tensor(P["a"], requires_grad=True), torch.tensor(P["b"], requires_grad=True)]
    opt_np, opt_t = T.Adam(P), train_mod.Adam(Pt)
    for step in range(6):
        G = {k: rng.standard_normal(v.shape).astype(np.float32) for k, v in P.items()}
        for p, k in zip(Pt, ("a", "b")):
            p.grad = torch.from_numpy(G[k].copy())
        opt_np.step(P, G, 0.05)
        opt_t.step(Pt, 0.05)
        if step == 2:                                                        # cull 3 -> 2 in the middle
            keep = np.array([2, 0])
            P = {k: v[keep] for k, v in P.items()}
            opt_np.subset(keep)
            Pt = [torch.tensor(p.detach().numpy()[keep], requires_grad=True) for p in Pt]
            opt_t.select(keep)
    d = max(np.abs(P["a"] - Pt[0].detach().numpy()).max(), np.abs(P["b"] - Pt[1].detach().numpy()).max())
    check("Adam vs prototype Adam (with select)", d < 1e-5, f"max abs diff {d:.1e}")


def test_model_and_export():
    for act in ("leaky_smelu", "relu", "leaky_relu"):
        cfg = np_model.ModelConfig(width=32, num_hidden=3, latent_channels=6, latent_height=16, latent_width=12,
                                   activation=act, beta=0.7, eps=0.05 if act != "relu" else 0.0).validate()
        m = nm.NeuralMaterial(cfg, k=3, seed=5)
        with torch.no_grad():
            for p in m.param_list():                                         # random weights incl. biases, on the fp16 grid
                p.copy_((torch.randn_like(p) * 0.4).half().float() if p is not m.latent else p.half().float())
        rng = np.random.default_rng(6)
        uv, wi, wo = rng.random((3000, 2)).astype(np.float32), hemi(rng, 3000), hemi(rng, 3000)
        with torch.no_grad():
            pred = m.predict(t32(uv), t32(wi), t32(wo)).numpy()              # (3, N, 3)
        arrays = m.to_arrays(1)
        d_np = (np.abs(pred[1] - np_model.forward_np(arrays, cfg, uv, wi, wo)) / (np.abs(pred[1]) + 1e-3)).max()
        mdl = export_nnm.build(arrays, cfg, num_tv=16)
        mdl2 = read_nnm(write_nnm(mdl))
        got = ref.decode(mdl2, np.concatenate([wi, wo], 1), uv)
        d_nnm = (np.abs(pred[1] - got) / (np.abs(pred[1]) + 1e-3)).max()
        check(f"model forward vs numpy forward ({act}, K=3, 3 hidden)", d_np < 1e-4, f"max rel {d_np:.1e}")
        check(f"exported .nnm decodes to the torch prediction ({act})", d_nnm < 2e-3, f"max rel {d_nnm:.1e}")
        before = {i: m.to_arrays(i) for i in (0, 2)}
        m.select([2, 0])
        same = (m.k == 2 and all(np.array_equal(m.to_arrays(0)[k], v) for k, v in before[2].items())
                and all(np.array_equal(m.to_arrays(1)[k], v) for k, v in before[0].items()))
        with torch.no_grad():
            pred2 = m.predict(t32(uv), t32(wi), t32(wo)).numpy()
        check(f"select([2, 0]) keeps instances ({act})", same and np.abs(pred2[0] - pred[2]).max() < 1e-5
              and np.abs(pred2[1] - pred[0]).max() < 1e-5)


def test_train_smoke(device):
    with tempfile.TemporaryDirectory() as td:
        d, _ = make_synthetic_dump.make(30000, seed=1)
        from dataset import save_dump
        dump = os.path.join(td, "d.npz")
        save_dump(dump, **d)
        common = ["--dump", dump, "--device", device, "--width", "16", "--latent-res", "16", "--latent-channels", "4",
                  "--steps", "80", "--schedule", "4:64,1:256", "--lr", "0.02", "--eval-size", "2000", "--eval-every", "40",
                  "--log-every", "10", "--val-frac", "0.1"]
        out = os.path.join(td, "run")
        train_mod.main(common + ["--out", out, "--stop-after", "25"])
        check("stop-after wrote a resume checkpoint", os.path.exists(os.path.join(out, "ckpt_last.pt")))
        train_mod.main(common + ["--out", out, "--resume", os.path.join(out, "ckpt_last.pt")])
        rows = open(os.path.join(out, "train_log.csv")).read().strip().splitlines()[1:]
        steps = [int(r.split(",")[0]) for r in rows]
        check("resume continues the log without a gap or repeat", steps == sorted(set(steps)) and steps[-1] == 80, str(steps))
        first, last = float(rows[0].split(",")[6]), float(rows[-1].split(",")[6])
        check("training loss falls", last < 0.9 * first, f"{first:.3f} -> {last:.3f}")
        ph = json.load(open(os.path.join(out, "phases.json")))
        check("culling 4 -> 1 happened", [p["k"] for p in ph] == [4, 1], str([p["k"] for p in ph]))
        check("model.nnm exported and verified by train.py", os.path.exists(os.path.join(out, "model.nnm")))
        for kind in ("model_final.pt", "model_final.npz"):
            res = export_nnm.export(os.path.join(out, kind), os.path.join(td, kind + ".nnm"), verbose=False)[1]
            check(f"export_nnm reads {kind}", res["fp16_rel_err_p99"] < 0.05, f"p99 fp16 {res['fp16_rel_err_p99']:.1e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", action="store_true", help="run the training smoke test on cuda instead of cpu")
    a = ap.parse_args()
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    torch.manual_seed(0)
    for fn in (test_encoding, test_activation, test_bilinear, test_loss, test_adam, test_model_and_export):
        fn()
    test_train_smoke("cuda" if a.cuda else "cpu")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
