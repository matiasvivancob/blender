#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Torch-free trainer for the smoke test: the numpy prototype's model, forward/backward and Adam
(experiments/neural_material_train/train.py) fed from a dump instead of its built-in sampler.

    python3 numpy_train_small.py synthetic.npz tiny.npz --steps 300

Writes the interchange .npz of np_model.py (latent, W0.., b0.., config_json), the input of export_nnm.py.
`--steps 0` writes the random initialisation. The prototype is fixed at a 64x64x8 latent grid, width 16 or 32,
2 hidden layers, LeakySmeLU(beta=1, eps=0.05), power loss n=3. Also useful as a numerical oracle for the torch
trainer: same data, same loss, same init distribution.
"""
import argparse
import time

import numpy as np

import _paths  # noqa: F401
import dataset
from np_model import ModelConfig, check_arrays, save_arrays_npz

T = _paths.load_proto()


def proto_to_arrays(P, k=0):
    """Prototype parameter dict (leading instance axis) -> interchange arrays of instance k."""
    res, lat = T.RES, T.LAT
    return {
        "latent": P["grid"][k].reshape(res, res, lat).astype(np.float32),
        "W0": np.concatenate([P["W1d"][k], P["W1z"][k]], 0).astype(np.float32),     # direction features first
        "b0": P["b1"][k, 0].astype(np.float32),
        "W1": P["W2"][k].astype(np.float32), "b1": P["b2"][k, 0].astype(np.float32),
        "W2": P["W3"][k].astype(np.float32), "b2": P["b3"][k, 0].astype(np.float32),
    }


def proto_config(width):
    return ModelConfig(width=width, num_hidden=2, latent_channels=T.LAT, latent_height=T.RES, latent_width=T.RES,
                       activation="leaky_smelu", beta=T.BETA, eps=T.LEAK, loss_power=T.POW_N).validate()


def train(records, width=16, steps=300, batch=4096, lr=0.04, seed=0, verbose=True):
    rng = np.random.default_rng(1000 + seed)
    P = T.init_params(np.random.default_rng(seed), 1, width)
    opt = T.Adam(P)
    losses = []
    t0 = time.time()
    for s in range(steps):
        idx = rng.integers(0, records.n, batch)
        B = T.Batch(records.tex_uv[idx].astype(np.float64), records.wi[idx], records.wo[idx], records.target[idx])
        out, cache = T.forward(P, B, "lsmelu", keep_cache=True)
        L, gout = T.loss_and_gout(out, B.t, "pow")
        G = T.backward(P, B, cache, gout, "lsmelu")
        opt.step(P, G, T.lr_at(s, steps, lr))
        losses.append(float(L[0]))
        if verbose and (s % 100 == 0 or s == steps - 1):
            print(f"  step {s:5d} loss {L[0]:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    return P, losses


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("out", help="output .npz (interchange format)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--width", type=int, default=16, choices=(16, 32))
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    records, reports, _ = dataset.load_records(a.dump)
    tr, _ = dataset.split_indices(records.n, 0.05, 262144, 0)
    print(f"{records.n} valid records ({reports[0]['dropped']} dropped), training on {len(tr)}")
    P, losses = train(records.take(tr), a.width, a.steps, a.batch, a.lr, a.seed)
    cfg = proto_config(a.width)
    arrays = proto_to_arrays(P)
    check_arrays(arrays, cfg)
    save_arrays_npz(a.out, arrays, cfg)
    print(f"wrote {a.out} (final train loss {losses[-1] if losses else float('nan'):.4f})")


if __name__ == "__main__":
    main()
