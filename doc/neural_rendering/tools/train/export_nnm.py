#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a trained checkpoint into a `.nnm` file (tools/nnm), embed test vectors, re-read it and verify.

    python3 export_nnm.py runs/a/model_final.pt  material.nnm            # torch checkpoint (needs torch)
    python3 export_nnm.py runs/a/model_final.npz material.nnm            # numpy arrays; torch not needed
    python3 export_nnm.py runs/a/ckpt_last.pt    material.nnm --instance 0
    python3 export_nnm.py runs/a/model_final.npz material.nnm --dump dump_a.npz dump_b.npz   # + Cycles proxy block
    python3 export_nnm.py runs/a/model_final.npz material.nnm --proxy 0.2 0.2 0.5 0.9 0.9 0.9  # explicit constants

The Cycles neural closure needs the proxy block (nnm_format.md section 3a): GGX alpha, cosine-lobe probability and
albedo. `--dump` fits it from the training dump (proxy_fit.py); `--proxy` writes six given numbers
(alpha_x alpha_y p_diffuse albedo_rgb). Without either the file has no proxy block and the Cycles node rejects it.

Accepted inputs:
  * .npz : latent, W0..Wn, b0..bn (+ optional config_json); np_model.py documents the layout.
  * .pt  : torch.save of {"format": "nnm_train_export_v1", "arrays": {...}, "config": {...}} (written by
           train.py), or a resume checkpoint {"params": {name: (K,...) tensor}, ...}; --instance picks one.
The activation / eps / beta / output transform come from the checkpoint config; flags override them.

Verification, all of it must pass or the script exits 1 (the file is written first so it can be inspected):
  1. nnm.format.read_nnm re-parses the file with full validation.
  2. every embedded test vector is reproduced by nnm.reference (the format's own tolerance, section 6).
  3. the float64 matmul reference decoder agrees on the same vectors.
  4. on random (uv, wi, wo) the fp16 .nnm decode matches the unquantised float64 forward of the training model
     (np_model.forward_np); the deviation is the fp16 quantisation error and is reported.
"""
import argparse
import sys

import numpy as np

import _paths  # noqa: F401
from nnm.export_numpy import build_model
from nnm.format import ENC_BSDF_DIRS, OUT_EXP, read_nnm, write_nnm
from nnm.reference import decode, decode_float64, decode_test_inputs
import np_model
from np_model import ModelConfig, forward_np
import proxy_fit


def load_checkpoint(path, instance=0):
    """-> (arrays dict float32 numpy, ModelConfig or None)."""
    if str(path).endswith(".npz"):
        return np_model.load_arrays_npz(path)
    try:
        import torch
    except ImportError:
        sys.exit("export_nnm: reading a .pt checkpoint needs torch. train.py also writes model_final.npz next to it; "
                 "pass that file instead.")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ModelConfig.from_dict(ck["config"]) if "config" in ck else None
    if "arrays" in ck:
        arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in ck["arrays"].items()}
    elif "params" in ck:
        arrays = {k: v.detach().cpu().numpy()[instance].astype(np.float32) for k, v in ck["params"].items()}
        # resume checkpoints keep biases as (K, 1, out)
        for k in list(arrays):
            if k.startswith("b"):
                arrays[k] = arrays[k].reshape(-1)
    else:
        raise ValueError("unrecognised checkpoint: no 'arrays' or 'params'")
    return arrays, cfg


def random_inputs(rng, n):
    """Random valid (uv, wi, wo): hemisphere directions with z >= 0.05."""
    def hemi():
        ct = rng.uniform(0.05, 1.0, n)
        st = np.sqrt(1 - ct * ct)
        ph = rng.uniform(0, 2 * np.pi, n)
        return np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1).astype(np.float32)
    return rng.random((n, 2)).astype(np.float32), hemi(), hemi()


def build(arrays, cfg: ModelConfig, num_tv=32, latent_levels=1, seed=0):
    np_model.check_arrays(arrays, cfg)
    aid, beta, eps = cfg.nnm_activation()
    layers = np_model.layers_from_arrays(arrays)
    return build_model(layers, cfg.width, activation=aid, beta=beta, eps=eps, output_transform=OUT_EXP,
                       encoding=ENC_BSDF_DIRS, latent=arrays["latent"], latent_levels=latent_levels,
                       num_test_vectors=num_tv, seed=seed)


def verify_file(path, arrays, cfg, n_random=4096, seed=1, fp16_tol=0.05):
    """Steps 1 to 4 of the module docstring. Returns a dict of numbers; raises AssertionError on failure."""
    m = read_nnm(path)                                                           # 1
    res = {"file_bytes": int(len(open(path, "rb").read())), "test_vectors": int(m.tv_inputs.shape[0])}
    if m.tv_inputs.shape[0]:
        got = decode_test_inputs(m, m.tv_inputs)                                 # 2
        bad = np.abs(got - m.tv_outputs) > 1e-4 * np.abs(m.tv_outputs) + 1e-5
        assert not bad.any(), f"{int(bad.sum())} test-vector outputs differ from nnm.reference"
        g64 = decode_float64(m, m.tv_inputs)                                     # 3
        rel = np.abs(g64 - m.tv_outputs) / (np.abs(m.tv_outputs) + 1e-5)
        res["tv_float64_max_rel"] = float(rel.max())
        assert rel.max() < 1e-3, f"float64 reference differs from the embedded vectors by {rel.max():.2e}"
    rng = np.random.default_rng(seed)
    uv, wi, wo = random_inputs(rng, n_random)
    caller = np.concatenate([wi, wo], 1)
    pred_nnm = np.concatenate([decode(m, caller[s:s + 8192], uv[s:s + 8192]) for s in range(0, n_random, 8192)])
    pred_fp = forward_np(arrays, cfg, uv, wi, wo)                                # 4
    rel = np.abs(pred_nnm - pred_fp) / (np.abs(pred_fp) + 1e-3)
    res["fp16_rel_err_median"] = float(np.median(rel))
    res["fp16_rel_err_p99"] = float(np.quantile(rel, 0.99))
    res["fp16_rel_err_max"] = float(rel.max())
    assert res["fp16_rel_err_p99"] < fp16_tol, (
        f".nnm decode deviates from the training model: p99 relative error {res['fp16_rel_err_p99']:.3e} "
        f"(limit {fp16_tol}). Check input order, activation parameters and the exp clamp.")
    return res


def fit_proxy_from_dumps(paths):
    """Fit the proxy block from training dump files -> (six floats, fit dict)."""
    import dataset
    rs, _, _ = dataset.load_records(list(paths))
    fit = proxy_fit.fit_proxy(rs.wi, rs.wo, rs.target, rs.albedo)
    return proxy_fit.proxy_tuple(fit), fit


def export(ckpt, out, num_tv=32, latent_levels=1, seed=0, instance=0, overrides=None, fp16_tol=0.05, verbose=True,
           proxy=None, dumps=None):
    arrays, cfg = load_checkpoint(ckpt, instance)
    cfg = np_model.infer_config(arrays, cfg)
    for k, v in (overrides or {}).items():
        if v is not None:
            setattr(cfg, k, v)
    cfg.validate()
    model = build(arrays, cfg, num_tv, latent_levels, seed)
    if dumps and proxy is None:
        proxy, fit = fit_proxy_from_dumps(dumps)
        if verbose:
            print(f"proxy fit from {len(dumps)} dump file(s): alpha {fit['alpha']:.4g}, p_diffuse {fit['p_diffuse']:.1f}, "
                  f"albedo {np.round(fit['albedo'], 4).tolist()}, second moment {fit['moment']:.4g}")
    if proxy is not None:
        model.proxy = tuple(float(v) for v in proxy)
    data = write_nnm(model, out)
    res = verify_file(out, arrays, cfg, fp16_tol=fp16_tol)
    if verbose:
        print(f"wrote {out}: {len(data)} bytes, input {model.input_dim}, {cfg.num_hidden}x{cfg.width}, latent "
              f"{cfg.latent_height}x{cfg.latent_width}x{cfg.latent_channels}, activation {cfg.activation}")
        print("verified: " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
    return model, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--instance", type=int, default=0, help="instance index for resume checkpoints (K > 1)")
    ap.add_argument("--tv", type=int, default=32, help="number of embedded test vectors (0..64)")
    ap.add_argument("--latent-levels", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--activation", choices=sorted(np_model.ACTIVATIONS), default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--fp16-tol", type=float, default=0.05, help="limit on the p99 relative fp16 deviation")
    ap.add_argument("--dump", nargs="+", default=None, help="training dump(s) to fit the Cycles proxy block from")
    ap.add_argument("--proxy", nargs=6, type=float, default=None,
                    metavar=("AX", "AY", "PD", "R", "G", "B"), help="write these proxy constants instead of fitting")
    a = ap.parse_args(argv)
    try:
        export(a.checkpoint, a.out, a.tv, a.latent_levels, a.seed, a.instance,
               dict(activation=a.activation, beta=a.beta, eps=a.eps), a.fp16_tol,
               proxy=a.proxy, dumps=a.dump)
    except AssertionError as e:
        print(f"VERIFY FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
