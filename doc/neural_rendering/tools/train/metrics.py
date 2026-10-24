# SPDX-License-Identifier: Apache-2.0
"""Error metrics on f*cos predictions versus targets (numpy). Same definitions as
experiments/neural_material_train/train.py:metrics_from_preds."""
import numpy as np


def pow_transform(x, n=3.0):
    return n * (np.power(np.maximum(x, 0.0), 1.0 / n) - 1.0)


def _ratio(num, den):
    return float(num / den) if den > 0 else float("nan")


def compute_metrics(pred, tgt, wi=None, wo=None, power=3.0, grazing_deg=75.0, top_frac=0.01):
    """pred, tgt (N,3) linear values; wi, wo (N,3) local-frame directions (for the grazing subset).

    Returns a dict of python floats:
      pow_loss          mean |M(pred) - M(tgt)|, M = power-transform with n = `power`  (the training loss)
      rel_l1            sum|pred - tgt| / sum(tgt)
      rel_l1_top1       same on the `top_frac` brightest targets (mean over RGB)
      rel_l1_grazing    same on records with theta_i or theta_o > grazing_deg
      peak_energy_top01 sum(pred) / sum(tgt) on the brightest 0.1 % (1.0 = peaks fully reproduced)
    """
    pred = np.asarray(pred, np.float64)
    tgt = np.asarray(tgt, np.float64)
    err = np.abs(pred - tgt)
    lum = tgt.mean(1)
    out = {"n": int(len(tgt))}
    out["pow_loss"] = float(np.abs(pow_transform(pred, power) - pow_transform(tgt, power)).mean())
    out["rel_l1"] = _ratio(err.sum(), tgt.sum())
    top = lum >= np.quantile(lum, 1.0 - top_frac)
    out["rel_l1_top1"] = _ratio(err[top].sum(), tgt[top].sum())
    top2 = lum >= np.quantile(lum, 0.999)
    out["peak_energy_top01"] = _ratio(pred[top2].sum(), tgt[top2].sum())
    if wi is not None and wo is not None:
        c = np.cos(np.radians(grazing_deg))
        graz = (np.asarray(wi)[:, 2] < c) | (np.asarray(wo)[:, 2] < c)
        out["grazing_frac"] = float(graz.mean())
        out["rel_l1_grazing"] = _ratio(err[graz].sum(), tgt[graz].sum()) if graz.any() else float("nan")
    return out
