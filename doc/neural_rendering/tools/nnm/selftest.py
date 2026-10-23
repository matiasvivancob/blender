# SPDX-License-Identifier: Apache-2.0
"""Self test: build random models, round-trip them through the file format and
check the reference decoder against the embedded test vectors.

    python3 -m nnm.selftest [--out DIR]       (run from doc/neural_rendering/tools)

With --out the generated .nnm files are kept; build_standalone_test.sh feeds
them to the C++ kernel header.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np

from .export_numpy import build_model, random_layers
from .format import (ACT_LEAKY_RELU, ACT_LEAKY_SMELU, ACT_RELU, ENC_BSDF_DIRS, ENC_RAW, NnmFormatError, NnmModel,
                     read_nnm, write_nnm)
from .reference import activation, decode_float64, decode_test_inputs, encode_bsdf_dirs

WIDTHS = (16, 32, 64)
ACTS = (ACT_RELU, ACT_LEAKY_RELU, ACT_LEAKY_SMELU)
XFORMS = (0, 1, 2)

# Structural variants cycled over the (width, activation, transform) grid.
# (encoding, caller_dim for raw, latent (C, H, W, levels) or None, num_hidden, output_dim)
VARIANTS = [
    (ENC_BSDF_DIRS, 0, (8, 16, 16, 3), 2, 3),
    (ENC_BSDF_DIRS, 0, None, 3, 3),
    (ENC_RAW, 5, None, 1, 1),
    (ENC_RAW, 7, (4, 8, 4, 2), 4, 9),
    (ENC_BSDF_DIRS, 0, (16, 5, 3, 1), 2, 8),
    (ENC_RAW, 1, None, 2, 16),
]


def act_params(act: int, rng: np.random.Generator):
    if act == ACT_RELU:
        return 0.0, 0.0
    if act == ACT_LEAKY_RELU:
        return 0.0, float(np.float32(rng.uniform(0.01, 0.2)))
    return float(np.float32(rng.uniform(0.5, 1.5))), float(np.float32(rng.uniform(0.01, 0.2)))


def make_model(width: int, act: int, xform: int, variant: int, seed: int, n_tv: int = 16,
               small_latent: bool = False) -> NnmModel:
    enc, raw_dim, lat, num_hidden, out_dim = VARIANTS[variant % len(VARIANTS)]
    rng = np.random.default_rng(seed)
    lat_c = lat[0] if lat else 0
    enc_dim = 12 if enc == ENC_BSDF_DIRS else raw_dim
    layers = random_layers(rng, enc_dim + lat_c, width, num_hidden, out_dim)
    latent, levels = None, 1
    if lat:
        c, h, w, levels = lat
        latent = rng.uniform(-1, 1, (h, w, c)).astype(np.float32)
    beta, eps = act_params(act, rng)
    return build_model(layers, width, activation=act, beta=beta, eps=eps, output_transform=xform, encoding=enc,
                       latent=latent, latent_levels=levels, num_test_vectors=n_tv, seed=seed + 1)


def model_name(width: int, act: int, xform: int, variant: int) -> str:
    return f"w{width}_act{act}_out{xform}_v{variant % len(VARIANTS)}"


def all_models(n_tv: int = 16):
    i = 0
    for width in WIDTHS:
        for act in ACTS:
            for xform in XFORMS:
                yield model_name(width, act, xform, i), make_model(width, act, xform, i, 1000 + i, n_tv)
                i += 1


def models_equal(a: NnmModel, b: NnmModel) -> bool:
    if (a.input_dim, a.output_dim, a.num_hidden, a.width, a.activation, a.output_transform, a.encoding,
            a.latent_channels) != (b.input_dim, b.output_dim, b.num_hidden, b.width, b.activation,
                                   b.output_transform, b.encoding, b.latent_channels):
        return False
    if np.float32(a.act_beta) != np.float32(b.act_beta) or np.float32(a.act_eps) != np.float32(b.act_eps):
        return False
    for (w0, b0), (w1, b1) in zip(a.layers, b.layers):
        if not (np.array_equal(w0, w1) and np.array_equal(b0, b1)):
            return False
    if len(a.latent_levels) != len(b.latent_levels):
        return False
    if not all(np.array_equal(x, y) for x, y in zip(a.latent_levels, b.latent_levels)):
        return False
    return np.array_equal(a.tv_inputs, b.tv_inputs) and np.array_equal(a.tv_outputs, b.tv_outputs)


def close(got, ref, rtol=1e-4, atol=1e-5) -> bool:
    return bool(np.all(np.abs(got - ref) <= rtol * np.abs(ref) + atol))


def check_model(name: str, m: NnmModel, errors: List[str]) -> None:
    data = write_nnm(m)
    r = read_nnm(data)
    if not models_equal(m, r):
        errors.append(f"{name}: read-back differs from written model")
    if write_nnm(r) != data:
        errors.append(f"{name}: rewrite is not byte identical")
    # The reference decoder must reproduce the embedded vectors from the read-back model, exactly.
    got = decode_test_inputs(r, r.tv_inputs)
    if not np.array_equal(got, r.tv_outputs):
        errors.append(f"{name}: reference decoder does not reproduce test vectors bit-exactly")
    # Independent float64 matmul decoder agrees (checks the input-major layout and the formulas).
    ref64 = decode_float64(r, r.tv_inputs)
    if not close(r.tv_outputs, ref64):
        worst = np.max(np.abs(r.tv_outputs - ref64) / (np.abs(ref64) + 1e-5))
        errors.append(f"{name}: float32 reference differs from float64 matmul decoder (rel {worst:.2e})")


def check_negative(m: NnmModel, errors: List[str]) -> None:
    data = bytearray(write_nnm(m))

    def expect_fail(label, mutate):
        d = bytearray(data)
        mutate(d)
        try:
            read_nnm(bytes(d))
        except NnmFormatError:
            return
        errors.append(f"negative test '{label}' was accepted")

    expect_fail("bad magic", lambda d: d.__setitem__(slice(0, 4), b"XXXX"))
    expect_fail("bad width", lambda d: d.__setitem__(28, 24))
    expect_fail("truncated", lambda d: d.__delitem__(slice(len(d) - 8, len(d))))
    expect_fail("bad layer offset", lambda d: d.__setitem__(132 + 4, d[132 + 4] ^ 16))
    expect_fail("bad version", lambda d: d.__setitem__(4, 2))
    expect_fail("bad activation", lambda d: d.__setitem__(32, 7))


def check_math(errors: List[str]) -> None:
    rng = np.random.default_rng(7)
    # Encoding: f32 reference versus an independent float64 Rodrigues rotation; wd' must be unit length.
    n = 4096
    ct = rng.uniform(0.02, 1, (2, n))
    ph = rng.uniform(0, 2 * np.pi, (2, n))
    d = np.stack([np.sqrt(1 - ct ** 2) * np.cos(ph), np.sqrt(1 - ct ** 2) * np.sin(ph), ct], axis=2)
    wi, wo = d[0], d[1]
    enc = encode_bsdf_dirs(wi, wo).astype(np.float64)
    wi64, wo64 = enc[:, 0:3], enc[:, 3:6]
    h = wi64 + wo64
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    v = np.stack([h[:, 1], -h[:, 0], np.zeros(n)], axis=1)
    c1 = np.cross(v, wi64)
    wd = wi64 + c1 + np.cross(v, c1) / (1 + h[:, 2])[:, None]
    if np.max(np.abs(enc[:, 6:9] - h)) > 1e-5 or np.max(np.abs(enc[:, 9:12] - wd)) > 1e-5:
        errors.append("encode_bsdf_dirs disagrees with float64 Rodrigues rotation")
    if np.max(np.abs(np.linalg.norm(enc[:, 9:12], axis=1) - 1)) > 1e-5:
        errors.append("wd' is not unit length")
    # Rotating h by the same shortest arc must give +z (checks the axis/sign convention).
    hr = h * h[:, 2:3] + np.cross(v, h) + (np.sum(v * h, axis=1) / (1 + h[:, 2]))[:, None] * v
    if np.max(np.abs(hr - np.array([0, 0, 1.0]))) > 1e-9:
        errors.append("shortest-arc formula does not map h to +z")
    # LeakySmeLU matches the piecewise definition and is C1.
    x = np.linspace(-3, 3, 2001)
    beta, eps = 0.8, 0.05
    g = np.where(x <= -beta, 0.0, np.where(x >= beta, x, (x + beta) ** 2 / (4 * beta)))
    ref = eps * x + (1 - eps) * g
    got = activation(x.astype(np.float32), ACT_LEAKY_SMELU, beta, eps).astype(np.float64)
    if np.max(np.abs(got - ref)) > 2e-6:
        errors.append("LeakySmeLU differs from its piecewise definition")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="directory to keep the generated .nnm files in")
    a = ap.parse_args(argv)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    errors: List[str] = []
    count = 0
    for name, m in all_models():
        before = len(errors)
        check_model(name, m, errors)
        if a.out:
            write_nnm(m, os.path.join(a.out, name + ".nnm"))
        count += 1
        print(f"{'PASS' if len(errors) == before else 'FAIL'} {name}")
    check_negative(make_model(32, ACT_LEAKY_SMELU, 1, 0, 5), errors)
    check_math(errors)
    if errors:
        print("\n".join("ERROR: " + e for e in errors))
        print(f"selftest FAILED ({len(errors)} errors)")
        return 1
    print(f"selftest OK: {count} models, negative and math checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
