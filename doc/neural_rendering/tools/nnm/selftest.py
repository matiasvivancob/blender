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
from .format import (ACT_LEAKY_RELU, ACT_LEAKY_SMELU, ACT_RELU, ENC_BSDF_DIRS, ENC_LEARNED_FRAMES, ENC_RAW,
                     HEADER_SIZE, NnmFormatError, NnmModel, pad8, read_nnm, validate_model, write_nnm)
from .reference import (activation, decode_float64, decode_test_inputs, encode_bsdf_dirs, encode_learned_frames,
                        make_orthonormals_a)

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


# ---------------------------------------------------------------------------
# learned_frames models (encoding 2)
# ---------------------------------------------------------------------------
FRAME_LATENT = (8, 16, 16, 2)  # C, H, W, levels
IDENTITY_FRAME = np.array([0, 0, 1, 1, 0, 0], np.float32)  # n = z axis, t = x axis


def hemisphere(rng: np.random.Generator, n: int) -> np.ndarray:
    ct = rng.uniform(0.05, 1.0, n)
    st = np.sqrt(1 - ct * ct)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)


def make_frames_model(K: int, width: int, act: int, xform: int, seed: int, mode: str = "random", n_tv: int = 16,
                      proxy=None) -> NnmModel:
    """Encoding-2 model with random latent and MLP. ``mode``: random frame weights (frames are rotated),
    "fallback_nt" (frame 0 has n = 0, frame 1 has t = 0), "fallback_cross" (frame 0: t = 2 n, frame 1: n = (c, c, c)
    and t = -3 n, so the cross product vanishes and both branches of make_orthonormals are used)."""
    rng = np.random.default_rng(seed)
    C, h, w, levels = FRAME_LATENT
    layers = random_layers(rng, 6 * K + C, width, 2, 3)
    latent = rng.uniform(-1, 1, (h, w, C)).astype(np.float32)
    beta, eps = act_params(act, rng)
    Wf = rng.uniform(-0.35, 0.35, (C, 6 * K)).astype(np.float32)
    bf = (np.tile(IDENTITY_FRAME, K) + rng.uniform(-0.25, 0.25, 6 * K)).astype(np.float32)
    if mode == "fallback_nt":
        Wf[:, 0:3] = 0
        bf[0:3] = 0           # frame 0: n = 0 -> (0, 0, 1)
        Wf[:, 9:12] = 0
        bf[9:12] = 0          # frame 1: t = 0 -> (1, 0, 0)
    elif mode == "fallback_cross":
        Wf[:, 3:6] = 2 * Wf[:, 0:3]
        bf[3:6] = 2 * bf[0:3]  # frame 0: t = 2 n exactly
        Wf[:, 6:9] = 0
        bf[6:9] = 0.5          # frame 1: n = (c, c, c) with all components equal
        Wf[:, 9:12] = 0
        bf[9:12] = -1.5        # t = -3 n
    elif mode != "random":
        raise ValueError(mode)
    from .export_numpy import make_mip_chain
    m = NnmModel(input_dim=6 * K + C, output_dim=3, num_hidden=2, width=width,
                 layers=[(wl.astype(np.float16), bl.astype(np.float16)) for wl, bl in layers],
                 activation=act, act_beta=beta, act_eps=eps, output_transform=xform, encoding=ENC_LEARNED_FRAMES,
                 latent_channels=C, latent_levels=make_mip_chain(latent, levels), proxy=proxy, frames_k=K,
                 frame_layer=(Wf.astype(np.float16), bf.astype(np.float16)))
    rng2 = np.random.default_rng(seed + 1)
    m.tv_inputs = np.concatenate([hemisphere(rng2, n_tv), hemisphere(rng2, n_tv), rng2.uniform(0, 1, (n_tv, 2))],
                                 axis=1).astype(np.float32)
    m.tv_outputs = np.zeros((n_tv, 3), np.float32)
    validate_model(m)
    m.tv_outputs = decode_test_inputs(m, m.tv_inputs)
    return m


def frames_model_name(K: int, width: int, act: int, xform: int, mode: str = "random") -> str:
    return f"frames_k{K}_w{width}_act{act}_out{xform}" + ("" if mode == "random" else "_" + mode)


def frames_models(n_tv: int = 16):
    """K = 2 and K = 1 over every width and activation (transforms cycled), plus the two fallback models."""
    i = 0
    for K in (2, 1):
        for width in WIDTHS:
            for act in ACTS:
                xform = XFORMS[(i + i // 3) % 3]
                yield frames_model_name(K, width, act, xform), make_frames_model(K, width, act, xform, 5000 + i, "random", n_tv)
                i += 1
    for mode in ("fallback_nt", "fallback_cross"):
        yield (frames_model_name(2, 32, ACT_LEAKY_SMELU, 1, mode),
               make_frames_model(2, 32, ACT_LEAKY_SMELU, 1, 6000 + i, mode, n_tv))
        i += 1


def models_equal(a: NnmModel, b: NnmModel) -> bool:
    pa = None if a.proxy is None else tuple(np.float32(x) for x in a.proxy)
    pb = None if b.proxy is None else tuple(np.float32(x) for x in b.proxy)
    if (a.frames_k, pa) != (b.frames_k, pb) or (a.frame_layer is None) != (b.frame_layer is None):
        return False
    if a.frame_layer is not None and not all(np.array_equal(x, y) for x, y in zip(a.frame_layer, b.frame_layer)):
        return False
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


def check_frames(errors: List[str]) -> None:
    rng = np.random.default_rng(11)
    n = 512
    wi, wo = hemisphere(rng, n).astype(np.float32), hemisphere(rng, n).astype(np.float32)
    z = rng.uniform(-1, 1, (n, 8)).astype(np.float32)

    # Identity frames (Wf = 0, bias = identity) give (wi, wo) repeated K times, exactly.
    for K in (1, 2, 3, 4):
        enc = encode_learned_frames(z, wi, wo, np.zeros((8, 6 * K), np.float16),
                                    np.tile(IDENTITY_FRAME, K).astype(np.float16), K)
        if not np.array_equal(enc, np.tile(np.concatenate([wi, wo], axis=1), (1, K))):
            errors.append(f"identity frames (K = {K}) do not reproduce (wi, wo) repeated")

    # Fixed rotation R in every frame: n = R e3, t = R e1 (so b = R e2), encoded = (R^T wi, R^T wo).
    # Frame vectors are scaled by arbitrary positive factors to exercise the normalisation.
    def rot(axis, ang):
        a = np.asarray(axis, np.float64)
        a /= np.linalg.norm(a)
        K_ = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(ang) * K_ + (1 - np.cos(ang)) * (K_ @ K_)
    exact = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float64) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)
    for label, R, tol in (("generic", rot([0.3, -0.5, 0.8], 0.9), 4e-3), ("axis-aligned", exact, 0.0)):
        K = 2
        bias = np.concatenate([np.concatenate([2.0 * R[:, 2], 0.5 * R[:, 0]]) for _ in range(K)]).astype(np.float16)
        enc = encode_learned_frames(z, wi, wo, np.zeros((8, 6 * K), np.float16), bias, K).astype(np.float64)
        exp = np.concatenate([wi.astype(np.float64) @ R, wo.astype(np.float64) @ R], axis=1)  # rows: R^T w
        exp = np.tile(exp, (1, K))
        err = np.max(np.abs(enc - exp))
        if err > tol:
            errors.append(f"rotation check ({label}): max error {err:.2e} > {tol}")
    # Rotation equivariance: nonzero Wf, rotating all frame vectors by R rotates the encoding the same way.
    C, K = 8, 2
    Wf = rng.uniform(-0.4, 0.4, (C, 6 * K)).astype(np.float16)
    bf = (np.tile(IDENTITY_FRAME, K) + rng.uniform(-0.2, 0.2, 6 * K)).astype(np.float16)
    R = exact  # signed permutation: rotated f16 weights and directions stay exact
    R16 = lambda a: a.astype(np.float64)
    def rotate_cols(a):  # rotate every 3-vector (n and t of every frame) by R, in float64 then to f16
        v = R16(a).reshape(a.shape[:-1] + (2 * K, 3))
        return (v @ R.T).reshape(a.shape).astype(np.float16)
    base = encode_learned_frames(z, wi, wo, Wf, bf, K).astype(np.float64)
    wi_r, wo_r = (wi.astype(np.float64) @ R.T).astype(np.float32), (wo.astype(np.float64) @ R.T).astype(np.float32)
    rotated = encode_learned_frames(z, wi_r, wo_r, rotate_cols(Wf), rotate_cols(bf), K).astype(np.float64)
    if np.max(np.abs(base - rotated)) > 2e-6:
        errors.append("rotating frames and directions together does not leave the encoding invariant "
                      f"(max error {np.max(np.abs(base - rotated)):.2e})")

    # make_orthonormals_a: unit, orthogonal to n, both branches.
    nn = np.concatenate([hemisphere(rng, 200), np.full((3, 3), 0.57735026), np.array([[0, 0, 1.0], [1.0, 0, 0]])])
    a = make_orthonormals_a(nn.astype(np.float32)).astype(np.float64)
    if np.max(np.abs(np.linalg.norm(a, axis=1) - 1)) > 1e-6 or np.max(np.abs(np.sum(a * nn.astype(np.float32), axis=1))) > 1e-6:
        errors.append("make_orthonormals_a is not a unit vector orthogonal to n")

    # Round trip with frames and proxy together, plus the usual decoder checks.
    proxy = (0.2, 0.3, 0.5, 0.1, 0.4, 0.9)
    m = make_frames_model(2, 32, ACT_LEAKY_RELU, 1, 77, "random", 8, proxy=proxy)
    check_model("frames+proxy", m, errors)
    r = read_nnm(write_nnm(m))
    if r.proxy is None or tuple(np.float32(x) for x in r.proxy) != tuple(np.float32(x) for x in proxy):
        errors.append("proxy lost in frames + proxy round trip")
    if r.frames_k != 2 or r.frame_layer is None:
        errors.append("frames lost in frames + proxy round trip")

    # Files without encoding 2 have zero bytes 184..256; with encoding 2 only 184..192 are used.
    plain = write_nnm(make_model(32, ACT_RELU, 0, 0, 3))
    if any(plain[184:256]):
        errors.append("encoding 1 file has non-zero bytes in 184..256")

    # -- rejection ---------------------------------------------------------------------------
    def expect_model_fail(label, mutate):
        mm = make_frames_model(2, 16, ACT_RELU, 0, 91, "random", 8)
        mutate(mm)
        try:
            write_nnm(mm)
        except NnmFormatError:
            return
        errors.append(f"frames model rejection '{label}' was accepted by the writer")

    def set_attr(**kw):
        return lambda mm: [setattr(mm, k, v) for k, v in kw.items()]

    expect_model_fail("frames_k 0", set_attr(frames_k=0))
    expect_model_fail("frames_k 5", set_attr(frames_k=5))
    expect_model_fail("wrong encoded width", set_attr(frames_k=1))
    expect_model_fail("missing frame layer", set_attr(frame_layer=None))
    expect_model_fail("wrong frame layer shape", set_attr(frame_layer=(np.zeros((8, 8), np.float16), np.zeros(8, np.float16))))
    expect_model_fail("non-finite frame weight", lambda mm: mm.frame_layer[0].__setitem__((0, 0), np.float16(np.inf)))
    expect_model_fail("float32 frame layer", set_attr(frame_layer=(mm_w32(), np.zeros(12, np.float32))))
    mm = make_frames_model(2, 16, ACT_RELU, 0, 91, "random", 0)
    mm.latent_channels, mm.latent_levels, mm.input_dim = 0, [], 12
    mm.layers[0] = (mm.layers[0][0][:12], mm.layers[0][1])
    mm.frame_layer = (np.zeros((0, 12), np.float16), np.zeros(12, np.float16))
    try:
        write_nnm(mm)
        errors.append("frames model without latent was accepted by the writer")
    except NnmFormatError:
        pass
    m13 = make_model(16, ACT_RELU, 0, 0, 9)  # encoding 1
    m13.frames_k = 2
    try:
        write_nnm(m13)
        errors.append("encoding 1 with frames_k != 0 accepted by the writer")
    except NnmFormatError:
        pass
    m13 = make_model(16, ACT_RELU, 0, 0, 9)
    m13.frame_layer = (np.zeros((8, 12), np.float16), np.zeros(12, np.float16))
    try:
        write_nnm(m13)
        errors.append("encoding 1 with a frame layer accepted by the writer")
    except NnmFormatError:
        pass

    good = write_nnm(make_frames_model(1, 16, ACT_RELU, 0, 92, "random", 8))  # K = 1: 6 of 8 columns used
    gm = read_nnm(good)

    def expect_bytes_fail(label, data0, mutate):
        d = bytearray(data0)
        mutate(d)
        try:
            read_nnm(bytes(d))
        except NnmFormatError:
            return
        errors.append(f"byte-level rejection '{label}' was accepted by the reader")

    from .format import frame_layer_size, frames_offset
    f_off = HEADER_SIZE + frames_offset(gm)
    expect_bytes_fail("frame padding", good, lambda d: d.__setitem__(f_off + 6 * 2, 1))
    expect_bytes_fail("frame bias padding", good, lambda d: d.__setitem__(f_off + 8 * 2 * 8 + 7 * 2, 0x3c))
    expect_bytes_fail("reserved 192", good, lambda d: d.__setitem__(192, 1))
    expect_bytes_fail("reserved 255", good, lambda d: d.__setitem__(255, 1))
    expect_bytes_fail("frames_k 0", good, lambda d: d.__setitem__(slice(184, 188), (0).to_bytes(4, "little")))
    expect_bytes_fail("frames_k 5", good, lambda d: d.__setitem__(slice(184, 188), (5).to_bytes(4, "little")))
    expect_bytes_fail("frames_k mismatching width", good, lambda d: d.__setitem__(slice(184, 188), (2).to_bytes(4, "little")))
    expect_bytes_fail("frames_offset wrong", good, lambda d: d.__setitem__(slice(188, 192), (frames_offset(gm) + 16).to_bytes(4, "little")))
    expect_bytes_fail("frames_offset zero", good, lambda d: d.__setitem__(slice(188, 192), (0).to_bytes(4, "little")))
    expect_bytes_fail("weight_size without frame layer", good,
                      lambda d: d.__setitem__(slice(80, 88), (frames_offset(gm)).to_bytes(8, "little")))
    if frame_layer_size(gm) != 9 * 8 * 2:
        errors.append("frame layer size is not (C + 1) * pad8(6K) * 2")
    enc1 = write_nnm(make_model(16, ACT_RELU, 0, 0, 9))
    expect_bytes_fail("frames_k with encoding 1", enc1, lambda d: d.__setitem__(slice(184, 188), (2).to_bytes(4, "little")))
    expect_bytes_fail("frames_offset with encoding 1", enc1, lambda d: d.__setitem__(slice(188, 192), (64).to_bytes(4, "little")))
    expect_bytes_fail("reserved bytes with encoding 1", enc1, lambda d: d.__setitem__(230, 1))
    enc0 = write_nnm(make_model(16, ACT_RELU, 0, 2, 9))  # raw
    expect_bytes_fail("frames_k with encoding 0", enc0, lambda d: d.__setitem__(slice(184, 188), (1).to_bytes(4, "little")))
    expect_bytes_fail("encoding 2 on an encoding 1 file", enc1, lambda d: d.__setitem__(40, 2))
    # Encoding 2 without a latent: patch latent_channels to 0 is caught by the other consistency rules.
    expect_bytes_fail("encoding 2 without latent", good, lambda d: d.__setitem__(44, 0))


def mm_w32():
    return np.zeros((8, 12), np.float32)


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
    for name, m in list(all_models()) + list(frames_models()):
        before = len(errors)
        check_model(name, m, errors)
        if a.out:
            write_nnm(m, os.path.join(a.out, name + ".nnm"))
        count += 1
        print(f"{'PASS' if len(errors) == before else 'FAIL'} {name}")
    check_frames(errors)
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
