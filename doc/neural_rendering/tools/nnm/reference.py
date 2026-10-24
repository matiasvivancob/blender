# SPDX-License-Identifier: Apache-2.0
"""numpy reference decoder for .nnm models.

Mirrors intern/cycles/kernel/util/mlp.h operation for operation: float32
arithmetic, fp16 weights promoted exactly, a per-output axpy accumulation loop
(bias first, inputs in ascending order) with a fused multiply-add emulated
through float64, and the same activation / transform / encoding formulas. All
functions take batches (leading axis N) and treat each row independently.
"""
from __future__ import annotations

import numpy as np

from .format import (ACT_LEAKY_RELU, ACT_LEAKY_SMELU, ACT_RELU, ENC_BSDF_DIRS, ENC_LEARNED_FRAMES, OUT_EXP, OUT_NONE,
                     OUT_SOFTPLUS, NnmModel)

F32 = np.float32


def f32(x) -> np.ndarray:
    return np.asarray(x, dtype=F32)


def fma32(a, b, c) -> np.ndarray:
    """Fused multiply-add on float32 operands.

    The product of two float32 values is exact in float64, so one float64 add
    followed by a rounding to float32 equals a hardware FMA except in rare
    double-rounding ties.
    """
    return (np.asarray(a, np.float64) * np.asarray(b, np.float64) + np.asarray(c, np.float64)).astype(F32)


# ---------------------------------------------------------------------------
# activations and output transforms
# ---------------------------------------------------------------------------
def activation(x, act_id: int, beta: float, eps: float) -> np.ndarray:
    x = f32(x)
    if act_id == ACT_RELU:
        return np.where(x > 0, x, F32(0)).astype(F32)
    eps = F32(eps)
    if act_id == ACT_LEAKY_RELU:
        return np.where(x > 0, x, eps * x).astype(F32)
    if act_id == ACT_LEAKY_SMELU:
        beta = F32(beta)
        c4 = F32(0.25) / beta
        s = np.minimum(np.maximum(x + beta, F32(0)), F32(2) * beta)
        g = fma32(s * s, c4, np.maximum(x - beta, F32(0)))
        return fma32(eps, x, (F32(1) - eps) * g)
    raise ValueError(f"unknown activation id {act_id}")


def log1p32(e) -> np.ndarray:
    """Kahan's log1p trick in float32, as in mlp.h (the kernel has no log1pf on all backends)."""
    e = f32(e)
    u = F32(1) + e
    d = u - F32(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(u) * (e / d)
    return np.where(d == 0, e, r).astype(F32)


def output_transform(x, xform_id: int) -> np.ndarray:
    x = f32(x)
    if xform_id == OUT_NONE:
        return x
    if xform_id == OUT_EXP:
        return np.exp(np.minimum(np.maximum(x, F32(-20)), F32(14))).astype(F32)
    if xform_id == OUT_SOFTPLUS:
        xs = np.minimum(x, F32(20))
        return np.where(x > F32(20), x, log1p32(np.exp(xs).astype(F32))).astype(F32)
    raise ValueError(f"unknown output transform id {xform_id}")


# ---------------------------------------------------------------------------
# input encoding
# ---------------------------------------------------------------------------
def encode_bsdf_dirs(wi, wo) -> np.ndarray:
    """wi, wo (N,3) -> (N,12) = [wi, wo, h, wd'], float32, same op order as the kernel."""
    wi, wo = f32(wi), f32(wo)
    wix, wiy, wiz = wi[:, 0], wi[:, 1], wi[:, 2]
    hx, hy, hz = wi[:, 0] + wo[:, 0], wi[:, 1] + wo[:, 1], wi[:, 2] + wo[:, 2]
    l2 = hx * hx + hy * hy + hz * hz
    ok = l2 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = F32(1) / np.sqrt(l2)
    hx = np.where(ok, hx * inv, F32(0)).astype(F32)
    hy = np.where(ok, hy * inv, F32(0)).astype(F32)
    hz = np.where(ok, hz * inv, F32(1)).astype(F32)
    # v = (hy, -hx, 0)
    vdw = hy * wix - hx * wiy
    k = vdw / np.maximum(F32(1) + hz, F32(1e-6))
    wdx = wix * hz + (-hx * wiz) + k * hy
    wdy = wiy * hz + (-hy * wiz) + k * (-hx)
    wdz = wiz * hz + (hx * wix + hy * wiy)
    return np.stack([wix, wiy, wiz, wo[:, 0], wo[:, 1], wo[:, 2], hx, hy, hz, wdx, wdy, wdz], axis=1).astype(F32)


# ---------------------------------------------------------------------------
# learned shading frames (05_learned_shading_frames.md section 2, nnm_format.md section 7)
# ---------------------------------------------------------------------------
FRAME_EPS2 = F32(1e-12)


def _dot3(a, b) -> np.ndarray:
    """(N,3) . (N,3) in the kernel's order: (x*x + y*y) + z*z, float32, no fused ops."""
    return (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) + a[:, 2] * b[:, 2]


def make_orthonormals_a(n) -> np.ndarray:
    """First output `a` of Cycles' make_orthonormals(N, &a, &b) (util/math_float3.h), float32, (N,3) -> (N,3).

    a = normalize((1,1,1) x N), or normalize((-1,1,1) x N) when N.x == N.y == N.z; unit and orthogonal to N.
    """
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    differ = (nx != ny) | (nx != nz)
    ax = nz - ny
    ay = np.where(differ, nx - nz, nx + nz).astype(F32)
    az = np.where(differ, ny - nx, -ny - nx).astype(F32)
    a = np.stack([ax, ay, az], axis=1).astype(F32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a / np.sqrt(_dot3(a, a))[:, None]).astype(F32)


def encode_learned_frames(z, wi, wo, W16, b16, K: int) -> np.ndarray:
    """z (N,C) latent, wi, wo (N,3), frame layer W16 (C,6K) / b16 (6K,) f16 -> (N,6K) float32.

    Same operation order as mlp.h `neural_encode_learned_frames`: f = bf + sum_c z[c] Wf[c] with one fma per
    product (ascending c), then per frame normalise n and t (division by sqrt of the plain-product squared
    length, fallbacks (0,0,1) and (1,0,0) when the squared length is <= 1e-12), b = normalise(n x t) with the
    make_orthonormals fallback, and out = (wi.t, wi.b, wi.n, wo.t, wo.b, wo.n).
    """
    z, wi, wo = f32(z), f32(wi), f32(wo)
    f = linear_axpy(z, W16, b16)
    out = np.zeros((z.shape[0], 6 * K), F32)
    for k in range(K):
        n = f[:, 6 * k:6 * k + 3]
        t = f[:, 6 * k + 3:6 * k + 6]
        l2n, l2t = _dot3(n, n), _dot3(t, t)
        with np.errstate(divide="ignore", invalid="ignore"):
            n = np.where((l2n > FRAME_EPS2)[:, None], n / np.sqrt(l2n)[:, None], np.array([0, 0, 1], F32)).astype(F32)
            t = np.where((l2t > FRAME_EPS2)[:, None], t / np.sqrt(l2t)[:, None], np.array([1, 0, 0], F32)).astype(F32)
        b = np.stack([n[:, 1] * t[:, 2] - n[:, 2] * t[:, 1],
                      n[:, 2] * t[:, 0] - n[:, 0] * t[:, 2],
                      n[:, 0] * t[:, 1] - n[:, 1] * t[:, 0]], axis=1).astype(F32)
        l2b = _dot3(b, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where((l2b > FRAME_EPS2)[:, None], b / np.sqrt(l2b)[:, None], make_orthonormals_a(n)).astype(F32)
        for j, w in enumerate((wi, wo)):
            out[:, 6 * k + 3 * j + 0] = _dot3(w, t)
            out[:, 6 * k + 3 * j + 1] = _dot3(w, b)
            out[:, 6 * k + 3 * j + 2] = _dot3(w, n)
    return out


def encode_learned_frames_f64(z, wi, wo, W16, b16, K: int) -> np.ndarray:
    """Independent float64 version of `encode_learned_frames` (plain matmul, np.cross, np.linalg.norm)."""
    z, wi, wo = (np.asarray(x, np.float64) for x in (z, wi, wo))
    f = z @ W16.astype(np.float64) + b16.astype(np.float64)
    out = np.zeros((z.shape[0], 6 * K))
    for k in range(K):
        n, t = f[:, 6 * k:6 * k + 3].copy(), f[:, 6 * k + 3:6 * k + 6].copy()
        ln, lt = np.sum(n * n, axis=1), np.sum(t * t, axis=1)
        n = np.where((ln > 1e-12)[:, None], n / np.sqrt(np.maximum(ln, 1e-300))[:, None], [0.0, 0.0, 1.0])
        t = np.where((lt > 1e-12)[:, None], t / np.sqrt(np.maximum(lt, 1e-300))[:, None], [1.0, 0.0, 0.0])
        b = np.cross(n, t)
        lb = np.sum(b * b, axis=1)
        # make_orthonormals(n): normalize((1,1,1) x n), or (-1,1,1) x n if all components are equal.
        eq = (n[:, 0] == n[:, 1]) & (n[:, 0] == n[:, 2])
        a = np.cross(np.where(eq[:, None], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]), n)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = np.where((lb > 1e-12)[:, None], b / np.sqrt(np.maximum(lb, 1e-300))[:, None], a)
        for j, w in enumerate((wi, wo)):
            out[:, 6 * k + 3 * j:6 * k + 3 * j + 3] = np.stack(
                [np.sum(w * t, axis=1), np.sum(w * b, axis=1), np.sum(w * n, axis=1)], axis=1)
    return out


# ---------------------------------------------------------------------------
# latent sampling
# ---------------------------------------------------------------------------
def latent_sample_bilinear(level0: np.ndarray, u, v) -> np.ndarray:
    """level0 (H,W,C) f16, u,v (N,) -> (N,C) float32. Corner-to-corner texel grid, clamped."""
    H, W, C = level0.shape
    tex = level0.astype(F32)
    u = np.minimum(np.maximum(f32(u), F32(0)), F32(1))
    v = np.minimum(np.maximum(f32(v), F32(0)), F32(1))
    x = u * F32(W - 1)
    y = v * F32(H - 1)
    ix = np.clip(np.floor(x).astype(np.int64), 0, max(W - 2, 0))
    iy = np.clip(np.floor(y).astype(np.int64), 0, max(H - 2, 0))
    fx = (x - ix.astype(F32)).astype(F32)
    fy = (y - iy.astype(F32)).astype(F32)
    ix1 = np.minimum(ix + 1, W - 1)
    iy1 = np.minimum(iy + 1, H - 1)
    one = F32(1)
    w00 = (one - fx) * (one - fy)
    w10 = fx * (one - fy)
    w01 = (one - fx) * fy
    w11 = fx * fy
    t00, t10, t01, t11 = tex[iy, ix], tex[iy, ix1], tex[iy1, ix], tex[iy1, ix1]
    acc = t00 * w00[:, None]
    acc = fma32(t10, w10[:, None], acc)
    acc = fma32(t01, w01[:, None], acc)
    acc = fma32(t11, w11[:, None], acc)
    return acc.astype(F32)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
def linear_axpy(x: np.ndarray, w16: np.ndarray, b16: np.ndarray) -> np.ndarray:
    """acc[o] = b[o]; for i ascending: acc[o] = fma(x[i], W[i][o], acc[o]). x (N,in) -> (N,out)."""
    w = w16.astype(F32)
    acc = np.broadcast_to(b16.astype(F32), (x.shape[0], w.shape[1])).copy()
    for i in range(w.shape[0]):
        acc = fma32(x[:, i:i + 1], w[i][None, :], acc)
    return acc


def mlp_eval(model: NnmModel, x) -> np.ndarray:
    """Hidden layers + output layer + output transform on the (N,input_dim) MLP input."""
    a = f32(x)
    for l, (w, b) in enumerate(model.layers):
        a = linear_axpy(a, w, b)
        if l < model.num_layers - 1:
            a = activation(a, model.activation, model.act_beta, model.act_eps)
    return output_transform(a, model.output_transform)


def encode_input(model: NnmModel, caller, uv=None) -> np.ndarray:
    """caller (N, caller_dim) [+ uv (N,2)] -> MLP input (N, input_dim)."""
    caller = f32(caller)
    z = None
    if model.latent_channels:
        uv = f32(uv)
        z = latent_sample_bilinear(model.latent_levels[0], uv[:, 0], uv[:, 1])
    if model.encoding == ENC_BSDF_DIRS:
        enc = encode_bsdf_dirs(caller[:, 0:3], caller[:, 3:6])
    elif model.encoding == ENC_LEARNED_FRAMES:
        # The frames depend on the latent sample, so the lookup comes first.
        enc = encode_learned_frames(z, caller[:, 0:3], caller[:, 3:6], *model.frame_layer, model.frames_k)
    else:
        enc = caller
    if z is not None:
        enc = np.concatenate([enc, z], axis=1)
    return enc.astype(F32)


def decode(model: NnmModel, caller, uv=None) -> np.ndarray:
    """Full pipeline: encode, latent lookup, MLP, output transform. Returns (N, output_dim) float32."""
    return mlp_eval(model, encode_input(model, caller, uv))


def decode_test_inputs(model: NnmModel, tv_inputs: np.ndarray) -> np.ndarray:
    """Decode rows laid out like the test-vector block's input part."""
    tv_inputs = f32(tv_inputs)
    cd = model.caller_dim
    uv = tv_inputs[:, cd:cd + 2] if model.latent_channels else None
    return decode(model, tv_inputs[:, :cd], uv)


def decode_float64(model: NnmModel, tv_inputs: np.ndarray) -> np.ndarray:
    """Independent check: plain float64 matmul decoder (x @ W + b) on the same quantised weights."""
    tv = np.asarray(tv_inputs, np.float64)
    cd = model.caller_dim
    caller = tv[:, :cd]
    z = None
    if model.latent_channels:
        lv = model.latent_levels[0].astype(np.float64)
        H, W, C = lv.shape
        u, v_ = np.clip(tv[:, cd], 0, 1), np.clip(tv[:, cd + 1], 0, 1)
        x, y = u * (W - 1), v_ * (H - 1)
        ix = np.clip(np.floor(x).astype(int), 0, max(W - 2, 0)); fx = (x - ix)[:, None]
        iy = np.clip(np.floor(y).astype(int), 0, max(H - 2, 0)); fy = (y - iy)[:, None]
        ix1, iy1 = np.minimum(ix + 1, W - 1), np.minimum(iy + 1, H - 1)
        z = ((1 - fx) * (1 - fy) * lv[iy, ix] + fx * (1 - fy) * lv[iy, ix1]
             + (1 - fx) * fy * lv[iy1, ix] + fx * fy * lv[iy1, ix1])
    if model.encoding == ENC_LEARNED_FRAMES:
        enc = encode_learned_frames_f64(z, caller[:, 0:3], caller[:, 3:6], *model.frame_layer, model.frames_k)
    elif model.encoding == ENC_BSDF_DIRS:
        wi, wo = caller[:, 0:3], caller[:, 3:6]
        h = wi + wo
        h /= np.linalg.norm(h, axis=1, keepdims=True)
        v = np.stack([h[:, 1], -h[:, 0], np.zeros(len(h))], axis=1)
        # Shortest-arc rotation (Rodrigues) taking h to +z, applied to wi.
        cr = np.cross(v, wi)
        wd = wi + cr + np.cross(v, cr) / (1 + h[:, 2])[:, None]
        enc = np.concatenate([wi, wo, h, wd], axis=1)
    else:
        enc = caller
    if z is not None:
        enc = np.concatenate([enc, z], axis=1)
    a = enc
    for l, (w, b) in enumerate(model.layers):
        a = a @ w.astype(np.float64) + b.astype(np.float64)
        if l < model.num_layers - 1:
            if model.activation == ACT_RELU:
                a = np.maximum(a, 0)
            elif model.activation == ACT_LEAKY_RELU:
                a = np.where(a > 0, a, model.act_eps * a)
            else:
                be, ep = float(model.act_beta), float(model.act_eps)
                g = np.where(a <= -be, 0.0, np.where(a >= be, a, (a + be) ** 2 / (4 * be)))
                a = ep * a + (1 - ep) * g
    if model.output_transform == OUT_EXP:
        a = np.exp(np.clip(a, -20, 14))
    elif model.output_transform == OUT_SOFTPLUS:
        a = np.where(a > 20, a, np.log1p(np.exp(np.minimum(a, 20))))
    return a
