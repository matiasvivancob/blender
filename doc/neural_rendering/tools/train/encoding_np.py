# SPDX-License-Identifier: Apache-2.0
"""numpy implementation of the `bsdf_dirs` input encoding and the bilinear latent lookup.

This is the second of three independent implementations of the encoding in nnm_format.md section 7:
  1. tools/nnm/reference.py   (the format's reference, written with the kernel op order)
  2. this file                (float32, written for batches with arbitrary leading dims, same op order as 3)
  3. tools/train/model.py     (torch, used in training)
test_torchfree.py checks 1 == 2 == float64 Rodrigues; crosscheck_torch.py (workstation) checks 3 == 2.
"""
import numpy as np

F32 = np.float32


def encode_bsdf_dirs(wi, wo):
    """wi, wo (..., 3) unit vectors, z > 0 -> (..., 12) float32 = [wi, wo, h, wd]."""
    wi = np.asarray(wi, F32)
    wo = np.asarray(wo, F32)
    wix, wiy, wiz = wi[..., 0], wi[..., 1], wi[..., 2]
    hx, hy, hz = wi[..., 0] + wo[..., 0], wi[..., 1] + wo[..., 1], wi[..., 2] + wo[..., 2]
    l2 = hx * hx + hy * hy + hz * hz
    ok = l2 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = F32(1) / np.sqrt(l2)
    hx = np.where(ok, hx * inv, F32(0)).astype(F32)
    hy = np.where(ok, hy * inv, F32(0)).astype(F32)
    hz = np.where(ok, hz * inv, F32(1)).astype(F32)
    vdw = hy * wix - hx * wiy                                   # v = (hy, -hx, 0), v . wi
    k = vdw / np.maximum(F32(1) + hz, F32(1e-6))
    wdx = wix * hz + (-hx * wiz) + k * hy
    wdy = wiy * hz + (-hy * wiz) + k * (-hx)
    wdz = wiz * hz + (hx * wix + hy * wiy)
    return np.stack([wix, wiy, wiz, wo[..., 0], wo[..., 1], wo[..., 2], hx, hy, hz, wdx, wdy, wdz],
                    axis=-1).astype(F32)


def encode_bsdf_dirs_f64(wi, wo):
    """Independent float64 version with an explicit Rodrigues rotation (used only by tests)."""
    wi = np.asarray(wi, np.float64)
    wo = np.asarray(wo, np.float64)
    h = wi + wo
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    v = np.stack([h[..., 1], -h[..., 0], np.zeros_like(h[..., 0])], axis=-1)
    cr = np.cross(v, wi)
    wd = wi + cr + np.cross(v, cr) / (1 + h[..., 2])[..., None]
    return np.concatenate([wi, wo, h, wd], axis=-1)


def bilinear_corners(uv, h, w):
    """Corner-to-corner bilinear lookup of nnm_format.md section 5.

    uv (..., 2) in [0,1] (clamped) -> flat texel indices (..., 4) [00, 10, 01, 11] into an (h*w) row-major
    texel array (index = y*w + x), and weights (..., 4) float32.
    """
    uv = np.asarray(uv, F32)
    u = np.clip(uv[..., 0], F32(0), F32(1))
    v = np.clip(uv[..., 1], F32(0), F32(1))
    x = u * F32(w - 1)
    y = v * F32(h - 1)
    ix = np.clip(np.floor(x).astype(np.int64), 0, max(w - 2, 0))
    iy = np.clip(np.floor(y).astype(np.int64), 0, max(h - 2, 0))
    fx = (x - ix.astype(F32)).astype(F32)
    fy = (y - iy.astype(F32)).astype(F32)
    ix1 = np.minimum(ix + 1, w - 1)
    iy1 = np.minimum(iy + 1, h - 1)
    idx = np.stack([iy * w + ix, iy * w + ix1, iy1 * w + ix, iy1 * w + ix1], axis=-1)
    one = F32(1)
    wt = np.stack([(one - fx) * (one - fy), fx * (one - fy), (one - fx) * fy, fx * fy], axis=-1).astype(F32)
    return idx, wt


def latent_lookup(latent_hwc, uv):
    """latent (H, W, C), uv (..., 2) -> (..., C) float32/float64 (dtype of the latent)."""
    h, w, c = latent_hwc.shape
    idx, wt = bilinear_corners(uv, h, w)
    flat = latent_hwc.reshape(h * w, c)
    return (flat[idx] * wt[..., None].astype(latent_hwc.dtype)).sum(axis=-2)
