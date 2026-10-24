"""Analytic two-layer spatially varying BRDF (ground truth) + procedural 64x64 texture.

Layering (all in the local shading frame, normal = +z):
  * top layer : dielectric GGX coat (F0 = 0.04, roughness r_c, alpha = r_c^2)
  * bottom    : GGX specular (F0 = mix(0.04, base_color, metallic), roughness r_b)
                + Lambert diffuse ((1 - metallic) * base_color / pi)
  * energy    : base is attenuated by (1 - F_c(cos_i)) * (1 - F_c(cos_o))
                (Schlick coat Fresnel on the way in and out), diffuse additionally by
                (1 - F_dielectric(wo.h)). Not physically exact, only approximately
                energy conserving; the point is spatial variation + sharp peaks.
Smith separable G1, Schlick Fresnel.  brdf() returns the BRDF (RGB); the training target
is f * cos(theta_i) (Zeltner et al. 2024).
"""
import numpy as np

TEX_RES = 64
PARAM_NAMES = ["base_r", "base_g", "base_b", "base_rough", "metallic", "coat_rough"]


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _value_noise(rng, res, cells, octaves=3):
    """Smooth value noise in [0,1], res x res (non-tiling, corner-to-corner)."""
    out = np.zeros((res, res))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = cells * (2 ** o)
        lat = rng.random((c + 1, c + 1))
        g = np.arange(res) / (res - 1) * c
        i0 = np.minimum(np.floor(g).astype(int), c - 1)
        t = g - i0
        t = t * t * (3 - 2 * t)
        rows = lat[i0] * (1 - t)[:, None] + lat[i0 + 1] * t[:, None]            # (res, c+1)
        field = rows[:, i0] * (1 - t)[None, :] + rows[:, i0 + 1] * t[None, :]   # (res, res)
        out += amp * field
        tot += amp
        amp *= 0.5
    out /= tot
    return (out - out.min()) / (out.max() - out.min())


def make_texture(seed=7, res=TEX_RES):
    """Returns float32 array (res*res, 6) of per-texel params, row-major (iy*res + ix)."""
    rng = np.random.default_rng(seed)
    n1, n2, n3, n4 = (_value_noise(rng, res, c) for c in (3, 2, 3, 4))
    iy, ix = np.mgrid[0:res, 0:res]
    checker = (((ix // 8) + (iy // 8)) % 2).astype(np.float64)       # 8x8 checker of 8-texel squares
    colA = np.array([0.80, 0.22, 0.12])   # red-ish
    colB = np.array([0.12, 0.35, 0.75])   # blue-ish
    gold = np.array([1.00, 0.77, 0.34])
    col = colA[None, None] * (1 - n1[..., None]) + colB[None, None] * n1[..., None]
    col = col * (1 - 0.85 * checker[..., None]) + gold[None, None] * (0.85 * checker[..., None])
    metallic = np.clip(0.95 * checker + 0.9 * _smoothstep(0.62, 0.72, n2) * (1 - checker), 0, 1)
    base_rough = 0.12 + 0.70 * n3
    coat_rough = 0.10 + 0.50 * n4 - 0.05 * checker
    tex = np.concatenate([col, base_rough[..., None], metallic[..., None], coat_rough[..., None]], -1)
    return tex.reshape(res * res, 6).astype(np.float32)


def bilinear_corners(uv, res=TEX_RES):
    """uv in [0,1]^2 -> corner indices (n,4) and weights (n,4); texels on a corner-to-corner grid."""
    x = uv[:, 0] * (res - 1)
    y = uv[:, 1] * (res - 1)
    ix = np.minimum(np.floor(x).astype(np.int64), res - 2)
    iy = np.minimum(np.floor(y).astype(np.int64), res - 2)
    fx = (x - ix).astype(np.float32)
    fy = (y - iy).astype(np.float32)
    b = iy * res + ix
    idx = np.stack([b, b + 1, b + res, b + res + 1], 1)
    w = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
    return idx, w


def lookup_params(tex, uv, res=TEX_RES):
    idx, w = bilinear_corners(uv, res)
    return (tex[idx] * w[..., None]).sum(1)          # (n, 6)


def _ggx_D(nh, a2):
    d = nh * nh * (a2 - 1.0) + 1.0
    return a2 / (np.pi * d * d)


def _smith_G1(c, a2):
    c2 = c * c
    tan2 = (1.0 - c2) / np.maximum(c2, 1e-8)
    return 2.0 / (1.0 + np.sqrt(1.0 + a2 * tan2))


def _schlick(f0, c):
    return f0 + (1.0 - f0) * (1.0 - np.clip(c, 0, 1)) ** 5


def brdf(params, wi, wo):
    """params (n,6), wi/wo (n,3) unit vectors in local frame (z up, both z>0). Returns f (n,3)."""
    params = np.asarray(params, np.float64)
    wi = np.asarray(wi, np.float64)
    wo = np.asarray(wo, np.float64)
    col = params[:, 0:3]
    rb = params[:, 3]
    met = params[:, 4:5]
    rc = params[:, 5]
    ci, co = wi[:, 2], wo[:, 2]
    h = wi + wo
    h = h / np.linalg.norm(h, axis=1, keepdims=True)
    nh = h[:, 2]
    oh = np.sum(wo * h, 1)
    denom = 4.0 * np.maximum(ci, 1e-4) * np.maximum(co, 1e-4)

    # coat: dielectric GGX
    ac2 = rc ** 4
    Fc = _schlick(0.04, oh)
    f_coat = Fc * _ggx_D(nh, ac2) * _smith_G1(ci, ac2) * _smith_G1(co, ac2) / denom       # (n,)

    # base
    ab2 = rb ** 4
    F0 = 0.04 * (1 - met) + col * met                                                       # (n,3)
    Fb = _schlick(F0, oh[:, None])
    f_bspec = Fb * (_ggx_D(nh, ab2) * _smith_G1(ci, ab2) * _smith_G1(co, ab2) / denom)[:, None]
    f_diff = (1 - met) * col / np.pi * (1.0 - _schlick(0.04, oh))[:, None]
    atten = ((1 - _schlick(0.04, ci)) * (1 - _schlick(0.04, co)))[:, None]
    return f_coat[:, None] + atten * (f_bspec + f_diff)


def target(params, wi, wo):
    """f * cos(theta_i): the quantity the network is fitted to."""
    return brdf(params, wi, wo) * np.asarray(wi)[:, 2:3]
