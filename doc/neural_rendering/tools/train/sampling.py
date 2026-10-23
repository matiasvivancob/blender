# SPDX-License-Identifier: Apache-2.0
"""Direction and texel sampling for BSDF dumps (numpy). Used by make_synthetic_dump.py and, unchanged, by
blender_dump.py (the Blender-side driver), so the synthetic and the real dumps share one distribution.

Conventions (match the Cycles eval_bsdf wrapper): `wi` points from the surface towards the viewer, `wo`
towards the light; the cosine factor in the target is cos(wo). Both are in the local shading frame, z up.

Distribution per record:
  * texel: uniform over the texture (uv in [0,1]^2); on a mesh use uniform_uv_on_mesh().
  * wo: cosine-weighted hemisphere (the distribution a path tracer would draw from for a diffuse lobe).
        With probability `grazing_frac`: uniform in cos(theta) restricted to theta > 75 deg.
  * wi: mixture. With prob `peak_frac`: GGX peak targeted, wi = reflect(wo about h), h ~ GGX(alpha),
        alpha log-uniform in [0.01, 0.5]; otherwise uniform in solid angle. With prob `grazing_frac`: grazing as above.
        (Reflecting the sampled wo about a GGX half vector puts a large share of the records inside the
        specular lobe, which is where the target is large and where fits fail first.)
"""
import numpy as np

MIN_COS = 0.005


def cosine_hemisphere(rng, n):
    u1, u2 = rng.random(n), rng.random(n)
    r, ph = np.sqrt(u1), 2 * np.pi * u2
    z = np.sqrt(np.maximum(1.0 - u1, 0.0))
    return np.stack([r * np.cos(ph), r * np.sin(ph), np.maximum(z, MIN_COS)], 1)


def _dirs_from_cos(ct, ph):
    st = np.sqrt(np.maximum(1.0 - ct * ct, 0.0))
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1)


def uniform_hemisphere(rng, n, cos_lo=MIN_COS, cos_hi=1.0):
    """Uniform in solid angle (uniform in cos theta) between cos_lo and cos_hi."""
    return _dirs_from_cos(rng.uniform(cos_lo, cos_hi, n), rng.random(n) * 2 * np.pi)


def ggx_half_vectors(rng, n, alpha_lo=0.01, alpha_hi=0.5):
    a = np.exp(rng.uniform(np.log(alpha_lo), np.log(alpha_hi), n))
    u = rng.random(n)
    tan2 = a * a * u / np.maximum(1 - u, 1e-12)
    return _dirs_from_cos(1 / np.sqrt(1 + tan2), rng.random(n) * 2 * np.pi)


def sample_directions(rng, n, peak_frac=0.5, grazing_frac=0.1, grazing_deg=75.0):
    """-> wi (n,3), wo (n,3) float64 unit vectors in the local frame with z >= MIN_COS."""
    cg = np.cos(np.radians(grazing_deg))
    wo = cosine_hemisphere(rng, n)
    g = rng.random(n) < grazing_frac
    wo[g] = uniform_hemisphere(rng, int(g.sum()), MIN_COS, cg)

    wi = uniform_hemisphere(rng, n)
    pk = rng.random(n) < peak_frac
    h = ggx_half_vectors(rng, n)
    refl = 2 * np.sum(wo * h, 1, keepdims=True) * h - wo
    ok = pk & (refl[:, 2] > MIN_COS)
    wi[ok] = refl[ok]
    g = rng.random(n) < grazing_frac
    wi[g] = uniform_hemisphere(rng, int(g.sum()), MIN_COS, cg)
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    wo /= np.linalg.norm(wo, axis=1, keepdims=True)
    return wi, wo


def uniform_uv_on_mesh(rng, n, tri_uvs):
    """Uniform samples over the used UV area of a mesh.

    tri_uvs (T, 3, 2): UV coordinates of the three corners of every triangle.
    Returns prims int32 (n,), bary float32 (n,2) = (u, v) with P = (1-u-v) v0 + u v1 + v v2 (Cycles
    convention), tex_uv float32 (n,2). Triangles are chosen proportionally to their UV area, so texels of the
    unwrapped texture are covered uniformly. Triangles with zero UV area are never chosen.
    """
    tri_uvs = np.asarray(tri_uvs, np.float64)
    e1, e2 = tri_uvs[:, 1] - tri_uvs[:, 0], tri_uvs[:, 2] - tri_uvs[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if not area.sum() > 0:
        raise ValueError("mesh has no UV area")
    prims = rng.choice(len(area), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    u, v = s * (1 - r2), s * r2                      # uniform in the triangle, weights of v1 and v2
    t = tri_uvs[prims]
    tex = (1 - u - v)[:, None] * t[:, 0] + u[:, None] * t[:, 1] + v[:, None] * t[:, 2]
    return prims.astype(np.int32), np.stack([u, v], 1).astype(np.float32), tex.astype(np.float32)


def locate_uv(tri_uvs, tex_uv, max_tris=256):
    """Brute-force inverse UV map for --fill: (prim, (u, v) barycentric in Cycles convention) per query."""
    if len(tri_uvs) > max_tris:
        raise RuntimeError(f"--fill supports at most {max_tris} triangles (use a quad); the mesh has {len(tri_uvs)}")
    prims = np.full(len(tex_uv), -1, np.int64)
    bary = np.zeros((len(tex_uv), 2), np.float32)
    for t, (a, b, c) in enumerate(tri_uvs):
        m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]], np.float64)
        if abs(np.linalg.det(m)) < 1e-12:
            continue
        uv = np.linalg.solve(m, (tex_uv - a).T.astype(np.float64)).T                 # (u, v) weights of b and c
        inside = (uv[:, 0] >= -1e-6) & (uv[:, 1] >= -1e-6) & (uv.sum(1) <= 1 + 1e-6) & (prims < 0)
        prims[inside] = t
        bary[inside] = uv[inside]
    if (prims < 0).any():
        raise RuntimeError(f"{int((prims < 0).sum())} query UVs are not covered by the mesh UV layout")
    return prims.astype(np.int32), bary
