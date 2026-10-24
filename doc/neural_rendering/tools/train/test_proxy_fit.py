# SPDX-License-Identifier: Apache-2.0
"""numpy-only tests of proxy_fit.py:  python3 test_proxy_fit.py"""
import numpy as np

import _paths  # noqa: F401
import proxy_fit as pf


def sphere(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def cosine_hemisphere(rng, n):
    u1, u2 = rng.random(n), rng.random(n)
    r, ph = np.sqrt(u1), 2 * np.pi * u2
    return np.stack([r * np.cos(ph), r * np.sin(ph), np.sqrt(np.maximum(1 - u1, 0))], 1)


def test_pdfs_integrate_to_the_visible_fraction():
    """Reflections whose wo falls below the horizon are lost, so the integral is <= 1. At normal incidence the lost
    fraction is exactly alpha^2 / (1 + alpha^2) (tan^2 of the half vector is distributed as t / (alpha^2 + t))."""
    rng = np.random.default_rng(0)
    wo = sphere(rng, 2_000_000)
    for cos_i, alpha, expect, tol in [(1.0, 0.2, 1 / (1 + 0.2 ** 2), 0.01), (1.0, 0.05, 1 / (1 + 0.05 ** 2), 0.01),
                                      (0.6, 0.3, None, None), (0.6, 0.05, None, None)]:
        wi = np.tile([np.sqrt(1 - cos_i ** 2), 0.0, cos_i], (len(wo), 1))
        integral = pf.ggx_reflect_pdf(wi, wo, alpha).mean() * 4 * np.pi
        if expect is not None:
            assert abs(integral - expect) < tol, (cos_i, alpha, integral, expect)
        else:
            assert 0.85 < integral <= 1.02, (cos_i, alpha, integral)
        mix = pf.proxy_pdf(wi, wo, alpha, 0.4).mean() * 4 * np.pi
        assert 0.4 + 0.6 * 0.85 < mix <= 1.02, (cos_i, alpha, mix)


def test_pdf_zero_below_horizon():
    wi = np.array([[0.0, 0.0, 1.0]] * 2)
    wo = np.array([[0.0, 0.6, -0.8], [0.0, 0.6, 0.8]])
    p = pf.ggx_reflect_pdf(wi, wo, 0.3)
    assert p[0] == 0.0 and p[1] > 0.0


def test_fit_diffuse_prefers_cosine_lobe():
    rng = np.random.default_rng(1)
    wo, wi = cosine_hemisphere(rng, 20000), cosine_hemisphere(rng, 20000)
    f = np.repeat((wo[:, 2] * 0.8 / np.pi)[:, None], 3, 1)
    fit = pf.fit_proxy(wi, wo, f, np.full((20000, 3), 0.8))
    assert fit["p_diffuse"] == 1.0, fit["p_diffuse"]
    assert np.allclose(fit["albedo"], 0.8)


def test_fit_glossy_prefers_narrow_lobe_and_small_p_diffuse():
    rng = np.random.default_rng(2)
    n, alpha_true = 30000, 0.1
    wi, wo = cosine_hemisphere(rng, n), cosine_hemisphere(rng, n)
    f = pf.ggx_reflect_pdf(wi, wo, alpha_true)[:, None].repeat(3, 1) * 0.9   # glossy-only material, f*cos ~ lobe
    fit = pf.fit_proxy(wi, wo, f, np.full((n, 3), 0.9))
    assert fit["p_diffuse"] <= 0.3 and 0.05 <= fit["alpha"] <= 0.2, (fit["alpha"], fit["p_diffuse"])
    # the fitted proxy must beat both extremes of the grid
    assert fit["moment"] < pf.second_moment(wi, wo, f.mean(1), 1.0, 1.0)


def test_fit_outputs_are_valid_for_the_format():
    rng = np.random.default_rng(3)
    wi, wo = cosine_hemisphere(rng, 5000), cosine_hemisphere(rng, 5000)
    fit = pf.fit_proxy(wi, wo, np.abs(rng.normal(size=(5000, 3))), rng.random((5000, 3)) * 2 - 0.5)
    ax, ay, pd, *alb = pf.proxy_tuple(fit)
    assert 1e-3 <= ax == ay <= 1.0 and 0.1 <= pd <= 1.0 and all(0.0 <= a <= 1.0 for a in alb)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
