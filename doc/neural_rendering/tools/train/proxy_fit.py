# SPDX-License-Identifier: Apache-2.0
"""Per-material constants for the Cycles neural closure (nnm_format.md section 3a): sampling proxy and albedo.

The closure samples a mixture of a cosine lobe (probability p_diffuse) and a GGX VNDF reflection lobe and
reports the pdf of that procedure, so the proxy changes noise, never the expected value. This module fits
(alpha, p_diffuse) offline from a training dump by minimising the second moment of the one-sample estimator
under the dump's cosine `wo` density (STATUS.md, decision (b)):

    E[(f_c / pdf_proxy)^2 * pdf_cos]  =  mean over records of  f_c^2 / (pdf_proxy * pdf_cos)

with f_c the RGB average of the dump's f*cos columns and pdf_cos = cos(theta_o) / pi. The albedo is the mean of
the dump's albedo columns, clamped to [0, 1].

`pdf_proxy` here is the same function as the kernel's (`bsdf_microfacet_eval` for an isotropic GGX reflection
with Fresnel off, `D / cos_NI * 0.25 / (1 + lambda_I)`, plus the cosine lobe). The slice-2 unit test checks the
two against each other.
"""
import numpy as np

ALPHAS = np.logspace(np.log10(0.01), 0.0, 15)
P_DIFFUSES = np.round(np.arange(1, 11) / 10.0, 1)
FALLBACK = (0.2, 0.5)  # (alpha, p_diffuse), acceptable per STATUS.md
EPS = 1e-12


def ggx_reflect_pdf(wi, wo, alpha):
    """pdf (w.r.t. solid angle of wo) of sampling `wo` by reflecting a GGX visible normal seen from `wi`.

    wi, wo: (N, 3) unit vectors in the local shading frame (z = normal), alpha scalar or (N,). Zero when either
    direction is below the horizon. Isotropic, Fresnel off: D / cos_NI * 0.25 / (1 + lambda_I).
    """
    wi, wo = np.asarray(wi, np.float64), np.asarray(wo, np.float64)
    a2 = np.asarray(alpha, np.float64) ** 2
    h = wi + wo
    h = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), EPS)
    cos_nh, cos_ni, cos_no = h[..., 2], wi[..., 2], wo[..., 2]
    d = a2 / (np.pi * (cos_nh * cos_nh * (a2 - 1.0) + 1.0) ** 2)
    tan2 = np.maximum(1.0 - cos_ni * cos_ni, 0.0) / np.maximum(cos_ni * cos_ni, EPS)
    lam = 0.5 * (-1.0 + np.sqrt(1.0 + a2 * tan2))
    pdf = d / np.maximum(cos_ni, EPS) * 0.25 / (1.0 + lam)
    return np.where((cos_ni > 0.0) & (cos_no > 0.0), pdf, 0.0)


def proxy_pdf(wi, wo, alpha, p_diffuse):
    """Mixture pdf: p_diffuse * cos(theta_o) / pi + (1 - p_diffuse) * GGX reflection pdf."""
    cos_no = np.maximum(np.asarray(wo, np.float64)[..., 2], 0.0)
    return p_diffuse * cos_no / np.pi + (1.0 - p_diffuse) * ggx_reflect_pdf(wi, wo, alpha)


def second_moment(wi, wo, f_c, alpha, p_diffuse):
    """mean(f_c^2 / (pdf_proxy * pdf_cos)) over records."""
    pdf_cos = np.maximum(np.asarray(wo, np.float64)[:, 2], EPS) / np.pi
    return float(np.mean(f_c * f_c / (np.maximum(proxy_pdf(wi, wo, alpha, p_diffuse), EPS) * pdf_cos)))


def fit_proxy(wi, wo, target, albedo, max_records=200_000, seed=0):
    """-> dict(alpha, p_diffuse, albedo (3,), moment, table). Falls back to FALLBACK if the fit is not finite.

    wi, wo (N,3) local-frame unit vectors; target (N,3) = f*cos(theta_o) from the dump; albedo (N,3).
    """
    wi, wo = np.asarray(wi, np.float64), np.asarray(wo, np.float64)
    f_c = np.asarray(target, np.float64).mean(axis=1)
    if len(wi) > max_records:
        sel = np.random.default_rng(seed).choice(len(wi), max_records, replace=False)
        wi, wo, f_c = wi[sel], wo[sel], f_c[sel]
    table = {(float(a), float(p)): second_moment(wi, wo, f_c, a, p) for a in ALPHAS for p in P_DIFFUSES}
    (alpha, p_d), moment = min(table.items(), key=lambda kv: kv[1])
    if not np.isfinite(moment):
        alpha, p_d = FALLBACK
    alb = np.clip(np.asarray(albedo, np.float64).mean(axis=0), 0.0, 1.0)
    return {"alpha": float(alpha), "p_diffuse": float(p_d), "albedo": [float(v) for v in alb],
            "moment": float(moment), "table": table}


def proxy_tuple(fit):
    """The six floats of the .nnm proxy block."""
    return (fit["alpha"], fit["alpha"], fit["p_diffuse"], *fit["albedo"])
