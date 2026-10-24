# SPDX-License-Identifier: Apache-2.0
"""Loader, writer and validator for SHADER_EVAL_BSDF dumps (numpy only). Format: dump_format.md."""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

FORMAT_VERSION = 1
FLAG_LOCAL_FRAME = 1


class DumpError(ValueError):
    """The file violates dump_format.md."""


@dataclass
class BsdfDump:
    prims: np.ndarray                     # int32 (N,)
    uv: np.ndarray                        # float32 (N,2)  barycentric (u, v) in the triangle
    wi: np.ndarray                        # float32 (N,3)
    wo: np.ndarray                        # float32 (N,3)
    frame_local: bool
    out: np.ndarray                       # float32 (N,8): bsdf rgb (f*cos), pdf, albedo rgb, flags
    tex_uv: Optional[np.ndarray] = None   # float32 (N,2) latent-grid coordinates in [0,1]^2
    meta: Dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.prims)

    def texture_uv(self):
        """Latent-grid coordinates. `tex_uv` if present, else `uv` when the meta says uv are texture coords."""
        if self.tex_uv is not None:
            return self.tex_uv
        if self.meta.get("uv_kind") == "texture":
            return self.uv
        raise DumpError("dump has no 'tex_uv' and meta['uv_kind'] != 'texture': cannot index a latent grid. "
                        "Compute tex_uv from the mesh UV layer (see blender_dump.py).")


def _arr(z, key, dtype, shape_tail):
    if key not in z.files:
        raise DumpError(f"missing array '{key}'")
    a = np.asarray(z[key])
    if dtype is np.int32:
        if a.dtype.kind not in "iu":
            raise DumpError(f"'{key}' must be an integer array, got {a.dtype}")
    elif a.dtype.kind != "f":
        raise DumpError(f"'{key}' must be a float array, got {a.dtype}")
    a = a.astype(dtype, copy=False)
    if a.ndim != 1 + len(shape_tail) or tuple(a.shape[1:]) != tuple(shape_tail):
        raise DumpError(f"'{key}' has shape {a.shape}, expected (N,{','.join(map(str, shape_tail))})".replace(",)", ")"))
    return a


def load_dump(path) -> BsdfDump:
    """Read and validate (shapes, dtypes, version). Does not drop records; see filter_records."""
    z = np.load(path, allow_pickle=False)
    ver = int(z["format_version"]) if "format_version" in z.files else None
    if ver != FORMAT_VERSION:
        raise DumpError(f"format_version {ver}, this loader reads {FORMAT_VERSION}")
    prims = _arr(z, "prims", np.int32, ())
    uv = _arr(z, "uv", np.float32, (2,))
    wi = _arr(z, "wi", np.float32, (3,))
    wo = _arr(z, "wo", np.float32, (3,))
    out = _arr(z, "out", np.float32, (8,))
    tex_uv = _arr(z, "tex_uv", np.float32, (2,)) if "tex_uv" in z.files else None
    if "frame_local" not in z.files:
        raise DumpError("missing 'frame_local'")
    frame_local = bool(np.asarray(z["frame_local"]).reshape(()))
    n = len(prims)
    if n == 0:
        raise DumpError("empty dump")
    for name, a in (("uv", uv), ("wi", wi), ("wo", wo), ("out", out), ("tex_uv", tex_uv)):
        if a is not None and len(a) != n:
            raise DumpError(f"'{name}' has {len(a)} records, 'prims' has {n}")
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return BsdfDump(prims, uv, wi, wo, frame_local, out, tex_uv, meta)


def save_dump(path, *, prims, uv, wi, wo, out, frame_local=True, tex_uv=None, meta=None):
    meta = dict(meta or {})
    meta.setdefault("format_version", FORMAT_VERSION)
    arrays = dict(prims=np.asarray(prims, np.int32), uv=np.asarray(uv, np.float32),
                  wi=np.asarray(wi, np.float32), wo=np.asarray(wo, np.float32),
                  out=np.asarray(out, np.float32))
    if tex_uv is not None:
        arrays["tex_uv"] = np.asarray(tex_uv, np.float32)
    np.savez_compressed(path, format_version=np.int32(FORMAT_VERSION), frame_local=np.bool_(frame_local),
                        meta_json=np.array(json.dumps(meta, sort_keys=True)), **arrays)


@dataclass
class RecordSet:
    """Valid training records, flat float32 arrays."""
    tex_uv: np.ndarray        # (M,2)
    wi: np.ndarray            # (M,3) local frame, z > 0
    wo: np.ndarray            # (M,3)
    target: np.ndarray        # (M,3) f * cos(wo), >= 0
    pdf: np.ndarray           # (M,)
    albedo: np.ndarray        # (M,3)

    @property
    def n(self):
        return len(self.tex_uv)

    def take(self, idx):
        return RecordSet(*(a[idx] for a in (self.tex_uv, self.wi, self.wo, self.target, self.pdf, self.albedo)))

    @staticmethod
    def concat(sets: List["RecordSet"]):
        return RecordSet(*(np.concatenate([getattr(s, f) for s in sets]) for f in
                           ("tex_uv", "wi", "wo", "target", "pdf", "albedo")))


def filter_records(d: BsdfDump, min_cos=1e-3, unit_tol=1e-2, require_local=True):
    """Drop records that cannot be training targets. Returns (RecordSet, report dict of counts).

    Dropped: out[:,7] != 0 (Cycles flags: 1 no closure, 2 backfacing, 4 invalid input); any non-finite input
    or output; |wi|, |wo| not unit within `unit_tol`; local-frame z < `min_cos` for wi or wo (the decoder's
    domain is the upper hemisphere); tex_uv outside [0,1] by more than 1e-4. Negative BSDF values are clamped
    to 0 and counted but not dropped. Reasons overlap, so the counts need not add up to `dropped`.
    """
    if require_local and not d.frame_local:
        raise DumpError("frame_local is False: the decoder is trained on local-frame directions. "
                        "Re-dump with flags bit 0 set.")
    tex = d.texture_uv()
    finite = (np.isfinite(d.out).all(1) & np.isfinite(d.wi).all(1) & np.isfinite(d.wo).all(1)
              & np.isfinite(tex).all(1))
    flagged = d.out[:, 7] != 0
    with np.errstate(invalid="ignore"):
        unit = (np.abs(np.linalg.norm(d.wi, axis=1) - 1) <= unit_tol) & (np.abs(np.linalg.norm(d.wo, axis=1) - 1) <= unit_tol)
        above = (d.wi[:, 2] >= min_cos) & (d.wo[:, 2] >= min_cos)
        in_uv = ((tex >= -1e-4) & (tex <= 1 + 1e-4)).all(1)
    keep = finite & ~flagged & unit & above & in_uv
    report = {"records": int(d.n), "kept": int(keep.sum()), "dropped": int(d.n - keep.sum()),
              "flagged": int(flagged.sum()), "nonfinite": int((~finite).sum()), "not_unit": int((~unit & finite).sum()),
              "below_horizon": int((~above & finite).sum()), "uv_out_of_range": int((~in_uv & finite).sum()),
              "negative_clamped": int((d.out[keep, 0:3] < 0).any(1).sum())}
    if keep.sum() == 0:
        raise DumpError(f"no valid records left: {report}")
    o = d.out[keep]
    rs = RecordSet(tex_uv=np.clip(tex[keep], 0, 1).astype(np.float32),
                   wi=(d.wi[keep] / np.linalg.norm(d.wi[keep], axis=1, keepdims=True)).astype(np.float32),
                   wo=(d.wo[keep] / np.linalg.norm(d.wo[keep], axis=1, keepdims=True)).astype(np.float32),
                   target=np.maximum(o[:, 0:3], 0).astype(np.float32), pdf=o[:, 3].copy(), albedo=o[:, 4:7].copy())
    return rs, report


def load_records(paths, **kw):
    """Load, validate and filter one or more dump files; concatenate. Returns (RecordSet, [report per file], metas)."""
    if isinstance(paths, str):
        paths = [paths]
    sets, reports, metas = [], [], []
    for p in paths:
        d = load_dump(p)
        rs, rep = filter_records(d, **kw)
        rep["path"] = str(p)
        sets.append(rs)
        reports.append(rep)
        metas.append(d.meta)
    return RecordSet.concat(sets), reports, metas


def require_uv_tangent_frame(metas):
    """Do not train a new-frame model from legacy edge-frame records."""
    for i, meta in enumerate(metas):
        if meta.get("frame") != "uv_tangent":
            raise DumpError(f"dump {i} has no UV tangent frame marker; regenerate it before training")


def split_indices(n, val_frac=0.05, val_max=262144, seed=0):
    """Deterministic train/validation split by record. -> (train_idx, val_idx) int64 arrays."""
    n_val = int(min(val_max, max(1, round(n * val_frac))))
    if n - n_val < 1:
        raise ValueError("too few records for a validation split")
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])
