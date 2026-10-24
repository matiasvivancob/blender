#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Headless render-and-compare tool for the neural-rendering work.

Two stages, one file.

Stage 1, inside Blender (needs `bpy`): configure Cycles, render one frame to a
32-bit float OpenEXR, read the pixels back through `bpy.data.images` into numpy
and store them as `<out>.npy` (+ `<out>.json` metadata). Optionally compare
against a reference or run the white-furnace check right away.

    blender -b scene.blend --python render_test.py -- \\
        --out out.exr --ref ref.exr --samples 64 --device OPTIX --report report.md
    blender -b furnace.blend --python render_test.py -- \\
        --out furnace.exr --samples 64 --device OPTIX --furnace
    blender -b --factory-startup --python render_test.py -- --list-devices

Stage 2, outside Blender (stdlib + numpy, no `bpy`, no OpenEXR module):

    python3 render_test.py compare out.npy ref.npy [--report r.md]
    python3 render_test.py furnace out.npy --expected 1.0 --tol 0.01
    python3 render_test.py selftest

Image formats understood by stage 2: `.npy` (H x W x C float, top row first),
`.pfm` (32-bit float, RGB or grey), `.f32`/`.raw` (little-endian float32 with a
`<file>.json` sidecar {"shape": [H, W, C]} or `--raw-shape HxWxC`). For `.exr`
the sibling `.npy` is used when present; otherwise optional readers
(OpenImageIO, imageio) are tried. Blender's own Python has no OpenEXR module,
which is why stage 1 always writes the `.npy` next to the EXR.

Exit codes: 0 pass, 1 check failed, 2 usage / environment error.

What "pass" means for `compare`:
  * no non-finite pixels in the test image,
  * shapes match,
  * every explicitly given threshold (--max-mse, --max-rel-mse, --max-abs) holds,
  * no statistically significant AND practically significant bias: the paired
    tile test below must not give |z| > --z-max together with |relative bias|
    > --max-bias. Both noisy images contain independent noise, so a plain
    threshold on the mean difference would either hide real bias or fail on
    noise; the z-score normalises by the observed noise of the difference.

Metrics (RGB only; alpha is checked separately):
  mse       mean((t - r)^2)
  rel_mse   mean((t - r)^2 / (r^2 + eps)), eps = 0.01 (Rousselle et al. 2011)
  max_abs   max |t - r|
  bias      mean(t - r) over all pixels/channels, relative to mean(r)
  z         bias / standard error, standard error from the spread of the
            per-tile mean differences (default 16 x 16 pixel tiles), which is
            robust against fireflies compared to a per-pixel estimate.
"""

import argparse
import json
import math
import os
import sys
import time

try:
    import numpy as np
except ImportError:  # Blender's Python always has numpy; the fallback is for odd installs.
    np = None

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


# -----------------------------------------------------------------------------
# Image I/O (numpy side).

def _require_numpy():
    if np is None:
        raise RuntimeError("numpy is required for this operation")


def _hwc(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError("expected a H x W x C array, got shape %s" % (a.shape,))
    return a


def write_pfm(path, arr):
    """Write a top-row-first H x W x C array as little-endian PFM (alpha dropped)."""
    _require_numpy()
    arr = _hwc(arr)
    c = arr.shape[2]
    if c == 1:
        magic, data = b"Pf", arr
    elif c >= 3:
        magic, data = b"PF", arr[:, :, :3]
    else:
        raise ValueError("PFM supports 1 or 3+ channels, got %d" % c)
    h, w = data.shape[:2]
    with open(path, "wb") as f:
        f.write(magic + b"\n%d %d\n-1.0\n" % (w, h))
        f.write(np.ascontiguousarray(data[::-1]).astype("<f4").tobytes())


def write_pfm_pure(path, flat_bottom_up, width, height, channels):
    """
    Numpy-free PFM writer for the Blender stage. `flat_bottom_up` is a flat
    sequence in Blender's pixel order (bottom row first, `channels` floats per
    pixel), which is also PFM's row order. Alpha is dropped.
    """
    from array import array
    if channels == 1:
        magic, take = b"Pf", 1
    elif channels >= 3:
        magic, take = b"PF", 3
    else:
        raise ValueError("unsupported channel count %d" % channels)
    out = array("f")
    for i in range(width * height):
        base = i * channels
        out.extend(flat_bottom_up[base:base + take])
    if sys.byteorder == "big":
        out.byteswap()
    with open(path, "wb") as f:
        f.write(magic + b"\n%d %d\n-1.0\n" % (width, height))
        f.write(out.tobytes())


def read_pfm(path):
    _require_numpy()
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic not in (b"PF", b"Pf"):
            raise ValueError("%s: not a PFM file" % path)
        dims = f.readline().split()
        while len(dims) < 2:  # tolerate dimensions split across lines
            dims += f.readline().split()
        w, h = int(dims[0]), int(dims[1])
        scale = float(f.readline().strip())
        c = 3 if magic == b"PF" else 1
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(f.read(w * h * c * 4), dtype=dtype)
    if data.size != w * h * c:
        raise ValueError("%s: truncated PFM" % path)
    return np.ascontiguousarray(data.reshape(h, w, c)[::-1]).astype(np.float32)


def parse_shape(text):
    parts = [int(p) for p in text.lower().replace(",", "x").split("x")]
    if len(parts) != 3:
        raise ValueError("shape must be HxWxC, got %r" % text)
    return tuple(parts)


def _load_exr_fallback(path):
    try:  # optional, best effort, not exercised by the self-test
        import OpenImageIO as oiio
        buf = oiio.ImageBuf(path)
        if not buf.has_error:
            return np.asarray(buf.get_pixels(oiio.FLOAT), dtype=np.float32)
    except Exception:
        pass
    try:
        import imageio.v3 as iio
        return np.asarray(iio.imread(path), dtype=np.float32)
    except Exception:
        pass
    return None


def load_image(path, raw_shape=None):
    """Load an image as float32 H x W x C, top row first."""
    _require_numpy()
    base, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".npy":
        return _hwc(np.load(path))
    if ext == ".pfm":
        return read_pfm(path)
    if ext in (".f32", ".raw"):
        shape = raw_shape
        if shape is None and os.path.exists(path + ".json"):
            with open(path + ".json") as f:
                shape = tuple(json.load(f)["shape"])
        if shape is None:
            raise ValueError("%s: raw file needs --raw-shape HxWxC or a .json sidecar" % path)
        data = np.fromfile(path, dtype="<f4")
        if data.size != shape[0] * shape[1] * shape[2]:
            raise ValueError("%s: size %d does not match shape %s" % (path, data.size, shape))
        return data.reshape(shape).astype(np.float32)
    if ext == ".exr":
        for sibling in (base + ".npy", base + ".pfm"):
            if os.path.exists(sibling):
                return load_image(sibling)
        arr = _load_exr_fallback(path)
        if arr is None:
            raise ValueError(
                "%s: cannot read EXR without Blender. Re-run the Blender stage (it writes %s.npy), "
                "or convert with `oiiotool in.exr -o out.pfm`." % (path, base))
        return _hwc(arr)
    raise ValueError("%s: unsupported extension %r" % (path, ext))


# -----------------------------------------------------------------------------
# Metrics.

def paired_bias_test(test, ref, tile=16):
    """
    Paired bias test of two noisy renders of the same scene.

    Per-pixel difference d = mean over RGB of (test - ref); d is averaged in
    `tile` x `tile` blocks; the mean of the block means is the bias estimate,
    and its standard error comes from the spread of the block means. Returns a
    dict with mean_diff, rel_bias (relative to the reference mean), z, n_tiles.
    """
    _require_numpy()
    d = (test[:, :, :3].astype(np.float64) - ref[:, :, :3].astype(np.float64)).mean(axis=2)
    ref_mean = float(ref[:, :, :3].astype(np.float64).mean())
    h, w = d.shape
    t = max(1, int(tile))
    while t > 1 and (h // t) * (w // t) < 8:  # need a handful of tiles for a spread estimate
        t //= 2
    th, tw = h // t, w // t
    blocks = d[:th * t, :tw * t].reshape(th, t, tw, t).mean(axis=(1, 3)).ravel()
    n = blocks.size
    mean_diff = float(blocks.mean())
    if n > 1:
        sd = float(blocks.std(ddof=1))
        se = sd / math.sqrt(n)
    else:
        sd, se = 0.0, 0.0
    if se > 0.0:
        z = mean_diff / se
    elif mean_diff == 0.0:
        z = 0.0
    else:
        z = math.copysign(math.inf, mean_diff)
    rel = mean_diff / ref_mean if ref_mean != 0.0 else (0.0 if mean_diff == 0.0 else math.inf)
    return {"mean_diff": mean_diff, "rel_bias": rel, "z": z, "n_tiles": int(n), "tile": t, "std_err": se}


def compare_images(test, ref, eps=1e-2, tile=16, max_mse=None, max_rel_mse=None, max_abs=None,
                   max_bias=0.02, z_max=4.0):
    """Compare two images. Returns a dict; `ok` is the verdict, `failures` lists reasons."""
    _require_numpy()
    test, ref = _hwc(test), _hwc(ref)
    res = {"ok": False, "failures": [], "shape_test": test.shape, "shape_ref": ref.shape}
    if test.shape != ref.shape:
        res["failures"].append("shape mismatch: test %s vs ref %s" % (test.shape, ref.shape))
        return res

    t, r = test[:, :, :3], ref[:, :, :3]
    fin_t, fin_r = np.isfinite(t), np.isfinite(r)
    res["nonfinite_test"] = int((~fin_t).sum())
    res["nonfinite_ref"] = int((~fin_r).sum())
    if res["nonfinite_test"]:
        res["failures"].append("%d non-finite values in the test image" % res["nonfinite_test"])
    if res["nonfinite_ref"]:
        res["failures"].append("%d non-finite values in the reference image" % res["nonfinite_ref"])

    # Metrics on pixels that are finite in both images (all three channels).
    ok_px = fin_t.all(axis=2) & fin_r.all(axis=2)
    if not ok_px.any():
        res["failures"].append("no finite pixels to compare")
        return res
    t = np.where(ok_px[:, :, None], t, 0.0).astype(np.float64)
    r = np.where(ok_px[:, :, None], r, 0.0).astype(np.float64)
    n = int(ok_px.sum()) * 3
    diff = t - r
    sq = diff * diff
    res["pixels"] = int(ok_px.sum())
    res["mse"] = float(sq.sum() / n)
    res["rmse"] = math.sqrt(res["mse"])
    res["rel_mse"] = float((sq / (r * r + eps)).sum() / n)
    ad = np.abs(diff).max(axis=2)
    idx = np.unravel_index(int(np.argmax(ad)), ad.shape)
    res["max_abs"] = float(ad[idx])
    res["max_abs_at_yx"] = (int(idx[0]), int(idx[1]))
    cnt = float(ok_px.sum())
    res["mean_test"] = [float((t[:, :, c] * ok_px).sum() / cnt) for c in range(3)]
    res["mean_ref"] = [float((r[:, :, c] * ok_px).sum() / cnt) for c in range(3)]

    # Paired bias on the finite-in-both pixels (non-finite ones are zeroed in both).
    res["bias"] = paired_bias_test(t.astype(np.float32), r.astype(np.float32), tile)

    if test.shape[2] >= 4:
        a_t, a_r = test[:, :, 3], ref[:, :, 3]
        res["alpha_max_abs"] = float(np.nanmax(np.abs(a_t - a_r)))

    b = res["bias"]
    if max_mse is not None and res["mse"] > max_mse:
        res["failures"].append("mse %.4g > %.4g" % (res["mse"], max_mse))
    if max_rel_mse is not None and res["rel_mse"] > max_rel_mse:
        res["failures"].append("rel_mse %.4g > %.4g" % (res["rel_mse"], max_rel_mse))
    if max_abs is not None and res["max_abs"] > max_abs:
        res["failures"].append("max_abs %.4g > %.4g" % (res["max_abs"], max_abs))
    if abs(b["z"]) > z_max and abs(b["rel_bias"]) > max_bias:
        res["failures"].append(
            "significant bias: relative %.3f%% (|z| = %.1f > %.1f and > %.2f%%)"
            % (100.0 * b["rel_bias"], abs(b["z"]), z_max, 100.0 * max_bias))
    res["ok"] = not res["failures"]
    return res


def furnace_check(img, expected=1.0, tol=0.01, use_alpha_mask=False):
    """
    White-furnace check. For a closed environment of constant radiance
    `expected` and surfaces that neither absorb nor emit, every pixel of the
    render has expectation `expected`. Reports per-channel mean/std over RGB
    pixels and passes when each channel mean is within `tol` (relative) of
    `expected`. Also flags (but does not fail on) an under-powered test: if
    three standard errors of the mean exceed half the tolerance, more samples
    or pixels are needed to make the test meaningful.
    """
    _require_numpy()
    img = _hwc(img)
    rgb = img[:, :, :3].astype(np.float64)
    res = {"ok": False, "failures": [], "expected": float(expected), "tol": float(tol),
           "pixels": int(rgb.shape[0] * rgb.shape[1])}
    finite = np.isfinite(rgb)
    res["nonfinite"] = int((~finite).sum())
    if res["nonfinite"]:
        res["failures"].append("%d non-finite values" % res["nonfinite"])
    px = finite.all(axis=2)
    if use_alpha_mask and img.shape[2] >= 4:
        px &= img[:, :, 3] > 0.5
    if not px.any():
        res["failures"].append("no valid pixels")
        return res
    vals = rgb[px]  # N x 3
    n = vals.shape[0]
    res["pixels_used"] = int(n)
    res["mean"] = [float(x) for x in vals.mean(axis=0)]
    res["std"] = [float(x) for x in vals.std(axis=0, ddof=1)] if n > 1 else [0.0] * 3
    res["sem"] = [s / math.sqrt(n) for s in res["std"]]
    res["min"] = [float(x) for x in vals.min(axis=0)]
    res["max"] = [float(x) for x in vals.max(axis=0)]
    res["rel_dev"] = [(m - expected) / expected if expected != 0 else math.inf for m in res["mean"]]
    res["frac_above_2x"] = float((vals > 2.0 * expected).any(axis=1).mean())
    res["underpowered"] = any(3.0 * s > 0.5 * tol * abs(expected) for s in res["sem"])
    for c, name in enumerate("RGB"):
        if abs(res["rel_dev"][c]) > tol:
            res["failures"].append("channel %s mean %.5f deviates %.3f%% from %.5g (tol %.2f%%)"
                                   % (name, res["mean"][c], 100.0 * res["rel_dev"][c], expected, 100.0 * tol))
    res["ok"] = not res["failures"]
    return res


# -----------------------------------------------------------------------------
# Markdown reports.

def _fmt(x, spec="%.5g"):
    return "inf" if isinstance(x, float) and math.isinf(x) else spec % x


def compare_report_md(title, res, meta=None, names=("test", "ref")):
    lines = ["### %s" % title, ""]
    if meta:
        lines += ["| setting | value |", "|---|---|"]
        lines += ["| %s | %s |" % (k, v) for k, v in meta.items()]
        lines.append("")
    lines.append("**%s**" % ("PASS" if res["ok"] else "FAIL"))
    lines.append("")
    if "mse" in res:
        b = res["bias"]
        lines += [
            "| metric | value |", "|---|---|",
            "| pixels compared | %d |" % res["pixels"],
            "| mse | %s |" % _fmt(res["mse"]),
            "| rmse | %s |" % _fmt(res["rmse"]),
            "| relative mse (eps 0.01) | %s |" % _fmt(res["rel_mse"]),
            "| max abs error | %s at (y=%d, x=%d) |" % ((_fmt(res["max_abs"]),) + res["max_abs_at_yx"]),
            "| mean %s RGB | %s |" % (names[0], ", ".join(_fmt(v, "%.5f") for v in res["mean_test"])),
            "| mean %s RGB | %s |" % (names[1], ", ".join(_fmt(v, "%.5f") for v in res["mean_ref"])),
            "| relative bias | %s %% |" % _fmt(100.0 * b["rel_bias"], "%.3f"),
            "| bias z-score (%d tiles of %d px) | %s |" % (b["n_tiles"], b["tile"], _fmt(b["z"], "%.2f")),
            "| non-finite (test / ref) | %d / %d |" % (res["nonfinite_test"], res["nonfinite_ref"]),
        ]
        if "alpha_max_abs" in res:
            lines.append("| alpha max abs error | %s |" % _fmt(res["alpha_max_abs"]))
        lines.append("")
    for f in res["failures"]:
        lines.append("- FAIL: %s" % f)
    return "\n".join(lines) + "\n"


def furnace_report_md(title, res, meta=None):
    lines = ["### %s" % title, ""]
    if meta:
        lines += ["| setting | value |", "|---|---|"]
        lines += ["| %s | %s |" % (k, v) for k, v in meta.items()]
        lines.append("")
    lines.append("**%s** (expected %.5g, tolerance %.2f %% relative)"
                 % ("PASS" if res["ok"] else "FAIL", res["expected"], 100.0 * res["tol"]))
    lines.append("")
    if "mean" in res:
        lines += ["| channel | mean | std | std. error | min | max | deviation |", "|---|---|---|---|---|---|---|"]
        for c, name in enumerate("RGB"):
            lines.append("| %s | %.5f | %.5f | %.2e | %.4f | %.4f | %+.3f %% |" % (
                name, res["mean"][c], res["std"][c], res["sem"][c], res["min"][c], res["max"][c],
                100.0 * res["rel_dev"][c]))
        lines.append("")
        lines.append("pixels used: %d, fraction of pixels above 2x expected (fireflies): %.4f"
                     % (res["pixels_used"], res["frac_above_2x"]))
        if res["underpowered"]:
            lines.append("")
            lines.append("WARNING: under-powered. 3 standard errors exceed half the tolerance; "
                         "a pass here does not rule out a bias of that size. Raise --samples or the resolution.")
    for f in res["failures"]:
        lines.append("- FAIL: %s" % f)
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Stage 2 command line (no Blender).

def _add_threshold_args(p):
    p.add_argument("--eps", type=float, default=1e-2, help="relative MSE denominator epsilon")
    p.add_argument("--tile", type=int, default=16, help="tile size (px) for the bias z-score")
    p.add_argument("--max-mse", type=float, default=None)
    p.add_argument("--max-rel-mse", type=float, default=None)
    p.add_argument("--max-abs", type=float, default=None)
    p.add_argument("--max-bias", type=float, default=0.02, help="relative bias that counts as a failure")
    p.add_argument("--z-max", type=float, default=4.0, help="|z| above which a bias is significant")


def _write_text(path, text):
    if path:
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


def cmd_compare(args):
    shape = parse_shape(args.raw_shape) if args.raw_shape else None
    test = load_image(args.test, shape)
    ref = load_image(args.ref, shape)
    res = compare_images(test, ref, args.eps, args.tile, args.max_mse, args.max_rel_mse, args.max_abs,
                         args.max_bias, args.z_max)
    md = compare_report_md("Compare %s vs %s" % (os.path.basename(args.test), os.path.basename(args.ref)), res)
    print(md)
    _write_text(args.report, md)
    return EXIT_PASS if res["ok"] else EXIT_FAIL


def cmd_furnace(args):
    shape = parse_shape(args.raw_shape) if args.raw_shape else None
    img = load_image(args.image, shape)
    res = furnace_check(img, args.expected, args.tol, args.alpha_mask)
    md = furnace_report_md("Furnace %s" % os.path.basename(args.image), res)
    print(md)
    _write_text(args.report, md)
    return EXIT_PASS if res["ok"] else EXIT_FAIL


def cmd_convert(args):
    img = load_image(args.src, parse_shape(args.raw_shape) if args.raw_shape else None)
    ext = os.path.splitext(args.dst)[1].lower()
    if ext == ".npy":
        np.save(args.dst, img)
    elif ext == ".pfm":
        write_pfm(args.dst, img)
    else:
        raise ValueError("convert writes .npy or .pfm, not %r" % ext)
    print("wrote %s %s" % (args.dst, img.shape))
    return EXIT_PASS


def build_stage2_parser():
    p = argparse.ArgumentParser(prog="render_test.py", description="Compare stage (no Blender needed).")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="compare a test image with a reference")
    c.add_argument("test")
    c.add_argument("ref")
    c.add_argument("--report")
    c.add_argument("--raw-shape")
    _add_threshold_args(c)
    c.set_defaults(func=cmd_compare)

    f = sub.add_parser("furnace", help="white-furnace check on one image")
    f.add_argument("image")
    f.add_argument("--expected", type=float, default=1.0)
    f.add_argument("--tol", type=float, default=0.01, help="relative tolerance of the channel means")
    f.add_argument("--alpha-mask", action="store_true", help="only use pixels with alpha > 0.5")
    f.add_argument("--report")
    f.add_argument("--raw-shape")
    f.set_defaults(func=cmd_furnace)

    v = sub.add_parser("convert", help="convert between .npy and .pfm (and read raw/exr siblings)")
    v.add_argument("src")
    v.add_argument("dst")
    v.add_argument("--raw-shape")
    v.set_defaults(func=cmd_convert)

    s = sub.add_parser("selftest", help="run the built-in checks on synthetic data")
    s.set_defaults(func=lambda a: selftest())
    return p


# -----------------------------------------------------------------------------
# Stage 1: inside Blender.

def build_render_parser():
    p = argparse.ArgumentParser(prog="render_test.py (Blender stage)",
                                description="Render with Cycles, dump .npy, compare. Arguments after `--`.")
    p.add_argument("--out", help="output EXR path (a .npy and .json are written next to it)")
    p.add_argument("--ref", help="reference image (.exr with sibling .npy, .npy or .pfm)")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--device", default="OPTIX", choices=["CPU", "CUDA", "OPTIX"])
    p.add_argument("--gpu-name", help="only use GPUs whose name contains this (case-insensitive), "
                                      "e.g. 3090 or 4070")
    p.add_argument("--res", help="WxH, e.g. 256x256 (default: keep the scene's)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scene", help="scene name (default: the active scene)")
    p.add_argument("--blend", help="open this .blend first (only needed when bpy is used as a module)")
    p.add_argument("--report", help="write a Markdown report here")
    p.add_argument("--update-ref", action="store_true", help="store the render as the reference if --ref is missing")
    p.add_argument("--furnace", action="store_true", help="white-furnace check instead of a reference compare")
    p.add_argument("--furnace-expected", type=float, default=None,
                   help="default: scene custom property nr_furnace_expected, else 1.0")
    p.add_argument("--furnace-tol", type=float, default=0.01)
    p.add_argument("--keep-settings", action="store_true",
                   help="do not neutralise clamping, adaptive sampling, denoising and the light threshold")
    p.add_argument("--denoise", action="store_true", help="leave denoising on (implies nothing else)")
    p.add_argument("--no-russian-roulette", action="store_true",
                   help="set min light/transparent bounces to max bounces (disables path termination)")
    p.add_argument("--list-devices", action="store_true", help="print the Cycles devices and exit")
    _add_threshold_args(p)
    return p


def _blender_argv():
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def list_cycles_devices():
    """Return [(type, name, id)] for every device type Cycles reports."""
    import bpy
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.refresh_devices()
    return [(d.type, d.name, d.id) for d in prefs.devices]


def configure_device(scene, device, gpu_name=None):
    """Select the compute device through the Cycles add-on preferences. Returns descriptions."""
    import bpy
    cycles = scene.cycles
    if device == "CPU":
        cycles.device = "CPU"
        return ["CPU"]
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = device
    except TypeError as e:  # enum item not available in this build / driver
        raise RuntimeError("compute device type %s is not available in this Blender build or on "
                           "this machine (%s). Available entries: %s"
                           % (device, e, sorted(set(t for t, _n, _i in list_cycles_devices())))) from e
    prefs.refresh_devices()
    used = []
    for d in prefs.devices:
        want = d.type == device and (not gpu_name or gpu_name.lower() in d.name.lower())
        d.use = bool(want)
        if want:
            used.append("%s:%s" % (d.type, d.name))
    if not used:
        raise RuntimeError("no %s device%s found. Devices seen: %s"
                           % (device, " matching %r" % gpu_name if gpu_name else "",
                              [(t, n) for t, n, _i in list_cycles_devices()]))
    cycles.device = "GPU"
    return used


def configure_scene(scene, args):
    """Apply the CLI options to the scene. Returns a dict describing what was set."""
    cycles = scene.cycles
    scene.render.engine = "CYCLES"
    info = {}
    info["devices"] = configure_device(scene, args.device, args.gpu_name)
    cycles.samples = args.samples
    cycles.seed = args.seed
    cycles.use_animated_seed = False
    cycles.time_limit = 0.0
    if args.res:
        w, h = [int(x) for x in args.res.lower().split("x")]
        scene.render.resolution_x, scene.render.resolution_y = w, h
        scene.render.resolution_percentage = 100

    if not args.keep_settings:
        # Anything that biases or randomises the sample count would make a
        # comparison of two renders meaningless. All of these exist in
        # intern/cycles/blender/addon/properties.py (CyclesRenderSettings).
        cycles.use_adaptive_sampling = False
        cycles.use_denoising = bool(args.denoise)
        cycles.sample_clamp_direct = 0.0
        cycles.sample_clamp_indirect = 0.0
        cycles.light_sampling_threshold = 0.0  # only used when the light tree is off
        scene.render.film_transparent = False
        scene.render.use_compositing = False
        scene.render.use_sequencer = False
        scene.view_settings.view_transform = "Standard"
    if args.no_russian_roulette:
        cycles.min_light_bounces = min(1024, cycles.max_bounces)
        cycles.min_transparent_bounces = min(1024, cycles.transparent_max_bounces)

    ims = scene.render.image_settings
    ims.file_format = "OPEN_EXR"
    ims.color_mode = "RGBA"
    ims.color_depth = "32"
    try:
        ims.exr_codec = "ZIP"
    except TypeError:
        pass
    scene.render.use_file_extension = True
    info["samples"] = cycles.samples
    info["resolution"] = "%dx%d" % (scene.render.resolution_x, scene.render.resolution_y)
    info["seed"] = cycles.seed
    return info


def read_back_exr(path):
    """Load a float EXR through bpy.data.images. Returns (array H x W x C top-row-first or None, w, h, c, flat)."""
    import bpy
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        c = img.channels
        if np is not None:
            buf = np.empty(w * h * c, dtype=np.float32)
            img.pixels.foreach_get(buf)
            return np.ascontiguousarray(buf.reshape(h, w, c)[::-1]), w, h, c, None
        return None, w, h, c, list(img.pixels)
    finally:
        bpy.data.images.remove(img)


def render_to_arrays(scene, out_exr, args, pre_render=None):
    """
    Render one still to `out_exr` (OpenEXR, 32-bit) and write `<base>.npy` and
    `<base>.json`. `pre_render(scene)`, if given, runs after the standard
    configuration and may change further settings. Returns (array or None, meta).
    """
    import bpy
    base, ext = os.path.splitext(os.path.abspath(out_exr))
    if ext.lower() != ".exr":
        raise ValueError("--out must end in .exr, got %r" % out_exr)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    scene.render.filepath = base  # use_file_extension appends .exr
    info = configure_scene(scene, args)
    if pre_render is not None:
        pre_render(scene)

    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True, scene=scene.name)
    seconds = time.perf_counter() - t0
    if not os.path.exists(base + ".exr"):
        raise RuntimeError("render finished but %s.exr was not written" % base)

    arr, w, h, c, flat = read_back_exr(base + ".exr")
    meta = dict(info)
    meta.update({
        "blender": bpy.app.version_string,
        "scene": scene.name,
        "engine": scene.render.engine,
        "render_seconds": round(seconds, 3),
        "exr": base + ".exr",
        "shape": [h, w, c],
        "light_tree": bool(scene.cycles.use_light_tree),
    })
    try:
        import _cycles
        meta["cycles_with_debug"] = bool(_cycles.with_debug)
    except Exception:
        pass
    if arr is not None:
        np.save(base + ".npy", arr)
    else:  # no numpy: the compare stage can still read a PFM
        write_pfm_pure(base + ".pfm", flat, w, h, c)
        meta["note"] = "numpy missing in this Python; wrote .pfm only"
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return arr, meta


def render_main(argv):
    args = build_render_parser().parse_known_args(argv)[0]  # tolerate --cycles-device etc.
    import bpy

    if args.list_devices:
        for t, n, i in list_cycles_devices():
            print("DEVICE %-6s %s  [%s]" % (t, n, i))
        return EXIT_PASS
    if not args.out:
        print("error: --out is required", file=sys.stderr)
        return EXIT_ERROR
    if args.blend:
        bpy.ops.wm.open_mainfile(filepath=args.blend)
    scene = bpy.data.scenes[args.scene] if args.scene else bpy.context.scene

    arr, meta = render_to_arrays(scene, args.out, args)
    base = os.path.splitext(os.path.abspath(args.out))[0]
    if arr is None:
        print("rendered %s.exr; numpy unavailable, compare outside Blender" % base)
        return EXIT_PASS

    title = "%s on %s, %d spp" % (os.path.basename(base), ", ".join(meta["devices"]), meta["samples"])
    meta_rows = {"blender": meta["blender"], "device": ", ".join(meta["devices"]), "samples": meta["samples"],
                 "resolution": meta["resolution"], "seed": meta["seed"],
                 "render time": "%.2f s" % meta["render_seconds"], "light tree": meta["light_tree"]}
    if "cycles_with_debug" in meta:
        meta_rows["WITH_CYCLES_DEBUG"] = meta["cycles_with_debug"]
    rc, md = EXIT_PASS, ""

    if args.furnace:
        expected = args.furnace_expected
        if expected is None:
            expected = float(scene.get("nr_furnace_expected", 1.0))
        res = furnace_check(arr, expected, args.furnace_tol)
        md = furnace_report_md("Furnace: " + title, res, meta_rows)
        rc = EXIT_PASS if res["ok"] else EXIT_FAIL
    elif args.ref:
        ref_base = os.path.splitext(os.path.abspath(args.ref))[0]
        have_ref = any(os.path.exists(ref_base + e) for e in (".npy", ".pfm")) or os.path.exists(args.ref)
        if not have_ref and args.update_ref:
            np.save(ref_base + ".npy", arr)
            md = "### %s\n\nReference %s.npy did not exist; stored this render as the reference.\n" % (title, ref_base)
        elif not have_ref:
            print("error: reference %s not found (use --update-ref to create it)" % args.ref, file=sys.stderr)
            return EXIT_ERROR
        else:
            ref = load_image(args.ref)
            res = compare_images(arr, ref, args.eps, args.tile, args.max_mse, args.max_rel_mse, args.max_abs,
                                 args.max_bias, args.z_max)
            md = compare_report_md("Compare: " + title, res, meta_rows)
            rc = EXIT_PASS if res["ok"] else EXIT_FAIL
    else:
        md = "### %s\n\nRendered %s.exr (%s); no --ref or --furnace given, nothing compared.\n" % (
            title, base, meta["shape"])
    print(md)
    _write_text(args.report, md)
    return rc


# -----------------------------------------------------------------------------
# Self-test (synthetic data, no Blender).

def _synthetic_scene(h=128, w=160, rng=None):
    """A smooth HDR-ish image with an edge and a bright spot, strictly positive."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.2 + 0.6 * (x / w)[:, :, None] * np.array([1.0, 0.8, 0.6], np.float32)[None, None, :]
    base = base + 0.3 * (y / h)[:, :, None]
    base[:, w // 2:, :] *= 1.5
    base[h // 4:h // 4 + 6, w // 4:w // 4 + 6, :] = 20.0
    return base.astype(np.float32)


def selftest():
    import tempfile
    _require_numpy()
    rng = np.random.default_rng(12345)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    sig = _synthetic_scene()
    h, w, _ = sig.shape

    def noisy(scale, mult=1.0, offset=0.0, tail=False):
        if tail:  # heavy-ish tailed, zero-mean multiplicative noise (exponential minus 1)
            n = rng.exponential(1.0, sig.shape).astype(np.float32) - 1.0
        else:
            n = rng.normal(0.0, 1.0, sig.shape).astype(np.float32)
        return (sig * mult + offset + scale * np.sqrt(sig + 0.05) * n).astype(np.float32)

    # compare ---------------------------------------------------------------
    r = compare_images(sig, sig)
    check("identical images pass", r["ok"] and r["mse"] == 0.0 and r["max_abs"] == 0.0)

    a, b = noisy(0.05), noisy(0.05)
    r = compare_images(a, b)
    check("independent noise on the same signal passes", r["ok"],
          "rel_bias=%.4f z=%.2f" % (r["bias"]["rel_bias"], r["bias"]["z"]))
    check("noise gives non-zero mse", r["mse"] > 0 and r["rel_mse"] > 0)

    a, b = noisy(0.05, tail=True), noisy(0.05, tail=True)
    r = compare_images(a, b)
    check("heavy-tailed noise passes", r["ok"], "z=%.2f" % r["bias"]["z"])

    # No false alarms over many null trials at the default |z| = 4.
    false_alarms = 0
    for _ in range(60):
        rr = compare_images(noisy(0.08), noisy(0.08))
        false_alarms += (not rr["ok"])
    check("no false alarms in 60 null trials", false_alarms == 0, "%d" % false_alarms)

    r = compare_images(noisy(0.05, mult=1.04), noisy(0.05))
    check("4 % brightness bias is detected", (not r["ok"]) and r["bias"]["z"] > 4,
          "rel_bias=%.4f z=%.1f" % (r["bias"]["rel_bias"], r["bias"]["z"]))

    r = compare_images(noisy(0.05, mult=1.004), noisy(0.05))
    check("0.4 % bias is below the practical threshold (default 2 %)", r["ok"],
          "rel_bias=%.4f z=%.1f" % (r["bias"]["rel_bias"], r["bias"]["z"]))
    r = compare_images(noisy(0.05, mult=1.004), noisy(0.05), max_bias=0.002)
    check("0.4 % bias is detected with --max-bias 0.002", not r["ok"])

    r = compare_images(noisy(0.05, mult=0.95), noisy(0.05))
    check("negative bias is detected", (not r["ok"]) and r["bias"]["z"] < -4)

    bad = noisy(0.05)
    bad[10, 10, 1] = np.nan
    bad[11, 11, 0] = np.inf
    r = compare_images(bad, noisy(0.05))
    check("NaN/Inf in test fails", (not r["ok"]) and r["nonfinite_test"] == 2)

    r = compare_images(sig[:-1], sig)
    check("shape mismatch fails", (not r["ok"]) and "shape" in r["failures"][0])

    spike = sig.copy()
    spike[3, 5, :] += 7.0
    r = compare_images(spike, sig, max_abs=1.0)
    check("max_abs threshold trips and locates the pixel", (not r["ok"]) and r["max_abs_at_yx"] == (3, 5)
          and abs(r["max_abs"] - 7.0) < 1e-4, str(r["max_abs_at_yx"]))

    r = compare_images(noisy(0.2), noisy(0.2), max_rel_mse=1e-6)
    check("max_rel_mse threshold trips", not r["ok"])
    r = compare_images(noisy(0.2), noisy(0.2), max_mse=1e-9)
    check("max_mse threshold trips", not r["ok"])

    rgba_t = np.concatenate([sig, np.ones((h, w, 1), np.float32)], axis=2)
    rgba_r = np.concatenate([sig, np.full((h, w, 1), 0.5, np.float32)], axis=2)
    r = compare_images(rgba_t, rgba_r)
    check("RGBA input: RGB identical, alpha difference reported", r["ok"] and abs(r["alpha_max_abs"] - 0.5) < 1e-6)

    md = compare_report_md("t", compare_images(noisy(0.05), noisy(0.05)), {"device": "OPTIX"})
    check("compare report is Markdown with a metric table", "| mse |" in md and "**PASS**" in md and "OPTIX" in md)

    # furnace -----------------------------------------------------------------
    flat = np.ones((96, 96, 3), np.float32)
    r = furnace_check(flat + 0.03 * rng.normal(size=flat.shape).astype(np.float32))
    check("furnace: unbiased noisy constant passes", r["ok"] and not r["underpowered"],
          "mean=%s std=%s" % (["%.4f" % m for m in r["mean"]], ["%.3f" % s for s in r["std"]]))
    check("furnace: per-channel mean/std reported", len(r["mean"]) == 3 and all(s > 0.02 for s in r["std"]))

    r = furnace_check(0.97 * flat + 0.03 * rng.normal(size=flat.shape).astype(np.float32))
    check("furnace: 3 % energy loss fails at 1 % tolerance", not r["ok"], "; ".join(r["failures"]))
    r = furnace_check(0.97 * flat + 0.03 * rng.normal(size=flat.shape).astype(np.float32), tol=0.05)
    check("furnace: same image passes at 5 % tolerance", r["ok"])

    tinted = flat.copy()
    tinted[:, :, 2] *= 0.9
    r = furnace_check(tinted)
    check("furnace: a single off channel fails and names it", (not r["ok"]) and "channel B" in r["failures"][0])

    nan_img = flat.copy()
    nan_img[0, 0, 0] = np.nan
    r = furnace_check(nan_img)
    check("furnace: NaN fails", not r["ok"] and r["nonfinite"] == 1)

    noisy_small = 1.0 + 0.8 * rng.normal(size=(8, 8, 3)).astype(np.float32)
    r = furnace_check(np.abs(noisy_small), tol=0.01)
    check("furnace: tiny noisy image is flagged under-powered", r["underpowered"])

    ex2 = furnace_check(2.0 * flat, expected=2.0)
    check("furnace: expected constant other than 1", ex2["ok"] and ex2["std"][0] == 0.0)

    fireflies = flat.copy()
    fireflies[5, 5, :] = 500.0
    r = furnace_check(fireflies, tol=0.5)
    check("furnace: firefly fraction reported", r["frac_above_2x"] > 0)

    md = furnace_report_md("f", furnace_check(flat), {"samples": 64})
    check("furnace report has per-channel table", "| R |" in md and "| G |" in md and "| B |" in md)

    # I/O ---------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        arr = _synthetic_scene(24, 32)
        npy = os.path.join(td, "a.npy")
        pfm = os.path.join(td, "a.pfm")
        np.save(npy, arr)
        write_pfm(pfm, arr)
        back = read_pfm(pfm)
        check("pfm round trip is exact", np.array_equal(back, arr))
        check("load_image(.npy)", np.array_equal(load_image(npy), arr))

        # EXR path resolves to the sibling .npy written by the Blender stage.
        exr = os.path.join(td, "a.exr")
        open(exr, "wb").write(b"not really an exr")
        check("load_image(.exr) uses the sibling .npy", np.array_equal(load_image(exr), arr))

        # Numpy-free writer used inside Blender when numpy is missing.
        bottom_up_flat = np.ascontiguousarray(
            np.concatenate([arr, np.ones(arr.shape[:2] + (1,), np.float32)], axis=2)[::-1]).ravel().tolist()
        pure = os.path.join(td, "pure.pfm")
        write_pfm_pure(pure, bottom_up_flat, arr.shape[1], arr.shape[0], 4)
        check("numpy-free PFM writer matches the numpy one",
              open(pure, "rb").read() == open(pfm, "rb").read())

        raw = os.path.join(td, "a.f32")
        arr.astype("<f4").tofile(raw)
        check("raw needs a shape", _raises(lambda: load_image(raw)))
        check("raw with --raw-shape", np.array_equal(load_image(raw, parse_shape("24x32x3")), arr))
        json.dump({"shape": [24, 32, 3]}, open(raw + ".json", "w"))
        check("raw with .json sidecar", np.array_equal(load_image(raw), arr))

        missing = os.path.join(td, "nothing.exr")
        check("EXR without sibling gives an actionable error", _raises(lambda: load_image(missing), "Blender"))

        # Command line end to end: exit codes and report file.
        rep = os.path.join(td, "r.md")
        np.save(os.path.join(td, "t.npy"), noisy(0.05))
        np.save(os.path.join(td, "r.npy"), noisy(0.05))
        np.save(os.path.join(td, "b.npy"), noisy(0.05, mult=1.1))
        p = build_stage2_parser()
        import io
        import contextlib
        def run_cli(*cli):
            a = p.parse_args(list(cli))
            return a.func(a)

        with contextlib.redirect_stdout(io.StringIO()):
            tn, rn, bn = (os.path.join(td, n) for n in ("t.npy", "r.npy", "b.npy"))
            rc_ok = run_cli("compare", tn, rn, "--report", rep)
            rc_bad = run_cli("compare", bn, rn)
            np.save(os.path.join(td, "f.npy"), flat)
            rc_f = run_cli("furnace", os.path.join(td, "f.npy"))
        check("CLI compare: pass -> 0, biased -> 1", rc_ok == 0 and rc_bad == 1, "%s %s" % (rc_ok, rc_bad))
        check("CLI furnace: pass -> 0", rc_f == 0)
        check("CLI wrote the report", os.path.exists(rep) and "PASS" in open(rep).read())

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, ok, detail in results:
        failed += (not ok)
        print("%-4s %s%s" % ("ok" if ok else "FAIL", name.ljust(width), ("  " + detail) if detail else ""))
    print("\n%d checks, %d failed" % (len(results), failed))
    return EXIT_PASS if failed == 0 else EXIT_FAIL


def _raises(fn, text=None):
    try:
        fn()
    except Exception as e:
        return text is None or text in str(e)
    return False


# -----------------------------------------------------------------------------

def _in_blender():
    try:
        import bpy  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None):
    if argv is None:
        if _in_blender() and "--" in sys.argv:
            return render_main(_blender_argv())
        argv = sys.argv[1:]
    if argv and argv[0] in ("compare", "furnace", "convert", "selftest"):
        args = build_stage2_parser().parse_args(argv)
        return args.func(args)
    if _in_blender():
        return render_main(argv)
    build_stage2_parser().print_help()
    print("\nThe render stage needs Blender: blender -b scene.blend --python %s -- --out out.exr ..." % sys.argv[0])
    return EXIT_ERROR


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:  # noqa: BLE001  keep the exit code meaningful inside Blender
        import traceback
        traceback.print_exc()
        print("error: %s" % e, file=sys.stderr)
        code = EXIT_ERROR
    sys.exit(code)
