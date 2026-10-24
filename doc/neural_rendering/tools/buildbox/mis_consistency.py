#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
MIS-consistency gate: render one scene with different light-sampling
strategies and assert that they converge to the same image.

This is the mandatory gate for any new closure (PROTOCOL.md). A closure whose
`eval`, `sample` and `pdf` disagree still gives a plausible image under the
default MIS strategy, because MIS blends two wrong estimators. Forcing one
strategy at a time exposes the disagreement as a brightness difference.

Run inside Blender (arguments after `--`):

    blender -b mis_sphere.blend --python mis_consistency.py -- \\
        --device OPTIX --samples 256 --out-dir out/mis

Exit codes: 0 all pairs agree, 1 at least one pair differs, 2 error.

What can be forced from Python (checked against the tree, Blender 5.x):

  Strategy knob (preferred). `scene.cycles.direct_light_sampling_type`, an enum
  in intern/cycles/blender/addon/properties.py with
      MULTIPLE_IMPORTANCE_SAMPLING  both strategies, power-heuristic weights
      FORWARD_PATH_TRACING          BSDF sampling only ("forward")
      NEXT_EVENT_ESTIMATION         light sampling only
  The kernel side is light_sample_mis_weight_forward()/..._nee() in
  intern/cycles/kernel/light/sample.h, which return 1/0 or 0/1 for these modes.
  It is a developer option: intern/cycles/blender/sync.cpp only reads it when
  `use_developer_ui` is true, i.e. when BOTH
      bpy.context.preferences.experimental.use_cycles_debug  and
      bpy.context.preferences.view.show_developer_ui
  are True (intern/cycles/blender/session.cpp). This script sets both
  in-process; they are not saved. It does not need WITH_CYCLES_DEBUG at build
  time (that CMake flag only adds ptxas reports, guiding debug passes and the
  extra sampling patterns).

  Not strategy knobs, and therefore NOT used to force anything:
    * scene.cycles.sample_clamp_direct / sample_clamp_indirect  bias the result
      on purpose; render_test.py sets them to 0.
    * scene.cycles.min_light_bounces / min_transparent_bounces  only delay
      Russian roulette (unbiased, variance only). Optional --no-russian-roulette.
    * scene.cycles.light_sampling_threshold  stops light samples with a small
      contribution probabilistically, which biases the image when the light
      tree is off. render_test.py sets it to 0.
    * scene.cycles.use_light_tree  changes which light is picked, not whether
      lights are sampled; both settings must converge to the same mean.
  There is no per-closure or per-material switch for BSDF-only / light-only
  sampling; it is global via the knob above.

Fallback. If `direct_light_sampling_type` does not exist in this build, or
`--mode lighttree` is given, the script compares light tree on, light tree off
and light tree on at 4x the samples with another seed. That checks light
sampling consistency and noise convergence but cannot isolate the BSDF
strategy, and the report says so.

Caveats for the scene: delta lights (point, spot, sun) cannot be hit by BSDF
sampling and delta BSDFs cannot be light-sampled, so avoid perfectly sharp
glossy surfaces; mis_sphere.blend (make_test_scenes.py) uses roughness 0.35 and
a finite-size area light. Forward-only on a small light is very noisy, so it
gets --forward-spp-mult times more samples by default.
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import render_test as rt  # noqa: E402

KNOB = "direct_light_sampling_type"
STRATEGIES = {
    "mis": "MULTIPLE_IMPORTANCE_SAMPLING",
    "nee_only": "NEXT_EVENT_ESTIMATION",
    "forward_only": "FORWARD_PATH_TRACING",
}


def build_parser():
    p = argparse.ArgumentParser(prog="mis_consistency.py", description="Arguments after `--`.")
    p.add_argument("--out-dir", default="mis_out")
    p.add_argument("--samples", type=int, default=256, help="samples of the MIS / NEE / light-tree-on renders")
    p.add_argument("--forward-spp-mult", type=int, default=4, help="forward-only gets this many times more samples")
    p.add_argument("--device", default="OPTIX", choices=["CPU", "CUDA", "OPTIX"])
    p.add_argument("--gpu-name")
    p.add_argument("--res", help="WxH (default: keep the scene's)")
    p.add_argument("--seed", type=int, default=100, help="base seed; every render uses a different one")
    p.add_argument("--scene")
    p.add_argument("--mode", default="auto", choices=["auto", "strategies", "lighttree"])
    p.add_argument("--extended", action="store_true", help="strategies mode: also MIS with the light tree off")
    p.add_argument("--no-russian-roulette", action="store_true")
    p.add_argument("--report", help="default: <out-dir>/mis_consistency.md")
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--tile", type=int, default=16)
    p.add_argument("--max-bias", type=float, default=0.02, help="relative brightness difference that fails")
    p.add_argument("--z-max", type=float, default=4.0)
    return p


def enable_developer_flags():
    """Make sync.cpp read direct_light_sampling_type. Returns True when both flags read back as set."""
    import bpy
    prefs = bpy.context.preferences
    try:
        prefs.experimental.use_cycles_debug = True
        prefs.view.show_developer_ui = True
    except AttributeError:
        return False
    return bool(prefs.experimental.use_cycles_debug and prefs.view.show_developer_ui)


def noise_proxy(img):
    """RMS residual of each pixel against its 4-neighbour mean. Informational; scene detail inflates it."""
    x = img[:, :, :3].mean(axis=2).astype("float64")
    nb = (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]) / 4.0
    d = x[1:-1, 1:-1] - nb
    return math.sqrt(float((d * d).mean()) / 1.25)


def plan(scene, args, mode):
    """List of configs: (name, apply(scene) callable, samples, description)."""
    n = args.samples
    cfgs = []
    if mode == "strategies":
        for name, enum in STRATEGIES.items():
            mult = args.forward_spp_mult if name == "forward_only" else 1

            def apply(sc, enum=enum):
                setattr(sc.cycles, KNOB, enum)
            cfgs.append((name, apply, n * mult, "%s = %s" % (KNOB, enum)))
        if args.extended:
            def apply_notree(sc):
                setattr(sc.cycles, KNOB, STRATEGIES["mis"])
                sc.cycles.use_light_tree = False
            cfgs.append(("mis_no_light_tree", apply_notree, n, "MIS, light tree off"))
    else:
        def tree_on(sc):
            sc.cycles.use_light_tree = True

        def tree_off(sc):
            sc.cycles.use_light_tree = False
        cfgs.append(("tree_on", tree_on, n, "light tree on"))
        cfgs.append(("tree_off", tree_off, n, "light tree off, light_sampling_threshold 0"))
        cfgs.append(("tree_on_4x", tree_on, n * 4, "light tree on, 4x samples"))
    return cfgs


def run(argv):
    args = build_parser().parse_known_args(argv)[0]
    import bpy
    scene = bpy.data.scenes[args.scene] if args.scene else bpy.context.scene

    has_knob = hasattr(scene.cycles, KNOB)
    dev_flags = enable_developer_flags() if has_knob else False
    if args.mode == "strategies" and not (has_knob and dev_flags):
        raise RuntimeError("--mode strategies needs scene.cycles.%s and the developer preferences "
                           "(experimental.use_cycles_debug, view.show_developer_ui)" % KNOB)
    mode = args.mode if args.mode != "auto" else ("strategies" if (has_knob and dev_flags) else "lighttree")

    os.makedirs(args.out_dir, exist_ok=True)
    cfgs = plan(scene, args, mode)
    images, rows = {}, []
    for i, (name, apply, spp, desc) in enumerate(cfgs):
        ra = argparse.Namespace(
            device=args.device, gpu_name=args.gpu_name, samples=spp, seed=args.seed + i + 1, res=args.res,
            keep_settings=False, denoise=False, no_russian_roulette=args.no_russian_roulette)
        out = os.path.join(args.out_dir, name + ".exr")
        arr, meta = rt.render_to_arrays(scene, out, ra, pre_render=apply)
        if arr is None:
            raise RuntimeError("numpy is required for the comparison stage inside Blender; "
                               "compare the .pfm files outside with render_test.py")
        images[name] = arr
        rows.append({"name": name, "desc": desc, "samples": spp, "seed": ra.seed,
                     "seconds": meta["render_seconds"], "mean": float(arr[:, :, :3].mean()),
                     "noise": noise_proxy(arr), "noise_1spp": noise_proxy(arr) * math.sqrt(spp)})
        print("rendered %-18s %5d spp  %.1fs  mean %.5f" % (name, spp, meta["render_seconds"], rows[-1]["mean"]))

    names = [c[0] for c in cfgs]
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            res = rt.compare_images(images[names[i]], images[names[j]], args.eps, args.tile,
                                    max_bias=args.max_bias, z_max=args.z_max)
            pairs.append((names[i], names[j], res))

    ok = all(p[2]["ok"] for p in pairs)
    md = report_md(args, mode, has_knob, dev_flags, rows, pairs, ok, scene)
    print(md)
    rt._write_text(args.report or os.path.join(args.out_dir, "mis_consistency.md"), md)
    return rt.EXIT_PASS if ok else rt.EXIT_FAIL


def report_md(args, mode, has_knob, dev_flags, rows, pairs, ok, scene):
    L = ["### MIS consistency: %s" % ("PASS" if ok else "FAIL"), ""]
    L += ["| setting | value |", "|---|---|",
          "| scene | %s |" % scene.name,
          "| device | %s%s |" % (args.device, " (%s)" % args.gpu_name if args.gpu_name else ""),
          "| mode | %s |" % mode,
          "| `%s` available | %s |" % (KNOB, has_knob),
          "| developer preferences enabled | %s |" % dev_flags,
          "| gate | relative bias > %.2f %% together with |z| > %.1f fails |" % (100 * args.max_bias, args.z_max), ""]
    if mode == "lighttree":
        L += ["Fallback mode: the strategy knob was not available, so only light-tree on/off and a "
              "higher-sample render are compared. This does not isolate BSDF sampling from light sampling.", ""]
    L += ["| render | settings | spp | seed | seconds | mean RGB | noise proxy | noise at 1 spp |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %d | %d | %.1f | %.5f | %.4g | %.4g |" % (
            r["name"], r["desc"], r["samples"], r["seed"], r["seconds"], r["mean"], r["noise"], r["noise_1spp"]))
    L += ["", "| pair | relative bias (a - b) | z | rel. mse | max abs | verdict |", "|---|---|---|---|---|---|"]
    for a, b, res in pairs:
        if "bias" in res:
            bb = res["bias"]
            L.append("| %s vs %s | %+.3f %% | %.2f | %.3g | %.3g | %s |" % (
                a, b, 100 * bb["rel_bias"], bb["z"], res["rel_mse"], res["max_abs"], "ok" if res["ok"] else "FAIL"))
        else:
            L.append("| %s vs %s | n/a | n/a | n/a | n/a | FAIL: %s |" % (a, b, "; ".join(res["failures"])))
    fails = [(a, b, f) for a, b, res in pairs for f in res["failures"]]
    if fails:
        L.append("")
        L += ["- FAIL %s vs %s: %s" % t for t in fails]
    by = {r["name"]: r for r in rows}
    if mode == "strategies" and "forward_only" in by and "mis" in by:
        if by["forward_only"]["noise_1spp"] <= 1.1 * by["mis"]["noise_1spp"]:
            L += ["", "NOTE (heuristic): forward-only is not noticeably noisier per sample than MIS. "
                  "If the scene has a small light this suggests the strategy knob did not take effect; "
                  "check that experimental.use_cycles_debug and view.show_developer_ui read back True."]
    if mode == "strategies":
        L += ["", "Reading a failure:",
              "- `nee_only` vs `forward_only` differ: `eval()` and `sample()` of some closure disagree "
              "(the value or pdf returned by `sample` is not the density it draws from, or differs from `eval`).",
              "- `nee_only` and `forward_only` agree but `mis` differs: the `pdf()` used for the MIS weights differs "
              "from the density `sample()` actually draws from.",
              "- all three differ pairwise: check the furnace test first, then each closure separately."]
    else:
        L += ["", "Reading a failure: light tree on vs off differing points at light selection pdfs "
              "(light tree / `light_sample_mis_weight_*` inputs); a difference only against the 4x render "
              "means the gate is under-sampled, raise --samples."]
    return "\n".join(L) + "\n"


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return run(argv)


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("error: %s" % e, file=sys.stderr)
        code = rt.EXIT_ERROR
    sys.exit(code)
