# SPDX-License-Identifier: Apache-2.0
"""Blender-side driver that writes a BSDF dump (dump_format.md) using the Cycles eval_bsdf wrapper.

*** Status: `engine.eval_bsdf` and this driver ran on CPU, CUDA and OptiX on 2026-10-23
(results/2026-10-23-eval-bsdf-parity), and --probe passed on 2026-10-24 on a plane and on a mesh with quads and
an n-gon (max albedo error 1.2e-7): the prim index is the `mesh.loop_triangles` index and the barycentric
convention matches. Re-run --probe for a new kind of geometry (curves, point clouds, instancing). ***

Run (from doc/neural_rendering/tools/train, or give the full path of this file), CPU or CUDA/OptiX Cycles device
as set in the scene's Cycles settings:

  # 1. training data: uniform texels, cosine-weighted wo, GGX-peak-targeted + uniform wi (sampling.py)
  blender -b scene.blend --python blender_dump.py -E BSDF_DUMP -f 1 -- \
      --object Plane --n 4000000 --out dump_a.npz --seed 0

  # 2. ground truth for the sphere render of evaluate.py (queries made by `python3 sphere.py q.npz`)
  blender -b scene.blend --python blender_dump.py -E BSDF_DUMP -f 1 -- \
      --object Plane --fill q.npz --out sphere_dump.npz

  # 3. check the prim-index and barycentric assumptions (prints PROBE PASS or PROBE FAIL)
  blender -b scene.blend --python blender_dump.py -E BSDF_DUMP -f 1 -- --object Plane --probe

--probe replaces the object's materials, for this run only (nothing is saved), with a diffuse BSDF whose color
is an affine function of the same UV layer the script reads: color = (0.1 + 0.8 u, 0.1 + 0.8 v, 0.5). Diffuse
albedo equals its color, so the albedo Cycles returns at (prim, barycentric) must reproduce the texture UV that
this script computed for that (prim, barycentric) from `loop_triangles`. A wrong triangle order, a missing prim
offset or a different barycentric convention shows up as a large error. Pass: max abs error below 1e-3 on 256
points. Run it once per new scene or mesh type before trusting a dump.

Requirements on the scene: a camera (any; without one -f never calls the engine and Blender exits 0), the
object has one material with the surface shader to be learned, an active UV
layer (or --uv-layer) that maps the area of interest into [0,1]^2 without overlaps (the latent grid is indexed
by that UV), and the Cycles device in the render settings is the one to use. Lights and world are irrelevant:
eval_bsdf evaluates the closures of the surface shader only. Shaders with transmission, emission only, volumes or
a displacement/bump that depends on the view are out of scope (records come back flagged).
"""
import os
import sys
import time

import numpy as np

import bpy
from cycles import CyclesRender
from cycles import engine as cycles_engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset  # noqa: E402
import sampling  # noqa: E402


def script_args():
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--uv-layer", default=None)
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peak-frac", type=float, default=0.5)
    ap.add_argument("--grazing-frac", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=1 << 20, help="records per eval_bsdf call")
    ap.add_argument("--out", default="dump.npz")
    ap.add_argument("--fill", default=None, help="queries .npz (e.g. from sphere.py): evaluate these instead of sampling")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-n", type=int, default=256)
    return ap.parse_args(argv)


PROBE_SCALE, PROBE_OFFSET = (0.8, 0.8, 0.0), (0.1, 0.1, 0.5)
PROBE_TOL = 1e-3


def uv_layer_name(obj, uv_layer=None):
    layer = obj.data.uv_layers[uv_layer] if uv_layer else obj.data.uv_layers.active
    if layer is None:
        raise RuntimeError("object has no UV layer")
    return layer.name


def install_probe_material(obj, uv_name):
    """Replace every material slot of obj with color = PROBE_OFFSET + PROBE_SCALE * uv on a diffuse BSDF."""
    mat = bpy.data.materials.new("BSDF_DUMP_probe")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = uv_name  # the same layer triangle_uvs() reads, not the render-active one
    ma = nt.nodes.new("ShaderNodeVectorMath")
    ma.operation = "MULTIPLY_ADD"
    ma.inputs[1].default_value = PROBE_SCALE
    ma.inputs[2].default_value = PROBE_OFFSET
    dif = nt.nodes.new("ShaderNodeBsdfDiffuse")
    dif.inputs["Roughness"].default_value = 0.0
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(uvn.outputs["UV"], ma.inputs[0])
    nt.links.new(ma.outputs["Vector"], dif.inputs["Color"])
    nt.links.new(dif.outputs["BSDF"], outn.inputs["Surface"])
    if not obj.material_slots:
        obj.data.materials.append(mat)
    for slot in obj.material_slots:
        slot.material = mat


def probe_check(tex, out):
    """Return (ok, report lines). tex (n,2) texture UV computed here, out (n,8) eval_bsdf output."""
    expect = np.column_stack([PROBE_OFFSET[0] + PROBE_SCALE[0] * tex[:, 0],
                              PROBE_OFFSET[1] + PROBE_SCALE[1] * tex[:, 1],
                              np.full(len(tex), PROBE_OFFSET[2])]).astype(np.float64)
    albedo = out[:, 4:7].astype(np.float64)
    flagged = out[:, 7] != 0
    err = np.abs(albedo - expect).max(axis=1)
    err[flagged] = np.inf
    worst = np.argsort(-err)[:5]
    lines = [f"probe n={len(tex)} flagged={int(flagged.sum())} max_abs_err={err.max():.3e} "
             f"median_abs_err={np.median(err):.3e} tol={PROBE_TOL:g}"]
    lines += [f"  worst i={i} expect={expect[i].round(5).tolist()} got={albedo[i].round(5).tolist()} "
              f"flag={int(out[i, 7])}" for i in worst]
    return bool(np.all(err < PROBE_TOL)), lines


def default_output_path():
    """-f writes the render result; Blender's default '/tmp/' becomes C:\\tmp on Windows. Keep it out of the way."""
    import tempfile
    r = bpy.context.scene.render
    if r.filepath in ("/tmp/", "/tmp\\", ""):
        r.filepath = os.path.join(tempfile.gettempdir(), "bsdf_dump", "render_")


def triangle_uvs(obj, uv_layer=None):
    """(T, 3, 2) UVs of the corners of every triangle of the evaluated mesh, in loop_triangles order."""
    mesh = obj.to_mesh()
    try:
        mesh.calc_loop_triangles()
        layer = mesh.uv_layers[uv_layer] if uv_layer else mesh.uv_layers.active
        if layer is None:
            raise RuntimeError("object has no UV layer")
        tris = mesh.loop_triangles
        loops = np.empty(len(tris) * 3, np.int32)
        tris.foreach_get("loops", loops)
        uv = np.empty(len(mesh.loops) * 2, np.float32)
        layer.data.foreach_get("uv", uv)
        return uv.reshape(-1, 2)[loops.reshape(-1, 3)]
    finally:
        obj.to_mesh_clear()


class BsdfDumpEngine(CyclesRender):
    bl_idname = "BSDF_DUMP"
    bl_label = "BSDF Dump"

    def render(self, depsgraph):
        # Blender swallows exceptions raised in render() and still exits 0, even with
        # --python-exit-code 1, so a failed dump would look like success to a harness.
        # This engine only runs in background dump/probe jobs: print the traceback and exit 1.
        try:
            self._render(depsgraph)
        except BaseException:
            import traceback
            traceback.print_exc()
            print("BSDF_DUMP ERROR: render failed, exiting with status 1", file=sys.stderr, flush=True)
            sys.stdout.flush()
            os._exit(1)

    def _render(self, depsgraph):
        a = script_args()
        obj = depsgraph.objects[a.object]
        tri_uvs = triangle_uvs(obj, a.uv_layer)
        rng = np.random.default_rng(a.seed)
        t0 = time.time()
        if a.probe:
            n = a.probe_n
            prims, bary, tex = sampling.uniform_uv_on_mesh(rng, n, tri_uvs)
            wi = np.tile(np.float32([0, 0, 1]), (n, 1))
            wo = np.tile(np.float32([0.6, 0, 0.8]), (n, 1))
            out = np.zeros((n, 8), np.float32)
            cycles_engine.eval_bsdf(self, depsgraph, obj, prims, bary, wi, wo, 1, out)
            ok, lines = probe_check(tex, out)
            print("\n".join(lines))
            print("PROBE PASS" if ok else "PROBE FAIL", flush=True)
            if not ok:
                raise RuntimeError("BSDF_DUMP probe failed")
            return
        if a.fill:
            q = dataset.load_dump(a.fill)
            tex = q.texture_uv()
            prims, bary = sampling.locate_uv(tri_uvs, tex)
            wi, wo = q.wi, q.wo
        else:
            prims, bary, tex = sampling.uniform_uv_on_mesh(rng, a.n, tri_uvs)
            wi, wo = sampling.sample_directions(rng, a.n, a.peak_frac, a.grazing_frac)
        wi, wo = np.ascontiguousarray(wi, np.float32), np.ascontiguousarray(wo, np.float32)
        n = len(prims)
        out = np.zeros((n, 8), np.float32)
        for s in range(0, n, a.batch):
            e = min(n, s + a.batch)
            cycles_engine.eval_bsdf(self, depsgraph, obj, prims[s:e], bary[s:e], wi[s:e], wo[s:e], 1, out[s:e])
            print(f"eval_bsdf {e}/{n} ({time.time() - t0:.0f}s)", flush=True)
        mat = obj.active_material.name if obj.active_material else ""
        meta = {"object_name": a.object, "material_name": mat, "blender_version": bpy.app.version_string,
                "sample_count": n, "uv_kind": "texture", "seed": a.seed, "uv_layer": a.uv_layer or "active",
                "sampler": {"peak_frac": a.peak_frac, "grazing_frac": a.grazing_frac}, "eval_seconds": time.time() - t0,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "sphere_queries" if a.fill else "training"}
        dataset.save_dump(a.out, prims=prims, uv=bary, tex_uv=tex, wi=wi, wo=wo, out=out, frame_local=True, meta=meta)
        flagged = float((out[:, 7] != 0).mean())
        print(f"wrote {a.out}: {n} records, {100 * flagged:.2f}% flagged")


def main():
    a = script_args()
    if bpy.context.scene.camera is None:
        # Without a camera, -f logs an error, never calls the engine and Blender still exits 0.
        print("BSDF_DUMP ERROR: the scene has no camera; -f would skip the engine silently. Add any camera.",
              file=sys.stderr, flush=True)
        sys.exit(1)
    if a.object not in bpy.data.objects:
        print(f"BSDF_DUMP ERROR: no object named {a.object!r}", file=sys.stderr, flush=True)
        sys.exit(1)
    default_output_path()
    if a.probe:
        obj = bpy.data.objects[a.object]
        install_probe_material(obj, uv_layer_name(obj, a.uv_layer))
    bpy.utils.register_class(BsdfDumpEngine)


main()
