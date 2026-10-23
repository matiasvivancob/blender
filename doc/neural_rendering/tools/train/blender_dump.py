# SPDX-License-Identifier: Apache-2.0
"""Blender-side driver that writes a BSDF dump (dump_format.md) using the Cycles eval_bsdf wrapper.

*** UNTESTED: written without a Blender build. The pieces it relies on are (a) `engine.eval_bsdf` in
intern/cycles/blender/addon/engine.py, which was itself not built when this was written, and (b) the
assumption that Cycles' triangle index `prim` equals the index into `mesh.loop_triangles` of the evaluated
mesh for a single-material object. Check (b) with --probe before trusting a dump. ***

Run (from doc/neural_rendering/tools/train, or give the full path of this file), CPU or CUDA/OptiX Cycles device
as set in the scene's Cycles settings:

  # 1. training data: uniform texels, cosine-weighted wo, GGX-peak-targeted + uniform wi (sampling.py)
  blender -b scene.blend --python blender_dump.py -E BSDF_DUMP -f 1 -- \
      --object Plane --n 4000000 --out dump_a.npz --seed 0

  # 2. ground truth for the sphere render of evaluate.py (queries made by `python3 sphere.py q.npz`)
  blender -b scene.blend --python blender_dump.py -E BSDF_DUMP -f 1 -- \
      --object Plane --fill q.npz --out sphere_dump.npz

  # 3. check the prim-index assumption on a handful of points (prints the barycentric round trip)
  blender -b scene.blend --python blender_dump.py -E BSDF_DUMP -f 1 -- --object Plane --probe

Requirements on the scene: the object has one material with the surface shader to be learned, an active UV
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
    return ap.parse_args(argv)


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
        a = script_args()
        obj = depsgraph.objects[a.object]
        tri_uvs = triangle_uvs(obj, a.uv_layer)
        rng = np.random.default_rng(a.seed)
        t0 = time.time()
        if a.probe:
            prims, bary, tex = sampling.uniform_uv_on_mesh(rng, 8, tri_uvs)
            wi = np.tile(np.float32([0, 0, 1]), (8, 1))
            out = np.zeros((8, 8), np.float32)
            cycles_engine.eval_bsdf(self, depsgraph, obj, prims, bary, wi, wi, 1, out)
            print("probe prims", prims, "\nprobe tex_uv", tex, "\nprobe out", out, sep="\n")
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


bpy.utils.register_class(BsdfDumpEngine)
