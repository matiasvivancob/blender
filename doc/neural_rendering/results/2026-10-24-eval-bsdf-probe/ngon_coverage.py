# Count how many of the probe's 256 records land on the n-gon (same seed and sampler as blender_dump.py --probe).
import sys, numpy as np, bpy
sys.path.insert(0, sys.argv[sys.argv.index("--") + 1]); import sampling
obj = bpy.data.objects["ProbeNgon"]; me = obj.data; me.calc_loop_triangles()
loops = np.empty(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get("loops", loops)
uv = np.empty(len(me.loops) * 2, np.float32); me.uv_layers.active.data.foreach_get("uv", uv)
prims, _, _ = sampling.uniform_uv_on_mesh(np.random.default_rng(0), 256, uv.reshape(-1, 2)[loops.reshape(-1, 3)])
poly = np.array([t.polygon_index for t in me.loop_triangles]); ngon = [p.index for p in me.polygons if len(p.vertices) > 4][0]
print(f"NGON polygon {ngon}: {int((poly == ngon).sum())} of {len(poly)} triangles; probe records on it: {int((poly[prims] == ngon).sum())} of 256")
