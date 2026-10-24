# Build probe_quad.blend from neural_material_probe.blend: the same material on ONE quad whose UV is the unit square.
# blender_dump.py --fill (sphere ground truth) accepts at most 256 triangles; the 16x16 probe plane has 512.
# blender -b <scenes>/neural_material_probe.blend --python-exit-code 1 --python make_probe_quad.py -- <abs out.blend>
import sys, bpy
scene = bpy.context.scene
mat = bpy.data.materials[scene["nr_probe_material"]]
bpy.data.objects.remove(bpy.data.objects[scene["nr_probe_object"]], do_unlink=True)
me = bpy.data.meshes.new("ProbeQuad")
me.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
uv = me.uv_layers.new(name=scene["nr_probe_uv_map"])
for loop, co in zip(me.loops, [(0, 0), (1, 0), (1, 1), (0, 1)]):
    uv.data[loop.index].uv = co
me.materials.append(mat)
obj = bpy.data.objects.new("ProbeQuad", me); scene.collection.objects.link(obj)
assert len(me.polygons) == 1 and scene.camera is not None
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
