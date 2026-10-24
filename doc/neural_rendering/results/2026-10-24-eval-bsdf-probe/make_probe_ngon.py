# Build probe_ngon.blend: a cube subdivided once (quads), one face dissolved into an 8-vertex n-gon, Smart UV Project.
# blender -b --factory-startup --python-exit-code 1 --python make_probe_ngon.py -- <abs out.blend>
import sys, bmesh, bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.camera_add(); bpy.context.scene.camera = bpy.context.active_object  # -f 1 aborts without a camera
bpy.ops.mesh.primitive_cube_add(); obj = bpy.context.active_object; obj.name = "ProbeNgon"
bm = bmesh.new(); bm.from_mesh(obj.data)
bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
top = [f for f in bm.faces if f.calc_center_median().z > 0.99]; assert len(top) == 4
bmesh.ops.dissolve_faces(bm, faces=top); bm.to_mesh(obj.data); bm.free()
assert sorted(len(p.vertices) for p in obj.data.polygons) == [4] * 20 + [8]  # 20 quads + one 8-gon
bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT"); bpy.ops.uv.smart_project(); bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.scene.render.engine = "CYCLES"; bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
