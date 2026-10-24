"""Cheap frame-defect confirmation requested by the advisor; reads saved probe and existing CPU residuals."""
from pathlib import Path
import collections
import json
import math
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

OUT = Path('C:/Users/Matia/workspace/procedural-ceramic-experiment')
scene = bpy.context.scene
ob = bpy.data.objects[scene['nr_probe_object']]
mesh = ob.data
mesh.calc_loop_triangles()
triangles = list(mesh.loop_triangles)
angles = []
for tri in triangles:
    p0,p1 = [mesh.vertices[i].co for i in tri.vertices[:2]]
    e = p1-p0
    angles.append(round(math.degrees(math.atan2(e.y,e.x)) % 180, 4))
print('TRIANGLE EDGES',[(tri.index,tri.polygon_index,round(angles[tri.index],3)) for tri in triangles[:12]])
world_vertices = [ob.matrix_world @ v.co for v in mesh.vertices]
bvh = BVHTree.FromPolygons(world_vertices,[tuple(t.vertices) for t in triangles],all_triangles=True)
d = np.load('D:/Github/Blender-RND/real/slice3_20261024/CPU/beauties.npz')
p,n = d['principled'],d['neural']
assert p.shape == n.shape == (512,512,3)
err = np.mean(np.abs(n-p)/np.maximum(np.abs(p),1e-3),axis=2)
cam = scene.camera
projection = cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(),x=512,y=512,scale_x=1,scale_y=1)
inverse = cam.matrix_world @ projection.inverted()
classes = collections.defaultdict(list)
for y in range(2,512,4):
    for x in range(2,512,4):
        a = inverse @ Vector((2*(x+0.5)/512-1,2*(y+0.5)/512-1,-1,1))
        b = inverse @ Vector((2*(x+0.5)/512-1,2*(y+0.5)/512-1,1,1))
        a = Vector(a[:3])/a.w
        b = Vector(b[:3])/b.w
        hit,normal,index,distance = bvh.ray_cast(a,(b-a).normalized())
        if index is not None:
            classes[angles[index]].append(float(err[y,x]))
result = {'scene':bpy.data.filepath,'object':ob.name,'edge_angles_counts':dict(collections.Counter(angles)),
          'first_triangles':[{'index':t.index,'polygon':t.polygon_index,'edge_angle_deg':angles[t.index]} for t in triangles[:12]],
          'residual_by_edge_angle':{str(k):{'pixels':len(v),'mean':float(np.mean(v)),'median':float(np.median(v)),
                                          'p99':float(np.quantile(v,.99))} for k,v in classes.items()},
          'method':'Existing CPU 1024spp residual; every fourth pixel, ray cast into probe plane; angle is object-space edge p1-p0 mod 180. No renderer changes.'}
assert sum(len(v) for v in classes.values()) > 1000
(OUT/'frame_confirmation.json').write_text(json.dumps(result,indent=2))
print('FRAME CONFIRMATION',json.dumps(result),flush=True)
