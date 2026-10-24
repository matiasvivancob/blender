"""Exercise UV-frame baking on both triangles of an anisotropic Principled quad.
Run factory-startup in Blender with --python-exit-code 1 --python this.py -E UV_FRAME_CHECK -f 1.
NR_DEVICE and NR_GPU select the device, NR_FRAME_OUT selects the JSON path.
"""
import json
import os
from pathlib import Path
import bpy
import numpy as np
from cycles import engine as cycles_engine
from cycles import CyclesRender

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
mesh = bpy.data.meshes.new('Frame quad')
mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
uv = mesh.uv_layers.new(name='UVMap')
for i, p in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
    uv.data[i].uv = p
obj = bpy.data.objects.new('Frame quad', mesh)
bpy.context.collection.objects.link(obj)
mat = bpy.data.materials.new('Anisotropic 0.8')
p = mat.node_tree.nodes.get('Principled BSDF')
p.inputs['Metallic'].default_value = 1
p.inputs['Roughness'].default_value = 0.4
p.inputs['Anisotropic'].default_value = 0.8
t = mat.node_tree.nodes.new('ShaderNodeTangent')
t.direction_type = 'UV_MAP'
t.uv_map = 'UVMap'
mat.node_tree.links.new(t.outputs['Tangent'], p.inputs['Tangent'])
mesh.materials.append(mat)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.camera_add(location=(0.5, 0.5, 3))
bpy.context.scene.camera = bpy.context.object
scene = bpy.context.scene
scene.render.resolution_x = scene.render.resolution_y = 8
scene.render.resolution_percentage = 100
prefs = bpy.context.preferences.addons['cycles'].preferences
device = os.environ.get('NR_DEVICE', 'CPU')
if device != 'CPU':
    prefs.compute_device_type = device
    prefs.refresh_devices()
    enabled = []
    for d in prefs.devices:
        d.use = d.type == device and os.environ.get('NR_GPU', '4070') in d.name
        if d.use:
            enabled.append(d.name)
    assert len(enabled) == 1, enabled
scene.cycles.device = 'CPU' if device == 'CPU' else 'GPU'

class FrameCheck(CyclesRender):
    bl_idname = 'UV_FRAME_CHECK'
    bl_label = 'UV tangent frame check'

    def render(self, depsgraph):
        try:
            target = depsgraph.objects['Frame quad']
            # Same UV-invariant surface, same UV-frame directions, different triangle vertex order.
            prim = np.array([0, 1], np.int32)
            bary = np.array([[0.2, 0.3], [0.2, 0.3]], np.float32)
            wi = np.tile(np.array([0.6, 0.0, 0.8], np.float32), (2, 1))
            wo = np.tile(np.array([-0.3, 0.4, np.sqrt(0.75)], np.float32), (2, 1))
            values = {}
            for label, flags in [('edge_frame', 1), ('uv_tangent', 3)]:
                out = np.zeros((2, 8), np.float32)
                cycles_engine.eval_bsdf(self, depsgraph, target, prim, bary, wi, wo, flags, out)
                assert np.isfinite(out).all() and not out[:, 7].any(), (label, out)
                assert out[:, :3].max() > 1e-6, out
                rel = np.abs(out[0, :3] - out[1, :3]) / np.maximum(np.abs(out[0, :3]), 1e-6)
                values[label] = {'out': out.tolist(), 'max_rel_triangle_difference': float(rel.max())}
            assert values['uv_tangent']['max_rel_triangle_difference'] < 1e-4, values
            assert values['edge_frame']['max_rel_triangle_difference'] > 0.01, 'fixture not discriminating'
            values['device'] = device
            Path(os.environ['NR_FRAME_OUT']).write_text(json.dumps(values, indent=2))
            print('UV FRAME ANISOTROPY PASS', json.dumps(values), flush=True)
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(1)

bpy.utils.register_class(FrameCheck)
