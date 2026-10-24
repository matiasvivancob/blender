"""Freeze the selected ceramic experiment and capture its native reference. Never save over the playground."""
from pathlib import Path
import hashlib
import json
import bpy
import numpy as np
from mathutils import Vector

ROOT = Path('C:/Users/Matia/workspace/procedural-ceramic-experiment')
ROOT.mkdir(parents=True, exist_ok=True)
source = Path(bpy.data.filepath)
before = hashlib.sha256(source.read_bytes()).hexdigest()
ob = bpy.data.objects['Sphere']
mat = ob.active_material
assert mat.name == ' Procedural Ceramic', repr(mat.name)
assert ob.type == 'MESH' and ob.data.uv_layers.active
# Leave all source shader nodes and values unchanged, including its emission.
root_nodes = []
for n in mat.node_tree.nodes:
    root_nodes.append({'name': n.name, 'type': n.bl_idname,
        'inputs': [{'index': i, 'name': s.name, 'value': list(s.default_value) if hasattr(s,'default_value') and hasattr(s.default_value,'__len__') and not isinstance(s.default_value,str) else str(getattr(s,'default_value',None)),
                    'links': [(l.from_node.name,l.from_socket.name) for l in s.links]} for i,s in enumerate(n.inputs)]})
# Normalize sidecar paths on this copy before relocating it.
for m in bpy.data.materials:
    if m.node_tree:
        for n in m.node_tree.nodes:
            if n.bl_idname == 'ShaderNodeBsdfNeural' and n.filepath:
                n.filepath = bpy.path.abspath(n.filepath)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.refresh_devices()
chosen = []
for d in prefs.devices:
    d.use = d.type == 'OPTIX' and '4070' in d.name
    if d.use:
        chosen.append(d.name)
assert len(chosen) == 1, chosen
scene.cycles.device = 'GPU'
scene.cycles.samples = 128
scene.cycles.use_denoising = True
scene.render.resolution_x = scene.render.resolution_y = 768
scene.render.resolution_percentage = 100
scene.render.use_compositing = False
# Add a reference camera, preserving the user's original camera and all lighting/geometry.
original_cam = scene.camera
cam = original_cam.copy()
cam.data = original_cam.data.copy()
cam.name = 'Ceramic comparison camera'
scene.collection.objects.link(cam)
scene.camera = cam
corners = [ob.matrix_world @ Vector(v) for v in ob.bound_box]
center = sum(corners, Vector()) / 8
radius = max((v-center).length for v in corners)
back = original_cam.rotation_euler.to_matrix() @ Vector((0,0,1))
cam.location = center + back * (radius * 5)
cam.rotation_euler = (center-cam.location).to_track_quat('-Z','Y').to_euler()
cam.data.type = 'ORTHO'
cam.data.ortho_scale = max(ob.dimensions) * 1.4
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(ROOT / 'native_reference.png')
scene['ceramic_experiment'] = 'Native reference only; no neural replacement or training yet'
scene['ceramic_receiver'] = ob.name
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / 'ceramic_reference.blend'), copy=True)
bpy.ops.render.render(write_still=True)
img = bpy.data.images.load(scene.render.filepath, check_existing=False)
pixels = np.asarray(img.pixels[:],np.float32)
assert np.isfinite(pixels).all() and pixels.reshape(-1,4)[:,:3].max() > 1e-3
assert hashlib.sha256(source.read_bytes()).hexdigest() == before
report = {'source':str(source),'source_sha256':before,'source_unchanged':True,'material':mat.name,'object':ob.name,
          'copy':str(ROOT/'ceramic_reference.blend'),'reference':scene.render.filepath,'camera':cam.name,
          'lighting_preserved':True,'node_values_preserved':True,'device':chosen,'samples':scene.cycles.samples,
          'training_started':False,'root_nodes':root_nodes,'uv_layers':[u.name for u in ob.data.uv_layers]}
(ROOT/'reference.json').write_text(json.dumps(report,indent=2))
print('CERAMIC REFERENCE PASS', json.dumps({k:v for k,v in report.items() if k!='root_nodes'}), flush=True)
