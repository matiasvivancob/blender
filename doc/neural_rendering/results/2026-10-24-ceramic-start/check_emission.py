"""Check whether the selected native ceramic actually emits, without modifying its node graph or saving."""
from pathlib import Path
import bpy
import json
import numpy as np

root = Path('C:/Users/Matia/workspace/procedural-ceramic-experiment')
s = bpy.context.scene
assert bpy.data.objects['Sphere'].active_material.name == ' Procedural Ceramic'
for ob in s.objects:
    if ob.type == 'LIGHT':
        ob.hide_render = True
    elif ob.type in {'MESH','META','CURVE','SURFACE','FONT'} and ob.name != 'Sphere':
        ob.hide_render = True
world = bpy.data.worlds.new('Black emission-only audit world')
s.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
s.cycles.device = 'CPU'
s.cycles.samples = 16
s.cycles.use_denoising = False
s.render.resolution_x = s.render.resolution_y = 256
s.render.resolution_percentage = 100
s.render.use_compositing = False
s.render.image_settings.file_format = 'OPEN_EXR'
s.render.image_settings.color_depth = '32'
s.render.filepath = str(root/'emission_only.exr')
bpy.ops.render.render(write_still=True)
im = bpy.data.images.load(s.render.filepath)
a = np.array(im.pixels[:],dtype=np.float32).reshape(-1,4)[:,:3]
assert np.isfinite(a).all()
r = {'method':'Native shader unchanged; no lights, black world, all other geometry hidden; CPU 16spp',
     'max_rgb':float(a.max()),'mean_rgb':float(a.mean()),'nonzero_pixels':int(np.count_nonzero(a.max(axis=1)>1e-7))}
(root/'emission_check.json').write_text(json.dumps(r,indent=2))
print('CERAMIC EMISSION CHECK',json.dumps(r),flush=True)
