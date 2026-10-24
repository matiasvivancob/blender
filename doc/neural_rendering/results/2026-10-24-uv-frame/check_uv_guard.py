"""Make a two-UV fixture and invoke the real dump entrypoint, to test its render-map guard."""
import bpy
import runpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_plane_add()
obj = bpy.context.object
obj.name = 'GuardPlane'
obj.data.uv_layers.active.name = 'UVMap'
obj.data.uv_layers.new(name='Other')
obj.data.uv_layers.active_index = 0
obj.data.uv_layers['UVMap'].active_render = True
bpy.ops.object.camera_add(location=(0, 0, 3))
bpy.context.scene.camera = bpy.context.object
bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.render.resolution_x = bpy.context.scene.render.resolution_y = 8
bpy.context.scene.render.filepath = 'C:/Users/Matia/workspace/uv-frame-fix/guard_render_'
runpy.run_path('D:/Github/Blender-RND/blender/doc/neural_rendering/tools/train/blender_dump.py', run_name='__main__')
