# Put a Neural BSDF on the one-quad probe scene (unit-square UV), for neural_closure_check.py.
#   NR_MODEL=<model_with_proxy.nnm> blender -b <probe_quad.blend> --python-exit-code 1 \
#       --python make_neural_quad_scene.py -- <abs out.blend>
import os, sys
import bpy

obj = bpy.data.objects["ProbeQuad"]
mat = bpy.data.materials.new("neural_probe_quad")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
neural = nt.nodes.new("ShaderNodeBsdfNeural")
neural.filepath = os.environ["NR_MODEL"]
nt.links.new(neural.outputs["BSDF"], out.inputs["Surface"])
obj.data.materials.clear()
obj.data.materials.append(mat)
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
