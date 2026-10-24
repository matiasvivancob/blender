# Build mis_sphere_neural.blend from mis_sphere.blend: the glossy sphere gets a Neural BSDF, everything else unchanged.
#   NR_MODEL=<model_with_proxy.nnm> blender -b <mis_sphere.blend> --python-exit-code 1 \
#       --python make_neural_mis_scene.py -- <abs out.blend>
import os, sys
import bpy

mat = bpy.data.materials["Glossy"]
nt = mat.node_tree
for n in [n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"]:
    nt.nodes.remove(n)
out = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"][0]
neural = nt.nodes.new("ShaderNodeBsdfNeural")
neural.filepath = os.environ["NR_MODEL"]
nt.links.new(neural.outputs["BSDF"], out.inputs["Surface"])
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
