# Slice 1 gate for the Neural BSDF node: node/RNA/DNA plumbing and the Cycles file upload, per device.
#
#   NR_DEVICE=CPU|CUDA|OPTIX NR_GPU=<substring> NR_MODEL=<model_with_proxy.nnm> NR_NOPROXY=<model_without_proxy.nnm> \
#   blender -b --factory-startup --python-exit-code 1 --python neural_node_check.py -- <abs scratch dir>
#
# Prints SLICE1 PASS or raises. Checks:
#  1. ShaderNodeBsdfNeural exists, has a `filepath`, and node type + path survive save and reload (DNA storage).
#  2. Neural node with a valid file renders identically to a Diffuse BSDF of the file's albedo (read here straight
#     from header bytes 172..183, independent of the C++ parser), so the file was loaded and uploaded.
#  3. Invalid path and a file without a proxy block render black (no crash, no garbage).
import os, struct, sys
import bpy
import numpy as np

work = sys.argv[sys.argv.index("--") + 1]
model, noproxy = os.environ["NR_MODEL"], os.environ["NR_NOPROXY"]
device, gpu = os.environ.get("NR_DEVICE", "CPU"), os.environ.get("NR_GPU", "")

data = open(model, "rb").read()
assert struct.unpack_from("<I", data, 12)[0] & 1, "model has no proxy block"
albedo = struct.unpack_from("<3f", data, 172)


def setup_device():
    scene = bpy.context.scene
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device == "CPU":
        scene.cycles.device = "CPU"
        return
    prefs.compute_device_type = device
    prefs.refresh_devices()
    on = []
    for d in prefs.devices:
        d.use = d.type == device and gpu in d.name
        if d.use:
            on.append(d.name)
    assert len(on) == 1, (device, gpu, on)
    scene.cycles.device = "GPU"
    print("SLICE1 device", device, on)


def make_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    c = scene.cycles
    c.samples, c.use_denoising, c.use_adaptive_sampling = 16, False, False
    c.sample_clamp_direct = c.sample_clamp_indirect = 0.0
    c.seed, c.use_animated_seed = 3, False
    c.max_bounces = 2
    scene.render.resolution_x = scene.render.resolution_y = 64
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.world = bpy.data.worlds.new("w")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0
    bpy.ops.mesh.primitive_plane_add(size=4)
    plane = bpy.context.active_object
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 3))
    bpy.context.active_object.data.energy = 600.0
    bpy.context.active_object.data.size = 2.0
    bpy.ops.object.camera_add(location=(0, 0, 4))
    scene.camera = bpy.context.active_object
    return scene, plane


def set_material(plane, builder):
    mat = bpy.data.materials.new("m")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = builder(nt)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    plane.data.materials.clear()
    plane.data.materials.append(mat)


def neural(path):
    def build(nt):
        n = nt.nodes.new("ShaderNodeBsdfNeural")
        n.filepath = path
        return n
    return build


def diffuse(col):
    def build(nt):
        n = nt.nodes.new("ShaderNodeBsdfDiffuse")
        n.inputs["Color"].default_value = (*col, 1.0)
        return n
    return build


def render(scene, name):
    path = os.path.join(work, name + ".exr")
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    px = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    px = px.reshape(-1, 4)[:, :3]
    assert np.isfinite(px).all(), name + ": non-finite pixels"
    return px


os.makedirs(work, exist_ok=True)
scene, plane = make_scene()
setup_device()

# 1. node, RNA, DNA round trip
set_material(plane, neural(model))
node = [n for n in bpy.data.materials["m"].node_tree.nodes if n.bl_idname == "ShaderNodeBsdfNeural"][0]
assert node.filepath == model, node.filepath
blend = os.path.join(work, "slice1.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend)
bpy.ops.wm.open_mainfile(filepath=blend)
node = [n for n in bpy.data.materials["m"].node_tree.nodes if n.bl_idname == "ShaderNodeBsdfNeural"][0]
# Saving makes the path blend-relative (PROP_PATH_SUPPORTS_BLEND_RELATIVE), so compare resolved paths.
norm = lambda s: os.path.normcase(os.path.normpath(bpy.path.abspath(s)))
assert node.filepath != model and node.filepath.startswith("//"), "path was not made blend-relative: " + node.filepath
assert norm(node.filepath) == norm(model), "filepath lost in save/load: " + node.filepath
scene, plane = bpy.context.scene, bpy.data.objects["Plane"]
setup_device()

# 2. valid file == diffuse of the file's albedo
a = render(scene, "neural_valid")
set_material(plane, diffuse(albedo))
b = render(scene, "diffuse_albedo")
diff = float(np.abs(a - b).max())
print(f"SLICE1 valid file: mean {a.mean():.5f}, diffuse(albedo={tuple(round(v, 4) for v in albedo)}) mean {b.mean():.5f}, max abs diff {diff:.3e}")
assert a.mean() > 1e-3, "valid neural file rendered black"
assert diff < 2e-3 * max(float(b.max()), 1.0), f"neural placeholder differs from diffuse(albedo): {diff}"

# 3. invalid inputs render black
for label, path in (("missing_file", os.path.join(work, "does_not_exist.nnm")), ("no_proxy_block", noproxy)):
    set_material(plane, neural(path))
    c = render(scene, label)
    print(f"SLICE1 {label}: mean {c.mean():.6f}, max {c.max():.6f}")
    assert c.max() < 1e-4, f"{label} should render black, max {c.max()}"

print("SLICE1 PASS", device)
