# Slice 2 furnace gate: the learned material against the Principled material it was trained from.
#
#   NR_DEVICE=CPU|CUDA|OPTIX NR_GPU=<substr> NR_MODEL=<model_with_proxy.nnm> NR_SPP=256 NR_SEEDS=4 \
#   blender -b <neural_material_probe.blend> --python-exit-code 1 --python neural_vs_principled.py -- <abs out dir>
#
# White furnace: world radiance 1, no lights, the probe plane seen from several angles (perspective camera,
# film transparent so the plane is the alpha mask). The Principled material from the probe scene is rendered
# first, then the same plane with a Neural BSDF node, each over NR_SEEDS seeds.
# Gate (STATUS.md, decision (d)): mean over the plane within 3 % of the Principled render, and no pixel of the
# neural render above 1 beyond 4 standard errors of its own noise (no energy gain).
import json, os, sys
import bpy
import numpy as np

out_dir = sys.argv[sys.argv.index("--") + 1]
model = os.environ["NR_MODEL"]
spp = int(os.environ.get("NR_SPP", "256"))
seeds = int(os.environ.get("NR_SEEDS", "4"))
device, gpu = os.environ.get("NR_DEVICE", "CPU"), os.environ.get("NR_GPU", "")
os.makedirs(out_dir, exist_ok=True)

scene = bpy.context.scene
prefs = bpy.context.preferences.addons["cycles"].preferences
if device == "CPU":
    scene.cycles.device = "CPU"
else:
    prefs.compute_device_type = device
    prefs.refresh_devices()
    on = []
    for d in prefs.devices:
        d.use = d.type == device and gpu in d.name
        if d.use:
            on.append(d.name)
    assert len(on) == 1, (device, gpu, on)
    scene.cycles.device = "GPU"
    print("FURNACE device", device, on)

c = scene.cycles
c.samples, c.use_denoising, c.use_adaptive_sampling = spp, False, False
c.sample_clamp_direct = c.sample_clamp_indirect = 0.0
c.max_bounces, c.use_animated_seed = 4, False
scene.render.resolution_x = scene.render.resolution_y = 96
scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"
scene.render.use_compositing = False

# White furnace world, no lights.
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (1, 1, 1, 1)
bg.inputs["Strength"].default_value = 1.0
for ob in list(bpy.data.objects):
    if ob.type == "LIGHT":
        bpy.data.objects.remove(ob, do_unlink=True)

plane = bpy.data.objects[scene["nr_probe_object"]]
src_mat = bpy.data.materials[scene["nr_probe_material"]]

# Oblique camera so wi varies over the plane.
cam = scene.camera
cam.data.type = "PERSP"
import mathutils
cam.location = (0.0, -2.6, 2.2)
cam.rotation_euler = (mathutils.Vector((0.0, 0.0, 0.0)) - cam.location).to_track_quat("-Z", "Y").to_euler()


def render(label, seed):
    c.seed = seed
    path = os.path.join(out_dir, f"{label}_{seed}.exr")
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    px = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    px = px.reshape(-1, 4)
    assert np.isfinite(px).all(), label + ": non-finite pixels"
    return px


def neural_material():
    mat = bpy.data.materials.new("neural_probe")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    n = nt.nodes.new("ShaderNodeBsdfNeural")
    n.filepath = model
    nt.links.new(n.outputs["BSDF"], out.inputs["Surface"])
    return mat


results = {}
for label, mat in (("principled", src_mat), ("neural", neural_material())):
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    results[label] = [render(label, s) for s in range(seeds)]

alpha = results["principled"][0][:, 3] > 0.99
assert alpha.sum() > 1000, f"plane mask too small: {int(alpha.sum())}"
assert np.array_equal(alpha, results["neural"][0][:, 3] > 0.99), "plane masks differ"

summary = {"device": device, "spp": spp, "seeds": seeds, "plane_pixels": int(alpha.sum())}
for label, imgs in results.items():
    rgb = np.stack([im[alpha, :3] for im in imgs])             # (seeds, pixels, 3)
    lum = rgb.mean(axis=2)                                      # (seeds, pixels)
    summary[label] = {"mean": float(lum.mean()), "mean_rgb": [float(v) for v in rgb.mean(axis=(0, 1))],
                      "per_seed_mean": [float(v) for v in lum.mean(axis=1)],
                      "max_pixel": float(lum.mean(axis=0).max())}
    results[label] = lum

ref, neu = summary["principled"]["mean"], summary["neural"]["mean"]
summary["mean_ratio_neural_over_principled"] = neu / ref
# Energy gain: per-pixel mean over seeds against 1 + 4 standard errors.
pix_mean = results["neural"].mean(axis=0)
pix_se = results["neural"].std(axis=0, ddof=1) / np.sqrt(seeds)
over = pix_mean > 1.0 + 4.0 * np.maximum(pix_se, 1e-6)
summary["pixels_above_one_beyond_4_sigma"] = int(over.sum())
summary["worst_excess"] = float((pix_mean - 1.0 - 4.0 * pix_se).max())
summary["gate_mean_within_3pct"] = bool(abs(neu / ref - 1.0) <= 0.03)
summary["gate_no_energy_gain"] = bool(over.sum() == 0)
json.dump(summary, open(os.path.join(out_dir, "furnace.json"), "w"), indent=1)
print("FURNACE", json.dumps(summary))
print("FURNACE", "PASS" if summary["gate_mean_within_3pct"] and summary["gate_no_energy_gain"] else "FAIL")
