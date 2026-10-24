# Slice 3: the same scene and camera rendered with the source Principled material and with the Neural BSDF.
#
#   NR_DEVICE=CPU|OPTIX|CUDA NR_GPU=<substr> NR_MODEL=<model_with_proxy.nnm> NR_RES=512 \
#   blender -b <neural_material_probe.blend> --python-exit-code 1 --python neural_vs_principled_render.py -- <abs out dir>
#
# Reports (1) image difference of one Principled render and one neural render at NR_SPP (default 1024), against the
# noise floor given by two independent Principled seeds, as mean and p99 relative error after NxN box averaging
# (so pixel noise does not dominate), and (2) diagnostic render time per sample from a two-point slope.
# Setup overhead is not guaranteed constant; report the complete intervals too, not a qualified speedup.
# The scene's own camera, lights and world are used unchanged.
import json, os, sys, time
import bpy
import numpy as np

out_dir = sys.argv[sys.argv.index("--") + 1]
model = os.environ["NR_MODEL"]
spp = int(os.environ.get("NR_SPP", "1024"))
res = int(os.environ.get("NR_RES", "512"))
box = int(os.environ.get("NR_BOX", "8"))
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

c = scene.cycles
c.use_denoising = c.use_adaptive_sampling = False
c.use_animated_seed = False
c.max_bounces = 4
scene.render.resolution_x = scene.render.resolution_y = res
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.render.use_compositing = False
scene.render.use_persistent_data = False
assert res > 0 and box > 0 and res % box == 0 and spp > 0
print("RENDERCMP SETTINGS", device, gpu, res, spp, "persistent_data", scene.render.use_persistent_data, flush=True)
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

plane = bpy.data.objects[scene["nr_probe_object"]]
src_mat = bpy.data.materials[scene["nr_probe_material"]]

nmat = bpy.data.materials.new("neural_probe")
nmat.use_nodes = True
nt = nmat.node_tree
nt.nodes.clear()
o = nt.nodes.new("ShaderNodeOutputMaterial")
n = nt.nodes.new("ShaderNodeBsdfNeural")
n.filepath = model
nt.links.new(n.outputs["BSDF"], o.inputs["Surface"])


def render(mat, samples, seed, tag):
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    c.samples, c.seed = samples, seed
    path = os.path.join(out_dir, f"{tag}.exr")
    scene.render.filepath = path
    t = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    dt = time.perf_counter() - t
    img = bpy.data.images.load(path)
    px = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    px = px.reshape(res, res, 4)[..., :3]
    assert np.isfinite(px).all(), tag + ": non-finite pixels"
    print("RENDERCMP RENDER", tag, "requested_samples", samples, "seconds", dt, flush=True)
    return px, dt


def slope(mat, label):
    """seconds per sample from two sample counts, best of 2 each (first render also warms kernels/caches)."""
    render(mat, 8, 0, f"{label}_warm")
    lo = min(render(mat, 64, s, f"{label}_lo")[1] for s in (0, 1))
    hi = min(render(mat, 512, s, f"{label}_hi")[1] for s in (0, 1))
    return (hi - lo) / (512 - 64), lo, hi


def boxavg(a):
    k = box
    return a[: res // k * k, : res // k * k].reshape(res // k, k, res // k, k, 3).mean(axis=(1, 3))


p0, _ = render(src_mat, spp, 0, "principled_s0")
p1, _ = render(src_mat, spp, 1, "principled_s1")
n0, _ = render(nmat, spp, 0, "neural_s0")

mask_lum = boxavg(p0).mean(axis=2)
mask = mask_lum > 1e-3
assert mask.sum() > 100, "empty image"


def rel(a, b):
    a, b = boxavg(a).mean(axis=2)[mask], boxavg(b).mean(axis=2)[mask]
    r = np.abs(a - b) / np.maximum(0.5 * (a + b), 1e-3)
    return {"mean": float(r.mean()), "p99": float(np.quantile(r, 0.99))}


def raw_rel(a, b):
    # Reference-defined mask, no box filtering; report RGB errors separately from luminance diagnostics.
    keep = p0.mean(axis=2) > 1e-3
    r = np.abs(a[keep] - b[keep]) / np.maximum(np.abs(b[keep]), 1e-3)
    return {"mean": float(r.mean()), "p99": float(np.quantile(r, 0.99))}


s_p, lo_p, hi_p = slope(src_mat, "principled")
s_n, lo_n, hi_n = slope(nmat, "neural")
assert s_p > 0 and s_n > 0, "non-positive time-vs-spp slope; timing is invalid"
np.savez(os.path.join(out_dir, "beauties.npz"), principled=p0, principled_repeat=p1, neural=n0)
summary = {
    "device": device + (" " + gpu if gpu else ""), "res": res, "spp": spp, "box": box,
    "pixels_in_mask_boxes": int(mask.sum()),
    "image_mean": {"principled": float(p0.mean()), "neural": float(n0.mean())},
    "mean_ratio_neural_over_principled": float(n0.mean() / p0.mean()),
    "noise_floor_principled_s0_vs_s1": rel(p0, p1),
    "neural_vs_principled": rel(n0, p0),
    "raw_rgb_neural_vs_principled": raw_rel(n0, p0),
    "raw_rgb_native_repeat": raw_rel(p1, p0),
    "timing_scope": "diagnostic best-of-two wall-time slope, not an equal-quality speedup",
    "ms_per_sample": {"principled": 1e3 * s_p, "neural": 1e3 * s_n},
    "cost_ratio_neural_over_principled": s_n / s_p,
    "seconds_64spp_512spp": {"principled": [lo_p, hi_p], "neural": [lo_n, hi_n]},
}
json.dump(summary, open(os.path.join(out_dir, "render_compare.json"), "w"), indent=1)
print("RENDERCMP", json.dumps(summary))
