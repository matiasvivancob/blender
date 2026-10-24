"""Pin a Cycles device before blender_dump.py runs. Env: NR_DEVICE=CPU|CUDA|OPTIX, NR_GPU=<name substring>.
Environment, not argv, because blender_dump.py parses everything after `--` strictly."""
import os
import bpy

device = os.environ["NR_DEVICE"]
gpu = os.environ.get("NR_GPU", "")
scene = bpy.context.scene
prefs = bpy.context.preferences.addons["cycles"].preferences
if device == "CPU":
    scene.cycles.device = "CPU"
    enabled = ["CPU"]
else:
    prefs.compute_device_type = device
    prefs.refresh_devices()
    enabled = []
    for d in prefs.devices:
        d.use = d.type == device and gpu in d.name
        if d.use:
            enabled.append(f"{d.type}:{d.name}")
    if len(enabled) != 1:
        raise SystemExit(f"expected exactly one {device} device matching {gpu!r}, got {enabled}")
    scene.cycles.device = "GPU"
print("NR_DEVICE", scene.cycles.device, prefs.compute_device_type, enabled, flush=True)
