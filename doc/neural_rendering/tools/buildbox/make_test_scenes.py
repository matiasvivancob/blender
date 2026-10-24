#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Build the three test scenes used by render_test.py, mis_consistency.py and the
later SHADER_EVAL_BSDF dumps. Run inside Blender, arguments after `--`:

    blender -b --factory-startup --python make_test_scenes.py -- --out-dir scenes
    blender -b --factory-startup --python make_test_scenes.py -- --out-dir scenes --only furnace

Writes (the .blend files are generated, not committed; see PROTOCOL.md):

  furnace.blend
      White furnace. World: uniform Background colour (1, 1, 1), strength 1.0.
      One sphere (radius 1) with a Diffuse BSDF of albedo exactly 1.0. Every
      pixel has expectation 1.0 for any energy-conserving integrator (the
      sphere is convex, so a camera ray sees the sphere once and the uniform
      environment through every bounce). Scene custom property
      `nr_furnace_expected` = 1.0, picked up by `render_test.py --furnace`.
      Note on "closed": an enclosing emissive shell would not give a constant
      radiance for albedo 1 (L = e / (1 - albedo) diverges), so the closed
      environment is the uniform world.

  mis_sphere.blend
      Glossy sphere (Principled, metallic 1, roughness 0.35) over a diffuse
      ground plane, lit by one small square area light (0.4 m, finite size so
      both light sampling and BSDF sampling can find it) and by an HDRI-like
      world: a generated 256x128 float equirect image with a sky gradient, a
      dim ground and a small bright sun disc, so environment importance
      sampling and the glossy lobe have to be combined by MIS. Intended for
      mis_consistency.py. No delta lights, no perfectly sharp BSDFs.

  neural_material_probe.blend
      A 2x2 m plane, 16 x 16 quads, with an explicit "UVMap" covering 0..1,
      and a layered Principled material: base colour, metallic 1, roughness
      driven by an image texture (a generated non-colour float image),
      clear coat weight 1.0 with coat roughness 0.05. Plus a camera looking
      straight down, an area light and a grey world so it can also be
      rendered. Custom properties on the scene record the probe object name,
      the UV map name and the material name for the SHADER_EVAL_BSDF dump.

Every socket name used below was checked against the node declarations in
source/blender/nodes/shader/nodes/node_shader_*.cc, every Cycles property
against intern/cycles/blender/addon/properties.py, every other RNA name
against source/blender/makesrna/intern. No `use_nodes` call is made: since
5.0 materials and worlds always use nodes (rna_material.cc, rna_world.cc), and
`bpy.data.materials.new()` / `worlds.new()` create a default node tree which
is cleared and rebuilt here.
"""

import argparse
import math
import os
import sys

import bpy
import bmesh
from mathutils import Vector

try:
    import numpy as np
except ImportError:
    np = None


# -----------------------------------------------------------------------------
# Helpers.

def clear_all():
    """Remove all datablocks of the types used here from the startup file."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for coll in (bpy.data.meshes, bpy.data.lights, bpy.data.cameras, bpy.data.materials,
                 bpy.data.worlds, bpy.data.images):
        for item in list(coll):
            coll.remove(item)


def setup_scene(name, samples, res, max_bounces=12):
    scene = bpy.context.scene
    scene.name = name
    scene.render.engine = "CYCLES"
    c = scene.cycles
    c.device = "CPU"
    c.samples = samples
    c.seed = 0
    c.use_animated_seed = False
    c.use_adaptive_sampling = False
    c.use_denoising = False
    c.sample_clamp_direct = 0.0
    c.sample_clamp_indirect = 0.0
    c.light_sampling_threshold = 0.0
    c.max_bounces = max_bounces
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    # The startup scene is reused for all three files: drop custom properties of the previous one.
    # Only "nr_*" keys: registered properties (scene.cycles ...) are stored as ID properties too.
    for key in [k for k in scene.keys() if k.startswith("nr_")]:
        del scene[key]
    return scene


def new_material(name):
    mat = bpy.data.materials.new(name)
    tree = mat.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    out.target = "ALL"
    return mat, tree, out


def new_world(scene, name):
    world = bpy.data.worlds.new(name)
    tree = world.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputWorld")
    bg = tree.nodes.new("ShaderNodeBackground")
    tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
    scene.world = world
    return world, tree, bg


def aim(obj, target=(0.0, 0.0, 0.0)):
    """Point an object's -Z axis (camera / area light emission) at `target`, +Y up."""
    direction = Vector(target) - Vector(obj.location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(scene, location, target, lens=50.0):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = location
    aim(cam, target)
    scene.camera = cam
    return cam


def add_area_light(scene, name, location, target, size, power_w, shape="SQUARE"):
    light = bpy.data.lights.new(name=name, type="AREA")
    light.shape = shape
    light.size = size
    light.energy = power_w
    light.color = (1.0, 1.0, 1.0)
    obj = bpy.data.objects.new(name, light)
    scene.collection.objects.link(obj)
    obj.location = location
    aim(obj, target)
    return obj


def add_uv_sphere(scene, name, radius, material, segments=96, rings=48):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius, calc_uvs=True)
    for f in bm.faces:
        f.smooth = True
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj


def add_grid_plane(scene, name, half_size, quads, material, uv_scale=1.0):
    """Flat grid in the XY plane at z = 0 with an explicit UV map 0..uv_scale."""
    n = quads
    verts = [(-half_size + 2.0 * half_size * i / n, -half_size + 2.0 * half_size * j / n, 0.0)
             for j in range(n + 1) for i in range(n + 1)]
    faces = [(j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i)
             for j in range(n) for i in range(n)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces, shade_flat=False)
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            uv.data[li].uv = (uv_scale * (co.x + half_size) / (2.0 * half_size),
                              uv_scale * (co.y + half_size) / (2.0 * half_size))
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj


def generated_image(name, width, height, rgb_func, is_data):
    """
    Create a float image from `rgb_func(width, height) -> (height, width, 3)`
    array (row 0 = bottom, Blender's order) and make it part of the .blend.
    Returns (image, needs_external_file). Packing a generated float image is
    attempted first; if that fails the caller passes the image to
    `save_external_images`, which writes an EXR next to the .blend.
    """
    if np is None:
        raise RuntimeError("numpy is required in Blender's Python to generate textures")
    rgb = np.asarray(rgb_func(width, height), dtype=np.float32)
    rgba = np.concatenate([rgb, np.ones((height, width, 1), np.float32)], axis=2)
    img = bpy.data.images.new(name, width, height, alpha=False, float_buffer=True, is_data=is_data)
    img.pixels.foreach_set(rgba.ravel())
    img.update()
    try:
        img.pack()
        return img, False
    except Exception:  # noqa: BLE001  packing generated images is the part I could not test without Blender
        return img, True


def save_external_images(images_needing_file, out_dir):
    for img in images_needing_file:
        path = os.path.join(out_dir, img.name + ".exr")
        img.filepath_raw = path
        img.file_format = "OPEN_EXR"
        img.save()
        print("note: could not pack %s, saved %s next to the .blend" % (img.name, path))


def save(scene_name, out_dir, external):
    os.makedirs(out_dir, exist_ok=True)
    save_external_images(external, out_dir)
    path = os.path.join(out_dir, scene_name + ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
    print("wrote", path)
    return path


# -----------------------------------------------------------------------------
# furnace.blend

def build_furnace(out_dir):
    clear_all()
    scene = setup_scene("furnace", samples=64, res=(128, 128))
    scene["nr_furnace_expected"] = 1.0
    scene["nr_description"] = ("White furnace: uniform world radiance 1.0, one diffuse sphere with albedo 1.0. "
                               "Every pixel must average 1.0.")

    world, tree, bg = new_world(scene, "FurnaceWorld")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 1.0

    mat, mtree, out = new_material("FurnaceDiffuse")
    diffuse = mtree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    diffuse.inputs["Roughness"].default_value = 0.0
    mtree.links.new(diffuse.outputs["BSDF"], out.inputs["Surface"])
    add_uv_sphere(scene, "FurnaceSphere", 1.0, mat)

    add_camera(scene, (0.0, -5.0, 0.0), (0.0, 0.0, 0.0))
    return save("furnace", out_dir, [])


# -----------------------------------------------------------------------------
# mis_sphere.blend

def hdri_like(width, height, sun_az_deg=40.0, sun_el_deg=35.0, sun_radius_deg=3.0, sun_peak=60.0):
    """Equirect (Blender convention, projection.h) sky: gradient + dim ground + sun disc. Row 0 = bottom."""
    v = (np.arange(height) + 0.5) / height
    u = (np.arange(width) + 0.5) / width
    lat = (v - 0.5) * math.pi                       # -pi/2 .. pi/2
    phi = math.pi - 2.0 * math.pi * u               # azimuth, see direction_to_equirectangular()
    cl, sl = np.cos(lat)[:, None], np.sin(lat)[:, None]
    dx, dy, dz = cl * np.cos(phi)[None, :], cl * np.sin(phi)[None, :], np.broadcast_to(sl, (height, width))

    horizon = np.array([1.00, 0.85, 0.70], np.float32)
    zenith = np.array([0.20, 0.40, 1.00], np.float32)
    ground = np.array([0.12, 0.11, 0.10], np.float32)
    t_up = np.clip(dz, 0.0, 1.0) ** 0.5
    sky = horizon[None, None, :] * (1 - t_up[..., None]) + zenith[None, None, :] * t_up[..., None]
    t_dn = np.clip(-dz * 4.0, 0.0, 1.0)
    rgb = sky * (1 - t_dn[..., None]) + ground[None, None, :] * t_dn[..., None]

    az, el = math.radians(sun_az_deg), math.radians(sun_el_deg)
    sx, sy, sz = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
    cosang = np.clip(dx * sx + dy * sy + dz * sz, -1.0, 1.0)
    ang = np.arccos(cosang)
    sun = sun_peak * np.exp(-((ang / math.radians(sun_radius_deg)) ** 2))
    rgb = rgb + sun[..., None] * np.array([1.0, 0.95, 0.85], np.float32)[None, None, :]
    return rgb.astype(np.float32)


def build_mis_sphere(out_dir):
    clear_all()
    scene = setup_scene("mis_sphere", samples=256, res=(160, 120))
    scene["nr_description"] = ("Glossy sphere (roughness 0.35) under a 0.4 m area light and a generated HDRI-like "
                               "world with a sun disc. For mis_consistency.py.")
    external = []

    env_img, needs_file = generated_image("mis_sphere_env", 256, 128, hdri_like, is_data=False)
    if needs_file:
        external.append(env_img)
    world, wtree, bg = new_world(scene, "MisWorld")
    env = wtree.nodes.new("ShaderNodeTexEnvironment")
    env.image = env_img
    wtree.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0

    gloss, gtree, gout = new_material("Glossy")
    p = gtree.nodes.new("ShaderNodeBsdfPrincipled")
    p.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
    p.inputs["Metallic"].default_value = 1.0
    p.inputs["Roughness"].default_value = 0.35
    gtree.links.new(p.outputs["BSDF"], gout.inputs["Surface"])
    sphere = add_uv_sphere(scene, "GlossySphere", 1.0, gloss)
    sphere.location = (0.0, 0.0, 1.0)

    ground, rtree, rout = new_material("Ground")
    q = rtree.nodes.new("ShaderNodeBsdfPrincipled")
    q.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    q.inputs["Roughness"].default_value = 1.0
    rtree.links.new(q.outputs["BSDF"], rout.inputs["Surface"])
    add_grid_plane(scene, "Ground", 6.0, 1, ground)

    # 0.4 m square light, ~3.8 m from the sphere centre: small but not a delta light. The powers
    # of both lights are rough (about the same order as the world), not calibrated.
    add_area_light(scene, "KeyArea", (2.5, -2.0, 3.5), (0.0, 0.0, 1.0), size=0.4, power_w=3000.0)
    add_camera(scene, (0.0, -6.0, 2.2), (0.0, 0.0, 0.9))
    return save("mis_sphere", out_dir, external)


# -----------------------------------------------------------------------------
# neural_material_probe.blend

def roughness_pattern(width, height):
    """Smooth radial gradient + diagonal stripes, values in [0.08, 0.92], grey. Row 0 = bottom."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    u, v = (x + 0.5) / width, (y + 0.5) / height
    radial = np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) / 0.7071
    stripes = 0.5 + 0.5 * np.sin(2.0 * math.pi * 6.0 * (u + v))
    r = 0.08 + 0.84 * np.clip(0.6 * radial + 0.4 * stripes, 0.0, 1.0)
    return np.repeat(r[..., None], 3, axis=2).astype(np.float32)


def build_probe(out_dir):
    clear_all()
    scene = setup_scene("neural_material_probe", samples=64, res=(256, 256))
    external = []

    rough_img, needs_file = generated_image("probe_roughness", 256, 256, roughness_pattern, is_data=True)
    if needs_file:
        external.append(rough_img)

    world, wtree, bg = new_world(scene, "ProbeWorld")
    bg.inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)
    bg.inputs["Strength"].default_value = 1.0

    mat, tree, out = new_material("LayeredProbe")
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.95, 0.64, 0.54, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.05
    texcoord = tree.nodes.new("ShaderNodeTexCoord")
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = rough_img
    tex.interpolation = "Linear"
    tex.extension = "REPEAT"
    tree.links.new(texcoord.outputs["UV"], tex.inputs["Vector"])
    tree.links.new(tex.outputs["Color"], bsdf.inputs["Roughness"])
    tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    plane = add_grid_plane(scene, "ProbePlane", 1.0, 16, mat)
    add_area_light(scene, "ProbeLight", (1.5, -1.5, 2.5), (0.0, 0.0, 0.0), size=1.0, power_w=1000.0)
    add_camera(scene, (0.0, 0.0, 3.0), (0.0, 0.0, 0.0), lens=50.0)

    scene["nr_probe_object"] = plane.name
    scene["nr_probe_uv_map"] = "UVMap"
    scene["nr_probe_material"] = mat.name
    scene["nr_description"] = ("Layered Principled probe: metallic base, roughness texture, clear coat. "
                               "UV map 'UVMap' covers 0..1 on ProbePlane (16x16 quads).")
    return save("neural_material_probe", out_dir, external)


BUILDERS = {
    "furnace": build_furnace,
    "mis_sphere": build_mis_sphere,
    "neural_material_probe": build_probe,
}


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="make_test_scenes.py")
    p.add_argument("--out-dir", default="scenes")
    p.add_argument("--only", choices=sorted(BUILDERS), action="append", help="build only this scene (repeatable)")
    args = p.parse_known_args(argv)[0]
    out_dir = os.path.abspath(args.out_dir)
    for name in (args.only or list(BUILDERS)):
        BUILDERS[name](out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
