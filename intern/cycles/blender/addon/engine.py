# SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def _configure_argument_parser():
    import argparse
    # No help because it conflicts with general Python scripts argument parsing
    parser = argparse.ArgumentParser(description="Cycles Addon argument parser",
                                     add_help=False)
    parser.add_argument("--cycles-print-stats",
                        help="Print rendering statistics to stderr",
                        action='store_true')
    parser.add_argument("--cycles-device",
                        help="Set the device to use for Cycles, overriding user preferences and the scene setting."
                             "Valid options are 'CPU', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI', or 'METAL'."
                             "Additionally, you can append '+CPU' to any GPU type for hybrid rendering.",
                        default=None)
    return parser


def _parse_command_line():
    import sys

    argv = sys.argv
    if "--" not in argv:
        return

    parser = _configure_argument_parser()
    args, _ = parser.parse_known_args(argv[argv.index("--") + 1:])

    if args.cycles_print_stats:
        import _cycles
        _cycles.enable_print_stats()

    if args.cycles_device:
        import _cycles
        if not _cycles.set_device_override(args.cycles_device):
            sys.exit(1)


def init():
    import bpy
    import _cycles
    import os.path

    path = os.path.dirname(__file__)
    user_path = os.path.dirname(os.path.abspath(bpy.utils.user_resource('CONFIG', path='')))

    _cycles.init(path, user_path, bpy.app.background)
    _parse_command_line()


def exit():
    import _cycles
    _cycles.exit()


def create(engine, data, region=None, v3d=None, rv3d=None, preview_osl=False):
    import _cycles
    import bpy

    data = data.as_pointer()
    prefs = bpy.context.preferences.as_pointer()
    screen = 0
    if region:
        screen = region.id_data.as_pointer()
        region = region.as_pointer()
    if v3d:
        screen = screen or v3d.id_data.as_pointer()
        v3d = v3d.as_pointer()
    if rv3d:
        screen = screen or rv3d.id_data.as_pointer()
        rv3d = rv3d.as_pointer()

    engine.session = _cycles.create(engine.as_pointer(), prefs, data, screen, region, v3d, rv3d, preview_osl)


def free(engine):
    if hasattr(engine, "session"):
        if engine.session:
            import _cycles
            _cycles.free(engine.session)
        del engine.session


def render(engine, depsgraph):
    import _cycles
    if hasattr(engine, "session"):
        _cycles.render(engine.session, depsgraph.as_pointer())


def render_frame_finish(engine):
    if not engine.session:
        return

    import _cycles
    _cycles.render_frame_finish(engine.session)


def draw(engine, depsgraph, space_image):
    if not engine.session:
        return

    depsgraph_ptr = depsgraph.as_pointer()
    space_image_ptr = space_image.as_pointer()
    screen_ptr = space_image.id_data.as_pointer()

    import _cycles
    _cycles.draw(engine.session, depsgraph_ptr, screen_ptr, space_image_ptr)


def bake(engine, depsgraph, obj, pass_type, pass_filter, width, height):
    import _cycles
    session = getattr(engine, "session", None)
    if session is not None:
        _cycles.bake(engine.session, depsgraph.as_pointer(), obj.as_pointer(), pass_type, pass_filter, width, height)


def eval_bsdf(engine, depsgraph, obj, prims, uvs, wi, wo, flags, out):
    """
    Evaluate the BSDF of the surface material of a mesh object at given points and directions,
    to generate training data for neural materials. Nothing is rendered. Uses the device of the
    Cycles session (CPU, CUDA or OptiX; the other devices are untested).

    All arrays are contiguous NumPy arrays with n records (a point plus a pair of directions):

    obj:   The evaluated object, e.g. `depsgraph.objects[name]` or `ob.evaluated_get(depsgraph)`.
    prims: int32 (n,)   Triangle index in the evaluated mesh of `obj` (as seen by Cycles,
                        meaning after modifiers and triangulation).
    uvs:   float32 (n, 2)  Barycentric coordinates (u, v) in the triangle, same convention as
                        Cycles hit points: P = (1 - u - v) * v0 + u * v1 + v * v2.
    wi:    float32 (n, 3)  Direction from the surface towards the viewer (Cycles ShaderData.wi,
                        the negated direction of travel of the camera ray).
    wo:    float32 (n, 3)  Direction from the surface towards the light (the sampled direction in
                        path tracing). Both directions point away from the surface.
    flags: int          Bit 0 set: wi and wo are in the local shading frame (z along the shading
                        normal, x along dPdu projected onto the tangent plane, y = z cross x).
                        Otherwise they are in world space.
    out:   float32 (n, 8) Filled with, per record:
                        [0:3] BSDF value as RGB, includes the cosine term and closure weights,
                        [3]   PDF of sampling wo,
                        [4:7] albedo as RGB (diffuse + glossy + transmission),
                        [7]   flags as bit mask: 1 = no BSDF closure at this point, 2 = wi is
                              on the backfacing side of the geometric normal, 4 = invalid input.

    Usage in a RenderEngine subclass, running in background mode (untested, there is no
    Cycles build with this function yet), for example with
    `blender -b scene.blend -E BSDF_DUMP -f 1`. Subclassing CyclesRender gives a session that is
    created and reset like for a regular final render, with the Cycles settings of the scene:

        import numpy as np
        import bpy
        from cycles import CyclesRender, engine as cycles_engine

        class BsdfDumpEngine(CyclesRender):
            bl_idname = "BSDF_DUMP"
            bl_label = "BSDF Dump"

            def render(self, depsgraph):
                obj = depsgraph.objects["Cube"]
                n = 4096
                prims = np.zeros(n, dtype=np.int32)
                uvs = np.full((n, 2), 1.0 / 3.0, dtype=np.float32)
                wi = np.tile(np.float32([0.0, 0.0, 1.0]), (n, 1))
                wo = np.random.normal(size=(n, 3)).astype(np.float32)
                wo[:, 2] = np.abs(wo[:, 2])
                wo /= np.linalg.norm(wo, axis=1, keepdims=True)
                out = np.zeros((n, 8), dtype=np.float32)
                cycles_engine.eval_bsdf(self, depsgraph, obj, prims, uvs, wi, wo, 1, out)
                np.save("/tmp/bsdf_out.npy", out)

        bpy.utils.register_class(BsdfDumpEngine)

    The object is synced as a bake target, so use a session dedicated to this and do not render
    with it afterwards.
    Raises RuntimeError on failure.
    """
    import _cycles
    import numpy as np

    session = getattr(engine, "session", None)
    if session is None:
        raise RuntimeError("Cycles session is not created")

    n = len(prims)
    prims = np.ascontiguousarray(prims, dtype=np.int32).reshape(n)
    uvs = np.ascontiguousarray(uvs, dtype=np.float32).reshape(n, 2)
    wi = np.ascontiguousarray(wi, dtype=np.float32).reshape(n, 3)
    wo = np.ascontiguousarray(wo, dtype=np.float32).reshape(n, 3)
    if (out.dtype != np.float32 or not out.flags["C_CONTIGUOUS"] or not out.flags["WRITEABLE"] or
            out.shape != (n, 8)):
        raise ValueError("out must be a writeable C-contiguous float32 array of shape (n, 8)")

    _cycles.eval_bsdf(
        session,
        depsgraph.as_pointer(),
        obj.as_pointer(),
        n,
        prims.ctypes.data,
        uvs.ctypes.data,
        wi.ctypes.data,
        wo.ctypes.data,
        int(flags),
        out.ctypes.data,
    )


def reset(engine, data, depsgraph):
    import _cycles
    import bpy

    prefs = bpy.context.preferences
    if prefs.experimental.use_cycles_debug and prefs.view.show_developer_ui:
        _cycles.debug_flags_update(depsgraph.scene.as_pointer())
    else:
        _cycles.debug_flags_reset()

    data = data.as_pointer()
    depsgraph = depsgraph.as_pointer()
    _cycles.reset(engine.session, data, depsgraph)


def sync(engine, depsgraph, data):
    import _cycles
    _cycles.sync(engine.session, depsgraph.as_pointer())


def view_draw(engine, depsgraph, region, v3d, rv3d):
    import _cycles
    depsgraph = depsgraph.as_pointer()
    v3d = v3d.as_pointer()
    rv3d = rv3d.as_pointer()

    # draw render image
    _cycles.view_draw(engine.session, depsgraph, v3d, rv3d)


def view_pause(engine, pause):
    import _cycles
    if engine.session:
        _cycles.view_pause(engine.session, pause)


def available_devices():
    import _cycles
    return _cycles.available_devices()


def with_osl():
    import _cycles
    return _cycles.with_osl


def osl_version():
    import _cycles
    return _cycles.osl_version


def with_path_guiding():
    import _cycles
    return _cycles.with_path_guiding


def system_info():
    import _cycles
    return _cycles.system_info()


def list_render_passes(scene, srl):
    import _cycles
    from bpy.app.translations import pgettext_n as n_

    crl = srl.cycles

    # Combined pass.
    yield ("Combined", "RGBA", 'COLOR')

    # Keep alignment for readability.
    # autopep8: off

    # Data passes.
    if srl.use_pass_z:                     yield (n_("Depth"),           "Z",    'VALUE')
    if srl.use_pass_mist:                  yield (n_("Mist"),            "Z",    'VALUE')
    if srl.use_pass_position:              yield (n_("Position"),        "XYZ",  'VECTOR')
    if srl.use_pass_normal:                yield (n_("Normal"),          "XYZ",  'VECTOR')
    if srl.use_pass_vector:                yield (n_("Vector"),          "XYZW", 'VECTOR')
    if srl.use_pass_uv:                    yield (n_("UV"),              "UVA",  'VECTOR')
    if srl.use_pass_object_index:          yield (n_("Object Index"),    "X",    'VALUE')
    if srl.use_pass_material_index:        yield (n_("Material Index"),  "X",    'VALUE')

    # Light passes.
    if srl.use_pass_diffuse_direct:        yield (n_("Diffuse Direct"),        "RGB",  'COLOR')
    if srl.use_pass_diffuse_indirect:      yield (n_("Diffuse Indirect"),      "RGB",  'COLOR')
    if srl.use_pass_diffuse_color:         yield (n_("Diffuse Color"),         "RGB",  'COLOR')
    if srl.use_pass_glossy_direct:         yield (n_("Glossy Direct"),         "RGB",  'COLOR')
    if srl.use_pass_glossy_indirect:       yield (n_("Glossy Indirect"),       "RGB",  'COLOR')
    if srl.use_pass_glossy_color:          yield (n_("Glossy Color"),          "RGB",  'COLOR')
    if srl.use_pass_transmission_direct:   yield (n_("Transmission Direct"),   "RGB",  'COLOR')
    if srl.use_pass_transmission_indirect: yield (n_("Transmission Indirect"), "RGB",  'COLOR')
    if srl.use_pass_transmission_color:    yield (n_("Transmission Color"),    "RGB",  'COLOR')
    if crl.use_pass_volume_direct:         yield (n_("Volume Direct"),         "RGB",  'COLOR')
    if crl.use_pass_volume_indirect:       yield (n_("Volume Indirect"),       "RGB",  'COLOR')
    if crl.use_pass_volume_scatter:        yield (n_("Volume Scatter"),        "RGB",  'COLOR')
    if crl.use_pass_volume_transmit:       yield (n_("Volume Transmit"),       "RGB",  'COLOR')
    if crl.use_pass_volume_majorant:       yield (n_("Volume Majorant"),       "Z",    'VALUE')
    if srl.use_pass_emit:                  yield (n_("Emission"),              "RGB",  'COLOR')
    if srl.use_pass_environment:           yield (n_("Environment"),           "RGB",  'COLOR')
    if srl.use_pass_ambient_occlusion:     yield (n_("Ambient Occlusion"),     "RGB",  'COLOR')
    if crl.use_pass_shadow_catcher:        yield (n_("Shadow Catcher"),        "RGB",  'COLOR')
    # autopep8: on

    # Debug passes.
    if crl.pass_debug_sample_count:
        yield (n_("Debug Sample Count"), "X", 'VALUE')
    if crl.pass_render_time:
        # Only yield the pass if rendering on CPU
        if scene.cycles.device == 'CPU':
            yield (n_("Render Time"), "X", "VALUE")

    # Cryptomatte passes.
    # NOTE: Name channels are lowercase RGBA so that compression rules check in OpenEXR DWA code
    # uses lossless compression. Reportedly this naming is the only one which works good from the
    # interoperability point of view. Using XYZW naming is not portable.
    crypto_depth = (min(16, srl.pass_cryptomatte_depth) + 1) // 2
    if srl.use_pass_cryptomatte_object:
        for i in range(0, crypto_depth):
            yield ("CryptoObject" + '{:02d}'.format(i), "rgba", 'COLOR')
    if srl.use_pass_cryptomatte_material:
        for i in range(0, crypto_depth):
            yield ("CryptoMaterial" + '{:02d}'.format(i), "rgba", 'COLOR')
    if srl.use_pass_cryptomatte_asset:
        for i in range(0, crypto_depth):
            yield ("CryptoAsset" + '{:02d}'.format(i), "rgba", 'COLOR')

    # Denoising passes.
    if scene.cycles.use_denoising and crl.use_denoising:
        yield (n_("Noisy Image"), "RGBA", 'COLOR')
        if crl.use_pass_shadow_catcher:
            yield (n_("Noisy Shadow Catcher"), "RGB", 'COLOR')
    if crl.denoising_store_passes:
        yield (n_("Denoising Albedo"), "RGB", 'COLOR')
        yield (n_("Denoising Specular Albedo"), "RGB", 'COLOR')
        yield (n_("Denoising Normal"), "XYZ", 'VECTOR')
        yield (n_("Denoising Roughness"), "X", 'VALUE')
        yield (n_("Denoising Depth"), "Z", 'VALUE')
        if crl.denoising_store_motion_passes:
            if not (scene.render.use_motion_blur and srl.use_motion_blur):
                yield (n_("Denoising Backward Motion"), "XYZ", 'VECTOR')
                yield (n_("Denoising Specular Motion"), "XYZ", 'VECTOR')

    # Custom AOV passes.
    for aov in srl.aovs:
        if not aov.is_valid:
            continue
        if aov.type == 'VALUE':
            yield (aov.name, "X", 'VALUE')
        else:
            yield (aov.name, "RGBA", 'COLOR')

    # Light groups.
    for lightgroup in srl.lightgroups:
        yield ("Combined_%s" % lightgroup.name, "RGB", 'COLOR')

    # Path guiding debug passes.
    if _cycles.with_debug and scene.cycles.use_guiding:
        yield (n_("Guiding Color"), "RGB", 'COLOR')
        yield (n_("Guiding Probability"), "X", 'VALUE')
        yield (n_("Guiding Average Roughness"), "X", 'VALUE')


def register_passes(engine, scene, view_layer):
    for name, channelids, channeltype in list_render_passes(scene, view_layer):
        engine.register_pass(scene, view_layer, name, len(channelids), channelids, channeltype)
