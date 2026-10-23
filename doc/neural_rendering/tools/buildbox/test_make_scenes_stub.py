#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Runs make_test_scenes.py against a strict fake `bpy` / `bmesh` / `mathutils`.

The fakes only know the socket names, node properties and RNA properties that
were verified in the tree (node_shader_*.cc, rna_*.cc, cycles properties.py).
Anything else the script touches raises, so typos and drifted names are caught
here. It does NOT prove the real API accepts the calls; the first run of
make_test_scenes.py in Blender does.

    python3 test_make_scenes_stub.py
"""

import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class Strict:
    """Attribute bag that rejects unknown names."""
    ALLOWED = ()

    def __init__(self, **kw):
        object.__setattr__(self, "_vals", {})
        for k, v in kw.items():
            self._vals[k] = v

    def __setattr__(self, k, v):
        if k not in self.ALLOWED:
            raise AttributeError("%s has no verified property %r" % (type(self).__name__, k))
        self._vals[k] = v

    def __getattr__(self, k):
        try:
            return object.__getattribute__(self, "_vals")[k]
        except KeyError:
            raise AttributeError("%s: %r not set" % (type(self).__name__, k))


class Cycles(Strict):
    ALLOWED = ("device", "samples", "seed", "use_animated_seed", "use_adaptive_sampling", "use_denoising",
               "sample_clamp_direct", "sample_clamp_indirect", "light_sampling_threshold", "max_bounces")


class Render(Strict):
    ALLOWED = ("engine", "resolution_x", "resolution_y", "resolution_percentage", "film_transparent")


class ViewSettings(Strict):
    ALLOWED = ("view_transform",)


class Socket:
    def __init__(self, name, is_output):
        self.name, self.is_output, self.default_value = name, is_output, None


class SockDict(dict):
    def __getitem__(self, k):
        if k not in self:
            raise KeyError("socket %r not found (known: %s)" % (k, sorted(self)))
        return dict.__getitem__(self, k)


# Socket names verified in source/blender/nodes/shader/nodes/node_shader_*.cc
PRINCIPLED_IN = ["Base Color", "Metallic", "Roughness", "IOR", "Alpha", "Thin Wall", "Normal", "Diffuse Roughness",
                 "Subsurface Weight", "Subsurface Radius", "Subsurface Scale", "Subsurface IOR",
                 "Subsurface Anisotropy", "Specular IOR Level", "Specular Tint", "Anisotropic",
                 "Anisotropic Rotation", "Tangent", "Transmission Weight", "Coat Weight", "Coat Roughness",
                 "Coat IOR", "Coat Tint", "Coat Normal", "Sheen Weight", "Sheen Roughness", "Sheen Tint",
                 "Emission Color", "Emission Strength"]
NODE_TABLE = {
    "ShaderNodeBsdfPrincipled": (PRINCIPLED_IN, ["BSDF"], ()),
    "ShaderNodeBsdfDiffuse": (["Color", "Roughness", "Normal"], ["BSDF"], ()),
    "ShaderNodeBackground": (["Color", "Strength"], ["Background"], ()),
    "ShaderNodeOutputMaterial": (["Surface", "Volume", "Displacement", "Thickness"], [], ("target",)),
    "ShaderNodeOutputWorld": (["Surface", "Volume"], [], ()),
    "ShaderNodeTexEnvironment": (["Vector"], ["Color"], ("image", "projection", "interpolation")),
    "ShaderNodeTexImage": (["Vector"], ["Color", "Alpha"], ("image", "projection", "interpolation", "extension")),
    "ShaderNodeTexCoord": ([], ["Generated", "Normal", "UV", "Object", "Camera", "Window", "Reflection"], ()),
}


class Node(Strict):
    def __init__(self, type_):
        ins, outs, props = NODE_TABLE[type_]
        super().__init__()
        object.__setattr__(self, "type", type_)
        object.__setattr__(self, "ALLOWED", props)
        object.__setattr__(self, "inputs", SockDict((n, Socket(n, False)) for n in ins))
        object.__setattr__(self, "outputs", SockDict((n, Socket(n, True)) for n in outs))


class Nodes(list):
    def new(self, type_):
        if type_ not in NODE_TABLE:
            raise RuntimeError("unverified node type %r" % type_)
        n = Node(type_)
        self.append(n)
        return n

    def clear(self):
        del self[:]


class Links(list):
    def new(self, out_sock, in_sock):
        assert isinstance(out_sock, Socket) and out_sock.is_output, "link source must be an output socket"
        assert isinstance(in_sock, Socket) and not in_sock.is_output, "link target must be an input socket"
        self.append((out_sock, in_sock))


class Tree:
    def __init__(self):
        self.nodes, self.links = Nodes(), Links()


class IDMat(Strict):
    ALLOWED = ()

    def __init__(self, name):
        super().__init__()
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "node_tree", Tree())
        object.__setattr__(self, "materials", [])


class Coll(list):
    def __init__(self, factory):
        super().__init__()
        self.factory = factory

    def new(self, *a, **kw):
        item = self.factory(*a, **kw)
        self.append(item)
        return item

    def remove(self, item):
        list.remove(self, item)


class Image(Strict):
    ALLOWED = ("filepath_raw", "file_format")

    def __init__(self, name, width, height, alpha=False, float_buffer=False, is_data=False):
        super().__init__()
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "size", (width, height))
        object.__setattr__(self, "is_data", is_data)
        object.__setattr__(self, "float_buffer", float_buffer)
        outer = self
        object.__setattr__(self, "stored", None)

        class Px:
            def foreach_set(self, flat):
                assert len(flat) == width * height * 4, (len(flat), width, height)
                object.__setattr__(outer, "stored", np.array(flat, np.float32).reshape(height, width, 4))
        object.__setattr__(self, "pixels", Px())
        object.__setattr__(self, "packed", False)

    def update(self):
        pass

    def pack(self):
        object.__setattr__(self, "packed", True)

    def save(self):
        pass


class UVLayer:
    def __init__(self):
        self.data = types.SimpleNamespace()
        self.data = Coll(lambda: None)


class Mesh:
    def __init__(self, name):
        self.name, self.materials = name, []
        self.vertices, self.loops, self.polygons = [], [], []
        self.uv_layers = types.SimpleNamespace(new=self._uv_new, layers=[])

    def from_pydata(self, verts, edges, faces, shade_flat=True):
        self.vertices = [types.SimpleNamespace(co=types.SimpleNamespace(x=v[0], y=v[1], z=v[2])) for v in verts]
        for f in faces:
            idx = []
            for vi in f:
                self.loops.append(types.SimpleNamespace(vertex_index=vi))
                idx.append(len(self.loops) - 1)
            self.polygons.append(types.SimpleNamespace(loop_indices=idx))

    def update(self):
        pass

    def _uv_new(self, name="UVMap", do_init=True):
        layer = types.SimpleNamespace(name=name, data=[types.SimpleNamespace(uv=None) for _ in self.loops])
        self.uv_layers.layers.append(layer)
        return layer


class Obj(Strict):
    ALLOWED = ("location", "rotation_euler")

    def __init__(self, name, data):
        super().__init__(location=(0, 0, 0))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "data", data)


class Light(Strict):
    ALLOWED = ("shape", "size", "energy", "color")


class CamData(Strict):
    ALLOWED = ("lens",)


class Scene(Strict):
    ALLOWED = ("name", "camera", "world")

    def __init__(self):
        super().__init__()
        object.__setattr__(self, "cycles", Cycles())
        object.__setattr__(self, "render", Render())
        object.__setattr__(self, "view_settings", ViewSettings())
        object.__setattr__(self, "collection", types.SimpleNamespace(objects=types.SimpleNamespace(link=lambda o: self.linked.append(o))))
        object.__setattr__(self, "linked", [])
        object.__setattr__(self, "props", {"cycles": "ID property group that must survive", "nr_old": 1})

    def keys(self):
        return list(self.props)

    def __setitem__(self, k, v):
        self.props[k] = v

    def __getitem__(self, k):
        return self.props[k]

    def __delitem__(self, k):
        del self.props[k]


def install_fakes(out_files):
    scene = Scene()
    scene.name = "Scene"
    data = types.SimpleNamespace()
    data.objects = Coll(lambda name, d: Obj(name, d))
    data.meshes = Coll(Mesh)
    data.lights = Coll(lambda name, type: _light(name, type))
    data.cameras = Coll(lambda name: CamData())
    data.materials = Coll(IDMat)
    data.worlds = Coll(IDMat)
    data.images = Coll(Image)

    def _light(name, type):
        assert type == "AREA"
        return Light()
    # Image.new takes keyword args in the script; Coll.new forwards them.

    def save_as_mainfile(filepath, compress):
        assert compress is True
        open(filepath, "wb").write(b"fake blend")
        out_files.append((filepath, scene.props.copy(), scene.cycles._vals.copy(), scene.camera, scene.world))

    bpy = types.ModuleType("bpy")
    bpy.data = data
    bpy.context = types.SimpleNamespace(scene=scene)
    bpy.ops = types.SimpleNamespace(wm=types.SimpleNamespace(save_as_mainfile=save_as_mainfile))

    class BM:
        def __init__(self):
            self.faces = [types.SimpleNamespace(smooth=False) for _ in range(4)]

        def to_mesh(self, mesh):
            mesh.vertices = [object()]

        def free(self):
            pass

    bmesh = types.ModuleType("bmesh")
    bmesh.new = BM

    def create_uvsphere(bm, u_segments, v_segments, radius, calc_uvs):
        assert u_segments > 2 and v_segments > 1 and radius > 0 and calc_uvs is True
    bmesh.ops = types.SimpleNamespace(create_uvsphere=create_uvsphere)

    class Vector:
        def __init__(self, v):
            self.v = tuple(v)

        def __sub__(self, o):
            return Vector(a - b for a, b in zip(self.v, o.v))

        def to_track_quat(self, track, up):
            assert (track, up) == ("-Z", "Y")
            return types.SimpleNamespace(to_euler=lambda: (0.0, 0.0, 0.0))

    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = Vector
    sys.modules.update(bpy=bpy, bmesh=bmesh, mathutils=mathutils)
    return bpy, scene


def main():
    results = []

    def check(name, cond, detail=""):
        results.append(bool(cond))
        print("%-4s %s%s" % ("ok" if cond else "FAIL", name, ("  " + detail) if detail else ""))

    files = []
    bpy, scene = install_fakes(files)
    sys.modules.pop("make_test_scenes", None)
    import make_test_scenes as mts

    with tempfile.TemporaryDirectory() as td:
        rc = mts.main(["--out-dir", td])
        check("all builders ran", rc == 0)
        names = sorted(os.path.basename(f[0]) for f in files)
        check("three .blend files written", names == ["furnace.blend", "mis_sphere.blend", "neural_material_probe.blend"],
              str(names))
        by = {os.path.basename(f[0]): f for f in files}

        f_props, f_cycles = by["furnace.blend"][1], by["furnace.blend"][2]
        check("furnace: expected constant stored", f_props.get("nr_furnace_expected") == 1.0)
        check("furnace: previous 'nr_' props dropped, registered group kept",
              "nr_old" not in f_props and "cycles" in f_props)
        check("furnace: unbiased settings", f_cycles["sample_clamp_indirect"] == 0.0 and not f_cycles["use_denoising"]
              and not f_cycles["use_adaptive_sampling"])
        check("mis_sphere: no leftover furnace property", "nr_furnace_expected" not in by["mis_sphere.blend"][1])
        check("probe: probe metadata stored",
              by["neural_material_probe.blend"][1].get("nr_probe_uv_map") == "UVMap"
              and by["neural_material_probe.blend"][1].get("nr_probe_object") == "ProbePlane")
        check("every file has camera and world", all(f[3] is not None and f[4] is not None for f in files))

        # Final state is the probe scene: inspect the material wiring that was built last.
        mat = [m for m in bpy.data.materials if m.name == "LayeredProbe"][0]
        types_ = [n.type for n in mat.node_tree.nodes]
        check("probe material nodes", sorted(types_) == sorted(
            ["ShaderNodeOutputMaterial", "ShaderNodeBsdfPrincipled", "ShaderNodeTexCoord", "ShaderNodeTexImage"]), str(types_))
        links = {(o.name, i.name) for o, i in mat.node_tree.links}
        check("probe wiring: texture -> Roughness, UV -> Vector, BSDF -> Surface",
              {("Color", "Roughness"), ("UV", "Vector"), ("BSDF", "Surface")} <= links, str(links))
        bsdf = [n for n in mat.node_tree.nodes if n.type == "ShaderNodeBsdfPrincipled"][0]
        check("probe: coat 1.0 / coat roughness 0.05 / metallic 1.0",
              bsdf.inputs["Coat Weight"].default_value == 1.0 and bsdf.inputs["Coat Roughness"].default_value == 0.05
              and bsdf.inputs["Metallic"].default_value == 1.0)
        imgs = {i.name: i for i in bpy.data.images}
        rough = imgs["probe_roughness"]
        check("roughness image is non-colour data, float, packed, in [0.08, 0.92]",
              rough.is_data and rough.float_buffer and rough.packed and
              0.079 < rough.stored[:, :, 0].min() and rough.stored[:, :, 0].max() < 0.921
              and np.allclose(rough.stored[:, :, 0], rough.stored[:, :, 1]))
        plane = [m for m in bpy.data.meshes if m.name == "ProbePlane"][0]
        uvs = np.array([d.uv for d in plane.uv_layers.layers[0].data])
        check("probe plane: 16x16 quads, UVs span 0..1", len(plane.polygons) == 256 and uvs.min() == 0.0
              and abs(uvs.max() - 1.0) < 1e-9)

        # mis_sphere and furnace again, separately, to inspect their content.
        files.clear()
        bpy2, scene2 = install_fakes(files)
        mts2 = sys.modules["make_test_scenes"]
        mts2.bpy, mts2.bmesh = bpy2, sys.modules["bmesh"]
        mts2.main(["--out-dir", td, "--only", "mis_sphere"])
        env = [i for i in bpy2.data.images if i.name == "mis_sphere_env"][0]
        e = env.stored[:, :, :3]
        check("env map: float, not data, 256x128, has a bright sun (> 20) and a dim ground (< 0.3)",
              not env.is_data and e.shape == (128, 256, 3) and e.max() > 20.0 and e[:20].mean() < 0.3)
        sun_row, sun_col = np.unravel_index(np.argmax(e.sum(axis=2)), e.shape[:2])
        lat = ((sun_row + 0.5) / 128 - 0.5) * np.pi
        phi = np.pi - 2 * np.pi * (sun_col + 0.5) / 256
        check("sun lands at the requested azimuth/elevation (40, 35 deg)",
              abs(np.degrees(lat) - 35) < 2 and abs(((np.degrees(phi) - 40 + 180) % 360) - 180) < 3,
              "el=%.1f az=%.1f" % (np.degrees(lat), np.degrees(phi)))
        lights = [o for o in bpy2.data.objects if o.name == "KeyArea"][0]
        check("area light is small and finite", lights.data.size == 0.4 and lights.data.shape == "SQUARE")
        gloss = [m for m in bpy2.data.materials if m.name == "Glossy"][0]
        p = [n for n in gloss.node_tree.nodes if n.type == "ShaderNodeBsdfPrincipled"][0]
        check("glossy sphere is not a delta BSDF", 0.2 < p.inputs["Roughness"].default_value < 0.6
              and p.inputs["Metallic"].default_value == 1.0)

    print("\n%d checks, %d failed" % (len(results), results.count(False)))
    return 1 if False in results else 0


if __name__ == "__main__":
    sys.exit(main())
