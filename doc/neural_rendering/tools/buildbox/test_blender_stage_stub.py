#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Exercises the Blender-side code of render_test.py and mis_consistency.py
against a hand-written fake `bpy`, so control flow, argument handling, exit
codes, file layout and report formatting can be checked on a machine without
Blender. It cannot check that the real bpy API behaves like the fake: that is
what the first run on the workstation is for. The fake mirrors only the
property names verified against intern/cycles/blender/addon/properties.py and
source/blender/makesrna.

    python3 test_blender_stage_stub.py
"""

import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class Ns(types.SimpleNamespace):
    pass


class FakeDevice:
    def __init__(self, type_, name, id_):
        self.type, self.name, self.id, self.use = type_, name, id_, False


class FakePrefs:
    available = ("NONE", "CUDA", "OPTIX")

    def __init__(self):
        self._type = "NONE"
        self.devices = [FakeDevice("CUDA", "NVIDIA GeForce RTX 3090", "CUDA_0"),
                        FakeDevice("CUDA", "NVIDIA GeForce RTX 4070 Ti SUPER", "CUDA_1"),
                        FakeDevice("OPTIX", "NVIDIA GeForce RTX 3090", "OPTIX_0"),
                        FakeDevice("OPTIX", "NVIDIA GeForce RTX 4070 Ti SUPER", "OPTIX_1"),
                        FakeDevice("CPU", "Fake CPU", "CPU")]

    @property
    def compute_device_type(self):
        return self._type

    @compute_device_type.setter
    def compute_device_type(self, v):
        if v not in self.available:
            raise TypeError("enum \"%s\" not found in %s" % (v, self.available))
        self._type = v

    def refresh_devices(self):
        pass


class FakeScene:
    def __init__(self, name="Scene", props=None):
        self.name = name
        self._props = props or {}
        self.cycles = Ns(device="CPU", samples=4096, seed=0, use_animated_seed=True, time_limit=5.0,
                         use_adaptive_sampling=True, use_denoising=True, sample_clamp_direct=3.0,
                         sample_clamp_indirect=10.0, light_sampling_threshold=0.01, use_light_tree=True,
                         min_light_bounces=0, min_transparent_bounces=0, max_bounces=12,
                         transparent_max_bounces=8, direct_light_sampling_type="MULTIPLE_IMPORTANCE_SAMPLING")
        self.render = Ns(engine="BLENDER_EEVEE", resolution_x=64, resolution_y=48, resolution_percentage=50,
                         film_transparent=True, use_compositing=True, use_sequencer=True, filepath="",
                         use_file_extension=False,
                         image_settings=Ns(file_format="PNG", color_mode="RGB", color_depth="8", exr_codec="NONE"))
        self.view_settings = Ns(view_transform="AgX")

    def get(self, key, default=None):
        return self._props.get(key, default)


class FakeImage:
    def __init__(self, arr_top_down):
        h, w, c = arr_top_down.shape
        self.size, self.channels = (w, h), c
        self._flat = np.ascontiguousarray(arr_top_down[::-1]).ravel()  # Blender order: bottom row first
        outer = self

        class Pixels:
            def foreach_get(self, buf):
                buf[:] = outer._flat

            def __iter__(self):
                return iter(outer._flat.tolist())

        self.pixels = Pixels()


class FakeBpy(types.ModuleType):
    """Minimal bpy. `renderer(scene)` returns the H x W x 4 image the 'render' produces."""

    def __init__(self, renderer, scene):
        super().__init__("bpy")
        self.renderer = renderer
        self.prefs = FakePrefs()
        self.scene = scene
        self.render_calls = []
        self.images_removed = 0
        self.app = Ns(version_string="5.x-fake")
        self.context = Ns(scene=scene,
                          preferences=Ns(addons={"cycles": Ns(preferences=self.prefs)},
                                         experimental=Ns(use_cycles_debug=False),
                                         view=Ns(show_developer_ui=False)))
        outer = self

        class Scenes(dict):
            pass

        class Images:
            def load(self, path, check_existing=True):
                assert os.path.exists(path), path
                return FakeImage(outer._last_render)

            def remove(self, img):
                outer.images_removed += 1

        self.data = Ns(scenes=Scenes({scene.name: scene}), images=Images())

        class RenderOps:
            def render(self, write_still=False, scene=None):
                s = outer.scene
                outer.render_calls.append(dict(scene=scene, settings=dict(vars(s.cycles)),
                                               filepath=s.render.filepath, devices=[
                                                   (d.type, d.name) for d in outer.prefs.devices if d.use]))
                outer._last_render = outer.renderer(s)
                assert write_still and s.render.image_settings.file_format == "OPEN_EXR"
                with open(s.render.filepath + ".exr", "wb") as f:
                    f.write(b"fake exr")

        self.ops = Ns(render=RenderOps(), wm=Ns(open_mainfile=lambda **kw: None))


def run_script(fake, module, argv):
    import contextlib
    import io
    sys.modules["bpy"] = fake
    old = sys.argv
    sys.argv = ["blender", "-b", "x.blend", "--python", module.__file__, "--"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return module.main()
    finally:
        run_script.stdout = buf.getvalue()
        sys.argv = old
        sys.modules.pop("bpy", None)


run_script.stdout = ""


def main():
    import render_test as rt
    rng = np.random.default_rng(7)
    sig = rt._synthetic_scene(64, 80)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print("%-4s %s%s" % ("ok" if cond else "FAIL", name, ("  " + detail) if detail else ""))

    def noisy_render(scale=1.0, bias=1.0):
        def f(scene):
            n = rng.normal(0, 1, sig.shape).astype(np.float32)
            rgb = sig * bias + 0.05 * np.sqrt(sig + 0.05) * n * scale
            return np.concatenate([rgb, np.ones(sig.shape[:2] + (1,), np.float32)], axis=2)
        return f

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "o", "out.exr")
        ref = os.path.join(td, "ref.exr")
        np.save(os.path.join(td, "ref.npy"), noisy_render()(None)[:, :, :4])

        # 1. Reference compare passes, scene is configured, files are written.
        scene = FakeScene()
        fake = FakeBpy(noisy_render(), scene)
        rep = os.path.join(td, "rep.md")
        rc = run_script(fake, rt, ["--out", out, "--ref", ref, "--samples", "32", "--device", "OPTIX",
                                   "--res", "80x64", "--report", rep, "--seed", "5"])
        c = fake.render_calls[0]
        check("compare run exits 0", rc == 0, str(rc))
        check("engine set to CYCLES, GPU device", scene.render.engine == "CYCLES" and c["settings"]["device"] == "GPU")
        check("OptiX devices (both cards) enabled, CPU and CUDA entries off",
              sorted(n for t, n in c["devices"] if t == "OPTIX") == sorted(
                  ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4070 Ti SUPER"]) and all(t == "OPTIX" for t, _ in c["devices"]),
              str(c["devices"]))
        check("samples, seed, resolution applied",
              c["settings"]["samples"] == 32 and c["settings"]["seed"] == 5 and
              (scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage) == (80, 64, 100))
        check("bias sources neutralised",
              c["settings"]["sample_clamp_direct"] == 0 and c["settings"]["sample_clamp_indirect"] == 0 and
              not c["settings"]["use_adaptive_sampling"] and not c["settings"]["use_denoising"] and
              c["settings"]["light_sampling_threshold"] == 0 and c["settings"]["time_limit"] == 0 and
              not scene.render.film_transparent and not scene.render.use_compositing)
        check("EXR 32-bit RGBA output, extension appended by Blender",
              scene.render.image_settings.color_depth == "32" and scene.render.use_file_extension and
              c["filepath"] == os.path.join(td, "o", "out"))
        check("npy + json + exr written", all(os.path.exists(os.path.join(td, "o", "out" + e))
                                              for e in (".npy", ".json", ".exr")))
        back = np.load(os.path.join(td, "o", "out.npy"))
        check("npy is top-row-first float32 H x W x 4",
              back.shape == (64, 80, 4) and back.dtype == np.float32 and
              abs(float(back[10:14, 20:24, 0].mean()) - float(fake._last_render[10:14, 20:24, 0].mean())) < 1e-6)
        check("image datablock removed after read", fake.images_removed == 1)
        check("report written with PASS", "PASS" in open(rep).read())

        # 2. Biased render fails the compare with exit code 1.
        fake = FakeBpy(noisy_render(bias=1.08), FakeScene())
        rc = run_script(fake, rt, ["--out", out, "--ref", ref, "--device", "CUDA", "--gpu-name", "4070"])
        check("8 % biased render exits 1", rc == 1, str(rc))
        check("--gpu-name selects only the 4070 Ti Super",
              fake.render_calls[0]["devices"] == [("CUDA", "NVIDIA GeForce RTX 4070 Ti SUPER")],
              str(fake.render_calls[0]["devices"]))

        # 3. CPU device does not touch the preferences.
        fake = FakeBpy(noisy_render(), FakeScene())
        rc = run_script(fake, rt, ["--out", out, "--ref", ref, "--device", "CPU"])
        check("CPU run: device CPU, no GPU enabled", rc == 0 and fake.render_calls[0]["settings"]["device"] == "CPU"
              and fake.render_calls[0]["devices"] == [])

        # 4. Missing device type -> clear error.
        fake = FakeBpy(noisy_render(), FakeScene())
        fake.prefs.available = ("NONE", "CUDA")
        try:
            run_script(fake, rt, ["--out", out, "--device", "OPTIX"])
            check("unavailable OPTIX raises a clear error", False)
        except RuntimeError as e:
            check("unavailable OPTIX raises a clear error", "OPTIX" in str(e) and "Available" in str(e), str(e)[:90])

        # 5. Missing reference: error unless --update-ref.
        missing = os.path.join(td, "new_ref.exr")
        fake = FakeBpy(noisy_render(), FakeScene())
        rc = run_script(fake, rt, ["--out", out, "--ref", missing])
        check("missing ref -> exit 2", rc == 2, str(rc))
        rc = run_script(fake, rt, ["--out", out, "--ref", missing, "--update-ref"])
        check("--update-ref stores the render", rc == 0 and os.path.exists(os.path.join(td, "new_ref.npy")))

        # 6. Furnace: constant 1 passes, scene property overrides expected, bias fails.
        def furnace_render(level):
            def f(scene):
                n = rng.normal(0, 0.03, (64, 64, 3)).astype(np.float32)
                return np.concatenate([level + n, np.ones((64, 64, 1), np.float32)], axis=2)
            return f
        fake = FakeBpy(furnace_render(1.0), FakeScene())
        rc = run_script(fake, rt, ["--out", out, "--furnace"])
        check("furnace passes at 1.0", rc == 0, str(rc))
        fake = FakeBpy(furnace_render(1.0), FakeScene(props={"nr_furnace_expected": 2.0}))
        rc = run_script(fake, rt, ["--out", out, "--furnace"])
        check("furnace uses scene nr_furnace_expected", rc == 1, str(rc))
        fake = FakeBpy(furnace_render(0.9), FakeScene())
        rc = run_script(fake, rt, ["--out", out, "--furnace", "--furnace-tol", "0.2"])
        check("furnace --furnace-tol widens the window", rc == 0)

        # 7. No numpy inside Blender: PFM fallback.
        saved = rt.np
        rt.np = None
        try:
            fake = FakeBpy(noisy_render(), FakeScene())
            rc = run_script(fake, rt, ["--out", os.path.join(td, "nn.exr")])
        finally:
            rt.np = saved
        pf = os.path.join(td, "nn.pfm")
        check("numpy-free fallback writes a PFM that reads back",
              rc == 0 and os.path.exists(pf) and np.allclose(rt.read_pfm(pf), fake._last_render[:, :, :3]))

        # 8. --no-russian-roulette and --keep-settings.
        fake = FakeBpy(noisy_render(), FakeScene())
        run_script(fake, rt, ["--out", out, "--no-russian-roulette", "--keep-settings", "--denoise"])
        s = fake.render_calls[0]["settings"]
        check("--no-russian-roulette sets min bounces = max bounces",
              s["min_light_bounces"] == 12 and s["min_transparent_bounces"] == 8)
        check("--keep-settings leaves clamps alone", s["sample_clamp_indirect"] == 10.0 and s["use_adaptive_sampling"])

        # 9. --list-devices.
        fake = FakeBpy(noisy_render(), FakeScene())
        rc = run_script(fake, rt, ["--list-devices"])
        listing = run_script.stdout
        check("--list-devices prints both cards", rc == 0 and "3090" in listing and "4070" in listing)

        # 10. mis_consistency end to end.
        import mis_consistency as mc
        sys.modules.pop("bpy", None)

        def strategy_render(scene):
            kind = scene.cycles.direct_light_sampling_type
            # Same expectation for all strategies; forward has the largest variance.
            std = {"MULTIPLE_IMPORTANCE_SAMPLING": 0.03, "NEXT_EVENT_ESTIMATION": 0.05, "FORWARD_PATH_TRACING": 0.15}[kind]
            n = rng.normal(0, 1, sig.shape).astype(np.float32)
            rgb = sig + std * np.sqrt(sig + 0.05) * n
            return np.concatenate([rgb, np.ones(sig.shape[:2] + (1,), np.float32)], axis=2)

        outdir = os.path.join(td, "mis")
        fake = FakeBpy(strategy_render, FakeScene())
        rc = run_script(fake, mc, ["--out-dir", outdir, "--samples", "16", "--device", "CPU"])
        kinds = [c["settings"]["direct_light_sampling_type"] for c in fake.render_calls]
        check("mis_consistency: three strategies rendered", kinds == [
            "MULTIPLE_IMPORTANCE_SAMPLING", "NEXT_EVENT_ESTIMATION", "FORWARD_PATH_TRACING"], str(kinds))
        check("mis_consistency: developer flags enabled before rendering",
              fake.context.preferences.experimental.use_cycles_debug and fake.context.preferences.view.show_developer_ui)
        check("mis_consistency: distinct seeds per render", len({c["settings"]["seed"] for c in fake.render_calls}) == 3)
        check("mis_consistency: forward gets more samples",
              fake.render_calls[2]["settings"]["samples"] > fake.render_calls[0]["settings"]["samples"])
        check("mis_consistency: unbiased strategies converge -> exit 0", rc == 0, str(rc))
        check("mis_consistency: report written",
              os.path.exists(os.path.join(outdir, "mis_consistency.md")) and
              "NEXT_EVENT_ESTIMATION" in open(os.path.join(outdir, "mis_consistency.md")).read())

        def biased_nee(scene):
            img = strategy_render(scene)
            if scene.cycles.direct_light_sampling_type == "NEXT_EVENT_ESTIMATION":
                img[:, :, :3] *= 0.9  # e.g. a closure whose pdf and eval disagree
            return img
        fake = FakeBpy(biased_nee, FakeScene())
        rc = run_script(fake, mc, ["--out-dir", outdir, "--samples", "16", "--device", "CPU"])
        check("mis_consistency: a 10 % biased NEE path fails the gate", rc == 1, str(rc))
        check("mis_consistency: report names the failing pair",
              "FAIL" in open(os.path.join(outdir, "mis_consistency.md")).read())

        # Fallback when the developer knob is missing from scene.cycles.
        class NoKnobScene(FakeScene):
            def __init__(self):
                super().__init__()
                del self.cycles.direct_light_sampling_type

        def tree_render(scene):
            return strategy_render(Ns(cycles=Ns(direct_light_sampling_type="MULTIPLE_IMPORTANCE_SAMPLING")))
        fake = FakeBpy(tree_render, NoKnobScene())
        rc = run_script(fake, mc, ["--out-dir", outdir, "--samples", "16", "--device", "CPU"])
        trees = [(c["settings"]["use_light_tree"], c["settings"]["samples"]) for c in fake.render_calls]
        check("mis_consistency fallback: light tree on / off / on at 4x samples",
              rc == 0 and trees == [(True, 16), (False, 16), (True, 64)], "%s rc=%s" % (trees, rc))

    failed = sum(1 for _, ok, _ in results if not ok)
    print("\n%d checks, %d failed" % (len(results), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
