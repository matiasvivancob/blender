#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Exercises the Windows code paths of check_env.py on any machine.

`sys.platform` is patched to "win32", the environment variables Windows would
have (ProgramFiles, ProgramData, SystemRoot, CUDA_PATH, ...) point into a
temporary fake file system, and every external command (`nvidia-smi`, `nvcc`,
`vswhere`, `cmake`, `gh`, ...) is answered by a table instead of being run.
It checks decisions (status, hints, the "Missing" list), not that the real
tools print what the table says; that needs the workstation.

    python3 test_check_env_windows.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check_env  # noqa: E402

NVIDIA_SMI_CSV = (
    "0, NVIDIA GeForce RTX 3090, 572.83, 8.6, 24576 MiB\n"
    "1, NVIDIA GeForce RTX 4070 Ti SUPER, 572.83, 8.9, 16376 MiB\n"
)


def touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


class FakeWindows:
    """Builds a fake Windows file system and patches the probes."""

    def __init__(self, root, repo_name="Blender-RND", good=True):
        self.root = root
        self.good = good
        j = lambda *p: os.path.join(root, *p)  # noqa: E731
        self.pf = j("Program Files")
        self.pf86 = j("Program Files (x86)")
        self.progdata = j("ProgramData")
        self.sysroot = j("Windows")
        self.repo = j("D", "Github", repo_name, "blender")
        self.build = j("D", "Github", repo_name, "build_windows_neural")
        self.vs22 = j("Program Files", "Microsoft Visual Studio", "2022", "Community")
        self.vs19 = j("Program Files", "Microsoft Visual Studio", "2019", "BuildTools")
        self.cuda = j("Program Files", "NVIDIA GPU Computing Toolkit", "CUDA", "v12.9")
        os.makedirs(self.repo, exist_ok=True)
        self.populate()

    def populate(self):
        touch(os.path.join(self.pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe"))
        touch(os.path.join(self.sysroot, "System32", "nvidia-smi.exe"))
        os.makedirs(os.path.join(self.pf86, "Windows Kits", "10", "Include", "10.0.22621.0"), exist_ok=True)
        os.makedirs(os.path.join(self.vs19, "VC", "Tools", "MSVC", "14.29.30133"), exist_ok=True)
        if self.good:
            os.makedirs(os.path.join(self.vs22, "VC", "Tools", "MSVC", "14.44.35217"), exist_ok=True)
            os.makedirs(os.path.join(self.vs22, "VC", "Tools", "MSVC", "14.43.34808"), exist_ok=True)
            touch(os.path.join(self.cuda, "bin", "nvcc.exe"))
            touch(os.path.join(self.progdata, "NVIDIA Corporation", "OptiX SDK 9.0.0", "include", "optix.h"),
                  "#define OPTIX_VERSION 90000\n")
            touch(os.path.join(self.vs22, "Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", "Ninja",
                               "ninja.exe"))
            touch(os.path.join(self.repo, "lib", "windows_x64", ".git"), "gitdir: ../../.git/modules/lib/windows_x64\n")
            touch(os.path.join(self.repo, "lib", "windows_x64", "python", "313", "bin", "python.exe"))
            touch(os.path.join(self.repo, "lib", "windows_x64", "zlib", "lib", "libz_st.lib"))
            touch(os.path.join(self.repo, "tests", "files", "render", "bsdf", "diffuse.blend"), "BLENDER-v500")
            touch(os.path.join(self.build, "bin", "blender.exe"))
            touch(os.path.join(self.build, "bin", "5.2", "scripts", "addons_core", "cycles", "lib",
                               "kernel_sm_86.cubin.zst"))
            touch(os.path.join(self.build, "bin", "5.2", "scripts", "addons_core", "cycles", "lib",
                               "kernel_sm_89.cubin.zst"))
            touch(os.path.join(self.build, "bin", "5.2", "scripts", "addons_core", "cycles", "lib",
                               "kernel_optix.ptx.zst"))
            touch(os.path.join(self.build, "CMakeCache.txt"),
                  "CMAKE_GENERATOR:INTERNAL=Ninja\nCMAKE_BUILD_TYPE:STRING=Release\n"
                  "CYCLES_CUDA_BINARIES_ARCH:STRING=sm_86;sm_89\nWITH_CYCLES_DEVICE_CUDA:BOOL=ON\n"
                  "WITH_CYCLES_DEVICE_OPTIX:BOOL=ON\nWITH_CYCLES_CUDA_BINARIES:BOOL=ON\nWITH_CYCLES_DEVICE_HIP:BOOL=OFF\n"
                  "WITH_CYCLES_DEVICE_ONEAPI:BOOL=OFF\nWITH_CYCLES_DEBUG:BOOL=ON\nWITH_STRICT_BUILD_OPTIONS:BOOL=ON\n"
                  "WITH_WINDOWS_SCCACHE:BOOL=ON\nOPTIX_ROOT_DIR:PATH=C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0\n"
                  "OPTIX_INCLUDE_DIR:PATH=C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include\n")
        else:
            touch(os.path.join(self.repo, "tests", "files", "render", "bsdf", "diffuse.blend"),
                  "version https://git-lfs.github.com/spec/v1\n")
            os.makedirs(os.path.join(self.repo, "lib", "windows_x64", "zlib"), exist_ok=True)  # no .git, no python

    # -- vswhere answers ------------------------------------------------------
    def installs(self):
        if self.good:
            return [
                {"installationPath": self.vs22, "installationVersion": "17.14.36301.6",
                 "displayName": "Visual Studio Community 2022", "isPrerelease": False},
                {"installationPath": self.vs19, "installationVersion": "16.11.40",
                 "displayName": "Visual Studio Build Tools 2019", "isPrerelease": False},
            ]
        # Broken: VS 2022 17.9 without the C++ workload, plus 2019 with C++ tools.
        return [
            {"installationPath": self.vs22, "installationVersion": "17.9.34728.123",
             "displayName": "Visual Studio Community 2022", "isPrerelease": False},
            {"installationPath": self.vs19, "installationVersion": "16.11.40",
             "displayName": "Visual Studio Build Tools 2019", "isPrerelease": False},
        ]

    def fake_run(self, cmd, timeout=20, env=None):
        exe = os.path.basename(cmd[0]).lower()
        args = cmd[1:]
        if exe == "cl.exe":
            folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cmd[0])))))
            v = check_env.parse_version(folder)
            return 2, "Microsoft C/C++ Compiler Version %s for x64" % check_env.vstr((v[0] + 5, *v[1:]))
        if exe == "vswhere.exe":
            allv = self.installs()
            if check_env.VS_COMPONENT_VC in args:
                sel = allv if self.good else [i for i in allv if i["installationPath"] == self.vs19]
            elif check_env.VS_WORKLOAD_CPP in args:
                sel = [i for i in allv if i["installationPath"] == self.vs22] if self.good else []
            else:
                sel = allv
            return 0, json.dumps(sel)
        if exe == "nvidia-smi.exe":
            if "compute_cap" in args[0]:
                return 0, NVIDIA_SMI_CSV.strip()
            return 0, "\n".join(l.rsplit(",", 2)[0] + "," + l.rsplit(",", 1)[1] for l in NVIDIA_SMI_CSV.strip().splitlines())
        if exe == "nvcc.exe":
            return 0, "Cuda compilation tools, release 12.9, V12.9.86"
        if exe == "cmake.exe":
            return 0, "cmake version 3.31.6"
        if exe == "ninja.exe":
            return 0, "1.12.1"
        if exe == "sccache.exe":
            return 0, "sccache 0.10.0"
        if exe == "git.exe" and args[:1] == ["--version"]:
            return 0, "git version 2.50.0.windows.1"
        if exe == "git.exe":
            return 0, "abcdef0" if "--short" in args else "claude/amazing-planck-xvnke7"
        if exe == "git-lfs.exe":
            return 0, "git-lfs/3.7.0 (GitHub; windows amd64; go 1.24)"
        if exe == "gh.exe":
            if args[:1] == ["auth"]:
                if self.good:
                    return 0, "github.com\n  - Logged in to github.com account someone (keyring)\n  - Token: gho_************"
                return 1, "You are not logged into any GitHub hosts. To log in, run: gh auth login"
            return 0, "gh version 2.80.0 (2026-01-01)"
        if exe == "blender.exe":
            return 0, "Blender 5.2.0 Alpha"
        return None, "not found"

    def which_table(self):
        pf = lambda *p: os.path.join(self.pf, *p)  # noqa: E731
        t = {"git": pf("Git", "cmd", "git.exe"), "git-lfs": pf("Git LFS", "git-lfs.exe"),
             "cmake": pf("CMake", "bin", "cmake.exe"), "gh": pf("GitHub CLI", "gh.exe")}
        if self.good:
            t["ninja"] = pf("ninja", "ninja.exe")
            t["sccache"] = pf("sccache", "sccache.exe")
            t["nvcc"] = os.path.join(self.cuda, "bin", "nvcc.exe")
        else:
            del t["cmake"]
        return t

    def env(self):
        e = {"ProgramFiles": self.pf, "ProgramFiles(x86)": self.pf86, "ProgramData": self.progdata,
             "SystemRoot": self.sysroot, "HOME": self.root, "USERPROFILE": self.root}
        if self.good:
            e["CUDA_PATH"] = self.cuda
            e["CUDA_PATH_V12_9"] = self.cuda
        return e

    @contextlib.contextmanager
    def patched(self, torch_cuda="12.8", torch_ok=True, tcnn=True):
        table = self.which_table()
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "2.9.0+cu128"
        fake_torch.version = types.SimpleNamespace(cuda=torch_cuda)
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: torch_ok, device_count=lambda: 2,
            get_device_capability=lambda i: (8, 6) if i == 0 else (8, 9),
            get_device_name=lambda i: "NVIDIA GeForce RTX 3090" if i == 0 else "NVIDIA GeForce RTX 4070 Ti SUPER")
        mods = {"torch": fake_torch}
        mods["tinycudann"] = types.ModuleType("tinycudann") if tcnn else None  # None => ImportError
        real_glob = check_env.glob.glob
        with mock.patch.object(check_env.glob, "glob", lambda pattern: [p for p in real_glob(pattern)
                                                                      if os.path.abspath(p).startswith(self.root + os.sep)]), \
                mock.patch.object(sys, "platform", "win32"), \
                mock.patch.dict(os.environ, self.env(), clear=True), \
                mock.patch.object(check_env, "REPO", self.repo), \
                mock.patch.object(check_env.shutil, "which", lambda n, *a, **k: table.get(n)), \
                mock.patch.object(check_env, "run", self.fake_run), \
                mock.patch.dict(sys.modules, mods):
            yield


def rows_by_item(rep):
    return {(r[0], r[1]): r for r in rep.rows}


class WindowsGood(unittest.TestCase):
    def test_serviced_compiler_version_not_directory_name(self):
        with mock.patch.object(check_env.glob, "glob", return_value=["VS/VC/Tools/MSVC/14.44.35207"]):
            for output, expected in [("Compiler Version 19.44.35228 for x64", (14, 44, 35228)),
                                     ("not found", None)]:
                with self.subTest(output=output), mock.patch.object(check_env, "run", return_value=(2, output)):
                    self.assertEqual(check_env.newest_msvc_toolset("VS"), expected)

    def test_everything_present(self):
        with tempfile.TemporaryDirectory() as td:
            fw = FakeWindows(td)
            with fw.patched():
                self.assertTrue(check_env.is_windows())
                rep = check_env.collect(fw.build, skip_python_gpu=False)
                text = check_env.render_markdown(rep, "test")
            R = rows_by_item(rep)
            self.assertEqual(rep.problems(), [], [p[:4] for p in rep.problems()])
            self.assertEqual(R[("GPU", "expected 3090 (sm_86)")][2], "OK")
            self.assertEqual(R[("GPU", "expected 4070 Ti SUPER (sm_89)")][2], "OK")
            self.assertEqual(R[("CUDA", "nvcc 12.9")][2], "OK")
            self.assertIn("OptiX SDK 9.0.0", R[("OptiX", "SDK 9.0.0")][3])
            # OPTIX_ROOT_DIR unset, SDK in ProgramData: informational, not a warning
            self.assertEqual(R[("OptiX", "OPTIX_ROOT_DIR")][2], "info")
            self.assertEqual(R[("Visual Studio", "usable VS 2022 (>= 17.14.14 with C++ tools)")][2], "OK")
            vs22 = [r for r in rep.rows if r[0] == "Visual Studio" and "2022 17.14" in r[1]][0]
            self.assertEqual(vs22[2], "OK")
            self.assertIn("C++ workload (Desktop development with C++): yes", vs22[3])
            self.assertIn("14.44.35217", vs22[3])
            vs19 = [r for r in rep.rows if r[0] == "Visual Studio" and "2019" in r[1]][0]
            self.assertNotEqual(vs19[2], "OK")
            self.assertEqual(R[("Tools", "ninja")][2], "OK")
            self.assertEqual(R[("Tools", "sccache")][2], "OK")
            self.assertEqual(R[("Tools", "gh auth status")][2], "OK")
            self.assertNotIn("gho_", text)  # never copy token lines
            self.assertEqual(R[("Python", "torch")][2], "OK")
            self.assertEqual(R[("Python", "torch CUDA vs nvcc")][2], "WARN")  # 12.8 vs 12.9: minor differs
            self.assertFalse(R[("Python", "torch CUDA vs nvcc")][4])           # ...and is not a required failure
            self.assertEqual(R[("Python", "tiny-cuda-nn")][2], "OK")
            self.assertEqual(R[("Repo", "precompiled libs lib/windows_x64")][2], "OK")
            self.assertEqual(R[("Repo", "render test data (LFS)")][2], "OK")
            self.assertEqual(R[("Build", "blender executable")][2], "OK")
            self.assertEqual(R[("Build", "WITH_WINDOWS_SCCACHE")][2], "OK")
            self.assertEqual(R[("Build", "installed CUDA kernels")][2], "OK")
            self.assertNotIn(("Build", "WITH_COMPILER_CCACHE"), R)
            self.assertEqual(R[("System", "source path has no spaces")][2], "OK")
            self.assertTrue(text.startswith("[result] test"))
            self.assertIn("**Missing**", text)
            self.assertIn("Nothing. Every required item is present", text)

    def test_nvidia_smi_only_in_driver_store(self):
        with tempfile.TemporaryDirectory() as td:
            fw = FakeWindows(td)
            os.remove(os.path.join(fw.sysroot, "System32", "nvidia-smi.exe"))
            touch(os.path.join(fw.sysroot, "System32", "DriverStore", "FileRepository", "nv_dispi.inf_amd64_abc",
                               "nvidia-smi.exe"))
            with fw.patched():
                self.assertIn("DriverStore", check_env.find_nvidia_smi())
                rep = check_env.collect(fw.build, skip_python_gpu=True)
            R = rows_by_item(rep)
            self.assertEqual(R[("GPU", "driver")][2], "OK")
            self.assertEqual(R[("GPU", "nvidia-smi location")][2], "info")

    def test_blender_exe_in_release_dir(self):
        with tempfile.TemporaryDirectory() as td:
            fw = FakeWindows(td)
            os.remove(os.path.join(fw.build, "bin", "blender.exe"))
            touch(os.path.join(fw.build, "bin", "Release", "blender.exe"))
            with fw.patched():
                rep = check_env.collect(fw.build, skip_python_gpu=True)
            R = rows_by_item(rep)
            self.assertIn("Release", R[("Build", "blender executable")][3])

    def test_cuda_only_via_programfiles_not_path(self):
        with tempfile.TemporaryDirectory() as td:
            fw = FakeWindows(td)
            with fw.patched():
                del os.environ["CUDA_PATH"]
                del os.environ["CUDA_PATH_V12_9"]
                check_env.shutil.which = lambda n, *a, **k: None if n == "nvcc" else fw.which_table().get(n)
                rep = check_env.collect(fw.build, skip_python_gpu=True)
            R = rows_by_item(rep)
            self.assertEqual(R[("CUDA", "nvcc 12.9")][2], "OK")  # found by scanning %ProgramFiles%\NVIDIA GPU Computing Toolkit
            self.assertEqual(R[("CUDA", "nvcc on PATH")][2], "WARN")


class WindowsBroken(unittest.TestCase):
    def test_missing_list(self):
        with tempfile.TemporaryDirectory() as td:
            fw = FakeWindows(td, repo_name="Blender RND", good=False)
            with fw.patched(torch_cuda="13.0", torch_ok=False, tcnn=False):
                rep = check_env.collect(fw.build, skip_python_gpu=False)
                text = check_env.render_markdown(rep, "test")
                with contextlib.redirect_stdout(io.StringIO()):
                    strict = check_env.main(["--build-dir", fw.build, "--skip-python-gpu", "--strict"])
            R = rows_by_item(rep)
            self.assertEqual(R[("System", "source path has no spaces")][2], "MISSING")
            self.assertEqual(R[("CUDA", "nvcc")][2], "MISSING")
            self.assertEqual(R[("OptiX", "SDK")][2], "MISSING")
            self.assertEqual(R[("Visual Studio", "usable VS 2022 (>= 17.14.14 with C++ tools)")][2], "MISSING")
            self.assertEqual(R[("Tools", "cmake")][2], "MISSING")
            self.assertEqual(R[("Tools", "ninja")][2], "MISSING")
            self.assertEqual(R[("Tools", "gh auth status")][2], "MISSING")
            self.assertEqual(R[("Repo", "precompiled libs lib/windows_x64")][2], "WARN")
            self.assertEqual(R[("Repo", "render test data (LFS)")][2], "MISSING")
            self.assertEqual(R[("Python", "tiny-cuda-nn")][2], "MISSING")
            self.assertEqual(R[("Python", "torch")][2], "WARN")
            self.assertEqual(R[("Tools", "sccache")][2], "WARN")
            self.assertEqual(R[("Build", "build dir")][2], "MISSING")
            self.assertEqual(strict, 1)
            tail = text.split("**Missing**")[1]
            for needle in ("Blender-RND", "cuda-toolkit-archive", "OptiX SDK 9.0.0", "Desktop development with C++",
                           "Kitware.CMake", "Ninja-build.Ninja", "gh auth login", "make_update.py --no-blender",
                           "TCNN_CUDA_ARCHITECTURES=86,89", "Mozilla.sccache", "git lfs pull"):
                self.assertIn(needle, tail, needle)
            self.assertIn("Required (install these", tail)
            self.assertIn("Optional", tail)
            self.assertNotIn("Nothing.", tail)

    def test_old_toolset_is_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            fw = FakeWindows(td)
            # newest toolset 14.43 only (VS 2022 17.13): below 19.44.35216
            import shutil as sh
            sh.rmtree(os.path.join(fw.vs22, "VC", "Tools", "MSVC", "14.44.35217"))
            with fw.patched():
                rep = check_env.collect(fw.build, skip_python_gpu=True)
            R = rows_by_item(rep)
            self.assertEqual(R[("Visual Studio", "usable VS 2022 (>= 17.14.14 with C++ tools)")][2], "MISSING")
            vs22 = [r for r in rep.rows if r[0] == "Visual Studio" and "2022 17.14" in r[1]][0]
            self.assertEqual(vs22[2], "WARN")
            self.assertIn("17.14.14", vs22[3])


class LinuxStillWorks(unittest.TestCase):
    def test_linux_flow(self):
        if sys.platform == "win32":
            self.skipTest("Linux path")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = check_env.main(["--skip-python-gpu", "--build-dir", "/nonexistent-build-dir"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertTrue(out.startswith("[result] "))
        self.assertIn("lib/linux_x64", out)
        self.assertIn("**Missing**", out)
        self.assertNotIn("Visual Studio", out)
        self.assertNotIn("sccache", out.replace("WITH_WINDOWS_SCCACHE", ""))

    def test_json_and_out_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "r.json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                check_env.main(["--skip-python-gpu", "--json", "--out", path, "--build-dir", td])
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertTrue(all("hint" in d for d in data))


if __name__ == "__main__":
    unittest.main(verbosity=2)
