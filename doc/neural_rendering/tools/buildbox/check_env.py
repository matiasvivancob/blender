#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Report the state of the neural-rendering build box as a Markdown block, ready
to paste into a `[result]` PR comment (see doc/neural_rendering/PROTOCOL.md).

Standard library only, runs with any Python 3.8+ outside Blender, on Linux and
on Windows 10/11:

    python3 check_env.py                      # prints Markdown (Windows: py -3 check_env.py)
    python3 check_env.py --build-dir ~/build_linux_neural
    python3 check_env.py --json               # machine-readable
    python3 check_env.py --strict             # exit 1 if a required item is missing
    python3 check_env.py --out report.md      # UTF-8 file (PowerShell `>` writes UTF-16)

The report ends with a "Missing" list that gives, for every missing or
suspicious item, where to get it and which version.

Run it with the same Python that is used for training, so the torch and
tiny-cuda-nn import checks mean something.

Checks (R = required for the Cycles build, T = required for training):
  R  NVIDIA GPUs and driver (nvidia-smi; on Windows also System32 and the
     driver store), both target cards present
  R  nvcc version, CUDA toolkits in common locations ($CUDA_PATH and
     "%ProgramFiles%\\NVIDIA GPU Computing Toolkit\\CUDA\\v*" on Windows)
  R  OptiX SDK: $OPTIX_ROOT_DIR and common install paths ("%ProgramData%\\NVIDIA
     Corporation\\OptiX SDK *" on Windows); version from include/optix.h
     `#define OPTIX_VERSION` decoded as FindOptiX.cmake does
     (major = v / 10000, minor = (v % 10000) / 100, micro = v % 100);
     intern/cycles/CMakeLists.txt requires >= 8.0.0
  R  cmake (>= 3.21, CMakeLists.txt), ninja, git, git-lfs, gh (+ `gh auth status`)
  R  Linux: host compiler. Windows: Visual Studio found with vswhere (version,
     "Desktop development with C++" workload, VC.Tools component, MSVC toolset
     >= 14.44.35216 = VS 2022 17.14.14, which platform_win32.cmake demands)
  .  ccache (Linux) / sccache (Windows) for build speed, Blender precompiled libs
     (lib/linux_x64, lib/windows_x64 with its .git entry), render test data
     (tests/files/render, Git LFS pointers vs real files)
  T  python, numpy, torch (+ CUDA, per-GPU capability, match with nvcc),
     tiny-cuda-nn
  .  Blender build directory: bin/blender (bin/Release/blender.exe or
     bin/blender.exe on Windows), version, and the CMake cache values that
     matter (arch list, OptiX, debug, ccache/sccache), plus which Cycles
     kernels were installed
"""

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *([os.pardir] * 4)))

EXPECTED_GPUS = (("3090", "8.6", "sm_86"), ("4070 Ti SUPER", "8.9", "sm_89"))
OPTIX_MIN = (8, 0, 0)       # intern/cycles/CMakeLists.txt: find_package(OptiX 8.0.0)
CMAKE_MIN = (3, 21)         # CMakeLists.txt: cmake_minimum_required(VERSION 3.21)

VS_MIN = (17, 14)           # build_files/cmake/platform/platform_win32.cmake: MSVC 2022 17.14.14 or newer
MSVC_TOOLSET_MIN = (14, 44, 35216)  # same file: CMAKE_CXX_COMPILER_VERSION >= 19.44.35216 (cl 19.x == toolset 14.x)
VS_COMPONENT_VC = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"   # what detect_msvc_vswhere.cmd -requires
VS_WORKLOAD_CPP = "Microsoft.VisualStudio.Workload.NativeDesktop"       # "Desktop development with C++"

OK, WARN, MISSING, INFO = "OK", "WARN", "MISSING", "info"


def is_windows():
    """Single switch for every platform decision (tests monkeypatch it or sys.platform)."""
    return sys.platform == "win32"


class Report:
    def __init__(self):
        self.rows = []  # (section, item, status, detail, required, hint)

    def add(self, section, item, status, detail="", required=False, hint=""):
        self.rows.append((section, item, status, detail, required, hint))

    def problems(self):
        return [r for r in self.rows if r[4] and r[2] in (MISSING, WARN)]

    def optional_gaps(self):
        """Non-required items that are absent and have an install hint."""
        return [r for r in self.rows if not r[4] and r[2] in (MISSING, WARN) and r[5]]


def run(cmd, timeout=20, env=None):
    """Return (returncode, combined output) or (None, reason) when the command cannot run."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                           universal_newlines=True, env=env, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout.strip()
    except FileNotFoundError:
        return None, "not found"
    except subprocess.TimeoutExpired:
        return None, "timed out after %ds" % timeout
    except OSError as e:
        return None, str(e)


def first_line(text):
    return text.splitlines()[0].strip() if text else ""


def parse_version(text, parts=3):
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text or "")
    if not m:
        return None
    v = tuple(int(x) if x is not None else 0 for x in m.groups())
    return v[:parts]


def vstr(v):
    return ".".join(str(x) for x in v)




def envv(name):
    return os.environ.get(name) or ""


def unique_paths(paths):
    seen, out = set(), []
    for d in paths:
        if not d:
            continue
        rp = os.path.normcase(os.path.realpath(d))
        if rp not in seen:
            seen.add(rp)
            out.append(d)
    return out


# -----------------------------------------------------------------------------
# Install hints, printed in the final "Missing" list. Versions come from the tree:
# build_files/buildbot/config/blender_version.cmake (CUDA 12.9.1 / 13.4.1, OptiX 9.0.0),
# platform_win32.cmake (MSVC 2022 17.14.14), CMakeLists.txt (cmake 3.21).

def hints():
    if is_windows():
        return {
            "driver": "Install the current NVIDIA driver (Game Ready or Studio) from https://www.nvidia.com/Download/index.aspx "
                      "and reboot; `nvidia-smi` is installed with it (C:\\Windows\\System32\\nvidia-smi.exe). Check the OptiX "
                      "SDK release notes for the minimum driver version.",
            "cuda": "CUDA Toolkit 12.9.1 (the version Blender's buildbot pins; 12.8 is fine too) from "
                    "https://developer.nvidia.com/cuda-toolkit-archive, custom install with Visual Studio Integration "
                    "unticked if VS is not installed yet. Install the version that matches the CUDA of your PyTorch wheel "
                    "(torch.version.cuda). Installer sets CUDA_PATH and adds %CUDA_PATH%\\bin to PATH; open a new shell afterwards.",
            "optix": "OptiX SDK 9.0.0 (buildbot pin; Cycles needs >= 8.0.0) from https://developer.nvidia.com/optix "
                     "(NVIDIA developer login), run the installer and keep the default folder "
                     "C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 9.0.0 (neural_dev.cmake finds it there); "
                     "otherwise setx OPTIX_ROOT_DIR \"<folder that contains include\\optix.h>\".",
            "vs": "Visual Studio 2022 17.14.14 or newer (Community is free; Build Tools also work): "
                  "https://visualstudio.microsoft.com/downloads/ , workload \"Desktop development with C++\" "
                  "(includes MSVC v143 x64/x86 build tools and a Windows 10/11 SDK). Update through the Visual Studio "
                  "Installer if it is older than 17.14.14.",
            "winsdk": "Windows 10/11 SDK: tick it in the Visual Studio Installer under the C++ desktop workload.",
            "cmake": "CMake >= 3.21: winget install Kitware.CMake (or https://cmake.org/download/, choose \"add to PATH\"). "
                     "make.bat needs cmake.exe on PATH BEFORE it sets up Visual Studio.",
            "ninja": "Ninja: winget install Ninja-build.Ninja (or pip install ninja; the Visual Studio component "
                     "\"C++ CMake tools for Windows\" also ships one, visible only inside a VS developer shell).",
            "sccache": "sccache (C/C++ compile cache, Ninja builds only): winget install Mozilla.sccache "
                       "(or scoop install sccache). Optional.",
            "git": "Git for Windows: winget install Git.Git (https://git-scm.com/download/win).",
            "git-lfs": "Git LFS (needed for lib/windows_x64 and the test data): bundled with Git for Windows, otherwise "
                       "winget install GitHub.GitLFS, then `git lfs install`.",
            "gh": "GitHub CLI: winget install GitHub.cli, then `gh auth login` (used by tools\\comms\\post.ps1 and watch.ps1).",
            "gh-auth": "Run `gh auth login` (GitHub.com, HTTPS, browser) and check `gh auth status`.",
            "python": "Python 3.12 or 3.13 (64-bit) from https://www.python.org/downloads/windows/ or "
                      "winget install Python.Python.3.12; create a venv for training: py -3.12 -m venv .venv",
            "numpy": "pip install numpy (inside the training venv).",
            "torch": "pip install torch --index-url https://download.pytorch.org/whl/cu128 -- use the cuXYZ tag that matches "
                     "the CUDA toolkit you installed (see https://pytorch.org/get-started/locally/); the nvcc used for "
                     "tiny-cuda-nn must have the same major version as torch.version.cuda.",
            "tcnn": "Needs MSVC and nvcc on PATH and torch already installed. From an x64 Native Tools / VS developer shell with "
                    "the venv active: git clone --recursive https://github.com/NVlabs/tiny-cuda-nn ; cd tiny-cuda-nn\\bindings\\torch ; "
                    "set TCNN_CUDA_ARCHITECTURES=86,89 ; pip install --no-build-isolation . (details: buildbox README).",
            "libs": "python build_files\\utils\\make_update.py --no-blender   (fetches lib\\windows_x64 through Git LFS; several GB; "
                    "avoids the git pull --rebase that `make.bat update` does).",
            "lfs-data": "git lfs pull --include=\"tests/files/render/**\"",
            "path-spaces": "Move or rename the work folder to a path without spaces, e.g. D:\\Github\\Blender-RND; "
                           "build_files\\windows\\check_spaces_in_path.cmd makes make.bat exit when the source path has one.",
            "build": "Not built yet: follow the Windows section of tools/buildbox/README.md.",
        }
    return {
        "driver": "Install the NVIDIA driver from your distribution or https://www.nvidia.com/Download/index.aspx.",
        "cuda": "CUDA Toolkit 12.9.1 or 13.x (12.8 is what Blender's Rocky 9 setup uses) from "
                "https://developer.nvidia.com/cuda-toolkit-archive ; put its bin directory on PATH or pass "
                "-DCUDA_TOOLKIT_ROOT_DIR.",
        "optix": "OptiX SDK 9.0.0 (Cycles needs >= 8.0.0) from https://developer.nvidia.com/optix ; unpack and "
                 "export OPTIX_ROOT_DIR=<folder that contains include/optix.h>.",
        "cmake": "CMake >= 3.21 from your package manager or https://cmake.org/download/.",
        "ninja": "ninja-build from your package manager.",
        "ccache": "ccache from your package manager; then `ccache --max-size=50G`. Optional.",
        "git": "git from your package manager.",
        "git-lfs": "git-lfs from your package manager, then `git lfs install`.",
        "gh": "GitHub CLI https://cli.github.com/ then `gh auth login`.",
        "gh-auth": "Run `gh auth login` and check `gh auth status`.",
        "python": "python3 >= 3.8 (3.12 or 3.13 for training with torch).",
        "numpy": "pip install numpy (inside the training venv).",
        "torch": "pip install torch (CUDA wheel matching your toolkit, https://pytorch.org/get-started/locally/).",
        "tcnn": "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch with "
                "TCNN_CUDA_ARCHITECTURES=86,89 and an nvcc whose major version matches torch.version.cuda.",
        "libs": "python3 build_files/utils/make_update.py --no-blender",
        "lfs-data": "git lfs pull --include=\"tests/files/render/**\"",
        "build": "Not built yet: follow tools/buildbox/README.md.",
    }


# -----------------------------------------------------------------------------

def total_ram_gb():
    """Physical RAM in GB, or None."""
    if is_windows():
        try:
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]
            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            return st.ullTotalPhys / 1024.0 ** 3
        except (AttributeError, OSError, ImportError):
            return None
    try:
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
        return kb / 1024.0 / 1024.0
    except (OSError, AttributeError, ValueError):
        return None


def windows_long_paths_enabled():
    """True/False from the registry, None when unknown."""
    try:
        import winreg  # type: ignore[import-not-found]
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return bool(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
    except (ImportError, OSError):
        return None


def check_system(rep, hint):
    rep.add("System", "OS", INFO, "%s %s, %s" % (platform.system(), platform.release(), platform.machine()))
    if not (is_windows() or platform.system() == "Linux"):
        rep.add("System", "platform", WARN, "this setup targets Linux and Windows", True)
    rep.add("System", "CPU threads", INFO, str(os.cpu_count()))
    gb = total_ram_gb()
    if gb is not None:
        rep.add("System", "RAM", OK if gb >= 30 else WARN,
                "%.0f GB%s" % (gb, "" if gb >= 30 else " (nvcc for several archs and the final link are memory hungry)"))
    try:
        du = shutil.disk_usage(REPO)
        rep.add("System", "free disk at repo", OK if du.free > 60e9 else WARN, "%.0f GB" % (du.free / 1e9))
    except OSError:
        pass
    if is_windows():
        # build_files/windows/check_spaces_in_path.cmd tests BLENDER_DIR (= the directory of make.bat) for a space and
        # exits with "spaces ... not supported". The build directory is not tested, but configure_ninja.cmd /
        # build_ninja.cmd use it unquoted, so keep both free of spaces.
        spaces = " " in REPO
        rep.add("System", "source path has no spaces", MISSING if spaces else OK,
                ("%s contains a space; make.bat will refuse to run" % REPO) if spaces else REPO, True,
                hint["path-spaces"] if spaces else "")
        rep.add("System", "source path length", OK if len(REPO) <= 40 else WARN,
                "%d characters%s" % (len(REPO), "" if len(REPO) <= 40 else
                                      " (keep it short: long paths in the build tree can hit the 260 character limit)"))
        lp = windows_long_paths_enabled()
        if lp is not None:
            rep.add("System", "Windows long paths", OK if lp else WARN,
                    "LongPathsEnabled=%d%s" % (lp, "" if lp else " (optional: registry HKLM\\SYSTEM\\CurrentControlSet\\"
                                               "Control\\FileSystem\\LongPathsEnabled=1 and `git config --global core.longpaths true`)"))


def find_nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if is_windows():
        sysroot = envv("SystemRoot") or envv("windir") or r"C:\Windows"
        pf = envv("ProgramFiles") or r"C:\Program Files"
        cands = [os.path.join(sysroot, "System32", "nvidia-smi.exe"),
                 os.path.join(pf, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")]
        cands += sorted(glob.glob(os.path.join(sysroot, "System32", "DriverStore", "FileRepository", "nv*",
                                               "nvidia-smi.exe")))
        for c in cands:
            if os.path.isfile(c):
                return c
    return None


def check_gpus(rep, hint):
    exe = find_nvidia_smi()
    if not exe:
        rep.add("GPU", "nvidia-smi", MISSING, "not on PATH, System32 or the driver store (NVIDIA driver not installed?)", True,
                hint["driver"])
        return []
    if not shutil.which("nvidia-smi"):
        rep.add("GPU", "nvidia-smi location", INFO, "%s (not on PATH)" % exe)
    q = "index,name,driver_version,compute_cap,memory.total"
    rc, out = run([exe, "--query-gpu=" + q, "--format=csv,noheader"])
    if rc != 0:  # older drivers do not know compute_cap
        q = "index,name,driver_version,memory.total"
        rc, out = run([exe, "--query-gpu=" + q, "--format=csv,noheader"])
    if rc != 0:
        rep.add("GPU", "nvidia-smi", MISSING, "failed: %s" % first_line(out), True, hint["driver"])
        return []
    fields = q.split(",")
    gpus = []
    for line in out.splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == len(fields):
            gpus.append(dict(zip(fields, vals)))
    drivers = sorted({g["driver_version"] for g in gpus})
    rep.add("GPU", "driver", OK if drivers else MISSING, ", ".join(drivers), True, "" if drivers else hint["driver"])
    for g in gpus:
        cc = g.get("compute_cap", "?")
        rep.add("GPU", "GPU %s" % g["index"], INFO, "%s, compute %s, %s" % (g["name"], cc, g["memory.total"]))
    names = " | ".join(g["name"] for g in gpus).lower()
    for key, cc, sm in EXPECTED_GPUS:
        found = key.lower() in names
        rep.add("GPU", "expected %s (%s)" % (key, sm), OK if found else MISSING,
                "present" if found else "not listed by nvidia-smi", True,
                "" if found else "Card not visible to the driver: reseat/power it, or check Device Manager / the driver install.")
    return gpus


def nvcc_candidates():
    cands = []
    w = shutil.which("nvcc")
    if w:
        cands.append(w)
    exe = "nvcc.exe" if is_windows() else "nvcc"
    for env in ("CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_ROOT_DIR"):
        if os.environ.get(env):
            cands.append(os.path.join(os.environ[env], "bin", exe))
    if is_windows():
        # The CUDA installer sets CUDA_PATH (latest) and CUDA_PATH_V<major>_<minor> for each toolkit.
        for k, v in sorted(os.environ.items()):
            if re.match(r"CUDA_PATH_V\d+_\d+$", k, re.I) and v:
                cands.append(os.path.join(v, "bin", exe))
        for pf in unique_paths([envv("ProgramFiles"), r"C:\Program Files"]):
            cands += sorted(glob.glob(os.path.join(pf, "NVIDIA GPU Computing Toolkit", "CUDA", "v*", "bin", exe)))
    else:
        cands += sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))
        cands += sorted(glob.glob("/opt/cuda*/bin/nvcc"))
    seen, out = set(), []
    for c in cands:
        rp = os.path.normcase(os.path.realpath(c))
        if rp not in seen and os.path.exists(c):
            seen.add(rp)
            out.append(c)
    return out


def check_cuda(rep, hint):
    found = []
    for nvcc in nvcc_candidates():
        rc, out = run([nvcc, "--version"])
        m = re.search(r"release (\d+\.\d+), V([\d.]+)", out or "")
        found.append((nvcc, m.group(1) if m else "?", m.group(2) if m else "?"))
    if is_windows():
        rep.add("CUDA", "%CUDA_PATH%", INFO, envv("CUDA_PATH") or "unset (the CUDA installer sets it; FindCUDA reads it)")
    if not found:
        where = "PATH, %CUDA_PATH% or %ProgramFiles%\\NVIDIA GPU Computing Toolkit\\CUDA" if is_windows() else \
            "PATH, $CUDA_HOME or /usr/local/cuda*"
        rep.add("CUDA", "nvcc", MISSING, "no nvcc on %s; set CUDA_TOOLKIT_ROOT_DIR for CMake" % where, True, hint["cuda"])
        return None
    default = shutil.which("nvcc")
    for nvcc, rel, full in found:
        major = rel.split(".")[0]
        tested = major in ("12", "13")  # kernel/device/cuda/CMakeLists.txt warns for other majors
        tag = " (PATH default)" if default and os.path.realpath(nvcc) == os.path.realpath(default) else ""
        rep.add("CUDA", "nvcc %s" % rel, OK if tested else WARN,
                "%s, V%s%s%s" % (nvcc, full, tag, "" if tested else "; Cycles only tested with CUDA 12 and 13"), True,
                "" if tested else hint["cuda"])
    if is_windows():
        if not default:
            rep.add("CUDA", "nvcc on PATH", WARN, "nvcc.exe is not on PATH: CMake finds it through CUDA_PATH, but "
                    "tiny-cuda-nn's setup.py runs `nvcc --version` and needs it on PATH", False,
                    "Add %CUDA_PATH%\\bin to PATH (System Properties, Environment Variables) and open a new shell.")
        rep.add("CUDA", "nvcc host compiler", INFO,
                "cl.exe of the Visual Studio shell/toolset that runs cmake (CMake's CUDA_HOST_COMPILER = CMAKE_C_COMPILER); "
                "nvcc has its own MSVC version window, see the Visual Studio rows")
    else:
        gcc = shutil.which("gcc")
        if gcc:
            rc, out = run([gcc, "-dumpfullversion"])
            v = parse_version(out)
            if v:
                rep.add("CUDA", "gcc (nvcc host compiler)", INFO,
                        "%s; if nvcc rejects it pass -DCUDA_HOST_COMPILER=/usr/bin/gcc-<n>" % vstr(v))
    # Release of the nvcc that CMake/torch will see first: the PATH one, else the first candidate.
    if default:
        for n, r, f in found:
            if os.path.realpath(n) == os.path.realpath(default):
                return r if r != "?" else None
    return found[0][1] if found[0][1] != "?" else None


def optix_dirs():
    dirs = []
    if os.environ.get("OPTIX_ROOT_DIR"):
        dirs.append(os.environ["OPTIX_ROOT_DIR"])
    home = os.path.expanduser("~")
    if is_windows():
        # FindOptiX.cmake has no default location; this is where the NVIDIA installer puts the SDK.
        for base in unique_paths([envv("ProgramData"), r"C:\ProgramData"]):
            dirs += sorted(glob.glob(os.path.join(base, "NVIDIA Corporation", "OptiX SDK *")))
        for pat in (os.path.join(home, "NVIDIA-OptiX-SDK-*"), os.path.join(home, "Downloads", "NVIDIA-OptiX-SDK-*"),
                    os.path.join(home, "sdk", "*ptiX*")):
            dirs += sorted(glob.glob(pat))
    else:
        for pat in ("/opt/NVIDIA-OptiX-SDK-*", "/opt/nvidia/optix*", "/opt/optix*", "/usr/local/NVIDIA-OptiX-SDK-*",
                    "/usr/local/optix*", "/usr/share/nvidia-optix*",
                    home + "/NVIDIA-OptiX-SDK-*", home + "/optix*", home + "/sdk/*ptiX*", home + "/sdk/optix*",
                    home + "/opt/*ptiX*", home + "/Downloads/NVIDIA-OptiX-SDK-*"):
            dirs += sorted(glob.glob(pat))
    return unique_paths(dirs)


def read_optix_version(root):
    header = os.path.join(root, "include", "optix.h")
    try:
        with open(header, errors="replace") as f:
            for line in f:
                m = re.match(r"\s*#\s*define\s+OPTIX_VERSION\s+(\d+)", line)
                if m:
                    v = int(m.group(1))
                    return (v // 10000, (v % 10000) // 100, v % 100)
    except OSError:
        return None
    return None


def check_optix(rep, hint):
    env = os.environ.get("OPTIX_ROOT_DIR")
    hits = []
    for d in optix_dirs():
        ver = read_optix_version(d)
        if ver is not None:
            hits.append((d, ver))
    if env:
        rep.add("OptiX", "OPTIX_ROOT_DIR", INFO, env)
    elif is_windows() and hits:
        rep.add("OptiX", "OPTIX_ROOT_DIR", INFO,
                "unset; neural_dev.cmake picks the newest SDK under %ProgramData%\\NVIDIA Corporation (FindOptiX.cmake "
                "alone would not find it)")
    else:
        rep.add("OptiX", "OPTIX_ROOT_DIR", WARN, "unset (CMake then needs -DOPTIX_ROOT_DIR=...)")
    if not hits:
        listed = optix_dirs()
        rep.add("OptiX", "SDK", MISSING,
                "no include/optix.h with OPTIX_VERSION found%s" % ((" in " + ", ".join(listed)) if listed else ""), True,
                hint["optix"])
        return
    for d, ver in hits:
        ok = ver >= OPTIX_MIN
        rep.add("OptiX", "SDK %s" % vstr(ver), OK if ok else WARN,
                "%s%s" % (d, "" if ok else " (Cycles needs >= %s)" % vstr(OPTIX_MIN)), True, "" if ok else hint["optix"])
    if env:
        v = read_optix_version(env)
        rep.add("OptiX", "OPTIX_ROOT_DIR is usable", OK if v else MISSING,
                ("optix.h version %s" % vstr(v)) if v else "%s/include/optix.h missing or has no OPTIX_VERSION" % env,
                True, "" if v else hint["optix"])


# -----------------------------------------------------------------------------
# Visual Studio (Windows). make.bat -> detect_msvc_vswhere.cmd runs
#   vswhere -latest -version [17.0,17.99) -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64
# and then vcvarsall.bat; platform_win32.cmake rejects cl older than 19.44.35216.

def find_vswhere():
    for pf in unique_paths([envv("ProgramFiles(x86)"), envv("ProgramFiles"), r"C:\Program Files (x86)"]):
        p = os.path.join(pf, "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if os.path.isfile(p):
            return p
    return shutil.which("vswhere")


def vswhere_query(vswhere, extra=()):
    rc, out = run([vswhere, "-all", "-products", "*", "-prerelease", "-utf8", "-format", "json"] + list(extra))
    if rc != 0:
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def newest_msvc_toolset(install_path):
    best = None
    for d in glob.glob(os.path.join(install_path, "VC", "Tools", "MSVC", "*")):
        # Servicing updates change cl.exe without renaming the toolset directory.
        _, output = run([os.path.join(d, "bin", "Hostx64", "x64", "cl.exe")])
        compiler = parse_version(output)
        v = (compiler[0] - 5, *compiler[1:]) if compiler else None
        if v and (best is None or v > best):
            best = v
    return best


def vs_bundled_tool(installs, relpath):
    for inst in installs:
        p = os.path.join(inst.get("installationPath", ""), *relpath)
        if os.path.isfile(p):
            return p
    return None


def check_visual_studio(rep, hint):
    """Returns the list of installs found (used to locate the VS-bundled cmake/ninja)."""
    vswhere = find_vswhere()
    if not vswhere:
        rep.add("Visual Studio", "vswhere", MISSING, "Microsoft Visual Studio\\Installer\\vswhere.exe not found "
                "(make.bat prints \"Visual Studio not found\")", True, hint["vs"])
        return []
    installs = vswhere_query(vswhere)
    with_vc = {i.get("installationPath") for i in vswhere_query(vswhere, ["-requires", VS_COMPONENT_VC])}
    with_wl = {i.get("installationPath") for i in vswhere_query(vswhere, ["-requires", VS_WORKLOAD_CPP])}
    if not installs:
        rep.add("Visual Studio", "installations", MISSING, "vswhere lists no Visual Studio or Build Tools", True, hint["vs"])
        return []
    usable17, usable18 = [], []
    for inst in installs:
        path = inst.get("installationPath", "")
        ver = parse_version(inst.get("installationVersion", "")) or (0, 0, 0)
        name = inst.get("displayName") or inst.get("productId") or "Visual Studio"
        vc, wl = path in with_vc, path in with_wl
        detail = "%s; C++ workload (Desktop development with C++): %s; VC.Tools x64 component: %s%s" % (
            path, "yes" if wl else "NO", "yes" if vc else "NO", "; preview" if inst.get("isPrerelease") else "")
        status = INFO
        toolset = newest_msvc_toolset(path) if vc else None
        if toolset:
            detail += "; newest MSVC toolset %s" % vstr(toolset)
        if vc and ver[0] == 17 and ver[:2] >= VS_MIN:
            if toolset is None or toolset < MSVC_TOOLSET_MIN:
                status = WARN
                detail += " (cl.exe unavailable or older than the required compiler 19.44.35216; VS 2022 17.14.14 or newer required)"
            else:
                status = OK
                usable17.append(inst)
        elif vc and ver[0] >= 18:
            status = OK
            usable18.append(inst)
        elif ver[0] == 17 and vc:
            status = WARN
            detail += " (17.%d is older than 17.14: update)" % ver[1]
        elif not vc:
            status = WARN
            detail += " (install the \"Desktop development with C++\" workload)"
        rep.add("Visual Studio", "%s %s" % (name, inst.get("installationVersion", "?")), status, detail)
    if usable17:
        rep.add("Visual Studio", "usable VS 2022 (>= 17.14.14 with C++ tools)", OK, "make.bat switch: 2022")
    elif usable18:
        rep.add("Visual Studio", "usable VS 2022 (>= 17.14.14 with C++ tools)", WARN,
                "only Visual Studio 2026 with C++ tools found (make.bat switch 2026; needs CMake >= 4.2); whether the CUDA "
                "toolkit's nvcc accepts its cl.exe is unverified, VS 2022 17.14 is the safe choice", True, hint["vs"])
    else:
        rep.add("Visual Studio", "usable VS 2022 (>= 17.14.14 with C++ tools)", MISSING,
                "no Visual Studio 2022 17.14+ with the VC.Tools x64 component", True, hint["vs"])
    sdk_dirs = []
    for pf in unique_paths([envv("ProgramFiles(x86)"), r"C:\Program Files (x86)"]):
        sdk_dirs += glob.glob(os.path.join(pf, "Windows Kits", "10", "Include", "10.*"))
    if sdk_dirs:
        rep.add("Visual Studio", "Windows SDK", OK, ", ".join(sorted(os.path.basename(d) for d in sdk_dirs)))
    else:
        rep.add("Visual Studio", "Windows SDK", WARN, "no Windows Kits\\10\\Include\\10.* found", True, hint["winsdk"])
    return installs


def check_tools(rep, hint, vs_installs):
    cm = shutil.which("cmake")
    bundled_cmake = vs_bundled_tool(vs_installs, ("Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", "CMake", "bin",
                                                  "cmake.exe")) if is_windows() else None
    if cm:
        v = parse_version(run([cm, "--version"])[1])
        ok = v is not None and v[:2] >= CMAKE_MIN
        rep.add("Tools", "cmake", OK if ok else WARN, "%s%s" % (vstr(v) if v else "?", "" if ok else " (need >= 3.21)"),
                True, "" if ok else hint["cmake"])
    elif bundled_cmake:
        rep.add("Tools", "cmake", WARN, "not on PATH; Visual Studio bundles one at %s (only on PATH inside a VS developer "
                "shell; make.bat needs cmake.exe on PATH before it sets that up)" % bundled_cmake, True, hint["cmake"])
    else:
        rep.add("Tools", "cmake", MISSING, "need >= 3.21", True, hint["cmake"])
    nj = shutil.which("ninja") or shutil.which("ninja-build")
    bundled_ninja = vs_bundled_tool(vs_installs, ("Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", "Ninja",
                                                  "ninja.exe")) if is_windows() else None
    if nj:
        rep.add("Tools", "ninja", OK, vstr(parse_version(run([nj, "--version"])[1]) or ()) or "?", is_windows())
    elif bundled_ninja:
        rep.add("Tools", "ninja", OK, "not on PATH; bundled with Visual Studio at %s (on PATH inside a VS developer shell, "
                "which is where make.bat and the README run it)" % bundled_ninja, False)
    elif is_windows():
        rep.add("Tools", "ninja", MISSING, "make.bat ninja and the README build need it", True, hint["ninja"])
    else:
        rep.add("Tools", "ninja", WARN, "not found; the GNUmakefile falls back to `make -s`", False, hint["ninja"])
    if is_windows():
        sc = shutil.which("sccache")
        if sc:
            rep.add("Tools", "sccache", OK, first_line(run([sc, "--version"])[1]), False)
        else:
            rep.add("Tools", "sccache", WARN, "not found; neural_dev.cmake then leaves WITH_WINDOWS_SCCACHE OFF (every "
                    "rebuild recompiles all C/C++)", False, hint["sccache"])
    else:
        cc = shutil.which("ccache")
        if cc:
            rc, out = run([cc, "--version"])
            v = parse_version(out)
            size = ""
            rc2, cfg = run([cc, "--show-config"])
            m = re.search(r"max_size\s*=\s*(.+)", cfg or "")
            if m:
                size = ", max_size " + m.group(1).strip()
            rep.add("Tools", "ccache", OK, (vstr(v) if v else "?") + size, False)
        else:
            rep.add("Tools", "ccache", WARN, "not found; WITH_COMPILER_CCACHE falls back to OFF (every rebuild recompiles "
                    "all CUDA kernels)", False, hint["ccache"])
    for tool in ("git", "git-lfs"):
        p = shutil.which(tool)
        if p:
            rc, out = run([p, "--version"])
            rep.add("Tools", tool, OK, first_line(out), True)
        else:
            rep.add("Tools", tool, MISSING, "needed by `make update` (precompiled libraries and test data are in LFS)",
                    True, hint[tool])
    gh = shutil.which("gh")
    if gh:
        rc, out = run([gh, "--version"])
        rep.add("Tools", "gh", OK, first_line(out), True)
        rc, out = run([gh, "auth", "status"])
        # Print only the account lines, never token lines (gh masks them, but there is no reason to copy them).
        who = [l.strip() for l in (out or "").splitlines() if re.search(r"Logged in|Active account|not logged", l, re.I)]
        rep.add("Tools", "gh auth status", OK if rc == 0 else MISSING,
                "; ".join(who) if who else first_line(out), True, "" if rc == 0 else hint["gh-auth"])
    else:
        rep.add("Tools", "gh", MISSING, "needed for the PR comments of PROTOCOL.md (comms/post.*)", True, hint["gh"])
    if not is_windows():
        for tool in ("g++", "clang++"):
            p = shutil.which(tool)
            if p:
                rc, out = run([p, "--version"])
                rep.add("Tools", tool, INFO, first_line(out))


def check_repo(rep, hint):
    if is_windows():
        arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
        libname = "windows_" + arch
    else:
        libname = "linux_x64"
    lib = os.path.join(REPO, "lib", libname)
    n = len([e for e in os.listdir(lib) if e != ".git"]) if os.path.isdir(lib) else 0
    # platform_win32.cmake: FATAL_ERROR unless ${LIBDIR}/.git exists (the submodule checkout marker).
    has_git = os.path.exists(os.path.join(lib, ".git"))
    if n and (has_git or not is_windows()):
        rep.add("Repo", "precompiled libs lib/%s" % libname, OK, "%d entries" % n)
    elif n:
        rep.add("Repo", "precompiled libs lib/%s" % libname, WARN,
                "%d entries but no .git entry: platform_win32.cmake requires lib/%s/.git (submodule checkout)" % (n, libname),
                True, hint["libs"])
    else:
        rep.add("Repo", "precompiled libs lib/%s" % libname, MISSING, "empty or missing: see the install hint", True, hint["libs"])
    if is_windows() and n:
        pys = glob.glob(os.path.join(lib, "python", "3*", "bin", "python.exe"))
        rep.add("Repo", "libs python (make.bat test uses it)", OK if pys else WARN,
                ", ".join(sorted(os.path.relpath(p, REPO) for p in pys)) if pys else
                "no lib/%s/python/3*/bin/python.exe: LFS files probably not pulled" % libname, False,
                "" if pys else hint["libs"])
    sample = os.path.join(REPO, "tests", "files", "render", "bsdf", "diffuse.blend")
    try:
        with open(sample, "rb") as f:
            head = f.read(40)
        if head.startswith(b"version https://git-lfs"):
            rep.add("Repo", "render test data (LFS)", MISSING,
                    "tests/files/render/** are LFS pointers: run `git lfs pull --include=\"tests/files/render/**\"`",
                    False, hint["lfs-data"])
        elif head.startswith(b"BLENDER"):
            rep.add("Repo", "render test data (LFS)", OK, "tests/files/render is present")
        else:
            rep.add("Repo", "render test data (LFS)", WARN, "unrecognised content in %s" % sample)
    except OSError:
        rep.add("Repo", "render test data (LFS)", MISSING, "%s not found" % sample, False, hint["lfs-data"])
    rc, out = run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"])
    if rc == 0:
        rep.add("Repo", "HEAD", INFO, out)
    rc, br = run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"])
    if rc == 0:
        rep.add("Repo", "branch", INFO, br)


def check_python(rep, skip_torch, hint, nvcc_release):
    rep.add("Python", "interpreter", INFO, "%s (%s)" % (sys.version.split()[0], sys.executable))
    try:
        import numpy
        rep.add("Python", "numpy", OK, numpy.__version__)
    except Exception as e:  # noqa: BLE001
        rep.add("Python", "numpy", MISSING, "%s: %s" % (type(e).__name__, e), True, hint["numpy"])
    if skip_torch:
        rep.add("Python", "torch / tiny-cuda-nn", INFO, "skipped (--skip-python-gpu)")
        return
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        detail = "%s, built for CUDA %s, cuda.is_available=%s" % (torch.__version__, torch.version.cuda, cuda_ok)
        rep.add("Python", "torch", OK if cuda_ok else WARN, detail, True,
                "" if cuda_ok else "%s (this torch has no working CUDA: CPU-only wheel, or driver problem)" % hint["torch"])
        if cuda_ok:
            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                rep.add("Python", "torch device %d" % i, INFO, "%s, sm_%d%d" % (torch.cuda.get_device_name(i), *cap))
        tcuda = getattr(torch.version, "cuda", None)
        if tcuda and nvcc_release:
            same_major = tcuda.split(".")[0] == nvcc_release.split(".")[0]
            same = tcuda == nvcc_release
            rep.add("Python", "torch CUDA vs nvcc", OK if same else (WARN if same_major else MISSING),
                    "torch %s, nvcc %s%s" % (tcuda, nvcc_release,
                                              "" if same else (" (minor versions differ: torch's cpp_extension only warns)"
                                                               if same_major else
                                                               " (major versions differ: building tiny-cuda-nn fails)")),
                    not same_major, "" if same_major else hint["cuda"])
    except Exception as e:  # noqa: BLE001
        rep.add("Python", "torch", MISSING, "%s: %s" % (type(e).__name__, str(e).splitlines()[0] if str(e) else ""), True,
                hint["torch"])
    try:
        import tinycudann
        rep.add("Python", "tiny-cuda-nn", OK, "import tinycudann ok (%s)" % getattr(tinycudann, "__version__", "no __version__"),
                True)
    except Exception as e:  # noqa: BLE001  also catches the OSError raised for a missing native extension
        rep.add("Python", "tiny-cuda-nn", MISSING, "%s: %s" % (type(e).__name__, str(e).splitlines()[0] if str(e) else ""),
                True, hint["tcnn"])


def read_cmake_cache(path):
    vals = {}
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = re.match(r"([A-Za-z0-9_\-]+)(?::[A-Z]+)?=(.*)$", line.rstrip("\n"))
                if m:
                    vals[m.group(1)] = m.group(2)
    except OSError:
        return None
    return vals


CACHE_EXPECT_COMMON = (
    ("CYCLES_CUDA_BINARIES_ARCH", "sm_86;sm_89"),
    ("WITH_CYCLES_DEVICE_CUDA", "ON"),
    ("WITH_CYCLES_DEVICE_OPTIX", "ON"),
    ("WITH_CYCLES_CUDA_BINARIES", "ON"),
    ("WITH_CYCLES_DEVICE_HIP", "OFF"),
    ("WITH_CYCLES_DEVICE_ONEAPI", "OFF"),
    ("WITH_CYCLES_DEBUG", "ON"),
    ("WITH_STRICT_BUILD_OPTIONS", "ON"),
)
CACHE_EXPECT_LINUX = (("WITH_COMPILER_CCACHE", "ON"),)
CACHE_EXPECT_WINDOWS = (("WITH_WINDOWS_SCCACHE", "ON"),)


def blender_exe_candidates(build_dir):
    if is_windows():
        # Ninja: CMAKE_INSTALL_PREFIX is <build>/bin/${BUILD_TYPE} and the Ninja install script leaves BUILD_TYPE empty
        # (CMakeLists.txt ~1300, tests/CMakeLists.txt), so blender.exe is in bin\. Visual Studio generators use bin\<config>.
        return [os.path.join(build_dir, "bin", "Release", "blender.exe"), os.path.join(build_dir, "bin", "blender.exe"),
                os.path.join(build_dir, "bin", "RelWithDebInfo", "blender.exe"),
                os.path.join(build_dir, "bin", "Debug", "blender.exe")]
    return [os.path.join(build_dir, "bin", "blender")]


def check_build(rep, build_dir, hint):
    if not os.path.isdir(build_dir):
        rep.add("Build", "build dir", MISSING, "%s does not exist (not built yet)" % build_dir)
        return
    rep.add("Build", "build dir", OK, build_dir)
    if is_windows() and " " in build_dir:
        rep.add("Build", "build path has no spaces", WARN,
                "configure_ninja.cmd / build_ninja.cmd use the build directory unquoted", False, hint["path-spaces"])
    cache = read_cmake_cache(os.path.join(build_dir, "CMakeCache.txt"))
    if cache is None:
        rep.add("Build", "CMakeCache.txt", MISSING, "not configured yet")
    else:
        gen = cache.get("CMAKE_GENERATOR", "?")
        rep.add("Build", "generator / type", INFO, "%s, %s" % (gen, cache.get("CMAKE_BUILD_TYPE", "?")))
        if is_windows() and "Ninja" not in gen:
            rep.add("Build", "generator", WARN, "%s: WITH_WINDOWS_SCCACHE only works with Ninja" % gen)
        expect = CACHE_EXPECT_COMMON + (CACHE_EXPECT_WINDOWS if is_windows() else CACHE_EXPECT_LINUX)
        for key, want in expect:
            got = cache.get(key)
            # sccache is optional: an OFF value is information, not a defect.
            rep.add("Build", key, OK if got == want else WARN, "%s%s" % (got, "" if got == want else " (expected %s)" % want))
        rep.add("Build", "OPTIX_ROOT_DIR", OK if cache.get("OPTIX_ROOT_DIR") else WARN, cache.get("OPTIX_ROOT_DIR") or "empty")
        rep.add("Build", "OPTIX_INCLUDE_DIR", OK if cache.get("OPTIX_INCLUDE_DIR", "").strip() and
                "NOTFOUND" not in cache.get("OPTIX_INCLUDE_DIR", "") else MISSING, cache.get("OPTIX_INCLUDE_DIR", "unset"))
    exes = [e for e in blender_exe_candidates(build_dir)
            if os.path.isfile(e) and (is_windows() or os.access(e, os.X_OK))]
    if exes:
        rc, out = run([exes[0], "--version"], timeout=60)
        rep.add("Build", "blender executable", OK if rc == 0 else WARN,
                "%s: %s" % (os.path.relpath(exes[0], build_dir), first_line(out) if out else "no output"))
        if len(exes) > 1:
            rep.add("Build", "other blender executables", INFO, ", ".join(os.path.relpath(e, build_dir) for e in exes[1:]))
    else:
        rep.add("Build", "blender executable", MISSING, "none of %s built yet" % ", ".join(
            os.path.relpath(e, build_dir) for e in blender_exe_candidates(build_dir)))
    libdirs = glob.glob(os.path.join(build_dir, "bin", "*", "scripts", "addons_core", "cycles", "lib"))
    libdirs += glob.glob(os.path.join(build_dir, "bin", "*", "*", "scripts", "addons_core", "cycles", "lib"))
    kernels = []
    for d in libdirs:
        kernels += [os.path.basename(p) for p in sorted(glob.glob(os.path.join(d, "kernel*")))]
    cubins = [k for k in kernels if re.match(r"kernel_sm_\d+\.cubin", k)]
    ptx = [k for k in kernels if k.startswith("kernel_optix")]
    if kernels:
        have = sorted(re.findall(r"sm_\d+", " ".join(cubins)))
        ok = set(have) == {"sm_86", "sm_89"}
        rep.add("Build", "installed CUDA kernels", OK if ok else WARN, ", ".join(cubins) or "none")
        rep.add("Build", "installed OptiX kernels", OK if ptx else WARN, ", ".join(ptx) or "none")
    else:
        rep.add("Build", "installed Cycles kernels", WARN, "none found under bin/*/scripts/addons_core/cycles/lib")


# -----------------------------------------------------------------------------

def render_markdown(rep, title):
    L = ["[result] %s" % title, ""]
    section = None
    for sec, item, status, detail, required, hint in rep.rows:
        if sec != section:
            if section is not None:
                L.append("")
            L += ["**%s**" % sec, "", "| item | status | detail |", "|---|---|---|"]
            section = sec
        L.append("| %s | %s | %s |" % (item.replace("|", "\\|"), status, detail.replace("|", "\\|").replace("\n", " ")))
    probs = rep.problems()
    L += ["", "**Summary:** %s" % ("all required items present" if not probs else
                                   "%d required item(s) missing or suspicious: %s" % (
                                       len(probs), "; ".join("%s %s" % (r[0], r[1]) for r in probs)))]
    opt = rep.optional_gaps()
    L += ["", "**Missing**", ""]
    if not probs and not opt:
        L.append("Nothing. Every required item is present; go on with the build.")
    else:
        if probs:
            L.append("Required (install these, nothing else, then re-run this script):")
            L.append("")
            for r in probs:
                L.append("- %s / %s: %s" % (r[0], r[1], r[5] or r[3]))
        if opt:
            if probs:
                L.append("")
            L.append("Optional (recommended for build speed or convenience):")
            L.append("")
            for r in opt:
                L.append("- %s / %s: %s" % (r[0], r[1], r[5]))
    return "\n".join(L) + "\n"


def default_build_dir():
    name = "build_windows_neural" if is_windows() else "build_linux_neural"
    return os.environ.get("BLENDER_BUILD_DIR") or os.path.join(os.path.dirname(REPO), name)


def collect(build_dir, skip_python_gpu):
    rep = Report()
    hint = hints()
    check_system(rep, hint)
    check_gpus(rep, hint)
    nvcc_release = check_cuda(rep, hint)
    check_optix(rep, hint)
    vs_installs = check_visual_studio(rep, hint) if is_windows() else []
    check_tools(rep, hint, vs_installs)
    check_repo(rep, hint)
    check_python(rep, skip_python_gpu, hint, nvcc_release)
    check_build(rep, build_dir, hint)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build-dir", default=None,
                   help="Blender build directory (default: $BLENDER_BUILD_DIR, else ../build_windows_neural on Windows "
                   "or ../build_linux_neural)")
    p.add_argument("--skip-python-gpu", action="store_true", help="do not import torch / tinycudann (slow)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true", help="exit 1 when a required item is missing")
    p.add_argument("--out", default=None, help="write the report to this file as UTF-8 instead of stdout")
    args = p.parse_args(argv)

    build_dir = os.path.abspath(os.path.expanduser(args.build_dir or default_build_dir()))
    rep = collect(build_dir, args.skip_python_gpu)

    if args.json:
        text = json.dumps([dict(section=s, item=i, status=st, detail=d, required=r, hint=h)
                           for s, i, st, d, r, h in rep.rows], indent=2) + "\n"
    else:
        text = render_markdown(rep, "Build box environment (%s, %s, %s)" % (
            platform.node() or "host", "Windows" if is_windows() else platform.system(), platform.machine()))
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("wrote %s" % args.out)
    else:
        try:
            sys.stdout.write(text)
        except UnicodeEncodeError:  # legacy Windows console code page
            sys.stdout.buffer.write(text.encode("utf-8", "replace"))
    return 1 if (args.strict and rep.problems()) else 0


if __name__ == "__main__":
    sys.exit(main())
