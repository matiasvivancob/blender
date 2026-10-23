#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Report the state of the neural-rendering build box as a Markdown block, ready
to paste into a `[result]` PR comment (see doc/neural_rendering/PROTOCOL.md).

Standard library only, runs with any Python 3.8+ outside Blender:

    python3 check_env.py                      # prints Markdown
    python3 check_env.py --build-dir ~/build_linux_neural
    python3 check_env.py --json               # machine-readable
    python3 check_env.py --strict             # exit 1 if a required item is missing

Run it with the same Python that is used for training, so the torch and
tiny-cuda-nn import checks mean something.

Checks (R = required for the Cycles build, T = required for training):
  R  NVIDIA GPUs and driver (nvidia-smi), both target cards present
  R  nvcc version, CUDA toolkits in common locations
  R  OptiX SDK: $OPTIX_ROOT_DIR and common install paths; version from
     include/optix.h `#define OPTIX_VERSION` decoded as FindOptiX.cmake does
     (major = v / 10000, minor = (v % 10000) / 100, micro = v % 100);
     intern/cycles/CMakeLists.txt requires >= 8.0.0
  R  cmake (>= 3.21, CMakeLists.txt), ninja, git, git-lfs, host compiler
  .  ccache (build speed), Blender precompiled libs (lib/linux_x64), render
     test data (tests/files/render, Git LFS pointers vs real files)
  T  python, numpy, torch (+ CUDA, per-GPU capability), tiny-cuda-nn
  .  Blender build directory: bin/blender, version, and the CMake cache values
     that matter (arch list, OptiX, debug, ccache), plus which Cycles kernels
     were installed
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

OK, WARN, MISSING, INFO = "OK", "WARN", "MISSING", "info"


class Report:
    def __init__(self):
        self.rows = []  # (section, item, status, detail, required)

    def add(self, section, item, status, detail="", required=False):
        self.rows.append((section, item, status, detail, required))

    def problems(self):
        return [r for r in self.rows if r[4] and r[2] in (MISSING, WARN)]


def run(cmd, timeout=20, env=None):
    """Return (returncode, combined output) or (None, reason) when the command cannot run."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                           universal_newlines=True, env=env)
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


# -----------------------------------------------------------------------------

def check_system(rep):
    rep.add("System", "OS", INFO, "%s %s, %s" % (platform.system(), platform.release(), platform.machine()))
    if platform.system() != "Linux":
        rep.add("System", "platform", WARN, "this setup targets Linux", True)
    rep.add("System", "CPU threads", INFO, str(os.cpu_count()))
    try:
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
        gb = kb / 1024.0 / 1024.0
        rep.add("System", "RAM", OK if gb >= 30 else WARN,
                "%.0f GB%s" % (gb, "" if gb >= 30 else " (nvcc for several archs and the final link are memory hungry)"))
    except (OSError, AttributeError, ValueError):
        pass
    try:
        du = shutil.disk_usage(REPO)
        rep.add("System", "free disk at repo", OK if du.free > 60e9 else WARN, "%.0f GB" % (du.free / 1e9))
    except OSError:
        pass


def check_gpus(rep):
    exe = shutil.which("nvidia-smi")
    if not exe:
        rep.add("GPU", "nvidia-smi", MISSING, "not on PATH (NVIDIA driver not installed?)", True)
        return []
    q = "index,name,driver_version,compute_cap,memory.total"
    rc, out = run([exe, "--query-gpu=" + q, "--format=csv,noheader"])
    if rc != 0:  # older drivers do not know compute_cap
        q = "index,name,driver_version,memory.total"
        rc, out = run([exe, "--query-gpu=" + q, "--format=csv,noheader"])
    if rc != 0:
        rep.add("GPU", "nvidia-smi", MISSING, "failed: %s" % first_line(out), True)
        return []
    fields = q.split(",")
    gpus = []
    for line in out.splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == len(fields):
            gpus.append(dict(zip(fields, vals)))
    drivers = sorted({g["driver_version"] for g in gpus})
    rep.add("GPU", "driver", OK if drivers else MISSING, ", ".join(drivers), True)
    for g in gpus:
        cc = g.get("compute_cap", "?")
        rep.add("GPU", "GPU %s" % g["index"], INFO, "%s, compute %s, %s" % (g["name"], cc, g["memory.total"]))
    names = " | ".join(g["name"] for g in gpus).lower()
    for key, cc, sm in EXPECTED_GPUS:
        found = key.lower() in names
        rep.add("GPU", "expected %s (%s)" % (key, sm), OK if found else MISSING,
                "present" if found else "not listed by nvidia-smi", True)
    return gpus


def nvcc_candidates():
    cands = []
    w = shutil.which("nvcc")
    if w:
        cands.append(w)
    for env in ("CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_ROOT_DIR"):
        if os.environ.get(env):
            cands.append(os.path.join(os.environ[env], "bin", "nvcc"))
    cands += sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))
    cands += sorted(glob.glob("/opt/cuda*/bin/nvcc"))
    seen, out = set(), []
    for c in cands:
        rp = os.path.realpath(c)
        if rp not in seen and os.path.exists(c):
            seen.add(rp)
            out.append(c)
    return out


def check_cuda(rep):
    found = []
    for nvcc in nvcc_candidates():
        rc, out = run([nvcc, "--version"])
        m = re.search(r"release (\d+\.\d+), V([\d.]+)", out or "")
        found.append((nvcc, m.group(1) if m else "?", m.group(2) if m else "?"))
    if not found:
        rep.add("CUDA", "nvcc", MISSING, "no nvcc on PATH, $CUDA_HOME or /usr/local/cuda*; set CUDA_TOOLKIT_ROOT_DIR "
                "for CMake", True)
        return
    default = shutil.which("nvcc")
    for nvcc, rel, full in found:
        major = rel.split(".")[0]
        tested = major in ("12", "13")  # kernel/device/cuda/CMakeLists.txt warns for other majors
        tag = " (PATH default)" if default and os.path.realpath(nvcc) == os.path.realpath(default) else ""
        rep.add("CUDA", "nvcc %s" % rel, OK if tested else WARN,
                "%s, V%s%s%s" % (nvcc, full, tag, "" if tested else "; Cycles only tested with CUDA 12 and 13"), True)
    gcc = shutil.which("gcc")
    if gcc:
        rc, out = run([gcc, "-dumpfullversion"])
        v = parse_version(out)
        if v:
            rep.add("CUDA", "gcc (nvcc host compiler)", INFO,
                    "%s; if nvcc rejects it pass -DCUDA_HOST_COMPILER=/usr/bin/gcc-<n>" % vstr(v))


def optix_dirs():
    dirs = []
    if os.environ.get("OPTIX_ROOT_DIR"):
        dirs.append(os.environ["OPTIX_ROOT_DIR"])
    home = os.path.expanduser("~")
    for pat in ("/opt/NVIDIA-OptiX-SDK-*", "/opt/nvidia/optix*", "/opt/optix*", "/usr/local/NVIDIA-OptiX-SDK-*",
                "/usr/local/optix*", "/usr/share/nvidia-optix*",
                home + "/NVIDIA-OptiX-SDK-*", home + "/optix*", home + "/sdk/*ptiX*", home + "/sdk/optix*",
                home + "/opt/*ptiX*", home + "/Downloads/NVIDIA-OptiX-SDK-*"):
        dirs += sorted(glob.glob(pat))
    seen, out = set(), []
    for d in dirs:
        rp = os.path.realpath(d)
        if rp not in seen:
            seen.add(rp)
            out.append(d)
    return out


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


def check_optix(rep):
    env = os.environ.get("OPTIX_ROOT_DIR")
    rep.add("OptiX", "$OPTIX_ROOT_DIR", INFO if env else WARN, env or "unset (CMake then needs -DOPTIX_ROOT_DIR=...)")
    hits = []
    for d in optix_dirs():
        ver = read_optix_version(d)
        if ver is not None:
            hits.append((d, ver))
    if not hits:
        listed = optix_dirs()
        rep.add("OptiX", "SDK", MISSING,
                "no include/optix.h with OPTIX_VERSION found%s. Download the OptiX SDK from NVIDIA and export "
                "OPTIX_ROOT_DIR" % ((" in " + ", ".join(listed)) if listed else ""), True)
        return
    for d, ver in hits:
        ok = ver >= OPTIX_MIN
        rep.add("OptiX", "SDK %s" % vstr(ver), OK if ok else WARN,
                "%s%s" % (d, "" if ok else " (Cycles needs >= %s)" % vstr(OPTIX_MIN)), True)
    if env:
        v = read_optix_version(env)
        rep.add("OptiX", "$OPTIX_ROOT_DIR is usable", OK if v else MISSING,
                ("optix.h version %s" % vstr(v)) if v else "%s/include/optix.h missing or has no OPTIX_VERSION" % env,
                True)


def check_tools(rep):
    cm = shutil.which("cmake")
    if cm:
        v = parse_version(run([cm, "--version"])[1])
        ok = v is not None and v[:2] >= CMAKE_MIN
        rep.add("Tools", "cmake", OK if ok else WARN, "%s%s" % (vstr(v) if v else "?", "" if ok else " (need >= 3.21)"),
                True)
    else:
        rep.add("Tools", "cmake", MISSING, "need >= 3.21", True)
    nj = shutil.which("ninja") or shutil.which("ninja-build")
    if nj:
        rep.add("Tools", "ninja", OK, vstr(parse_version(run([nj, "--version"])[1]) or ()) or "?", False)
    else:
        rep.add("Tools", "ninja", WARN, "not found; the GNUmakefile falls back to `make -s`", False)
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
                "all CUDA kernels)", False)
    for tool in ("git", "git-lfs"):
        p = shutil.which(tool)
        if p:
            rc, out = run([p, "--version"])
            rep.add("Tools", tool, OK, first_line(out), True)
        else:
            rep.add("Tools", tool, MISSING, "needed by `make update` (precompiled libraries and test data are in LFS)",
                    True)
    for tool in ("g++", "clang++"):
        p = shutil.which(tool)
        if p:
            rc, out = run([p, "--version"])
            rep.add("Tools", tool, INFO, first_line(out))


def check_repo(rep):
    lib = os.path.join(REPO, "lib", "linux_x64")
    n = len(os.listdir(lib)) if os.path.isdir(lib) else 0
    rep.add("Repo", "precompiled libs lib/linux_x64", OK if n else MISSING,
            "%d entries" % n if n else "empty: run `python3 build_files/utils/make_update.py --no-blender` (see README)",
            True)
    sample = os.path.join(REPO, "tests", "files", "render", "bsdf", "diffuse.blend")
    try:
        with open(sample, "rb") as f:
            head = f.read(40)
        if head.startswith(b"version https://git-lfs"):
            rep.add("Repo", "render test data (LFS)", MISSING,
                    "tests/files/render/** are LFS pointers: run `git lfs pull --include=\"tests/files/render/**\"`",
                    False)
        elif head.startswith(b"BLENDER"):
            rep.add("Repo", "render test data (LFS)", OK, "tests/files/render is present")
        else:
            rep.add("Repo", "render test data (LFS)", WARN, "unrecognised content in %s" % sample)
    except OSError:
        rep.add("Repo", "render test data (LFS)", MISSING, "%s not found" % sample)
    rc, out = run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"])
    if rc == 0:
        rep.add("Repo", "HEAD", INFO, out)
    rc, br = run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"])
    if rc == 0:
        rep.add("Repo", "branch", INFO, br)


def check_python(rep, skip_torch):
    rep.add("Python", "interpreter", INFO, "%s (%s)" % (sys.version.split()[0], sys.executable))
    try:
        import numpy
        rep.add("Python", "numpy", OK, numpy.__version__)
    except Exception as e:  # noqa: BLE001
        rep.add("Python", "numpy", MISSING, "%s: %s" % (type(e).__name__, e), True)
    if skip_torch:
        rep.add("Python", "torch / tiny-cuda-nn", INFO, "skipped (--skip-python-gpu)")
        return
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        detail = "%s, built for CUDA %s, cuda.is_available=%s" % (torch.__version__, torch.version.cuda, cuda_ok)
        rep.add("Python", "torch", OK if cuda_ok else WARN, detail, True)
        if cuda_ok:
            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                rep.add("Python", "torch device %d" % i, INFO, "%s, sm_%d%d" % (torch.cuda.get_device_name(i), *cap))
    except Exception as e:  # noqa: BLE001
        rep.add("Python", "torch", MISSING, "%s: %s" % (type(e).__name__, str(e).splitlines()[0] if str(e) else ""), True)
    try:
        import tinycudann
        rep.add("Python", "tiny-cuda-nn", OK, "import tinycudann ok (%s)" % getattr(tinycudann, "__version__", "no __version__"),
                True)
    except Exception as e:  # noqa: BLE001  also catches the OSError raised for a missing native extension
        rep.add("Python", "tiny-cuda-nn", MISSING, "%s: %s" % (type(e).__name__, str(e).splitlines()[0] if str(e) else ""),
                True)


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


CACHE_EXPECT = (
    ("CYCLES_CUDA_BINARIES_ARCH", "sm_86;sm_89"),
    ("WITH_CYCLES_DEVICE_CUDA", "ON"),
    ("WITH_CYCLES_DEVICE_OPTIX", "ON"),
    ("WITH_CYCLES_CUDA_BINARIES", "ON"),
    ("WITH_CYCLES_DEVICE_HIP", "OFF"),
    ("WITH_CYCLES_DEVICE_ONEAPI", "OFF"),
    ("WITH_CYCLES_DEBUG", "ON"),
    ("WITH_COMPILER_CCACHE", "ON"),
    ("WITH_STRICT_BUILD_OPTIONS", "ON"),
)


def check_build(rep, build_dir):
    exe = os.path.join(build_dir, "bin", "blender")
    if not os.path.isdir(build_dir):
        rep.add("Build", "build dir", MISSING, "%s does not exist" % build_dir)
        return
    rep.add("Build", "build dir", OK, build_dir)
    cache = read_cmake_cache(os.path.join(build_dir, "CMakeCache.txt"))
    if cache is None:
        rep.add("Build", "CMakeCache.txt", MISSING, "not configured yet")
    else:
        gen = cache.get("CMAKE_GENERATOR", "?")
        rep.add("Build", "generator / type", INFO, "%s, %s" % (gen, cache.get("CMAKE_BUILD_TYPE", "?")))
        for key, want in CACHE_EXPECT:
            got = cache.get(key)
            rep.add("Build", key, OK if got == want else WARN, "%s%s" % (got, "" if got == want else " (expected %s)" % want))
        rep.add("Build", "OPTIX_ROOT_DIR", OK if cache.get("OPTIX_ROOT_DIR") else WARN, cache.get("OPTIX_ROOT_DIR") or "empty")
        rep.add("Build", "OPTIX_INCLUDE_DIR", OK if cache.get("OPTIX_INCLUDE_DIR", "").strip() and
                "NOTFOUND" not in cache.get("OPTIX_INCLUDE_DIR", "") else MISSING, cache.get("OPTIX_INCLUDE_DIR", "unset"))
    if os.path.isfile(exe) and os.access(exe, os.X_OK):
        rc, out = run([exe, "--version"], timeout=60)
        rep.add("Build", "bin/blender", OK if rc == 0 else WARN, first_line(out) if out else "no output")
    else:
        rep.add("Build", "bin/blender", MISSING, "%s not built yet" % exe)
    libdirs = glob.glob(os.path.join(build_dir, "bin", "*", "scripts", "addons_core", "cycles", "lib"))
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
    for sec, item, status, detail, required in rep.rows:
        if sec != section:
            if section is not None:
                L.append("")
            L += ["**%s**" % sec, "", "| item | status | detail |", "|---|---|---|"]
            section = sec
        L.append("| %s | %s | %s |" % (item, status, detail.replace("|", "\\|")))
    probs = rep.problems()
    L += ["", "**Summary:** %s" % ("all required items present" if not probs else
                                   "%d required item(s) missing or suspicious: %s" % (
                                       len(probs), "; ".join("%s %s" % (r[0], r[1]) for r in probs)))]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build-dir", default=os.environ.get("BLENDER_BUILD_DIR") or
                   os.path.join(os.path.dirname(REPO), "build_linux_neural"),
                   help="Blender build directory (default: $BLENDER_BUILD_DIR or ../build_linux_neural)")
    p.add_argument("--skip-python-gpu", action="store_true", help="do not import torch / tinycudann (slow)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true", help="exit 1 when a required item is missing")
    args = p.parse_args(argv)

    rep = Report()
    check_system(rep)
    check_gpus(rep)
    check_cuda(rep)
    check_optix(rep)
    check_tools(rep)
    check_repo(rep)
    check_python(rep, args.skip_python_gpu)
    check_build(rep, os.path.abspath(os.path.expanduser(args.build_dir)))

    if args.json:
        print(json.dumps([dict(section=s, item=i, status=st, detail=d, required=r) for s, i, st, d, r in rep.rows],
                         indent=2))
    else:
        print(render_markdown(rep, "Build box environment (%s, %s)" % (platform.node() or "host", platform.machine())))
    return 1 if (args.strict and rep.problems()) else 0


if __name__ == "__main__":
    sys.exit(main())
