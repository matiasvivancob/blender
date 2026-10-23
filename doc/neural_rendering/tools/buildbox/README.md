# Build box: NVIDIA-only Cycles on an RTX 3090 + RTX 4070 Ti Super

Setup, test harness and iteration-time measurement for plan step 1
("Infrastructure") in `../../00_synthesis.md`. The primary target is the
owner's **Windows 11** workstation; Linux is kept in section 11. Everything here
was derived from the files of this tree; the file each fact comes from is named
in brackets. Nothing in this directory has been run against a real Blender, a
CUDA toolkit, Visual Studio or a GPU: the cloud session that wrote it has none
(the Windows code paths were exercised with mocks only). Section 10 lists what
only the workstation can confirm.

| File | Purpose |
|---|---|
| `neural_dev.cmake` | CMake `-C` preset: CUDA + OptiX only, `sm_86;sm_89`, ccache (Linux) / sccache (Windows), debug, strict options. Same file on both platforms |
| `check_env.py` | Markdown environment report for the `[result]` PR comment, ends with a "Missing" list with install hints (stdlib only, Linux and Windows) |
| `iter_tools.py` | `edit add/remove` (a real one-line change to a header and its exact undo) and `ninjalog` (slowest build steps) for section 8; same commands in every shell |
| `render_test.py` | Headless render, `.npy` dump, compare to reference, white-furnace check, self-test |
| `mis_consistency.py` | Same scene under MIS / NEE-only / forward-only, asserts the images agree |
| `make_test_scenes.py` | Builds `furnace.blend`, `mis_sphere.blend`, `neural_material_probe.blend` |
| `test_blender_stage_stub.py`, `test_make_scenes_stub.py`, `test_check_env_windows.py` | Run the Blender-side scripts against a fake `bpy`, and the Windows paths of `check_env.py` against a fake file system, on any machine |

## 1. First day on the Windows box, in order

1. **Make the path space-free** (section 2). The folder `D:\Github\Blender RND`
   contains a space and `make.bat` refuses to run from it. Use
   `D:\Github\Blender-RND`.
2. **Run the environment check first** and post its output as a `[result]`:
   ```powershell
   cd D:\Github\Blender-RND\blender
   python doc\neural_rendering\tools\buildbox\check_env.py --out env_report.md
   Get-Content env_report.md        # ends with the "Missing" list
   .\doc\neural_rendering\tools\comms\post.ps1 result "Build box environment (Windows)" env_report.md
   ```
   (`py -3` instead of `python` if only the launcher is installed. Use `--out`
   rather than `>`: Windows PowerShell 5.1 would write UTF-16.) If even Python is
   missing, install it first (section 3), everything else is reported by the script.
3. **Install only what the "Missing" list names** (section 3 explains each item),
   open a new shell, re-run the check until the required list is empty.
4. **Fetch libraries and test data** (section 4), **configure and build**
   (section 5), then **post the build times** (section 8).

## 2. The space in `D:\Github\Blender RND`

What `make.bat` does [`make.bat`, `build_files/windows/check_spaces_in_path.cmd`,
`parse_arguments.cmd`, `configure_ninja.cmd`, `build_ninja.cmd`]:

- `make.bat` sets `BLENDER_DIR=%~dp0` (the directory that contains `make.bat`,
  i.e. the **source checkout**, with a trailing backslash) and, before parsing
  any argument, calls `check_spaces_in_path.cmd`. That script compares
  `%BLENDER_DIR%` with `%BLENDER_DIR: =%` and, if they differ, prints "There
  are spaces detected in the build path ..., this is currently not supported,
  exiting...." and exits with code 1. A space in **any** component of the source
  path (drive, `Github`, `Blender RND`, ...) therefore stops the build. Arguments
  to `make.bat` are not looked at yet.
- The build directory is not tested, but its default is `%BLENDER_DIR%..\build_windows_<arch>_vc<ver>_<type>`,
  a sibling of the source, so it would carry the space too. A `builddir`
  override is stored with the quotes the user typed (`set BUILD_DIR_OVERRRIDE=%2`)
  and `configure_ninja.cmd` / `build_ninja.cmd` then use `%BUILD_DIR%` unquoted
  (`mkdir %BUILD_DIR%`, `cd %BUILD_DIR%`, `-B%BUILD_DIR%`, `-H%BLENDER_DIR%`),
  which breaks on a space. So both directories must be space-free for `make.bat`.
- Going around `make.bat` (the direct CMake route of section 5) avoids those
  scripts, and CMake itself accepts spaces, but the paths also flow into nvcc
  command lines, sccache and generated custom commands, which nobody tests with
  spaces. Not worth the risk on a build that already has many moving parts.

**Recommendation: use `D:\Github\Blender-RND`** (hyphen instead of the space) for
the work folder and keep this layout:

```
D:\Github\Blender-RND\blender                  source (this repository)
D:\Github\Blender-RND\build_windows_neural     build directory (check_env.py default)
D:\Github\Blender-RND\venv-neural              Python venv for torch / tiny-cuda-nn
```

If the repository already lives in `D:\Github\Blender RND`, renaming the folder
is enough for git (submodule gitdir links are relative) but **any existing CMake
build directory and Python venv inside it must be deleted or recreated** (they
store absolute paths). Keep the path short (Windows' 260 character limit applies
to some tools; `check_env.py` warns above 40 characters for the source root).
Fallback if the old name must stay (not tested): a junction, `mklink /J D:\bl
"D:\Github\Blender RND\blender"` and run `make.bat` from `D:\bl`; `cmd` keeps the
junction path in `%~dp0`, so the check passes, but whether CMake and Ninja later
resolve it to the real, space-containing path was not verified.

## 3. What to install, and where the tree looks for it

`check_env.py` reports each of these and prints the install hint for the missing
ones. The versions come from the tree.

| Item | Version | Where the build looks for it |
|---|---|---|
| NVIDIA driver | current Game Ready / Studio; check the OptiX SDK release notes for its minimum | `nvidia-smi` (PATH, `C:\Windows\System32`, driver store) |
| **Visual Studio 2022** | **17.14.14 or newer**, workload "Desktop development with C++" (MSVC v143 x64/x86 build tools + Windows SDK); Community or Build Tools | `platform_win32.cmake` raises FATAL_ERROR for `cl` older than 19.44.35216 ("MSVC 2022 17.14.14 or newer is required"). `make.bat` finds it with `vswhere ... -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64` [`detect_msvc_vswhere.cmd`] and supports `2019` (too old now), `2022`, `2026` switches; without a switch it tries 2019, 2022, 2026 in that order [`autodetect_msvc.cmd`], so **pass `2022`** to `make.bat` if an old VS 2019 is also installed. VS 2026 (`18.x`, needs CMake >= 4.2 [`CMakeLists.txt` ~1205]) is supported by the build scripts but whether the CUDA toolkit's nvcc accepts its `cl.exe` is unverified: stay on 2022 |
| **CUDA Toolkit** | 12.9.1 (buildbot pin) or 12.8; 13.4.1 is also pinned by the buildbot [`build_files/buildbot/config/blender_version.cmake`]. **Pick the one whose major.minor matches the CUDA of your PyTorch wheel**, see section 6 | `find_package(CUDA)` [`intern/cycles/cmake/external_libs.cmake`]: `CUDA_TOOLKIT_ROOT_DIR`, else `nvcc.exe` in `%CUDA_PATH%\bin` (set by the installer; FindCUDA reads `ENV CUDA_PATH`), else PATH. Default folder `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9` |
| **OptiX SDK** | 9.0.0 (pinned [`blender_version.cmake`]; Cycles requires >= 8.0.0 [`intern/cycles/CMakeLists.txt`]). Header-only at build time; NVIDIA developer login to download | `FindOptiX.cmake` only searches `${OPTIX_ROOT_DIR}/include` (CMake variable or environment) and has **no default install path**. The installer's default folder is `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0` (the buildbot config uses the same `OptiX SDK <version>` name); `neural_dev.cmake` globs that location when `OPTIX_ROOT_DIR` is unset, or set it: `setx OPTIX_ROOT_DIR "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0"` (new shells only) |
| CMake | >= 3.21 [`CMakeLists.txt:29`]; `winget install Kitware.CMake` | `make.bat` needs `cmake.exe` **on PATH before** it sets up Visual Studio [`find_dependencies.cmd`]. Inside a VS developer shell the VS-bundled CMake is also on PATH |
| Ninja | `winget install Ninja-build.Ninja`, or the VS component "C++ CMake tools for Windows" (on PATH inside a VS developer shell only) | `make.bat ninja` runs `ninja --version` [`configure_ninja.cmd`] |
| sccache | `winget install Mozilla.sccache` (optional) | `WITH_WINDOWS_SCCACHE`, see section 5 |
| Git, Git LFS | `winget install Git.Git` (includes LFS; else `GitHub.GitLFS`), then `git lfs install` | libraries and test data are LFS objects |
| GitHub CLI | `winget install GitHub.cli`, `gh auth login` | `comms\post.ps1`, `watch.ps1` |
| Python | 3.12 or 3.13, 64-bit, for the tools here and for training | `check_env.py` and the venv in section 6 |

Free space: plan for 100 GB (checkout, `lib\windows_x64`, build tree, CUDA, torch).
Check the actual `cl.exe` banner, not the `VC/Tools/MSVC` directory name: serviced
VS 2022 can keep directory `14.44.35207` while shipping compiler `19.44.35228`,
which meets this tree's `19.44.35216` minimum. `check_env.py` probes the executable.
The Windows checker tests isolate filesystem globs so installed host SDKs cannot
leak into simulated missing-tool cases.
RAM: 32 GB or more is comfortable for nvcc plus the final link; with less,
`-DWITH_CYCLES_CUDA_BUILD_SERIAL=ON` builds the cubins one after another.

**Windows Defender.** Real-time scanning of thousands of small object files and of
`ninja`/`cl`/`nvcc` child processes can slow a build noticeably. If you accept
the trade-off, add folder exclusions for the source, build and sccache folders
(Windows Security, Virus & threat protection, Manage settings, Exclusions, or
from an administrator PowerShell
`Add-MpPreference -ExclusionPath "D:\Github\Blender-RND","D:\sccache"`). This
reduces protection for those folders, so it is the owner's decision; nothing
here does it for you.

## 4. Source, precompiled libraries, test data

Windows libraries are a Git LFS submodule, `lib/windows_x64` [`.gitmodules`:
`update = none` until enabled]. `platform_win32.cmake` stops with "Windows requires
pre-compiled libs at ... Please run `make update`" unless `lib\windows_x64\.git`
exists. Two ways to fetch them (several GB, the LFS part has no progress
output for a long time):

- **Recommended: the update script without the Blender part.** `make.bat update`
  (and `make update` on Linux) runs `build_files/utils/make_update.py` **including
  `git pull --rebase` on the current branch**, which `PROTOCOL.md` forbids for
  this branch. Skip the Blender part:
  ```powershell
  cd D:\Github\Blender-RND\blender                  # branch claude/amazing-planck-xvnke7
  python build_files\utils\make_update.py --no-blender
  ```
  `--no-blender` keeps `git lfs install`, the LFS fallback remote, enabling
  `lib/windows_x64` for the detected platform (`sys.platform == "win32"`, x64) and
  checking it out with its LFS files [`make_update.py`: `ensure_git_lfs`,
  `lfs_fallback_setup`, `initialize_precompiled_libraries`, `submodules_lib_update`].
- Alternative: run `make.bat ninja 2022` once from the source root. When
  `lib/windows_x64/.git` is missing, `check_libraries.cmd` asks "Would you like to
  download them? (y/n)" and fetches the submodule and `git lfs pull` without
  rebasing (it uses paths relative to the current directory, so run `make.bat`
  from the source root). It then goes on and configures with default options, so
  stop it with Ctrl+C afterwards.

Check: `lib\windows_x64\python\313\bin\python.exe` must exist (`find_dependencies.cmd`
looks for it; `make.bat test` uses it) and be a real file, not an LFS pointer.

Test files (`tests\files\**`, including `.blend` files and the reference PNGs) are
LFS objects too, and `--no-blender` skips the step that fetches them:

```powershell
git lfs pull --include="tests/files/render/**"
Format-Hex tests\files\render\bsdf\diffuse.blend -Count 7      # must read BLENDER, not "version"
```

Building the libraries from source (`make deps` on Linux) is for platform maintainers only; do not.

## 5. Configure and build on Windows

### What `make.bat` can and cannot do for this branch

`make.bat` [`parse_arguments.cmd`] accepts only its own switches: `ninja`, `sccache`,
`builddir <dir>`, `2019|2022|2022b|2026...`, `debug`, `verbose`, `nobuild`,
`nobuildinfo`, `with_tests`, `clean`, `test`, `update`, `code_update`, and the preset
files `full`, `lite`, `cycles`, `headless`, `bpy`, `release`, `developer`. Each
preset appends `-C"...\build_files\cmake\config\<name>.cmake"` to
`BUILD_CMAKE_ARGS`, and `configure_ninja.cmd` adds `-G "Ninja"`, `-DCMAKE_BUILD_TYPE=%BUILD_TYPE%`
and, for `sccache`, `-DWITH_WINDOWS_SCCACHE=On`. **There is no pass-through for
extra CMake arguments or for another `-C` file**: `reset_variables.cmd` clears
`BUILD_CMAKE_ARGS` at the start of `make.bat`, so setting it in the environment does
nothing, and an unknown word aborts with `Command "..." unknown, aborting!`.
`release` is no substitute (it turns `WITH_DLSS` on, excluded by `PROTOCOL.md`;
`blender_release.cmake`) and `developer` knows nothing of CUDA or OptiX
(on Windows it does not enable ASAN). So **configure with CMake directly** and
use `make.bat` afterwards only as a convenience (below).

`make.bat` also ends up with an interesting property that is useful here:
`configure_ninja.cmd` configures only if `<build>\build.ninja` does not exist
(or with `nobuild`), then `build_ninja.cmd` runs `ninja install` in the build
directory and writes the start/end clock times to `<build>\buildtime.txt`.

### Step 1: a Visual Studio x64 developer shell

CMake must run with `cl.exe` and the Windows SDK on the environment. Either start
"x64 Native Tools Command Prompt for VS 2022" from the Start menu (and run
`powershell` in it), or in a normal PowerShell:

```powershell
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * `
        -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
cl 2>&1 | Select-Object -First 1      # Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35216 or newer ... for x64
nvcc --version | Select-String release
```

(This is what `make.bat` does internally: `vswhere`, then
`vcvarsall.bat x64` [`detect_msvc_vswhere.cmd`]. `make.bat` itself runs from any
plain `cmd` or PowerShell and needs no developer prompt.) Open this shell for
every build session.

### Step 2: configure with `neural_dev.cmake`

```powershell
$SRC   = "D:\Github\Blender-RND\blender"
$BUILD = "D:\Github\Blender-RND\build_windows_neural"
# Optional: neural_dev.cmake finds "C:\ProgramData\NVIDIA Corporation\OptiX SDK *" itself
# $env:OPTIX_ROOT_DIR = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0"
Set-Location $SRC
cmake -G Ninja `
      -C "$SRC\doc\neural_rendering\tools\buildbox\neural_dev.cmake" `
      -S $SRC -B $BUILD `
      -DCMAKE_BUILD_TYPE=Release
```

The `-C` path must be absolute. Quote any `-D` that contains a space, and never put a
`;`-separated list on the PowerShell command line unquoted (`neural_dev.cmake`
already contains `sm_86;sm_89`). Optional toolkit pins when several CUDA
toolkits are installed:
`"-DCUDA_TOOLKIT_ROOT_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"`
[`FindCUDA`; per major version `-DCUDA12_TOOLKIT_ROOT_DIR=... -DCUDA12_NVCC_EXECUTABLE=...`
are read by `cuda_get_toolkit_for_arch` in `kernel/device/cuda/CMakeLists.txt`].
The nvcc host compiler is the `cl.exe` CMake found, passed as `-ccbin`; the tree
overrides it only for clang-cl [`platform_win32.cmake`, `kernel/device/cuda/CMakeLists.txt`
`cuda_add_common_flags`], and there is no MSVC version check of its own beyond
`-D _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`, which only relaxes the STL check.
nvcc's own table of accepted `cl.exe` versions is not in this tree: if nvcc
rejects the host compiler, the error says so, and the answer is a CUDA toolkit
or a Visual Studio build on the other side of that table.

The configure step must end with `Found CUDA ...`, `Cycles CUDA: Using CUDA
toolkit ... for sm_86` / `sm_89` and the OptiX lines, and **must not** print "not
found, disabling": with `WITH_STRICT_BUILD_OPTIONS=ON` a missing OptiX SDK or nvcc
is a configure error instead of a silent CPU-only Blender.

`WITH_WINDOWS_SCCACHE` [`CMakeLists.txt`, `platform_win32.cmake`]: only with Ninja
(with a Visual Studio generator the option is switched off with a warning), sets
`CMAKE_C/CXX_COMPILER_LAUNCHER=sccache` and `/Z7` debug info. It does **not**
cover nvcc: the CUDA kernel commands are wrapped only when `WITH_COMPILER_CCACHE AND
CCACHE_PROGRAM` (Linux/macOS), so on Windows the cubins and the OptiX PTX are
rebuilt whenever a kernel header changes. `neural_dev.cmake` turns the option on
only if `sccache` is on PATH. Recommended once: `$env:SCCACHE_CACHE_SIZE="30G"`
and `$env:SCCACHE_DIR="D:\sccache"` (user environment variables; restart the
server with `sccache --stop-server`), statistics with `sccache --show-stats`.

### Step 3: build

```powershell
Set-Location $BUILD
ninja install 2>&1 | Tee-Object -FilePath "$BUILD\..\build_cold.log"
```

(`ninja install` is what `make.bat ninja` runs [`build_ninja.cmd`].) First build:
expect 30 to 60 minutes plus the CUDA kernels (`00_synthesis.md` estimate, not a
measurement). Where the executable lands: with Ninja the install prefix is
`<build>\bin\${BUILD_TYPE}` and Ninja's install script leaves `BUILD_TYPE` empty
(`CMakeLists.txt` ~1282-1308, `tests/CMakeLists.txt` takes `blender.exe` from the same
variable with the config part removed for single-config generators), so the
expectation is **`<build>\bin\blender.exe`**; the Visual Studio generators use
`<build>\bin\Release\blender.exe`. `check_env.py` looks for both and says which exists.

```powershell
$BLENDER = "$BUILD\bin\blender.exe"          # or $BUILD\bin\Release\blender.exe, see check_env.py
& $BLENDER --version
```

### Optional: `make.bat` afterwards

After the direct configure, `build.ninja` exists, so these only build or test
(from the source root, plain `cmd` or PowerShell; the second needs `ctest.exe` on PATH):

```cmd
make.bat ninja 2022 builddir D:\Github\Blender-RND\build_windows_neural
make.bat test  2022 builddir D:\Github\Blender-RND\build_windows_neural
```

`test` runs `lib\windows_x64\python\313\bin\python.exe -B build_files\utils\make_test.py
--config=Release <build dir>`, i.e. `ctest . --output-on-failure -C Release -O tests\log_Release.txt`
in the build directory [`test.cmd`, `make_test.py`]. `builddir` takes an absolute path
(second character `:`) or a name relative to the **parent** of the source.
Never use `make.bat update` on this branch (rebase, section 4).

Changing an option later (do **not** repeat `-C`, it would force everything back):

```powershell
cmake -DWITH_CYCLES_OSL=ON $BUILD          # example
```

## 6. Python environment: PyTorch + tiny-cuda-nn on Windows

This is probably the hardest install on the box. Use a venv outside the Blender
tree, with a Python version PyTorch ships wheels for (3.12 or 3.13):

```powershell
py -3.12 -m venv D:\Github\Blender-RND\venv-neural
D:\Github\Blender-RND\venv-neural\Scripts\Activate.ps1     # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -U pip wheel setuptools ninja numpy
pip install torch --index-url https://download.pytorch.org/whl/cu130    # cu130 for a CUDA 13.x toolkit; cu129/cu128 for 12.9/12.8. Python 3.12 or 3.13: 3.14 wheels are unverified
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

**Matching rule.** tiny-cuda-nn is compiled by `torch.utils.cpp_extension`, which
refuses to build when the major version of the `nvcc` it finds differs from
`torch.version.cuda` and only warns when the minor version differs. Install the
CUDA Toolkit (section 3) with the **same major.minor as the torch wheel** (cu128
-> CUDA 12.8; Blender itself is happy with 12.x and 13.x) and keep
`%CUDA_PATH%\bin` on PATH: tiny-cuda-nn's `setup.py` runs `nvcc --version` from PATH
(`check_env.py` reports "torch CUDA vs nvcc"). Check the pytorch.org selector for
the wheel tags that exist today.

Build tiny-cuda-nn from a VS developer shell (section 5, step 1) with the venv
active, because it compiles C++/CUDA with MSVC and nvcc. Its `setup.py` calls
`where cl.exe`, sets `DISTUTILS_USE_SDK=1` when `cl` is on PATH, reads
`TCNN_CUDA_ARCHITECTURES` (comma separated, bare numbers) and falls back to the
installed GPU:

```powershell
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn D:\Github\Blender-RND\tiny-cuda-nn
Set-Location D:\Github\Blender-RND\tiny-cuda-nn\bindings\torch
$env:TCNN_CUDA_ARCHITECTURES = "86,89"          # 3090 and 4070 Ti Super; each extra arch adds build time
pip install --no-build-isolation .              # setup.py imports torch, so build isolation must be off
python -c "import tinycudann; print('tcnn ok')"
```

tiny-cuda-nn is outside this tree (BSD-3, allowed by `PROTOCOL.md`); its README names
Visual Studio 2019 or 2022 and CUDA >= 11.5 for Windows. The recipe above was **not**
run here. If it fails: (1) match CUDA major and minor with the torch wheel exactly,
(2) prefer CUDA 12.x for this step even though Cycles also accepts 13 (an older
cutlass inside tiny-cuda-nn may predate the newest toolkit; unverified), (3) read the
first error, not the last, in the `ninja` output, (4) fall back to training inside
WSL2 Ubuntu (CUDA on WSL2 sees both cards; the `.nnm` model files are plain
files) while Blender stays on Windows, and report which route worked in the
`[result]` comment. Re-run `check_env.py` with this venv's Python so its torch
and `tinycudann` checks mean something.

## 7. What `neural_dev.cmake` sets and why

| Setting | Value | Source / reason |
|---|---|---|
| `WITH_CYCLES_DEVICE_CUDA`, `WITH_CYCLES_DEVICE_OPTIX` | ON | `CMakeLists.txt:674-675`, `if(NOT APPLE)` block |
| `WITH_CYCLES_CUDA_BINARIES` | ON | default OFF; without it **no cubin and no OptiX PTX is built** (`kernel/device/{cuda,optix}/CMakeLists.txt`) |
| `CYCLES_CUDA_BINARIES_ARCH` | `sm_86;sm_89` | `CMakeLists.txt:679` default is `sm_50 sm_52 sm_60 sm_70 sm_75 sm_86 sm_120 compute_75`. The runtime loads `kernel_sm_<major><minor>.cubin.zst` and falls back to lower minors (`device/cuda/device_impl.cpp`), so each card gets an exact cubin. No `compute_` entry: it only serves GPUs we do not own |
| OptiX PTX arch | `compute_50` or `compute_75` | not controlled by the variable above; chosen in `kernel/device/optix/CMakeLists.txt` |
| `WITH_CYCLES_DEVICE_HIP`, `_HIP_BINARIES`, `_HIPRT`, `_DEVICE_ONEAPI`, `_ONEAPI_BINARIES` | OFF | `CMakeLists.txt:709-740`. Metal is `if(APPLE)` only |
| `WITH_DLSS` | OFF | default OFF; needs the DLSS SDK (`FindDLSS.cmake`, `DLSS_SDK_ROOT`), excluded by `PROTOCOL.md` |
| `WITH_CYCLES_DEBUG` | ON | nvcc gets `-D WITH_CYCLES_DEBUG --ptxas-options="-v"` (`cuda_add_common_flags`), see section 9 |
| `WITH_CYCLES_NATIVE_ONLY` | ON | CPU kernel for this CPU only (`CMakeLists.txt:649`, "for development only") |
| `WITH_CYCLES_OSL` | OFF | OSL adds eight relocatable-code OptiX PTX modules (`kernel/device/optix/CMakeLists.txt`); the neural closures are SVM-only for now. Re-enable before touching code shared with OSL |
| `WITH_CYCLES_PATH_GUIDING` | ON | unchanged default; the NRC plan refactors the OpenPGL calls and must not break that build. CPU only (`cycles_render_tests.py`: "No path guiding on GPU") |
| `CYCLES_TEST_DEVICES` | `CPU;CUDA;OPTIX` | `CMakeLists.txt:657`; generates ctest names `cycles_<subject>_<device>` (`tests/python/CMakeLists.txt:1043-1080`) |
| `WITH_GTESTS` | ON | `intern/cycles/test/*`; `kernel_camera_projection_test.cpp` is the pattern for the neural-stage golden-vector tests |
| `WITH_COMPILER_CCACHE` (`if(UNIX)`) | ON | Linux: sets both launchers (`platform_unix.cmake:1109`) **and** makes `kernel/device/cuda/CMakeLists.txt` prefix every nvcc cubin command with `ccache`. Setting only the launchers would leave the dominant cost (nvcc cubins) uncached. The OptiX PTX commands are plain custom commands and are not wrapped. The option does not exist on Windows |
| `WITH_WINDOWS_SCCACHE` (`if(WIN32)`, only if `sccache` found) | ON | Windows, Ninja only (section 5); C/C++ only, no nvcc |
| `WITH_STRICT_BUILD_OPTIONS` | ON | without it a missing OptiX SDK or nvcc silently turns the backend OFF with a STATUS line (`set_and_warn_library_found` in `build_files/cmake/macros.cmake`) and you get a CPU-only Blender |
| `WITH_BUILDINFO`, `WITH_DOC_MANPAGE`, `CMAKE_EXPORT_COMPILE_COMMANDS` | OFF, OFF, ON | same as `blender_developer.cmake` (which also sets `WITH_COMPILER_ASAN` on non-Windows: not used here, ASAN slows every render) |
| `OPTIX_ROOT_DIR` | environment, else newest `%ProgramData%\NVIDIA Corporation\OptiX SDK *` (Windows) | section 3 |

Not set on purpose: `WITH_CYCLES_PARALLEL_DEVICE_KERNEL_BUILD` (advanced, builds
different devices' kernels concurrently, "uses more memory"; try it on a box
with 64 GB or more) and `WITH_CYCLES_CUDA_BUILD_SERIAL`. With two architectures
and nvcc >= 12.9 the CMake adds `--split-compile=0` automatically
(`_num_binaries LESS_EQUAL 2`). Since `-C` only pre-populates the cache and every
value is FORCEd, change a value later with a plain `cmake -D... <build dir>`.

Verify that `sm_86;sm_89` is in effect:

```powershell
Select-String -Path "$BUILD\CMakeCache.txt" -Pattern '^(CYCLES_CUDA_BINARIES_ARCH|WITH_CYCLES_(CUDA_BINARIES|DEVICE_OPTIX|DEBUG)|WITH_WINDOWS_SCCACHE|OPTIX_ROOT_DIR)'
Get-ChildItem "$BUILD\bin" -Recurse -Filter 'kernel_*' | Where-Object FullName -Match 'cycles\\lib' | Select-Object -ExpandProperty Name
#   expected: kernel_sm_86.cubin.zst  kernel_sm_89.cubin.zst  kernel_optix*.ptx.zst
python doc\neural_rendering\tools\buildbox\check_env.py     # does the same checks and prints Markdown
```

## 8. Measuring iteration time (STATUS.md asks for these numbers)

Report four numbers for the `[result]` comment. A `touch` is not an edit: compiler
caches hash file contents, so touching `bsdf.h` is a cache hit. The honest
edit-compile-run cost is a content change, which `iter_tools.py edit add` makes
(and `edit remove` undoes exactly, without disturbing other uncommitted edits in
the file; do not use `git checkout --` for that).

Windows (PowerShell, in the VS developer shell; use a fresh, empty sccache directory so the
measurement does not wipe your working cache; the sccache server must be restarted to
pick up the variable):

```powershell
Set-Location $SRC
function Build-Time($log) { [math]::Round((Measure-Command { ninja -C $BUILD install *> $log }).TotalSeconds, 1) }
$LOGS = Split-Path $BUILD
sccache --stop-server; $env:SCCACHE_DIR = "D:\sccache_timing"; Remove-Item -Recurse -Force $env:SCCACHE_DIR -ErrorAction SilentlyContinue
$CFG = @('-G','Ninja','-C',"$SRC\doc\neural_rendering\tools\buildbox\neural_dev.cmake",'-S',$SRC,'-B',$BUILD,'-DCMAKE_BUILD_TYPE=Release')

# 1. cold build, empty sccache (the real compile cost)
Remove-Item -Recurse -Force $BUILD -ErrorAction SilentlyContinue; cmake @CFG | Out-Null
"1 cold, empty cache : $(Build-Time "$LOGS\build_cold.log") s"

# 2. cold build, warm sccache (fresh build directory, cache filled by step 1; nvcc kernels are NOT cached on Windows)
Remove-Item -Recurse -Force $BUILD; cmake @CFG | Out-Null
"2 cold, warm sccache: $(Build-Time "$LOGS\build_cold_cache.log") s"

# 3. incremental after a real one-line change to the closure header
python doc\neural_rendering\tools\buildbox\iter_tools.py edit add intern\cycles\kernel\closure\bsdf.h
"3 incr. after edit  : $(Build-Time "$LOGS\build_incr_edit.log") s"

# 4. incremental after undoing the edit, and after only touching the file
python doc\neural_rendering\tools\buildbox\iter_tools.py edit remove intern\cycles\kernel\closure\bsdf.h
"4a incr. after undo  : $(Build-Time "$LOGS\build_incr_undo.log") s"
(Get-Item intern\cycles\kernel\closure\bsdf.h).LastWriteTime = Get-Date
"4b incr. after touch : $(Build-Time "$LOGS\build_incr_touch.log") s"

# where the time goes (ten slowest steps of the last build directory state)
python doc\neural_rendering\tools\buildbox\iter_tools.py ninjalog $BUILD -n 10
sccache --show-stats
```

(With `make.bat` the same numbers are in `<build>\buildtime.txt`, but only the
start and end clock times.) Expect from `00_synthesis.md` (estimates, not measurements): 30 to 60 min
cold, 5 to 15 min per CUDA architecture for an uncached kernel edit; the OptiX PTX
modules are not cached on either platform, so steps 2 to 4 still pay for them: that is the number to
watch. Paste the four wall times, the ten slowest steps, logical CPU count, RAM
and the sccache statistics into the `[result]` comment after the `check_env.py`
block. Linux equivalent: section 11.

## 9. Running Cycles headless, tests, float tools

```powershell
$BLENDER = "$BUILD\bin\blender.exe"          # check_env.py prints the real location
& $BLENDER -b --factory-startup --python doc\neural_rendering\tools\buildbox\render_test.py -- --list-devices
#   DEVICE OPTIX  NVIDIA GeForce RTX 3090 ...  /  DEVICE OPTIX  NVIDIA GeForce RTX 4070 Ti SUPER ...
```

Bash: `BLENDER=../build_linux_neural/bin/blender` and the same arguments.

Render one blend file on one device [`source/creator/creator_args.cc` help,
`intern/cycles/blender/addon/engine.py`]:

```powershell
& $BLENDER -b scene.blend -E CYCLES -o //out_#### -F PNG -f 1 -- --cycles-device OPTIX
& $BLENDER -b scene.blend -E CYCLES -f 1 -- --cycles-device CUDA+CPU --cycles-print-stats
```

`--cycles-device` (after `--`) overrides the scene and the user preferences
(`CPU`, `CUDA`, `OPTIX`, optionally `+CPU`). Blender executes arguments in
order, so anything configuring the scene (`--python`, `--python-expr`) must
come before `-f`, as `cycles_render_tests.py` does. By default Cycles uses
every device of the chosen type; to isolate one card use
`render_test.py --gpu-name 3090` / `--gpu-name 4070` (CUDA's device order is not
necessarily `nvidia-smi`'s, so do not rely on `CUDA_VISIBLE_DEVICES=0` meaning
the 3090). If PowerShell ever swallows the `--`, put the arguments in an array
and call `& $BLENDER @args`.

Which kernel was loaded: `blender --log-list-categories` lists the logging
categories; `device_impl.cpp` logs "Testing for pre-compiled kernel ..." and
"Using precompiled kernel." at info level (`--log <category> --log-level info`).
Pick the category from the list; it is not guessed here. The first OptiX or CUDA
render after a kernel change is slower than the rest (module creation / driver
JIT); discard it when timing renders.

### The Cycles regression tests

How they work [`tests/python/cycles_render_tests.py`,
`tests/python/modules/render_report.py`, `tests/python/CMakeLists.txt`]:

- One ctest per (subject directory, device): `cycles_<subject>_<cpu|cuda|optix>`.
  Subjects are the directories under `tests/files/render/` (`bsdf`, `shader`,
  `light`, `principled_bsdf`, `integrator`, ...), plus `bake`, `bake_raytrace`,
  `osl`, `updates`. GPU tests are `RUN_SERIAL`.
- Each `.blend` is rendered by `blender --background --factory-startup
  --enable-autoexec ... -E CYCLES -o ... -F PNG ... -f 1 -- --cycles-device
  <DEVICE>`. Output is an **8-bit PNG**, not EXR.
- The reference for `x.blend` is `tests/files/render/<subject>/cycles_renders/x.png`
  (`eevee_renders/` for the EEVEE tests, `cycles_ray_marching_renders/` etc. for
  variants). Reference images are generic across devices; GPU runs are compared
  to the same reference, with per-subject threshold overrides and the
  `BLOCKLIST_*` lists in `cycles_render_tests.py` for known GPU differences.
- Comparison is `oiiotool ref.png new.png --fail 0.016 --failpercent 1 --diff`:
  a pixel fails if any channel differs by more than **0.016** (of 1.0), and the
  test fails if more than **1 %** of the pixels fail. Overrides in `main()`:
  0.032 for `motion_blur`/`integrator`/`displacement`, 0.02/4 % for `light` and
  `camera`, 0.048/3 % for `volume`, and so on.
- An HTML report with diff images is written to
  `<build>/tests/cycles/<subject>/report.html` (`<outdir>/<subject>/diff/`).
- Environment variables read by the harness: `BLENDER_TEST_FILTER=name1:name2-excluded`
  (substring filter on file names), `BLENDER_VERBOSE=1`,
  `BLENDER_TEST_IGNORE_BLOCKLIST=1`, `CYCLESTEST_ARGS="..."` (extra Blender
  arguments), `CYCLESTEST_SPP_MULTIPLIER=2`. **`BLENDER_TEST_UPDATE=1`
  overwrites the reference images; never set it on this branch** unless a
  reference change is intended and reviewed.

```powershell
Set-Location $BUILD                                    # ctest.exe comes with CMake and must be on PATH
ctest -N | Select-String ' cycles_' | Select-Object -First 10          # what exists
ctest -C Release -R '^cycles_bsdf_optix$' --output-on-failure           # one subject, one device
ctest -C Release -R '^cycles_.*_(cpu|cuda|optix)$' --output-on-failure *> "$BUILD\..\cycles_tests_baseline.log"
# or through make.bat from the source root:  make.bat test 2022 builddir D:\Github\Blender-RND\build_windows_neural
```

Bash: `cd ../build_linux_neural; ctest -R '^cycles_bsdf_optix$' --output-on-failure`, or `make test BUILD_DIR=...`.

Record a **baseline before the first kernel change**: some GPU tests may fail
on this hardware for reasons unrelated to our work, and "failing set unchanged"
is the useful regression criterion afterwards.

A single subject without ctest (the `--oiiotool` path is what CMake resolved
from the precompiled libraries; print it with
`ctest -C Release -R '^cycles_bsdf_optix$' -N -V | Select-String -Pattern '--oiiotool'`):

```powershell
$env:BLENDER_TEST_FILTER = "diffuse"
python tests\python\cycles_render_tests.py --blender $BLENDER --testdir tests\files\render\bsdf `
    --outdir "$BUILD\tests\cycles" --oiiotool <path from the ctest output> --device OPTIX
```

The PNG tests are coarse (8-bit, 1.6 % per-pixel tolerance, noise-limited).
They catch gross regressions, not the subtle bias a wrong closure produces.
That is what the float tools below are for.

### Float render tests: `render_test.py`, furnace, MIS gate

These tools exist because PNG comparisons cannot see a 2 % energy error.

Build the test scenes:

```powershell
& $BLENDER -b --factory-startup --python doc\neural_rendering\tools\buildbox\make_test_scenes.py -- --out-dir scenes
```

| Scene | Content |
|---|---|
| `furnace.blend` | uniform world radiance 1.0, one Diffuse BSDF sphere with albedo exactly 1.0; scene property `nr_furnace_expected = 1.0` |
| `mis_sphere.blend` | glossy sphere (Principled, metallic 1, roughness 0.35), diffuse ground, 0.4 m square area light, generated HDRI-like world with a sun disc |
| `neural_material_probe.blend` | 2x2 m plane, 16x16 quads, explicit `UVMap` 0..1, Principled: metallic base + image-texture roughness + clear coat; scene properties name the probe object, UV map and material for the later `SHADER_EVAL_BSDF` dumps |

The `.blend` files are generated, not committed (`PROTOCOL.md`: no large
binaries). All three have clamping, adaptive sampling and denoising off.
"Closed sphere with emission 1.0 world" was implemented as the standard white
furnace (uniform world, albedo-1 convex object): an enclosing emissive shell
with albedo 1 has no constant radiance (`L = e / (1 - albedo)` diverges), so a
closed shell is not a usable furnace.

```powershell
$RT = "doc\neural_rendering\tools\buildbox\render_test.py"
# render + compare against a stored reference (EXR with a .npy next to it)
& $BLENDER -b scenes\mis_sphere.blend --python $RT -- --out out\mis.exr --ref ref\mis.exr --samples 64 --device OPTIX --report out\mis.md
# first time: create the reference
& $BLENDER -b scenes\mis_sphere.blend --python $RT -- --out ref\mis.exr --ref ref\mis.exr --update-ref --device CPU --samples 1024
# white furnace
& $BLENDER -b scenes\furnace.blend --python $RT -- --out out\furnace.exr --samples 64 --device OPTIX --furnace
# compare / furnace outside Blender, plain Python + numpy
python $RT compare out\mis.npy ref\mis.npy
python $RT furnace out\furnace.npy --expected 1.0 --tol 0.01
python $RT selftest
```

What the Blender stage does: selects the device through
`preferences.addons['cycles'].preferences` (`compute_device_type`,
`refresh_devices()`, per-device `use`, then `scene.cycles.device = 'GPU'`), sets
samples, seed and resolution, and **neutralises everything that biases or
randomises a comparison** (all names exist in
`intern/cycles/blender/addon/properties.py`): clamp direct/indirect to 0,
adaptive sampling off, denoising off, `light_sampling_threshold` 0, film not
transparent, compositor/sequencer off, view transform Standard. `--keep-settings`
disables that, `--no-russian-roulette` sets min bounces to max bounces. It
renders a 32-bit RGBA OpenEXR, reloads it through `bpy.data.images`, and writes
`<out>.npy` (H x W x 4 float32, top row first) and `<out>.json` (device,
samples, time, `WITH_CYCLES_DEBUG`). Blender's Python has no OpenEXR module,
hence the `.npy`; if even numpy were missing it falls back to a `.pfm`.

Metrics: MSE, relative MSE (`eps = 0.01`), max abs error and its location, per
channel means, and a **paired bias test**: relative bias of the mean plus a
z-score from the spread of 16 x 16 tile means. A comparison fails on
non-finite pixels, shape mismatch, any explicit `--max-mse/--max-rel-mse/--max-abs`,
or a bias that is both statistically (`|z| > 4`) and practically
(`> 2 %`, `--max-bias`) significant. Raw MSE between two noisy renders cannot
tell noise from bias; the z-score can. The furnace check passes when each
channel mean is within `--furnace-tol` (default 1 % relative) of the expected
constant and reports mean, std, standard error, min/max and the firefly
fraction, and warns when the test is too noisy to resolve the tolerance.
Exit codes: 0 pass, 1 check failed, 2 usage/environment error (verify that
Blender passes `sys.exit` codes through, see section 10).

#### `mis_consistency.py` (mandatory gate for any new closure)

```powershell
& $BLENDER -b scenes\mis_sphere.blend --python doc\neural_rendering\tools\buildbox\mis_consistency.py -- --device OPTIX --samples 256 --out-dir out\mis
```

Renders the scene as MIS (default), next-event-estimation only, and
forward/BSDF-sampling only, with different seeds, and asserts that all pairs
have the same mean image (same paired bias test). What is and is not
controllable from Python in this tree:

| Knob | Exists | Use |
|---|---|---|
| `scene.cycles.direct_light_sampling_type` = `MULTIPLE_IMPORTANCE_SAMPLING` / `FORWARD_PATH_TRACING` / `NEXT_EVENT_ESTIMATION` | yes (`properties.py`, kernel `light/sample.h`) | **the** way to force BSDF-only or light-only sampling. It is a developer option: `sync.cpp` only reads it if `preferences.experimental.use_cycles_debug` **and** `preferences.view.show_developer_ui` are both true (`session.cpp`); the script sets both in-process. Needs no special build flag |
| `scene.cycles.use_light_tree` | yes | which light is chosen; used by the fallback mode and `--extended` |
| `scene.cycles.light_sampling_threshold` | yes | biases when the light tree is off; forced to 0 |
| `scene.cycles.sample_clamp_direct/indirect` | yes | bias on purpose; forced to 0 |
| `scene.cycles.min_light_bounces`, `min_transparent_bounces` | yes | only delay Russian roulette (variance, not mean); `--no-russian-roulette` |
| per-closure / per-material BSDF-only switch | **no** | the strategy knob is global |
| per-light MIS flag | `light.cycles.use_multiple_importance_sampling` | a light property, not a strategy switch |

If the strategy knob is missing the script falls back to light tree on / light
tree off / light tree on at 4x samples and says so in the report (this cannot
isolate the BSDF side). Caveat: delta lights cannot be hit by BSDF sampling and
delta BSDFs cannot be light-sampled, so scenes for this gate need finite-size
lights and non-zero roughness (`mis_sphere.blend` does). Forward-only is very
noisy for a small light, so it gets `--forward-spp-mult` (default 4) times the
samples. The report prints a heuristic note if forward-only is not noisier than
MIS per sample, which would suggest the knob did not take effect.

Gate procedure for a new closure: furnace test passes (energy), then
`mis_consistency.py` passes on a scene that uses the closure, then the regular
ctest set is unchanged against the baseline.

#### Reading `ptxas` register reports

With `WITH_CYCLES_DEBUG=ON` every nvcc kernel compile prints `ptxas info`
lines (registers, spills, stack) [`cuda_add_common_flags`]. Capture them:

```powershell
ninja -C $BUILD -v cycles_kernel_cuda 2>&1 | Tee-Object -FilePath "$BUILD\..\ptxas.log" | Out-Null
Select-String -Path "$BUILD\..\ptxas.log" -Pattern 'ptxas info.*(Used|spill)'
# per-kernel numbers from the finished cubin (uncompressed copy next to the .zst):
cuobjdump --dump-resource-usage "$BUILD\intern\cycles\kernel\device\cuda\kernel_sm_89.cubin"
```

Bash: `ninja -C ../build_linux_neural -v cycles_kernel_cuda 2>&1 | tee ../ptxas.log; grep -E 'ptxas info.*(Used|spill)' ../ptxas.log`.
On Linux a ccache hit may replay no report: set `CCACHE_DISABLE=1` for that build
(nvcc is never cached on Windows). Compare numbers before and after a closure
change: register growth in the megakernel is the cost that shows up as lost
occupancy.

## 10. What only the workstation can validate

1. That `neural_dev.cmake` configures and builds on both platforms: the CMake file was exercised
   against a dummy project (values land in the cache; the Windows branch with a
   simulated `WIN32`, a fake `%ProgramData%` and a fake `sccache`), not against this
   tree's configure, which needs the libraries and toolkits.
2. On Windows: that `make.bat`'s space check behaves as read (section 2); that the
   `Enter-VsDevShell` snippet gives a shell where `cl` is 19.44+; that nvcc accepts
   your Visual Studio's `cl.exe` (the tree does not state the window); that
   `bin\blender.exe` (Ninja) is where the executable lands; that `sccache` works with
   `/Z7` on this tree; that `post.ps1` / `watch.ps1` run (no PowerShell was available
   to execute them); the `Add-MpPreference` and `winget` package ids.
3. nvcc accepts the host compiler; `sm_86`/`sm_89` cubins build; `--split-compile=0`
   behaves with your nvcc.
4. `WITH_STRICT_BUILD_OPTIONS=ON` does not trip over an unrelated optional
   dependency (flip it off and grep the configure log for "not found, disabling").
5. All Blender-side scripts: they ran against fake `bpy` modules that reject
   unknown names (`test_blender_stage_stub.py`, `test_make_scenes_stub.py`),
   which catches typos and logic errors, not API behaviour. Specifically
   unconfirmed: `Image.pixels.foreach_get` on a loaded float EXR returns the
   raw linear RGBA; `bpy.data.images.new(..., float_buffer=True, is_data=...)`
   followed by `pack()` works for generated images (the script falls back to
   saving an EXR next to the `.blend`); `bpy.ops.render.render(write_still=True)`
   writes `<filepath>.exr` when `use_file_extension` is on; `sys.exit(n)` from a
   `--python` script becomes Blender's exit code (otherwise add
   `--python-exit-code 1` and rely on the report text); numpy is importable in
   the built Blender (`& $BLENDER -b --python-expr "import numpy; print(numpy.__version__)"`).
6. That `direct_light_sampling_type` actually changes the sampling with the two
   developer preferences set from a background script (the report prints a
   heuristic note when it does not), and that the three strategies agree on the
   unmodified tree. Run `mis_consistency.py` once **before** changing any
   kernel code: a failure on stock Cycles would be a property of the scene or
   of the knob, not of a new closure.
7. That the furnace sphere gives 1.000 on stock Cycles with `OPTIX`, `CUDA`
   and `CPU`. Same reasoning: establish the baseline before the first closure.
8. Which Cycles regression tests pass on each GPU path before any change
   (section 9 baseline), and the iteration times of section 8.
9. That `--gpu-name` selects one card and both cards render the same image
   within the bias gate (run `render_test.py` once per card against the same
   reference).
10. tiny-cuda-nn builds against the installed PyTorch and imports (section 6).

## 11. Linux (condensed)

Same tree, same `neural_dev.cmake` (its Windows parts are skipped), same tools.

**Prerequisites.** Linux x86_64 with an NVIDIA driver recent enough for the OptiX
SDK you install (compare the SDK's release notes with `nvidia-smi`);
`cmake >= 3.21`, `ninja`, `git`, `git-lfs`, `ccache`, `python3`; CUDA toolkit 12.x
or 13.x (the kernel CMake warns for any other major version; the buildbot uses
12.9.1 and 13.4.1; Blender's Rocky 9 setup uses 12.8
[`build_files/build_environment/linux/linux_rocky9_setup.sh`], which is also the
authoritative list of system packages); OptiX SDK >= 8.0.0 unpacked somewhere
with `export OPTIX_ROOT_DIR=$HOME/sdk/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64`
(`FindOptiX.cmake` searches `${OPTIX_ROOT_DIR}/include` and reads the environment
variable when the CMake variable is empty; `neural_dev.cmake` copies it into the
cache; `CYCLES_RUNTIME_OPTIX_ROOT_DIR` only matters for compiling OptiX kernels at
run time, which does not happen when `WITH_CYCLES_CUDA_BINARIES` is on
[`CMakeLists.txt:692-696`, `intern/cycles/device/optix/device_impl.cpp`]). nvcc only
accepts host compilers up to a limit: if its step rejects the default gcc, pass
`-DCUDA_HOST_COMPILER=/usr/bin/gcc-<n>` (consumed as `-ccbin`,
`cuda_add_common_flags`). The build asks the toolkit with `nvcc --list-gpu-code`
and, with `WITH_STRICT_BUILD_OPTIONS`, fails if it cannot build an architecture.

**Libraries and test data.** `lib/linux_x64` is a Git LFS submodule. `make update`
runs `make_update.py` **including `git pull --rebase`**, forbidden on this branch
by `PROTOCOL.md`. Use:

```bash
cd ~/src/blender                      # branch claude/amazing-planck-xvnke7
python3 build_files/utils/make_update.py --no-blender
git lfs pull --include="tests/files/render/**"
head -c 7 tests/files/render/bsdf/diffuse.blend     # must print BLENDER, not "version"
```

`--no-blender` keeps `git lfs install`, the LFS fallback remote, and enabling
and checking out `lib/linux_x64` with its LFS files; it skips fetching the test
data, hence the second command. `make deps` is for platform maintainers only.

**Configure and build.** Build directory convention `../build_linux_neural`
(`check_env.py` default; override with `--build-dir` or `$BLENDER_BUILD_DIR`).

```bash
cd ~/src/blender
export OPTIX_ROOT_DIR=$HOME/sdk/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64
# Option A: through the GNUmakefile (BUILD_CMAKE_ARGS is copied into CMAKE_CONFIG_ARGS; -C needs an absolute path)
make ninja BUILD_DIR="$PWD/../build_linux_neural" \
     BUILD_CMAKE_ARGS="-C$PWD/doc/neural_rendering/tools/buildbox/neural_dev.cmake"
# Option B: plain CMake (CMAKE_BUILD_TYPE_INIT:STRING=Release is what the GNUmakefile passes)
cmake -G Ninja -C "$PWD/doc/neural_rendering/tools/buildbox/neural_dev.cmake" \
      -S "$PWD" -B "$PWD/../build_linux_neural" -DCMAKE_BUILD_TYPE_INIT:STRING=Release
ninja -C ../build_linux_neural install        # binary: ../build_linux_neural/bin/blender
```

Do not add the `developer` goal: `blender_developer.cmake` turns on
`WITH_COMPILER_ASAN`, which slows every render (its `WITH_BUILDINFO` and
`WITH_CYCLES_NATIVE_ONLY` are already in `neural_dev.cmake`). Unlike the Windows `make.bat`,
the GNUmakefile forwards `BUILD_CMAKE_ARGS`.
Optional toolkit pins: `-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8
-DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-12.8/bin/nvcc` (read by `find_package(CUDA)`)
or per major `-DCUDA12_TOOLKIT_ROOT_DIR=... -DCUDA12_NVCC_EXECUTABLE=...`.

**ccache.** The task asked for `CMAKE_C(XX)_COMPILER_LAUNCHER`; the tree's own
switch is `WITH_COMPILER_CCACHE`, which sets both launchers
(`platform_unix.cmake:1109`) **and** makes `kernel/device/cuda/CMakeLists.txt`
prefix every nvcc cubin command with `ccache`. Setting only the launchers would
leave the dominant cost (the nvcc cubins) uncached. The OptiX PTX commands are
plain custom commands and are not wrapped, so expect them to recompile on every
kernel-header change. Recommended once: `ccache --max-size=50G`.

**Iteration timing (bash).** Same four numbers as section 8, with a dedicated
ccache directory:

```bash
cd ~/src/blender; export OPTIX_ROOT_DIR=...
export CCACHE_DIR=$HOME/.cache/ccache_timing            # fresh, empty
BUILD=$PWD/../build_linux_neural; TIMEFORMAT='%R s wall'
CFG=(cmake -G Ninja -C "$PWD/doc/neural_rendering/tools/buildbox/neural_dev.cmake" -S "$PWD" -B "$BUILD" -DCMAKE_BUILD_TYPE_INIT:STRING=Release)
IT="python3 doc/neural_rendering/tools/buildbox/iter_tools.py"
rm -rf "$BUILD" "$CCACHE_DIR"; "${CFG[@]}" >/dev/null; time ninja -C "$BUILD" install >"$BUILD/../build_cold.log" 2>&1       # 1
rm -rf "$BUILD"; "${CFG[@]}" >/dev/null; time ninja -C "$BUILD" install >"$BUILD/../build_cold_ccache.log" 2>&1             # 2
$IT edit add intern/cycles/kernel/closure/bsdf.h; time ninja -C "$BUILD" install >"$BUILD/../build_incr_edit.log" 2>&1      # 3
$IT edit remove intern/cycles/kernel/closure/bsdf.h; time ninja -C "$BUILD" install >"$BUILD/../build_incr_undo.log" 2>&1   # 4a
touch intern/cycles/kernel/closure/bsdf.h; time ninja -C "$BUILD" install >"$BUILD/../build_incr_touch.log" 2>&1            # 4b
$IT ninjalog "$BUILD" -n 10; ccache --show-stats
```

(In step 2 ccache is warm, so mostly the uncached OptiX PTX is paid again.) Add `nproc`,
RAM and `ccache --show-stats` to the `[result]` comment.
