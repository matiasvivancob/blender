[result] Build box environment (Matias-PC, Windows, AMD64)

**System**

| item | status | detail |
|---|---|---|
| OS | info | Windows 11, AMD64 |
| CPU threads | info | 24 |
| RAM | OK | 96 GB |
| free disk at repo | OK | 80 GB |
| source path has no spaces | OK | D:\Github\Blender-RND\blender |
| source path length | OK | 29 characters |
| Windows long paths | OK | LongPathsEnabled=1 |

**GPU**

| item | status | detail |
|---|---|---|
| driver | OK | 616.92 |
| GPU 0 | info | NVIDIA GeForce RTX 3090, compute 8.6, 24576 MiB |
| GPU 1 | info | NVIDIA GeForce RTX 4070 Ti SUPER, compute 8.9, 16376 MiB |
| expected 3090 (sm_86) | OK | present |
| expected 4070 Ti SUPER (sm_89) | OK | present |

**CUDA**

| item | status | detail |
|---|---|---|
| %CUDA_PATH% | info | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3 |
| nvcc 13.3 | OK | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.EXE, V13.3.73 (PATH default) |
| nvcc host compiler | info | cl.exe of the Visual Studio shell/toolset that runs cmake (CMake's CUDA_HOST_COMPILER = CMAKE_C_COMPILER); nvcc has its own MSVC version window, see the Visual Studio rows |

**OptiX**

| item | status | detail |
|---|---|---|
| OPTIX_ROOT_DIR | info | unset; neural_dev.cmake picks the newest SDK under %ProgramData%\NVIDIA Corporation (FindOptiX.cmake alone would not find it) |
| SDK 9.1.0 | OK | C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0 |

**Visual Studio**

| item | status | detail |
|---|---|---|
| Visual Studio Build Tools 2026 18.7.11925.98 | OK | C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools; C++ workload (Desktop development with C++): NO; VC.Tools x64 component: yes; newest MSVC toolset 14.51.36231 |
| Visual Studio Community 2026 18.7.11925.98 | OK | C:\Program Files\Microsoft Visual Studio\18\Community; C++ workload (Desktop development with C++): yes; VC.Tools x64 component: yes; newest MSVC toolset 14.51.36231 |
| Visual Studio Build Tools 2019 16.11.37530.7 | info | C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools; C++ workload (Desktop development with C++): NO; VC.Tools x64 component: yes; newest MSVC toolset 14.29.30133 |
| Visual Studio Build Tools 2022 17.14.37614.0 | WARN | C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools; C++ workload (Desktop development with C++): NO; VC.Tools x64 component: yes; newest MSVC toolset 14.44.35207 (platform_win32.cmake needs toolset >= 14.44.35216, i.e. VS 2022 17.14.14: update) |
| usable VS 2022 (>= 17.14.14 with C++ tools) | WARN | only Visual Studio 2026 with C++ tools found (make.bat switch 2026; needs CMake >= 4.2); whether the CUDA toolkit's nvcc accepts its cl.exe is unverified, VS 2022 17.14 is the safe choice |
| Windows SDK | OK | 10.0.19041.0, 10.0.26100.0 |

**Tools**

| item | status | detail |
|---|---|---|
| cmake | OK | 4.4.2 |
| ninja | OK | 1.13.2 |
| sccache | WARN | not found; neural_dev.cmake then leaves WITH_WINDOWS_SCCACHE OFF (every rebuild recompiles all C/C++) |
| git | OK | git version 2.53.0.windows.3 |
| git-lfs | OK | git-lfs/3.7.1 (GitHub; windows amd64; go 1.25.1; git b84b3384) |
| gh | OK | gh version 2.96.0 (2026-07-02) |
| gh auth status | OK | ✓ Logged in to github.com account matiasvivancob (keyring); - Active account: true |

**Repo**

| item | status | detail |
|---|---|---|
| precompiled libs lib/windows_x64 | MISSING | empty or missing: see the install hint |
| render test data (LFS) | MISSING | tests/files/render/** are LFS pointers: run `git lfs pull --include="tests/files/render/**"` |
| HEAD | info | e533f32aa26 |
| branch | info | claude/amazing-planck-xvnke7 |

**Python**

| item | status | detail |
|---|---|---|
| interpreter | info | 3.14.7 (C:\Users\Matia\AppData\Local\hermes\installs\6af1caa198970d3b\environments\814c2fa76c11450093fabe0233daeebe\venv\Scripts\python.exe) |
| numpy | OK | 2.4.3 |
| torch | MISSING | ModuleNotFoundError: No module named 'torch' |
| tiny-cuda-nn | MISSING | ModuleNotFoundError: No module named 'tinycudann' |

**Build**

| item | status | detail |
|---|---|---|
| build dir | MISSING | D:\Github\Blender-RND\build_windows_neural does not exist (not built yet) |

**Summary:** 4 required item(s) missing or suspicious: Visual Studio usable VS 2022 (>= 17.14.14 with C++ tools); Repo precompiled libs lib/windows_x64; Python torch; Python tiny-cuda-nn

**Missing**

Required (install these, nothing else, then re-run this script):

- Visual Studio / usable VS 2022 (>= 17.14.14 with C++ tools): Visual Studio 2022 17.14.14 or newer (Community is free; Build Tools also work): https://visualstudio.microsoft.com/downloads/ , workload "Desktop development with C++" (includes MSVC v143 x64/x86 build tools and a Windows 10/11 SDK). Update through the Visual Studio Installer if it is older than 17.14.14.
- Repo / precompiled libs lib/windows_x64: python build_files\utils\make_update.py --no-blender   (fetches lib\windows_x64 through Git LFS; several GB; avoids the git pull --rebase that `make.bat update` does).
- Python / torch: pip install torch --index-url https://download.pytorch.org/whl/cu128 -- use the cuXYZ tag that matches the CUDA toolkit you installed (see https://pytorch.org/get-started/locally/); the nvcc used for tiny-cuda-nn must have the same major version as torch.version.cuda.
- Python / tiny-cuda-nn: Needs MSVC and nvcc on PATH and torch already installed. From an x64 Native Tools / VS developer shell with the venv active: git clone --recursive https://github.com/NVlabs/tiny-cuda-nn ; cd tiny-cuda-nn\bindings\torch ; set TCNN_CUDA_ARCHITECTURES=86,89 ; pip install --no-build-isolation . (details: buildbox README).

Optional (recommended for build speed or convenience):

- Tools / sccache: sccache (C/C++ compile cache, Ninja builds only): winget install Mozilla.sccache (or scoop install sccache). Optional.
- Repo / render test data (LFS): git lfs pull --include="tests/files/render/**"
