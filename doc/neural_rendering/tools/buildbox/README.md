# Build box: NVIDIA-only Cycles on an RTX 3090 + RTX 4070 Ti Super

Setup, test harness and iteration-time measurement for plan step 1
("Infrastructure") in `../../00_synthesis.md`. Everything here was derived from
the files of this tree; the file each fact comes from is named in brackets.
Nothing in this directory has been run against a real Blender, a CUDA toolkit
or a GPU: the cloud session that wrote it has none. Section 9 lists what only
the workstation can confirm.

| File | Purpose |
|---|---|
| `neural_dev.cmake` | CMake `-C` preset: CUDA + OptiX only, `sm_86;sm_89`, ccache, debug, strict options |
| `check_env.py` | Markdown environment report for the `[result]` PR comment (stdlib only) |
| `render_test.py` | Headless render, `.npy` dump, compare to reference, white-furnace check, self-test |
| `mis_consistency.py` | Same scene under MIS / NEE-only / forward-only, asserts the images agree |
| `make_test_scenes.py` | Builds `furnace.blend`, `mis_sphere.blend`, `neural_material_probe.blend` |
| `test_blender_stage_stub.py`, `test_make_scenes_stub.py` | Run the Blender-side scripts against a fake `bpy` on any machine |

## 1. Prerequisites on the workstation

- Linux x86_64, NVIDIA driver recent enough for the OptiX SDK you install
  (check the SDK's release notes against `nvidia-smi`).
- A C/C++ toolchain. Blender's own Linux build environment is Rocky with
  `gcc-toolset-14` and CUDA 12.8
  [`build_files/build_environment/linux/linux_rocky9_setup.sh`]. Any recent
  gcc works for the host code, but nvcc only accepts host compilers up to a
  limit: if CMake's nvcc step rejects the default gcc, pass
  `-DCUDA_HOST_COMPILER=/usr/bin/gcc-<n>` (consumed as `-ccbin`)
  [`intern/cycles/kernel/device/cuda/CMakeLists.txt`, `cuda_add_common_flags`].
- `cmake >= 3.21` [`CMakeLists.txt:29`], `ninja`, `git`, `git-lfs`, `ccache`,
  `python3` (to run `make_update.py` and the tools here).
- CUDA toolkit 12.x or 13.x. The kernel CMake warns for any other major
  version and the buildbot uses 12.9.1 and 13.4.1
  [`kernel/device/cuda/CMakeLists.txt`, `build_files/buildbot/config/blender_version.cmake`].
  `sm_86` and `sm_89` are both supported by 12 and 13; the build asks the
  toolkit with `nvcc --list-gpu-code` and, with `WITH_STRICT_BUILD_OPTIONS`,
  fails if it cannot build an architecture.
- OptiX SDK >= 8.0.0 (Cycles requires it, `find_package(OptiX 8.0.0)` in
  `intern/cycles/CMakeLists.txt`; the buildbot pins 9.0.0). It is a header-only
  dependency at build time: only `include/optix.h` is needed. Download it from
  NVIDIA (developer login), unpack it, and
  ```bash
  export OPTIX_ROOT_DIR=$HOME/sdk/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64   # contains include/optix.h
  ```
  `FindOptiX.cmake` searches `${OPTIX_ROOT_DIR}/include` and reads the
  environment variable when the CMake variable is empty
  [`build_files/cmake/Modules/FindOptiX.cmake`]; `neural_dev.cmake` also copies
  the variable into the CMake cache so `CMakeCache.txt` records it. The
  separate cache variable `CYCLES_RUNTIME_OPTIX_ROOT_DIR` (and the runtime
  environment variable `OPTIX_ROOT_DIR`) only matter for compiling OptiX
  kernels at run time, which does not happen when `WITH_CYCLES_CUDA_BINARIES`
  is on [`CMakeLists.txt:692-696`, `intern/cycles/device/optix/device_impl.cpp`].
- Linux system packages for the windowing and audio stack come from your
  distribution; the list in `linux_rocky9_setup.sh` is the authoritative one
  for this tree. Let CMake's configure errors tell you what is missing.

Run `python3 check_env.py` (section 7) before and after; it reports all of the
above.

## 2. Source, precompiled libraries, test data

The libraries are a Git LFS submodule, `lib/linux_x64`
[`.gitmodules`, `update = none` until enabled]. `make update` is the documented
way to fetch them (`GNUmakefile` target `update` runs
`build_files/utils/make_update.py`), **but it also runs `git pull --rebase` on
the current branch**, which `PROTOCOL.md` forbids for this branch. Skip the
Blender part:

```bash
cd ~/src/blender                      # branch claude/amazing-planck-xvnke7
python3 build_files/utils/make_update.py --no-blender
```

`--no-blender` keeps the steps that matter here: `git lfs install`, the LFS
fallback remote, enabling `lib/linux_x64` and checking it out with its LFS
files [`make_update.py`: `initialize_precompiled_libraries`,
`submodules_lib_update`]. `lib/linux_x64` must be non-empty afterwards;
CMake's `platform_unix.cmake` picks it up by that name.

Test files (`tests/files/**`, including the `.blend` files and the reference
PNGs) are LFS objects in this checkout. `--no-blender` skips the step that
fetches them, so do it by hand:

```bash
git lfs pull --include="tests/files/render/**"
head -c 7 tests/files/render/bsdf/diffuse.blend     # must print BLENDER, not "version"
```

`make deps` builds the libraries from source and is "intended only for
platform maintainers" [`GNUmakefile` help text]; do not use it.

## 3. Configure and build

Build directory convention: a sibling of the source tree named like the
GNUmakefile's default (`../build_linux`) with a suffix, so tools find it
without arguments: **`../build_linux_neural`**. `check_env.py` defaults to it;
override with `--build-dir` or `$BLENDER_BUILD_DIR`.

### Option A: through the GNUmakefile (what Blender developers normally do)

`BUILD_CMAKE_ARGS` is copied into `CMAKE_CONFIG_ARGS` and passed to `cmake`
[`GNUmakefile`]; `ninja` adds `-G Ninja`; the default goal runs
`<tool> -C $BUILD_DIR -j $NPROCS install`. The `-C` file must be an absolute
path.

```bash
cd ~/src/blender
export OPTIX_ROOT_DIR=$HOME/sdk/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64
make ninja BUILD_DIR="$PWD/../build_linux_neural" \
     BUILD_CMAKE_ARGS="-C$PWD/doc/neural_rendering/tools/buildbox/neural_dev.cmake"
```

Do not add the `developer` goal: `blender_developer.cmake` turns on
`WITH_COMPILER_ASAN`, which slows every render. The two settings from it that
help here (`WITH_BUILDINFO`, `WITH_CYCLES_NATIVE_ONLY`) are already in
`neural_dev.cmake`. The later `make` runs reconfigure and rebuild; once
`build.ninja` exists the makefile uses `ninja` on its own.

### Option B: plain CMake (the exact invocation)

```bash
cd ~/src/blender
export OPTIX_ROOT_DIR=$HOME/sdk/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64
cmake -G Ninja \
      -C "$PWD/doc/neural_rendering/tools/buildbox/neural_dev.cmake" \
      -S "$PWD" -B "$PWD/../build_linux_neural" \
      -DCMAKE_BUILD_TYPE_INIT:STRING=Release
ninja -C ../build_linux_neural install
```

`CMAKE_BUILD_TYPE_INIT:STRING=Release` is what the GNUmakefile passes. The
binary ends up at `../build_linux_neural/bin/blender`
[`GNUmakefile`: `BLENDER_BIN`].

Optional toolkit pins, if the default `nvcc` on `PATH` is not the one you want:
`-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-12.8/bin/nvcc`
(read by `find_package(CUDA)` in `intern/cycles/cmake/external_libs.cmake`), or
per major version `-DCUDA12_TOOLKIT_ROOT_DIR=... -DCUDA12_NVCC_EXECUTABLE=...`
(`cuda_get_toolkit_for_arch` in `kernel/device/cuda/CMakeLists.txt`).

Changing an option after the first configure (do **not** repeat `-C`, it would
force everything back):

```bash
cmake -DWITH_CYCLES_OSL=ON ../build_linux_neural     # example
```

### What `neural_dev.cmake` sets and why

| Setting | Value | Source / reason |
|---|---|---|
| `WITH_CYCLES_DEVICE_CUDA`, `WITH_CYCLES_DEVICE_OPTIX` | ON | `CMakeLists.txt:674-675`, `if(NOT APPLE)` block |
| `WITH_CYCLES_CUDA_BINARIES` | ON | default OFF; without it **no cubin and no OptiX PTX is built** (`kernel/device/{cuda,optix}/CMakeLists.txt`) |
| `CYCLES_CUDA_BINARIES_ARCH` | `sm_86;sm_89` | `CMakeLists.txt:679` default is `sm_50 sm_52 sm_60 sm_70 sm_75 sm_86 sm_120 compute_75`. The runtime loads `kernel_sm_<major><minor>.cubin.zst` and falls back to lower minors (`device/cuda/device_impl.cpp`), so each card gets an exact cubin. No `compute_` entry: it only serves GPUs we do not own |
| OptiX PTX arch | `compute_50` or `compute_75` | not controlled by the variable above; chosen in `kernel/device/optix/CMakeLists.txt` |
| `WITH_CYCLES_DEVICE_HIP`, `_HIP_BINARIES`, `_HIPRT`, `_DEVICE_ONEAPI`, `_ONEAPI_BINARIES` | OFF | `CMakeLists.txt:709-740`. Metal is `if(APPLE)` only (`:730`), absent on Linux |
| `WITH_DLSS` | OFF | default OFF; needs the DLSS SDK (`FindDLSS.cmake`, `DLSS_SDK_ROOT`), excluded by `PROTOCOL.md` |
| `WITH_CYCLES_DEBUG` | ON | nvcc gets `-D WITH_CYCLES_DEBUG --ptxas-options="-v"` (`cuda_add_common_flags`), see section 6 |
| `WITH_CYCLES_NATIVE_ONLY` | ON | CPU kernel for this CPU only (`CMakeLists.txt:649`, "for development only") |
| `WITH_CYCLES_OSL` | OFF | OSL adds eight relocatable-code OptiX PTX modules (`kernel/device/optix/CMakeLists.txt`); the neural closures are SVM-only for now. Re-enable before touching code shared with OSL |
| `WITH_CYCLES_PATH_GUIDING` | ON | unchanged default; the NRC plan refactors the OpenPGL calls and must not break that build. CPU only (`cycles_render_tests.py`: "No path guiding on GPU") |
| `CYCLES_TEST_DEVICES` | `CPU;CUDA;OPTIX` | `CMakeLists.txt:657`; generates ctest names `cycles_<subject>_<device>` (`tests/python/CMakeLists.txt:1043-1080`) |
| `WITH_GTESTS` | ON | `intern/cycles/test/*`; `kernel_camera_projection_test.cpp` is the pattern for the neural-stage golden-vector tests |
| `WITH_COMPILER_CCACHE` | ON | see below |
| `WITH_STRICT_BUILD_OPTIONS` | ON | without it a missing OptiX SDK or nvcc silently turns the backend OFF with a STATUS line (`set_and_warn_library_found` in `build_files/cmake/macros.cmake`) and you get a CPU-only Blender |
| `WITH_BUILDINFO`, `WITH_DOC_MANPAGE`, `CMAKE_EXPORT_COMPILE_COMMANDS` | OFF, OFF, ON | same as `blender_developer.cmake` |
| `OPTIX_ROOT_DIR` | from the environment | see section 1 |

**ccache.** The task asked for `CMAKE_C(XX)_COMPILER_LAUNCHER`; the tree's own
switch is `WITH_COMPILER_CCACHE`, which sets both launchers
(`build_files/cmake/platform/platform_unix.cmake:1109`) **and** makes
`kernel/device/cuda/CMakeLists.txt` prefix every nvcc cubin command with
`ccache`. Setting only the launchers would leave the dominant cost (the nvcc
cubins) uncached. The OptiX PTX commands in `kernel/device/optix/CMakeLists.txt`
are plain custom commands and are not wrapped, so expect them to recompile on
every kernel-header change. Recommended once: `ccache --max-size=50G` (CUDA
kernel objects are big).

Not set on purpose: `WITH_CYCLES_PARALLEL_DEVICE_KERNEL_BUILD` (advanced, builds
different devices' kernels concurrently, "uses more memory"; try it on a box
with 64 GB or more) and `WITH_CYCLES_CUDA_BUILD_SERIAL`. With two architectures
and nvcc >= 12.9 the CMake adds `--split-compile=0` automatically
(`_num_binaries LESS_EQUAL 2`).

### Verify that `sm_86;sm_89` is in effect

```bash
grep -E '^(CYCLES_CUDA_BINARIES_ARCH|WITH_CYCLES_(CUDA_BINARIES|DEVICE_OPTIX|DEBUG)|WITH_COMPILER_CCACHE)' ../build_linux_neural/CMakeCache.txt
ls ../build_linux_neural/bin/*/scripts/addons_core/cycles/lib/ | grep -E 'kernel_(sm|optix)'
#   expected: kernel_sm_86.cubin.zst  kernel_sm_89.cubin.zst  kernel_optix*.ptx.zst
python3 doc/neural_rendering/tools/buildbox/check_env.py     # does the same checks and prints Markdown
```

## 4. Running Cycles headless

```bash
BLENDER=../build_linux_neural/bin/blender
$BLENDER -b --factory-startup --python doc/neural_rendering/tools/buildbox/render_test.py -- --list-devices
#   DEVICE OPTIX  NVIDIA GeForce RTX 3090 ...  /  DEVICE OPTIX  NVIDIA GeForce RTX 4070 Ti SUPER ...
```

Render one blend file on one device [`source/creator/creator_args.cc` help,
`intern/cycles/blender/addon/engine.py`]:

```bash
$BLENDER -b scene.blend -E CYCLES -o //out_#### -F PNG -f 1 -- --cycles-device OPTIX
$BLENDER -b scene.blend -E CYCLES -f 1 -- --cycles-device CUDA+CPU --cycles-print-stats
```

`--cycles-device` (after `--`) overrides the scene and the user preferences
(`CPU`, `CUDA`, `OPTIX`, optionally `+CPU`). Blender executes arguments in
order, so anything configuring the scene (`--python`, `--python-expr`) must
come before `-f`, as `cycles_render_tests.py` does. By default Cycles uses
every device of the chosen type; to isolate one card use
`render_test.py --gpu-name 3090` / `--gpu-name 4070`, or hide the other from the
driver with `CUDA_VISIBLE_DEVICES`.

Which kernel was loaded: `blender --log-list-categories` lists the logging
categories; `device_impl.cpp` logs "Testing for pre-compiled kernel ..." and
"Using precompiled kernel." at info level (`--log <category> --log-level info`).
Pick the category from the list; it is not guessed here.

The first OptiX or CUDA render after a kernel change is slower than the rest
(module creation / driver JIT); discard it when timing renders.

## 5. The Cycles regression tests

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

```bash
cd ../build_linux_neural
ctest -N | grep ' cycles_' | head                 # what exists
ctest -R '^cycles_bsdf_optix$' --output-on-failure  # one subject, one device
ctest -R '^cycles_.*_(cpu|cuda|optix)$' --output-on-failure 2>&1 | tee ../cycles_tests_baseline.log
# or everything, from the source tree (runs `ctest . --output-on-failure` in the build dir):
#   make test BUILD_DIR="$PWD/../build_linux_neural"
```

Record a **baseline before the first kernel change**: some GPU tests may fail
on this hardware for reasons unrelated to our work, and "failing set unchanged"
is the useful regression criterion afterwards.

A single subject without ctest (the `--oiiotool` path is what CMake resolved
from the precompiled libraries; print it with
`ctest -R '^cycles_bsdf_optix$' -N -V | grep -o -- '--oiiotool [^ ]*'`):

```bash
BLENDER_TEST_FILTER=diffuse python3 tests/python/cycles_render_tests.py \
    --blender ../build_linux_neural/bin/blender \
    --testdir tests/files/render/bsdf --outdir ../build_linux_neural/tests/cycles \
    --oiiotool <path from the ctest output> --device OPTIX
```

The PNG tests are coarse (8-bit, 1.6 % per-pixel tolerance, noise-limited).
They catch gross regressions, not the subtle bias a wrong closure produces.
That is what the float tools below are for.

## 6. Float render tests: `render_test.py`, furnace, MIS gate

These tools exist because PNG comparisons cannot see a 2 % energy error.

### Build the test scenes

```bash
$BLENDER -b --factory-startup --python doc/neural_rendering/tools/buildbox/make_test_scenes.py -- --out-dir scenes
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

### `render_test.py`

```bash
# render + compare against a stored reference (EXR with a .npy next to it)
$BLENDER -b scenes/mis_sphere.blend --python doc/neural_rendering/tools/buildbox/render_test.py -- \
    --out out/mis.exr --ref ref/mis.exr --samples 64 --device OPTIX --report out/mis.md
# first time: create the reference
... -- --out ref/mis.exr --ref ref/mis.exr --update-ref --device CPU --samples 1024
# white furnace
$BLENDER -b scenes/furnace.blend --python .../render_test.py -- --out out/furnace.exr --samples 64 --device OPTIX --furnace
# compare / furnace outside Blender, plain Python + numpy
python3 doc/neural_rendering/tools/buildbox/render_test.py compare out/mis.npy ref/mis.npy
python3 doc/neural_rendering/tools/buildbox/render_test.py furnace out/furnace.npy --expected 1.0 --tol 0.01
python3 doc/neural_rendering/tools/buildbox/render_test.py selftest
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
Blender passes `sys.exit` codes through, see section 9).

### `mis_consistency.py` (mandatory gate for any new closure)

```bash
$BLENDER -b scenes/mis_sphere.blend --python doc/neural_rendering/tools/buildbox/mis_consistency.py -- \
    --device OPTIX --samples 256 --out-dir out/mis
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

### Reading `ptxas` register reports

With `WITH_CYCLES_DEBUG=ON` every nvcc kernel compile prints `ptxas info`
lines (registers, spills, stack) [`cuda_add_common_flags`]. Capture them:

```bash
ninja -C ../build_linux_neural -v cycles_kernel_cuda 2>&1 | tee ../ptxas.log
grep -E 'ptxas info.*(Used|spill)' ../ptxas.log
# per-kernel numbers from the finished cubin (uncompressed copy next to the .zst):
cuobjdump --dump-resource-usage ../build_linux_neural/intern/cycles/kernel/device/cuda/kernel_sm_89.cubin | less
```

A ccache hit may replay no report; set `CCACHE_DISABLE=1` for that build if the
output is empty. Compare numbers before and after a closure change: register
growth in the megakernel is the cost that shows up as lost occupancy.

## 7. Environment report for the PR

```bash
python3 doc/neural_rendering/tools/buildbox/check_env.py                  # Markdown, starts with "[result]"
python3 doc/neural_rendering/tools/buildbox/check_env.py --build-dir ../build_linux_neural --strict
```

Run it with the Python used for training so the `torch` and `tinycudann`
imports mean something. It reports: GPU names, compute capability and driver
(`nvidia-smi`), both target cards, `nvcc` versions found, OptiX SDK paths with
the version decoded from `optix.h` as `FindOptiX.cmake` does, cmake/ninja/
ccache/git/git-lfs/compilers, whether `lib/linux_x64` and the LFS test data are
real, numpy/torch/tiny-cuda-nn, and for the build directory the relevant
`CMakeCache.txt` values, `bin/blender --version`, and which kernels were
installed. `--json` gives the same data for scripts. It only reads state
(version queries, `nvidia-smi`, `git rev-parse`, `blender --version`).

tiny-cuda-nn is outside this tree (BSD-3, allowed by `PROTOCOL.md`). Its
README describes `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
with `TCNN_CUDA_ARCHITECTURES=86,89` and an `nvcc` matching the CUDA version of
the installed PyTorch; this was not verified here.

## 8. Measuring iteration time (STATUS.md asks for these numbers)

Report four numbers for the `[result]` comment. Use a dedicated ccache
directory so the measurement does not wipe the cache you work with, and note
that **a `touch` is not an edit**: ccache hashes file contents, so touching
`bsdf.h` is a cache hit. The honest edit-compile-run cost is a content change.

```bash
cd ~/src/blender
export OPTIX_ROOT_DIR=...                       # as in section 3
export CCACHE_DIR=$HOME/.cache/ccache_timing    # fresh, empty
BUILD=$PWD/../build_linux_neural
TIMEFORMAT='%R s wall'
CFG=(cmake -G Ninja -C "$PWD/doc/neural_rendering/tools/buildbox/neural_dev.cmake" -S "$PWD" -B "$BUILD" -DCMAKE_BUILD_TYPE_INIT:STRING=Release)

# 1. cold build, empty ccache (the real compile cost)
rm -rf "$BUILD" "$CCACHE_DIR"; "${CFG[@]}" > /dev/null
time ninja -C "$BUILD" install > "$BUILD/../build_cold.log" 2>&1

# 2. cold build, warm ccache (fresh build directory, cache filled by step 1)
rm -rf "$BUILD"; "${CFG[@]}" > /dev/null
time ninja -C "$BUILD" install > "$BUILD/../build_cold_ccache.log" 2>&1

# 3. incremental after a real one-line change to the closure header (ccache miss)
printf '\n/* iteration-timing %s */\n' "$(date +%s%N)" >> intern/cycles/kernel/closure/bsdf.h
time ninja -C "$BUILD" install > "$BUILD/../build_incr_edit.log" 2>&1

# 4. incremental after only touching the file (ccache hit; shows what ccache buys)
sed -i '/iteration-timing/d' intern/cycles/kernel/closure/bsdf.h      # restores the file, nothing else is touched
time ninja -C "$BUILD" install > "$BUILD/../build_incr_restore.log" 2>&1
touch intern/cycles/kernel/closure/bsdf.h
time ninja -C "$BUILD" install > "$BUILD/../build_incr_touch.log" 2>&1
```

Do not use `git checkout -- intern/cycles/kernel/closure/bsdf.h` to undo step
3: it would also discard anyone else's uncommitted edit to that file. The OptiX
PTX modules are not wrapped by ccache (section 3), so steps 2 to 4 still pay
for them; that is the number to watch.

Where the time goes (durations in `.ninja_log`):

```bash
python3 - "$BUILD/.ninja_log" <<'PY'
import sys
rows = []
for line in open(sys.argv[1]):
    if line.startswith('#'):
        continue
    start, end, _mtime, out, _hash = line.rstrip('\n').split('\t')
    rows.append(((int(end) - int(start)) / 1000.0, out))
for secs, out in sorted(rows, reverse=True)[:10]:
    print('%8.1f s  %s' % (secs, out))
PY
```

Expected from `00_synthesis.md` (estimates, not measurements): 30 to 60 min cold,
5 to 15 min per CUDA architecture for an uncached kernel edit. Paste the four
wall times, the ten slowest steps, `nproc`, RAM and `ccache --show-stats` into
the `[result]` comment after the `check_env.py` block.

## 9. What only the workstation can validate

1. That `neural_dev.cmake` configures and builds: the CMake file was exercised
   against a dummy project (values land in the cache), not against this tree's
   configure, which needs the libraries and toolkits.
2. nvcc accepts the host gcc; `sm_86`/`sm_89` cubins build; `--split-compile=0`
   behaves with your nvcc.
3. `WITH_STRICT_BUILD_OPTIONS=ON` does not trip over an unrelated optional
   dependency (flip it off and grep the configure log for "not found, disabling").
4. All Blender-side scripts: they ran against fake `bpy` modules that reject
   unknown names (`test_blender_stage_stub.py`, `test_make_scenes_stub.py`),
   which catches typos and logic errors, not API behaviour. Specifically
   unconfirmed: `Image.pixels.foreach_get` on a loaded float EXR returns the
   raw linear RGBA; `bpy.data.images.new(..., float_buffer=True, is_data=...)`
   followed by `pack()` works for generated images (the script falls back to
   saving an EXR next to the `.blend`); `bpy.ops.render.render(write_still=True)`
   writes `<filepath>.exr` when `use_file_extension` is on; `sys.exit(n)` from a
   `--python` script becomes Blender's exit code (otherwise add
   `--python-exit-code 1` and rely on the report text); numpy is importable in
   the built Blender (`$BLENDER -b --python-expr "import numpy; print(numpy.__version__)"`).
5. That `direct_light_sampling_type` actually changes the sampling with the two
   developer preferences set from a background script (the report prints a
   heuristic note when it does not), and that the three strategies agree on the
   unmodified tree. Run `mis_consistency.py` once **before** changing any
   kernel code: a failure on stock Cycles would be a property of the scene or
   of the knob, not of a new closure.
6. That the furnace sphere gives 1.000 on stock Cycles with `OPTIX`, `CUDA`
   and `CPU`. Same reasoning: establish the baseline before the first closure.
7. Which Cycles regression tests pass on each GPU path before any change
   (section 5 baseline), and the iteration times of section 8.
8. That `--gpu-name` selects one card and both cards render the same image
   within the bias gate (run `render_test.py` once per card against the same
   reference).
9. tiny-cuda-nn builds against the installed PyTorch and imports.
