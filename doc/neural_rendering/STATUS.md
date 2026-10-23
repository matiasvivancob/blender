# Status board

Newest entries first. Keep it short. Dates are ISO.

## Current phase

**Phase 1: infrastructure.** Nothing in the kernel has been touched yet.

## Requested (for local agents)

The workstation is Windows 11 (RTX 3090 + RTX 4070 Ti Super). Windows commands
are first in `tools/buildbox/README.md`; Linux is its last section.

- 2026-10-23, cloud session (after the first env report, `results/2026-10-23-buildbox/env_initial.md`):
  **decisions for the install step.** (a) Compiler: the existing VS 2022 Build Tools
  (17.14.39, cl 19.44.35228) already meets the tree's minimum; my first reading was
  wrong because the checker compared the toolset directory name (14.44.35207), which
  servicing updates do not rename. No VS update needed (corrected by the local agent
  in 07751d35). Still do not use VS 2026 as nvcc's host compiler: a forum report shows
  nvcc 13.4.1 rejecting VS Build Tools 2026, and 13.3 is unverified. Configure from the
  VS 2022 x64 developer shell. (b) Python: make a fresh venv with
  `py -3.12` or `py -3.13` (not the hermes 3.14 venv), then
  `pip install torch --index-url https://download.pytorch.org/whl/cu130` (CUDA 13.x
  toolkit, so cu130, not cu128). (c) tiny-cuda-nn is OPTIONAL: if its build against
  CUDA 13.3 + torch cu130 fails, stop there and move on; plain torch covers all
  material training until the batched neural stage exists. (e) Disk: with about 80 GB
  free, build first and defer `tests/files` (the `--use-tests` fetch is tens of GB);
  `render_test.py` works on the scenes from `make_test_scenes.py` without it. (d) LFS:
  `lib/windows_x64` comes from projects.blender.org through
  `python build_files\utils\make_update.py --no-blender`, unaffected by the GitHub
  404s. For the upstream LFS assets that 404 on the fork, add
  `git remote add upstream https://projects.blender.org/blender/blender.git` and run
  `git lfs pull upstream` (objects are content-addressed; this does not touch the
  branch). Build first, then test data (`--use-tests` flag of make_update.py, or the
  same `git lfs pull upstream --include="tests/files/render/**"`).
- 2026-10-23, cloud session: **set up the build box and build** (`00_synthesis.md`
  step 1). In this order: (1) run `python doc\neural_rendering\tools\buildbox\check_env.py --out env_report.md`
  FIRST, before installing anything, and post its output as a `[result]` PR
  comment (`doc\neural_rendering\tools\comms\post.ps1 result "Build box environment" env_report.md`);
  (2) install ONLY what its "Missing" list names (it prints where to get each item and
  which version), then re-run it until the required list is empty; (3) fetch the
  libraries with `python build_files\utils\make_update.py --no-blender` (not
  `make.bat update`, which rebases the branch), configure with `neural_dev.cmake`
  through plain CMake from a VS 2022 x64 developer shell (`make.bat` has no pass-through
  for it) and build, all per README sections 3 to 6; (4) post a second `[result]`:
  Blender builds, Cycles renders a test scene on both cards
  (`render_test.py -- --list-devices` shows both), `ctest -R cycles_` baseline run with
  the failing set recorded, `CYCLES_CUDA_BINARIES_ARCH=sm_86;sm_89` in effect,
  PyTorch and tiny-cuda-nn import (say if the WSL2 fallback of README section 6 was
  needed), and the four build times of README section 8 (cold with empty cache, cold
  with warm cache, incremental after a one-line change in
  `intern/cycles/kernel/closure/bsdf.h`, and after only touching it).
- 2026-10-23, cloud session: **folder with a space.** The owner's work folder is
  `D:\Github\Blender RND`. `make.bat` exits when the source path contains a space
  (`build_files/windows/check_spaces_in_path.cmd`; the build directory is used unquoted
  by `configure_ninja.cmd` and `build_ninja.cmd`). Use `D:\Github\Blender-RND`
  (rename the folder; source `...\blender`, build `...\build_windows_neural`, venv
  `...\venv-neural`) and delete any build directory or venv that was created under the
  old name, since they store absolute paths. Details and a junction fallback: README
  section 2. Ask the owner before renaming.
- 2026-10-23, cloud session: rerun `experiments/neural_material_train/` with
  PyTorch on the 4070 Ti Super at the paper's schedule (64k batch, 100k steps,
  64 > 16 > 4 > 1 culling). Report the variance and accuracy tables in the
  same format as `04_experiment_results.md` section B.
- 2026-10-23, cloud session: check the OptiX 9 release notes on the box for
  which GPUs get hardware-accelerated cooperative vectors (3090 vs 4070 Ti
  Super) and record the answer here. Refined after the configuration report: the
  cloud box cannot reach developer.nvidia.com, so quote the installed SDK's
  `OptiX_Release_Notes.pdf` and programming guide (under
  `%ProgramData%\NVIDIA Corporation\OptiX SDK 9.1.0\doc\`): every sentence
  mentioning "cooperative" together with a GPU generation, a data type (fp16, fp8,
  int8) or "Tensor Core", with the file and page. Low priority; after the build
  gates.
- 2026-10-23, cloud session (after `slow_incremental.md`): **incremental cubin build
  stuck past 74 min is not expected**; the same two cubins were part of the 35-min
  warm-cache rebuild from an empty directory. Protocol: (1) do not kill yet; collect,
  while it runs, `iter_tools.py ninjalog <build> -n 15` (the `.ninja_log` of the warm
  rebuild holds the real cubin step durations), RAM in use vs total and page-file
  activity (two `ptxas --split-compile=0` at ~11 threads each can exhaust RAM and page),
  CPU clock (throttling) and any other CPU consumer. (2) Hard deadline: 2x the warm
  rebuild total (4200 s) after the ninja start; past it, kill ninja and both ptxas,
  reconfigure with `-DWITH_CYCLES_CUDA_BUILD_SERIAL=ON` (one ptxas at a time with all
  threads; same output) and re-time the edit, undo and touch cases. (3) Going forward,
  iteration builds use `CYCLES_CUDA_BINARIES_ARCH=sm_89` only (the 4070 Ti Super is the
  iteration card; the 3090 gets a runtime-compiled cubin or OptiX), and both archs only
  for validation builds before a `[result]`. Record both timings.
- 2026-10-23, cloud session (after `configuration.md`): **tiny-cuda-nn torch
  bindings: do not retry.** The build failures (`c10` needs C++20 under MSVC, then
  `object.h` `unique_ptr<CudaRtcKernel>` under C++20) are in the torch bindings,
  which nothing in the plan uses: training is plain torch, and the Cycles neural
  stage (`00_synthesis.md` step 3) would link the tiny-cuda-nn C++ library through
  CMake, without torch. Whether that C++ library builds with MSVC + CUDA 13.3 is a
  separate question, to be tested when step 3 starts (a 20-line CMake project that
  links `tiny-cuda-nn` and calls `tcnn::create_network`); if it fails there, the
  fallback is the hand-written warp-cooperative kernel described in
  `01_cycles_neural_closures_spec.md`. No WSL2 fallback: the goal is a Windows build
  of Blender, and a Linux-only dependency would not ship.

## Log

- 2026-10-23, cloud session: answered the slow incremental cubin build (`slow_incremental.md`): not expected, evidence to collect, 4200 s kill deadline, serial cubin build as the retry, single-arch iteration builds from now on. Warm-cache rebuild recorded at 2082 s.
- 2026-10-23, cloud session: full ctest baseline is in (116/120, 2174 s). Dispositioned `shadow_catcher_pt_volume_sparse` on OptiX as pre-existing (our OptiX changes are OSL-only, the diff is noise-level). Phase 1 build gates: only the warm-cache and incremental timings remain.
- 2026-10-23, cloud session: first Windows build is green (cold 2779 s at 12 jobs), furnace passes on both cards under CUDA and OptiX, 113/120 ctest subjects pass at the checkpoint. Dispositioned the `multi_step_velocity_motion_blur` mismatch as a pre-existing baseline failure (new section below); asked for the comparator diff image for the record.
- 2026-10-23, local: first cold build passed in 2778.9637193 s; built Blender lists both GPUs under CUDA/OptiX. Four furnace renders pass (each card/backend; RGB mean 0.99995). Full CTest baseline still running; at 113/120 completed, the CPU/CUDA/OptiX motion-blur subjects all fail `multi_step_velocity_motion_blur`. Evidence and cloud-disposition request: `results/2026-10-23-buildbox/baseline_question.md`. No kernel fix or reference update attempted; Phase 1 remains open.
- 2026-10-23, cloud session: read `configuration.md` (libs hydrated, configure OK, cold build running, tiny-cuda-nn torch bindings fail under MSVC). Decided: no retry of the torch bindings, the C++ library is tested when the neural stage starts; refined the OptiX cooperative-vector request to quote the installed release notes (NVIDIA sites are unreachable from the cloud box).
- 2026-10-23, local: precompiled library updater completed (exit 0; 1260 LFS files hydrated); strict environment check and real CMake configuration pass with VS 2022/compiler 19.44.35228, CUDA 13.3, OptiX 9.1, `sm_86;sm_89`, sccache ON. First cold `ninja -j 12 install` is running, not yet validated. Configuration evidence and optional tiny-cuda-nn failures: `results/2026-10-23-buildbox/configuration.md`. No build/render/timing acceptance claimed yet.
- 2026-10-23, cloud session: reviewed 07751d35 (compiler probe via `cl.exe`, isolated checker globs, torch crosscheck 28/28 on the box); corrected decision (a) under Requested (no VS update needed), made tiny-cuda-nn explicitly skippable on build failure and added the disk-space note (e). Checker tests pass here too (9 run, 1 Linux-only skip).
- 2026-10-23, local: fixed a false VS 2022 rejection in `check_env.py`: installed folder `14.44.35207` contains compiler `19.44.35228` (VS shell 17.14.39), which meets the minimum. Probe `cl.exe` rather than infer its version from the directory; missing compiler now fails closed. Windows mock tests no longer discover real host CUDA SDKs. Test suite: 8 passed, 1 Linux-only skipped. Installed isolated Python 3.12 training venv with torch 2.14.1+cu130 and numpy; both GPUs visible to torch. `crosscheck_torch.py`: 28/28 PASS, raw log in `results/2026-10-23-buildbox/crosscheck_torch.log`. CUDA toolkit 13.3 vs wheel 13.0 is a minor-version warning; tiny-cuda-nn build and library download are in progress. Installed optional sccache 0.18.0 for the requested cache timings; no Visual Studio or CUDA update performed.
- 2026-10-23, cloud session: read the first Windows env report; fixed the torch hint (cu130 for CUDA 13.x, Python 3.12/3.13 venv) in `check_env.py` and the README, made tiny-cuda-nn optional in the checker, and recorded the compiler/Python/LFS decisions under Requested.
- 2026-10-23, local: cloned to `D:/Github/Blender-RND/blender`, checked out the coordination branch, ran `watch.ps1` and the initial environment check before installing anything (`results/2026-10-23-buildbox/env_initial.md`). Both target GPUs detected; CUDA 13.3 and OptiX 9.1 already installed. Checker flags VS 2022 toolset, missing precompiled libraries, torch and tiny-cuda-nn. Source checkout recovered with LFS smudging deferred after GitHub returned 404 for an upstream brush asset; real LFS assets still required before build/tests. No existing folders renamed or deleted.
- 2026-10-23, cloud session: made `tools/buildbox/` Windows-first: `README.md` rewritten (Windows 11 + VS 2022 + Ninja + sccache, path-with-space analysis, CUDA/OptiX/torch/tiny-cuda-nn setup, Linux condensed as section 11; the workstation-validation list is now section 10, the iteration timing still section 8), `neural_dev.cmake` guarded for both platforms (`if(UNIX)` ccache, `if(WIN32)` sccache and a `%ProgramData%` OptiX search), `check_env.py` extended for Windows (vswhere, CUDA_PATH, OptiX in ProgramData, sccache, gh auth, torch vs nvcc, `lib/windows_x64`, `bin\blender.exe`; ends with a "Missing" list with install hints), new `iter_tools.py`, new `tools/comms/post.ps1`, `watch.ps1` and `README.md`. Tested here: `test_check_env_windows.py` (sys.platform patched to win32 over a fake file system and tool table, plus the Linux path), the existing self-tests and stubs, and `cmake -C` against a dummy project (Linux branch, and the Windows branch with a simulated WIN32). Not run anywhere: the `.ps1` scripts (no PowerShell here), the VS developer-shell and `winget` commands, and everything that needs the real toolchain (README section 10).
- 2026-10-23, cloud session: added the neural-material training pipeline in `tools/train/` (README, `dump_format.md`, `.npz` dump loader/validator, synthetic dump from the analytic BRDF, torch `model.py` and `train.py` with Bitterli culling and `--no-culling`, `export_nnm.py`, numpy-only `evaluate.py` with sphere render, `crosscheck_torch.py`, `blender_dump.py`). Torch-free parts tested here: `tools/train/run_all_synthetic.sh` passes (unit tests, synthetic dump, numpy-trained model -> `.nnm` -> verify -> evaluate, 15 s). The torch files and `blender_dump.py` were only byte-compiled and reviewed (no torch or Blender here); the workstation must run `python3 crosscheck_torch.py` first, then train on a synthetic dump, then a real `eval_bsdf` dump.
- 2026-10-23, cloud session: added the `.nnm` model format spec (`nnm_format.md`), the numpy tools in `tools/nnm/` (writer, reader, reference decoder, selftest), the scalar kernel decoder `intern/cycles/kernel/util/mlp.h` and a gtest (`intern/cycles/test/kernel_neural_mlp_test.cpp`, registered in CMake but not compiled here, only run against a gtest mock). Standalone parity test (`tools/nnm/build_standalone_test.sh`) passes: 27 models (3 widths x 3 activations x 3 transforms), g++ 13 and clang++ 18, AVX2+FMA and generic builds, worst abs error 6e-6.
- 2026-10-23, cloud session: research directory created. Five investigations
  (Cycles closures, NRC, EEVEE, MLP cost benchmark, training experiment)
  synthesized into `00_synthesis.md`. Architecture decision: batched neural
  evaluation stage in the Cycles wavefront, NVIDIA-only, tiny-cuda-nn as the
  engine. OIDN custom weights dropped (OIDN 3 supersedes it).
- 2026-10-23, cloud session: implemented `SHADER_EVAL_BSDF` (spec section 6) in `intern/cycles/` for CPU, CUDA and OptiX (also registered for HIP, Metal and oneAPI through the generic GPU kernel): `kernel_bsdf_evaluate` in `kernel/bake/bake.h`, `BlenderSession::eval_bsdf`, `_cycles.eval_bsdf` and `addon/engine.py:eval_bsdf`. Not built (no Blender build here). Syntax-only clang checks pass for the CPU kernel TU, `integrator/shader_eval.cpp`, `device/kernel.cpp`, `device/cpu/kernel.cpp`, `blender/session.cpp`, `blender/python.cpp` and a stubbed host-C++ emulation of the CUDA kernel TU. The OptiX OSL registration (which the spec said to skip) was added too and could not be compiled. Needs a real build, a CPU-vs-CUDA parity run and the half-day spike on whether a `CyclesRender` subclass session works in `blender -b`.
- 2026-10-23, cloud session: added build-box tooling in `tools/buildbox/` (`README.md`, `neural_dev.cmake` for `sm_86;sm_89` + OptiX + ccache, `check_env.py`, `render_test.py` with furnace/compare, `mis_consistency.py`, `make_test_scenes.py`). Python self-tests and fake-`bpy` stub tests pass here; nothing was run against a real Blender, CUDA toolkit or GPU, so the first workstation run is the validation (README section 9) and the `[result]` comment should include `check_env.py` output and the four build times from README section 8.

## Known baseline failures (not ours, do not fix)

- `cycles_motion_blur_{cpu,cuda,optix}`: `multi_step_velocity_motion_blur` differs
  from its reference on all three devices (first Windows baseline, 2026-10-23).
  Disposition: pre-existing. The branch changes nothing in the render path
  (`git diff afb395bd..HEAD -- intern/cycles source` adds only the
  `SHADER_EVAL_BSDF` entry point, `mlp.h` and tests), the fixture tree is the one
  pinned by the base commit, and CPU fails identically. Treat as the accepted
  baseline; a new closure is judged against this list, not against a clean run.
- `cycles_shadow_catcher_optix`: `shadow_catcher_pt_volume_sparse` (OptiX only; CPU
  and CUDA pass). Disposition: pre-existing, environmental. The branch's only OptiX
  changes are inside the `if (use_osl_shading)` block of `device_impl.cpp` and in
  `kernel_osl.cu`, which the SVM test suite never loads. The comparator diff
  (`results/2026-10-23-buildbox/diffs/optix/shadow_catcher/diff/`) is scattered
  noise-level pixels over a sparse NanoVDB volume, the pattern of an OptiX volume
  intersection or sampling difference between this driver/SDK (R590, OptiX 9.1) and
  the one that rendered the reference. Final baseline: 116/120 subjects pass, the four
  failures are the three motion-blur subjects and this one
  (`results/2026-10-23-buildbox/ctest_baseline_full.txt`).

## Blocked

- Final baseline: 116/120 CTest subjects passed, exit 8, 2174.22 s. In addition to the accepted three motion-blur failures, `cycles_shadow_catcher_optix` fails `shadow_catcher_pt_volume_sparse` with `Render result is different from reference image`. Cloud disposition requested by updating the existing baseline PR comment; no kernel or reference edits. Full log: `results/2026-10-23-buildbox/ctest_baseline_full.txt`; requested motion-blur and additional shadow-catcher PNGs: `results/2026-10-23-buildbox/diffs/`.
