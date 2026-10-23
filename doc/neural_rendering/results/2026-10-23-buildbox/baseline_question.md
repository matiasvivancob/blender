# Baseline motion-blur mismatch: request disposition, no local kernel fix

Branch: `claude/amazing-planck-xvnke7`. Tested build source includes `da3963af977`; subsequent integrated changes through `177754114f5e2908366d0ca37503ef2c793b17ed` are documentation only. The posting commit is in the PR comment header.

| Executed check | Result |
|---|---|
| First cold build, 12 jobs, empty sccache | exit 0; 2778.9637192999944 seconds |
| Built Blender | 5.4.0 Alpha |
| Device listing | RTX 3090 and RTX 4070 Ti SUPER present under CUDA and OptiX |
| Furnace 128x128, 64 spp, each GPU/backend separately | all four PASS; RGB mean 0.99995, deviation -0.005% |
| CTest baseline | in progress at checkpoint; 113 of 120 subjects completed |
| Failed subjects at checkpoint | `cycles_motion_blur_cpu`, `cycles_motion_blur_cuda`, `cycles_motion_blur_optix` |

Command, from `D:/Github/Blender-RND/build_windows_neural`:
```text
ctest -R cycles_ --output-on-failure
```

Each failing subject reports the same fixture mismatch:
```text
[       OK ] multi_step_motion_blur
 Render result is different from reference image
[ FAILED   ] multi_step_velocity_motion_blur
```

Test fixture is under `tests/files/render/motion_blur/`; no source file/line is identified by the image comparator. No kernel edits or reference updates attempted. First foreground baseline hit the 180-second tool timeout; its log is retained locally. A complete background baseline was then started and is still running. No test data was explicitly fetched after the build; the updater's initial upstream fallback had already fetched assets, so CTest found real fixtures.

Please review whether this is an accepted pre-existing baseline mismatch or needs a cloud-authored fix. I will record the final failing set and can continue independent setup/timing work, but will not patch kernel code or update reference images.

Committed evidence in this directory:
- `build_cold_timing.json`, `build_cold_last200.txt`
- `devices.txt`
- `cuda_3090.md`, `optix_3090.md`, `cuda_4070.md`, `optix_4070.md`
- `ctest_checkpoint.txt` (partial, not a completed baseline)

Local full baseline log: `D:/Github/Blender-RND/ctest_baseline_full.log`. This request is for disposition of the shared motion-blur mismatch, not a claim that Phase 1 is complete.
