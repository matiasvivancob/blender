# Status board

Newest entries first. Keep it short. Dates are ISO.

## Current phase

**Phase 1: infrastructure.** Nothing in the kernel has been touched yet.

## Requested (for local agents)

- 2026-10-23, cloud session: set up the build box per `00_synthesis.md` step 1.
  Confirm with a `[result]` PR comment: Blender builds, Cycles renders a test
  scene on both cards, `tests/python/cycles_render_tests.py` runs,
  `CYCLES_CUDA_BINARIES_ARCH=sm_86;sm_89` is in effect, PyTorch and
  tiny-cuda-nn import. Report the cold and incremental build times for a
  one-line change in `intern/cycles/kernel/closure/bsdf.h`.
- 2026-10-23, cloud session: rerun `experiments/neural_material_train/` with
  PyTorch on the 4070 Ti Super at the paper's schedule (64k batch, 100k steps,
  64 > 16 > 4 > 1 culling). Report the variance and accuracy tables in the
  same format as `04_experiment_results.md` section B.
- 2026-10-23, cloud session: check the OptiX 9 release notes on the box for
  which GPUs get hardware-accelerated cooperative vectors (3090 vs 4070 Ti
  Super) and record the answer here.

## Log

- 2026-10-23, cloud session: added the `.nnm` model format spec (`nnm_format.md`), the numpy tools in `tools/nnm/` (writer, reader, reference decoder, selftest), the scalar kernel decoder `intern/cycles/kernel/util/mlp.h` and a gtest (`intern/cycles/test/kernel_neural_mlp_test.cpp`, registered in CMake but not compiled here, only run against a gtest mock). Standalone parity test (`tools/nnm/build_standalone_test.sh`) passes: 27 models (3 widths x 3 activations x 3 transforms), g++ 13 and clang++ 18, AVX2+FMA and generic builds, worst abs error 6e-6.
- 2026-10-23, cloud session: research directory created. Five investigations
  (Cycles closures, NRC, EEVEE, MLP cost benchmark, training experiment)
  synthesized into `00_synthesis.md`. Architecture decision: batched neural
  evaluation stage in the Cycles wavefront, NVIDIA-only, tiny-cuda-nn as the
  engine. OIDN custom weights dropped (OIDN 3 supersedes it).

## Blocked

- Nothing yet.
