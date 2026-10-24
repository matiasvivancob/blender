# SHADER_EVAL_BSDF: first real build, CPU vs CUDA vs OptiX parity on a Principled material

Result: **it builds and runs on all three devices; albedo agrees to 1e-4; f·cos and pdf agree to
7e-5 median / 0.5 % p99, with a magnitude-dependent spread at the sharp clear-coat peak (max 11 %)
and a +0.36 % total-energy offset (GPU higher), almost all of it in the brightest 0.4 % of records.**
CUDA and OptiX are bit-identical to each other (the shading kernel is the shared CUDA one).

## What ran

- Executable: the first cold build of this branch, preserved as
  `D:/Github/Blender-RND/install/baseline-first-build/blender.exe` (`git diff da3963af977 HEAD --
  intern source scripts` is empty, so it matches the current renderer source; coherent
  `sm_86;sm_89`; the live build directory is now sm_89-only for iteration).
- Scene: `make_test_scenes.py` `neural_material_probe.blend`, object `ProbePlane`
  (Principled: metallic 1 base, image-texture roughness 0.08..0.92, clear coat 1.0 with coat
  roughness 0.05).
- Driver: `tools/train/blender_dump.py`, unchanged, plus a 20-line pre-script that pins the
  Cycles device (exactly one device enabled, readback printed) via environment variables,
  because `blender_dump.py` parses everything after `--` strictly.
- Commands (per device; `NR_DEVICE` = CPU | CUDA | OPTIX, `NR_GPU=3090`):
  ```text
  blender.exe -b neural_material_probe.blend --python-exit-code 1 --python pin_device.py
      --python tools/train/blender_dump.py -o <abs>/render_<dev> -E BSDF_DUMP -f 1 --
      --object ProbePlane --n 200000 --seed 7 --out <abs>/dump_<dev>.npz
  ```
  Device readbacks: `CPU`, `CUDA:NVIDIA GeForce RTX 3090`, `OPTIX:NVIDIA GeForce RTX 3090`.
  Each: exit 0, 200,000 records, 0 flagged, 1.5 to 1.8 s process wall time.
- Same seed, so the query arrays (`prims`, `uv`, `wi`, `wo`) are asserted identical before
  comparing `out`. Comparator has an assert self-test (mismatched queries are rejected).

## Numbers (CUDA vs CPU; OptiX vs CPU is identical to every digit)

| column | median rel | p99 rel | max rel | max abs | CPU mean | GPU mean |
|---|---|---|---|---|---|---|
| f·cos R | 7.2e-05 | 4.9e-03 | 0.111 | 3705 | 51.817 | 51.999 |
| f·cos G | 7.1e-05 | 5.1e-03 | 0.111 | 3705 | 51.404 | 51.586 |
| f·cos B | 7.1e-05 | 5.2e-03 | 0.111 | 3705 | 51.274 | 51.457 |
| pdf | 7.3e-05 | 5.1e-03 | 0.111 | 4093 | 56.990 | 57.196 |
| albedo R | 5.0e-07 | 5.2e-06 | 1.4e-05 | 1.3e-05 | 0.95378 | 0.95378 |
| albedo G | 3.4e-06 | 3.3e-05 | 8.3e-05 | 4.2e-05 | 0.69121 | 0.69121 |
| albedo B | 4.2e-06 | 4.0e-05 | 9.9e-05 | 4.1e-05 | 0.61280 | 0.61280 |

Where the f·cos (G) difference sits, by CPU magnitude:

| CPU f·cos range | records | mean signed rel (GPU-CPU) | max abs rel | share of total energy |
|---|---|---|---|---|
| [0, 1) | 161,509 | +2.0e-06 | 3.1e-03 | 0.002 |
| [1, 10) | 28,017 | +9.1e-05 | 1.4e-02 | 0.008 |
| [10, 100) | 7,267 | +8.0e-04 | 4.3e-02 | 0.022 |
| [100, 1000) | 2,406 | +2.7e-03 | 1.1e-01 | 0.075 |
| [1000, inf) | 801 | +4.0e-03 | 1.1e-01 | 0.892 |

GPU/CPU total-energy ratio 1.00355. Ratio histogram is continuous and two-sided (0.9 to 1.11,
0.54 % of records beyond 1 %); f and pdf move by the same factor per record, so the spread is in a
shared term, most plausibly the GGX D of the coat at alpha ≈ 0.0025 near the mirror direction
(`--use_fast_math` on nvcc vs precise math on CPU). That is a hypothesis, not isolated.

## Not verified

- **The prim-index assumption** (`prim` = index into the evaluated mesh's `loop_triangles`).
  `--probe` ran (exit 0, 8 records, finite values, albedo 0.92 to 0.95) but it only prints prims,
  tex UV and outputs; it does **not** perform the barycentric round trip its docstring promises.
  The parity above is device-vs-device on the same queries, so it cannot detect a wrong mapping.
- A side effect: `-E BSDF_DUMP -f 1` saves the render result; without `-o` it wrote
  `C:\tmp\0001.png` during the probe run. The parity runs pass `-o` to keep outputs local.

## Questions

1. Is a +0.36 % energy offset and 11 % max deviation at a coat-roughness-0.05 peak acceptable for
   training data, or should GPU dumps be avoided / should the probe material use a less extreme
   coat for parity?
2. Should `--probe` get a real prim/UV round-trip check (cloud-side change to
   `tools/train/blender_dump.py`), before any real-material dump is trusted?

Raw dumps and logs stay local (`D:/Github/Blender-RND/parity/`). Committed next to this file:
`parity.json` (full comparator output), `compare_dumps.py` (run `--selftest`), `pin_device.py`. Training on the 4070 was running concurrently; the three
dumps took under 6 s in total and are noted against run `w16_baseline_s0`'s wall time.
