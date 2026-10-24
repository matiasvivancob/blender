# Slice 2 final-head gates: one accepted baseline failure passed

Source: `neural-p2-closure` @ `d03695edf723728778565b5530edfa2988a58b96`, including coordination head `0731add6d0f` (encoding-2 decoder and loader). No rebase. Slice 2 is **not merged**.

The build and all functional gates passed. The sequencing harness stopped because its literal equality check expected four baseline failures and got three: the OptiX shadow-catcher subject passed. No new failure or skipped subject. Please confirm whether 117/120 with this subset satisfies your "failing set unchanged" condition; I have not waived it or rerun the full suite.

## Commands and results

Executed by `python D:/Github/Blender-RND/qualify_slice2.py` (runner and large outputs local, `real/final_slice2_20261024/`).

| Command / gate | Result |
|---|---|
| `python validation_build.py "sm_86;sm_89"` | exit 0, 137.3 s; configure/build/install coherent both architectures |
| `ctest -R "^cycles$" --output-on-failure --output-junit unit.xml` | exit 0; aggregate cycles unit suite passed (228 gtests requested), including updated embedded encoding-2 models |
| `python doc/neural_rendering/tools/buildbox/iter_tools.py depcheck D:/Github/Blender-RND/build_windows_neural` | PASS; 9/201 zero-dependency objects, 4.5% < 10% limit |
| furnace CPU | PASS, neural/Principled mean 0.9989171, zero pixels above one beyond 4 sigma |
| furnace CUDA 3090 | PASS, mean ratio 0.9989120, zero excess pixels |
| furnace CUDA 4070 | PASS, mean ratio 0.9989121, zero excess pixels |
| furnace OptiX 3090 | PASS, mean ratio 0.9989121, zero excess pixels |
| furnace OptiX 4070 | PASS, mean ratio 0.9989121, zero excess pixels |
| `ctest -R "^cycles_" --output-on-failure --output-junit regression.xml` | exit 8, 2269.8 s; 120 subjects, 117 passed, 3 failed, none skipped |

Each furnace invocation used `blender.exe -b --factory-startup --disable-autoexec scenes/neural_material_probe.blend --python-exit-code 1 --python doc/neural_rendering/tools/buildbox/neural_vs_principled.py -- <absolute case dir>`, `NR_MODEL=real/cull_w32/model_proxy.nnm`, `NR_SPP=256`, `NR_SEEDS=4`, and explicit NR_DEVICE/NR_GPU. Separate local preference directories; desktop Blender untouched.

Actual failures: `cycles_motion_blur_cpu`, `cycles_motion_blur_cuda`, `cycles_motion_blur_optix`. `cycles_shadow_catcher_optix` **passed**. No renderer or reference-image fix attempted.

Exact harness error (`qualify_slice2.py:51`, not a renderer exception):

```text
AssertionError: baseline failures changed: ['cycles_motion_blur_cpu', 'cycles_motion_blur_cuda', 'cycles_motion_blur_optix']
```

One additional packaging fix in this head: registered `closure/bsdf_neural.h` in `intern/cycles/kernel/CMakeLists.txt`, which was omitted in slice 2. Verified the installed copy now exists and matches source bytes; this also makes device custom-command dependencies include the header. Earlier precompiled CUDA/OptiX runtime tests were valid but installed-source/JIT completeness had not been verified.

Evidence here: `unit.log`, `depcheck.log`, `regression.xml`, and `summary.json` (complete furnace numbers and commands). Full regression log remains local. Slice 3 comparison renders are proceeding independently while the merge stays blocked; their completion will not silently waive this condition.
