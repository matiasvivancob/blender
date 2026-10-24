# [review] Slice 1: Neural BSDF node, `.nnm` loader, NeuralManager (branch `neural-p2`, head `5d0d064d60d`)

Per your go-ahead (`ef03d9fd`). Based on `44ee1d3433d` of the coordination branch; stacked work (slice 2, closure) is on
`neural-p2-closure` and **not** part of this review. Nothing is merged. Please review before I merge.

```text
git diff 44ee1d3433d..neural-p2 -- . ':!doc'     # 22 files, +1122 -2
```

## What it adds

| area | files | note |
|---|---|---|
| Blender node | `node_shader_bsdf_neural.cc` (+register, CMake), `BKE_node_legacy_types.hh` (`SH_NODE_BSDF_NEURAL 726`), `DNA_node_types.h` (`NodeShaderBsdfNeural { char filepath[1024]; }`), `rna_nodetree.cc` (`filepath`, `PROP_PATH_SUPPORTS_BLEND_RELATIVE`), `node.cc` (blend path remap), `node_add_menu_shader.py` | Cycles only (`object_cycles_shader_nodes_poll`, no `gpu_fn`). Output `BSDF`; inputs `Normal`, `UV` (hidden values) |
| Cycles loader | `scene/neural.{h,cpp}` | `NeuralFile::load` validates every rule of `nnm_format.md` section 8 (sizes, alignment, overlap, finite f16, proxy ranges, reserved bytes) and then **runs the kernel decoder on all embedded test vectors**; `NeuralManager` slots with MD5 sharing like IES, packs `neural_models` (uint records of `KernelNeuralModel`), `neural_weights`, `neural_latents` (level 0 only) |
| Cycles node | `shader_nodes.{h,cpp}` `NeuralBsdfNode`, `blender/shader.cpp` sync, `scene.{h,cpp}`, `devicescene.{h,cpp}`, `kernel/data_arrays.h` | slice 1 shading = a diffuse closure with the file's albedo, black if the file is invalid or has no proxy block |
| Tools | `proxy_fit.py`, `test_proxy_fit.py`, `export_nnm.py --dump/--proxy`, `neural_node_check.py` | proxy fit as you specified (grid search of the second moment) |

## Gates (all executed, logs on the workstation)

| gate | result |
|---|---|
| Build | sm_89 incremental and a coherent two-architecture validation build (`sm_86;sm_89`, serial cubins): exit 0, 644 s |
| Header dependencies (`iter_tools.py depcheck`) | PASS |
| `ctest -R "^cycles$"` (unit tests) | all pass: 228 gtests including 4 new `SceneNeural.*` (every embedded Python-written `.nnm` loads and its fields match the test parser; 15 corrupted copies are rejected with the right reason; each proxy range check; manager sharing, freeing and reuse of slots) |
| `ctest -R "^cycles_"` | 116/120, failing set **identical** to the accepted one: `motion_blur` cpu/cuda/optix and `shadow_catcher` optix. 2067.8 s |
| `neural_node_check.py`, CPU, CUDA 3090, CUDA 4070, OptiX 3090, OptiX 4070 | PASS on all five. Node exists, `filepath` survives save and reload (blend-relative), the neural node renders **identically to a Diffuse BSDF of the file's albedo** (read independently from header bytes 172..183): max abs difference 0 on CPU, 1.4e-6 on GPUs; a missing file and a file without proxy block render exactly black, no crash |
| `test_proxy_fit.py` | 5 numpy tests pass (pdf integrates to its analytic value `1/(1+alpha^2)` at normal incidence, zero below the horizon, fit recovers diffuse and glossy cases) |

## Decisions you may want to look at

1. **No EEVEE support, deliberately.** No `gpu_fn`, and I did not add the node to the two BSDF lists in
   `node_shader_tree.cc` (EEVEE Weight handling). I did not render it in EEVEE.
2. **OSL**: the node's OSL `compile` falls back to `node_diffuse_bsdf` (build has OSL off; untested).
3. **Placeholder semantic**: the invalid-file path logs one error per node compile and renders black. A louder
   fallback (magenta) is a one-line change if you prefer.
4. **`KernelNeuralModel`/format untouched.** The proxy block is carried to the device in a separate
   `neural_proxy` array (added in slice 2) so your struct and its 64-byte `static_assert` stay as they are.
5. **Dependencies tracking**: this work was built with the `showincludes_fix.cmake` setup from the previous result;
   without it, header edits would not have rebuilt the CPU code.
6. Plain `git status` check: no model names in commits; no binaries.

## Not done in this slice

Real shading (slice 2, in progress), per-kernel register counts (no kernel code changed in slice 1, only three
new global array pointers in `data_arrays.h`; I will report register counts before and after slice 2).
