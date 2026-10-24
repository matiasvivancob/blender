# UV-tangent frame correction: ready for code review, final gates running

Plain result: the new frame is consistent across triangles, and retraining reduces the probe image error from **1.944% to 1.576%**. The large triangle-cell pattern is gone from the inspected residual; fine grid/stripe errors remain. Ceramic training has not started.

Branch: `neural-p4` @ `341469b0c0b` (renderer commit `6518ed887bc`). Base: coordination `f2f134c2de8`. No rebase. Please review the implementation while the remaining gates finish; **do not merge yet**.

## Implemented

- Shared raw UV tangent lookup in `kernel/geom/triangle.h`, with projection and degenerate-frame math in registered/installed `kernel/closure/bsdf_neural_frame.h`. Mirrored determinant sign retained; missing UV/non-triangle inputs fall back to dPdu.
- Neural node requests its UV attribute (default or directly connected named UV Map), passes its attribute id through `SVMNodeNeuralBsdfData`, and shading uses the same helper as baking.
- `KERNEL_BSDF_EVAL_UV_TANGENT_FRAME` bit 1; driver uses flags 3 and records `frame: uv_tangent`; trainer rejects old-frame dumps. Analytic synthetic producers explicitly mark their texture-axis frames. Two Python frame-marker tests pass.
- Three C++ frame-math tests cover rotated/mirrored UVs, both quad triangles, degenerate UVs and normal projection. The full `cycles` unit target passes.
- An actual first-frame anisotropic Principled check (metallic 1, roughness .4, anisotropy .8, explicit UV tangent) distinguishes the old frame and requires both triangles to match in the new one: **old relative difference .50264; new 0**, CPU/CUDA/OptiX 4070 all pass.
- That check initially caught a missing source-UV export: named Tangent/Generated/Object-only graphs may not request ATTR_STD_UV. Added session-scoped `Scene::bsdf_eval_uv` request before BSDF sync and tag a first-time resync, so the helper does not silently fall back. The corrected test proves this path rather than forcing an extra UV texture into the fixture.

## Results already complete

Same 2x32, latent 64x64x8, culling, seed 0, batch 65536, 100000 steps, LR .02. Fresh CPU dumps 2x4M and fresh sphere ground truth in the UV frame. Training wall time 731.73 s. No weights/datasets in git.

| metric | old edge frame | corrected UV frame |
|---|---:|---:|
| held-out relative L1 | .227887 | .204835 |
| sphere relative L1 | .025849 | .025038 |
| same-camera CPU image mean raw RGB relative error | 1.944% | 1.576% |
| same-camera CPU image p99 | 8.272% | 7.385% |
| same-camera OptiX image mean / p99 | 1.944% / 8.274% | 1.576% / 7.385% |
| same-camera CPU mean RGB change vs native | -0.0583% | -0.6472% |

The CPU native image is byte-identical in old and new comparisons. Both comparisons use the same 512x512 camera, scene, 1024 spp, no denoise/adaptive/compositor. See `comparison_uv_frame.png`: top native / old neural / corrected neural; bottom native repeat error x8 / old error x8 / corrected error x8. Shared display -2 EV, Reinhard, sRGB. Metrics come from unclipped scene-linear arrays. The coarse triangle pattern visibly disappears, but approximation error remains.

The requested pre-build numpy 45-degree rotation diagnostic ran on 65536 paired samples: median record-relative difference .02931, mean .04453, p99 .24347 (floor 1e-6); this demonstrates model azimuthal error, not causal attribution of every image residual. Its JSON includes floors and self-checks.

Both full comparison processes exit 0; export fp16 checks and evaluation exit 0. No speedup claim; new wall-time slope ratios are ~4.93x CPU and ~3.13x OptiX (diagnostic, variable setup and desktop contention, not qualified timings).

## Build blocker resolved without closing owner's Blender

The first build compiled all code but failed `LNK1104` replacing the executable used by the owner's live playground. Preserved the original process and pre-frame runtime; reused compiled objects while redirecting only the Blender target's runtime output via a root-directory deferred CMake hook, retaining the MSVC/sccache include-prefix fix. Installation is now `D:/Github/Blender-RND/install/neural-p4/blender.exe`. CTest regenerated against this install path. No source workaround or user process closure.

Exact intermediate command: `python -u C:/Users/Matia/workspace/uv-frame-fix/run_pipeline.py`. Source/runtime inputs correspond to this branch; first build was dirty but those exact edits are now committed above. The pipeline stage JSON, anisotropy outputs and metrics are retained. Header installation verified.

## Remaining gates — currently RUNNING, not accepted

`python -u C:/Users/Matia/workspace/uv-frame-fix/finish_gates.py` is an automatically chained tracked job: coherent sm_86+sm_89 build/install, unit/depcheck, furnace on all five device/backend combinations with the **newly retrained model**, MIS CPU+OptiX, then the 120-subject regression with no skipped/new failures. Artifacts: `D:/Github/Blender-RND/real/uv_frame_p4/final_gates/`. I will update this same review with results; no merge or ceramic training before the prerequisites pass.

## Ceramic UV preparation in parallel

Read-only audit of the frozen ceramic copy found that active/render map `automap` is unsuitable: 15088/15872 triangles extend outside 0-1 and extensive overlaps. Existing `UVMap` is within 0-1, no degenerate triangles, but exact clipping reports eight very small overlap slivers (~1.05e-9 to 1.86e-9 UV area). Inspection used a VIEWPORT depsgraph with subdivision levels aligned to saved render levels (2 rather than viewport 4), not a claim of an actual render-depsgraph inspection.

Proposed ceramic path: select `UVMap` as active/render on the **experiment copy**, retain original shader Object/Generated behavior and emission, and document/check those tiny boundary slivers before dumping. Do you accept them as numerical boundary artifacts, or require a separate cleaned bake UV? No user UVs were edited. Owner has already approved the hybrid scattering/native-emission contract.
