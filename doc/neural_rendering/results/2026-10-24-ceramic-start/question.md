# Procedural Ceramic selected; reference captured, two prerequisites for training

## Current decision — owner approval and advisor disposition

The owner explicitly approved preserving emission separately: **"Let's do that for the emission."** This confirms the hybrid approach approved by the advisor in [comment 6385746382](https://github.com/matiasvivancob/blender/pull/1#issuecomment-6385746382) and coordination commit `bc0f8b18098`.

- **Replacement:** `Add Shader(Neural BSDF scattering, original native emission branch)`. Preserve the emission color, strength and dependencies; retain only emission in the native contribution, not the original scattering a second time. Do not drop the glow, bake it into the BSDF, or extend `.nnm` for this experiment. Measure retained native work and the complete hybrid render cost explicitly.
- **Source:** ` Procedural Ceramic` on `Sphere`. Keep the owner's playground and original material intact; first qualification is an object-specific bake on this sphere.
- **Other issue resolved for implementation:** proceed with `06_uv_tangent_frame.md` on `neural-p4`. The advisor corrected the earlier diagnostic criterion: the confirmed 0/45-degree edge alternation is sufficient; similar average errors in the two classes do not disprove the defect. No further attribution claim is inferred from those averages.
- **Order:** numpy 45-degree frame-rotation decoder check → shared UV-tangent helper and dump-frame marker → build/tests, retrain and revalidate the existing probe → review → ceramic sampling and training in the corrected frame. Check non-overlapping sphere UVs and choose latent resolution from visible detail; start at 128x128 if 64x64 loses detail.
- **Current execution state:** native reference and emission-only check complete; correction, retraining and hybrid ceramic implementation are not yet complete. This update records approval, not a claim that training has started or the issues are fixed.

The original request and evidence below are retained as history; the two design questions are answered by the decisions above.

---

## Original request and evidence

Owner explicitly selected the material ` Procedural Ceramic` (leading space preserved) on object `Sphere` in the saved playground. This is now the selected next real-material experiment; not a new generated material. Source branch for existing runtime: `neural-p3` @ `ff5115d7d9d` (runtime renderer source `d03695edf72`). Coordination currently `cb836ce6582` before this report.

## 1. Native reference and material contract

Executed in Blender 5.4 Alpha from the verified neural build:

```text
blender.exe -b --factory-startup --disable-autoexec D:/Github/Blender-RND/demo/neural_material_playground.blend --python-exit-code 1 --python C:/Users/Matia/workspace/procedural-ceramic-experiment/capture_reference.py
```

Exit 0, `CERAMIC REFERENCE PASS`. Copied the scene, retained shader node values and all lighting/geometry, and added a close-up comparison camera. Native reference is 768x768, 128spp, OptiX 4070, denoised for appearance only. The owner's source hash is unchanged: `f518e4c13670ab666a5ca9d82ad299ab91316e735037a4da477d9afe1ad74e82`. No training or speed measurement yet.

Actual material surface is an Add Shader combining a ceramic group and a `Celestial Starfield` group. It includes Object/Generated coordinates, bump, multiple reflected lobes and **emission**. The root output's Displacement is unlinked. For the first comparison we will preserve the source sphere and treat this as an object-specific UV bake, not a reusable arbitrary-object material.

Emission is not only a dormant socket: a second no-light/black-world render with every other geometry object hidden and the source shader unchanged gives:

| native emission-only CPU test, 256x256, 16spp | value |
|---|---:|
| max linear RGB | 0.0241839215 |
| mean linear RGB | 0.0000904148 |
| pixels with RGB > 1e-7 | 22599 |

Command: same build `-b --factory-startup --disable-autoexec C:/Users/Matia/workspace/procedural-ceramic-experiment/ceramic_reference.blend --python-exit-code 1 --python C:/Users/Matia/workspace/procedural-ceramic-experiment/check_emission.py`; exit 0, `CERAMIC EMISSION CHECK` emitted. The local `.blend`, EXR, native PNG and original third-party node groups are deliberately not in git.

**Design question:** current BSDF `.nnm` has no emission output. For this owner's exact material, should we preserve emission as a separate native branch / UV bake and train only scattering, or extend the learned representation? My smallest complete proposal is hybrid neural scattering + preserved emission, with explicit measurement of retained native work; no whole-graph replacement or speed claim until measured. I have not removed emission or launched training.

## 2. Requested cheap tangent-frame confirmation

Executed your prescribed geometry-angle check before any renderer edit, using the existing CPU slice-3 residual:

```text
blender.exe -b --factory-startup --disable-autoexec D:/Github/Blender-RND/scenes/neural_material_probe.blend --python-exit-code 1 --python C:/Users/Matia/workspace/procedural-ceramic-experiment/confirm_frame_issue.py
```

Exit 0. `loop_triangles` edges alternate exactly **0 degrees / 45 degrees** within each quad (256 triangles each), confirming the changing dPdu edge frame.

| triangle edge class | sampled plane-hit pixels | mean raw relative RGB residual | median | p99 |
|---|---:|---:|---:|---:|
| 0 degrees | 6947 | 0.02277035 | 0.01796664 | 0.08475945 |
| 45 degrees | 6977 | 0.02213545 | 0.01829166 | 0.07241106 |

Method: every fourth pixel of retained 512x512 CPU beauty arrays; inverse camera projection, world-space BVH ray hit into the probe plane, associate hit triangle class; `mean(abs(neural-native)/max(abs(native),1e-3))` per RGB pixel. This excludes pixels missing the plane, unlike the earlier whole-image metric. No render rerun for this check.

The class means differ only 2.87% relative, and medians reverse slightly; tails differ more. This is **weak class separation, not a decisive attribution**. Per your instruction to post/stop if the residual does not clearly differ, I am holding the kernel frame edit for your reading. Do you consider this sufficient to proceed with `06_uv_tangent_frame.md`, or want a more specific diagnostic?

No build/test error or new traceback: these are experiment-contract/design questions. Local artifacts and reproducible scripts live under `C:/Users/Matia/workspace/procedural-ceramic-experiment/`; the angle/residual JSON and emission metrics are committed with this report, not the source material graph.
