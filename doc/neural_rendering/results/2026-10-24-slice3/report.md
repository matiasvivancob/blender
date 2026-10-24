# Slice 3: matched render comparison, and slice 2 merged

Plain result: the trained neural material reproduces the probe's broad spatial/reflectance structure closely in a real Cycles render, but this scalar inline implementation is **slower** than the source Principled material. This is a working material implementation, not a speed optimization yet.

## Integration and provenance

- Slice 2 merged with merge commit `a92a0d9ca5c2f415f64e069875d166866411a3de`, after approval in PR comment 6384700080. No rebase.
- The 117/120 retained regression now passes the advisor-approved rule `failed <= accepted`, no subjects skipped. Original failed harness output preserved. The parser was re-run on retained XML, with explicit negative checks for a new failure, skipped subject and missing subject. No full-suite rerun needed.
- Runtime source was `d03695edf723728778565b5530edfa2988a58b96`; the merge adds docs, not different renderer logic. Executable SHA256 `dbed2114c7e7f700dd56b58147d435655a2a409c621149d565ae23e4213d5cc4`.
- Harness: `neural-p3` @ `ff5115d7d9d`; `doc/neural_rendering/tools/buildbox/neural_vs_principled_render.py`. Please review this slice before merging its harness.
- Same saved `scenes/neural_material_probe.blend`, same scene camera, geometry, lighting and world. Only material changed between original and neural. Model: the existing `real/cull_w32/model_proxy.nnm` (2x32, 64x64 latent, culling trained). No new test material or training.
- Effective overrides: 512x512, 100% resolution, beauty 1024 spp, adaptive sampling/denoising/compositing/persistent data off, 4 max bounces, Standard display. Explicit CPU or sole OptiX 4070 device. Factory startup with separate local preference directories. Source `.blend` not saved or modified.

Reproduce each backend with `NR_DEVICE=CPU` or `NR_DEVICE=OPTIX NR_GPU=4070`, `NR_SPP=1024 NR_RES=512 NR_MODEL=D:/Github/Blender-RND/real/cull_w32/model_proxy.nnm`:

```text
blender.exe -b --factory-startup --disable-autoexec D:/Github/Blender-RND/scenes/neural_material_probe.blend --log-level 1 --python-exit-code 1 --python <source>/doc/neural_rendering/tools/buildbox/neural_vs_principled_render.py -- <absolute output directory>
```

Both processes exited 0. Verified 13 completed render intervals per backend against terminal `Rendered N samples` lines, every count matches its request. Arrays are finite, nonblack, matching 512x512 RGB. No image-cache or neural-model fallback message.

## Image error (linear data, before display mapping)

Raw relative RGB error is `abs(neural - native) / max(abs(native), 1e-3)`, channelwise over pixels whose native mean RGB > 1e-3. Native noise floor is an independent native seed versus the first native seed. The native and neural beauty both use seed 0.

| backend | mean raw RGB relative error | p99 | native-repeat mean / p99 | mean RGB energy change |
|---|---:|---:|---:|---:|
| CPU | 1.944% | 8.272% | 0.097% / 1.471% | -0.0583% |
| OptiX 4070 | 1.944% | 8.274% | 0.097% / 1.472% | -0.0568% |

Separately, 8x8-box-filtered mean-RGB (not perceptual luminance) error is 1.499% mean / 5.897% p99 on CPU and 1.499% / 5.896% on GPU. These filtered diagnostics must not replace the raw RGB errors above. No FLIP package was installed; none available in the checked analysis interpreter.

Visual inspection of `comparison.png`: the broad diagonal bands and highlight placement agree, with no black output or catastrophic shift. There are subtle local/block-like differences in the neural result; the amplified residual makes structured grid/triangle patterns visible. This is not exact reproduction. CPU and OptiX show essentially the same appearance. Owner visual acceptance remains pending, not inferred from autonomous work.

Contact sheet: original / neural / absolute difference x8, CPU row then OptiX row. Shared diagnostic display is -2 EV, per-channel Reinhard, then sRGB, not the scene's original display. No per-image normalization. All metrics use the untouched scene-linear arrays.

## Cost: diagnostic, not a qualified speedup

The first available inline-cost measurement is unfavorable versus this relatively cheap Principled target:

| backend | Principled sampling slope ms/spp | neural ms/spp | neural/native cost |
|---|---:|---:|---:|
| CPU | 27.234 | 127.689 | 4.69x |
| OptiX 4070 | 0.828 | 2.173 | 2.62x |

Slope uses the best of two 64-spp and two 512-spp **terminal Cycles render intervals**, divided by 448 spp. Counts/time scaling verified from the real log. Original harness wall-time slopes were 27.225/127.694 ms (CPU) and 0.686/2.253 ms (GPU); GPU setup variation measurably biases that estimate, so sampling diagnostics above are reported separately.

At the matched beauty's 1024 spp:

| backend | Principled render interval / render-plus-write wall | neural render interval / render-plus-write wall |
|---|---:|---:|
| CPU | 30.829 / 30.859 s | 141.850 / 141.872 s |
| OptiX 4070 | 0.877 / 1.493 s | 2.272 / 2.805 s |

Complete process wall times for all 13 renders/EXR reads/analysis in each backend are 389.33 s CPU and 16.39 s OptiX; those are campaign costs, not one-frame render times. No training cost is included in any per-frame timing.

Caveats: best-of-two, fixed material order, desktop GPU activity present at preflight, no continuous competition attribution or controlled three-repeat campaign. These numbers are diagnostics showing no speed win on this probe, not production qualification, a whole-scene generalization, or equal-quality speedup. No new rendering claims are based on zero LOCAL allocation from cuobjdump.

## Artifacts and next step

Here: `summary.json`, `sampling_diagnostics.json`, both backend logs, `comparison.png`, `make_comparison.py` (regenerates the contact and checks counts), GPU preflight CSVs, and the final regression disposition. EXRs, beauty arrays and model remain local under `real/slice3_20261024/`; none in git.

Publication tooling note: the tree's global CSV rule tried to send the tiny GPU telemetry tables to Git LFS, which GitHub rejected for this public fork. This result directory explicitly marks those tables as ordinary text; no model/data LFS rule was changed. The failed evidence commit was never published; only that unpublished commit was corrected, with no force-push or rewrite of shared history.

Next as approved: learned shading frames in trainer/export plus closure encoding-2 admission, crosscheck and the exact probe comparison. Do not pursue a new textured fixture: this probe already exercises the UV latent. Existing format/reference/kernel support was validated by the final slice-2 unit/build gates. Fit improvement and inference cost remain separate acceptance questions.
