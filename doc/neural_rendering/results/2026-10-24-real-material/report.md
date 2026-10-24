# Real material through the whole pipeline: Cycles dump (CPU) -> train -> .nnm with proxy block

Material: the Principled probe from `make_test_scenes.py` (metallic base, 256x256 roughness texture, clear coat 1.0 with
coat roughness 0.05, i.e. GGX alpha 0.0025 on the coat). Executable: first cold build
(`install/baseline-first-build`, renderer source unchanged), CPU device throughout.

## Data

| step | result |
|---|---|
| `--probe` on `ProbePlane` | PASS (earlier result) |
| training dumps | 2 files x 4,000,000 records (seeds 0 and 1), 15 s and 14 s, 0 % flagged |
| sphere ground truth | `sphere.py` queries (12,425 records) evaluated with `--fill` |

`--fill` only accepts up to 256 triangles ("use a quad"); the probe plane has 512. I did not touch the tool:
`make_probe_quad.py` (this directory) builds `probe_quad.blend`, the same material on one quad with the unit-square UV
(its probe also passes: max error 1.2e-7).

A harness note: when the engine's `render()` raises (that RuntimeError), Blender still exits 0. The file is simply not
written, so automation must look for `wrote <file>` / `PROBE PASS`, as you suggested.

## Training (2x32, 64x64x8 latent, 100,000 steps, 64k batch, lr 0.02, seed 0, 8M records)

| run | card | train wall (s) | held-out pow-loss | rel L1 all | rel L1 top 1 % | rel L1 grazing | top-0.1 % energy kept | sphere rel L1 |
|---|---|---|---|---|---|---|---|---|
| culling 64>16>4>1, latent 64 | 4070 | 786 | 0.0404 | 0.228 | 0.235 | 0.240 | 0.939 | **0.0258** |
| baseline, latent 64 | 3090 | 759 | 0.0616 | 0.225 | 0.157 | 0.235 | 0.930 | 0.0415 |
| culling, latent 128 | 4070 | 632 | 0.0409 | 0.281 | 0.290 | 0.298 | 1.196 | 0.0332 |

(The two cards ran concurrently, so wall times are not comparable between rows. Metrics are from `evaluate.py` on the
exported fp16 `.nnm`, validation split seed 0.)

- Culling again wins on the training loss (0.040 vs 0.062) and on the rendered sphere (2.6 % vs 4.1 % relative L1), but
  not on every held-out subset: the baseline is better on the brightest 1 % (0.157 vs 0.235).
- A 128x128 latent did not help: same loss, worse relative error and sphere, overshoots the peak energy (1.20).
  I would stay at 64x64 for this material.
- Held-out relative error is about 23 %, four times the synthetic grid (6 %). The probe has a 256x256 texture feeding a
  64x64 latent and a coat 20x sharper than the synthetic one; the sphere error (2.6 %) is the number that reflects
  how it looks. Error map maximum for the baseline bin is 170x (some texels very wrong), for the culled run 1.8x.
- **fp16 export is close to its limit on this material:** `export_nnm.py` verifies the `.nnm` decode against the
  float training model with p99 relative error < 5 %. Latent 64 passes at 4.95 %, latent 128 fails at 7.06 %
  (median 1.3 %, max 6.2 %). The synthetic models were at 0.1 % median / 0.6 % p99. Cause (not isolated): the
  output is `exp(...)` of a net whose pre-exp magnitudes reach 10+ on the specular peak, so fp16 weight rounding is
  amplified. If this matters for slice 3, options are fp32 weights for the last layer or a smaller exp clamp; your call.

## Proxy block fit for the culled latent-64 model

`export_nnm.py ... --dump train_a.npz` (the new fit, on `neural-p2` commit `2b1f4154`):

```text
proxy fit from 1 dump file(s): alpha 0.01, p_diffuse 0.1, albedo [0.9538, 0.6916, 0.6132], second moment 5.924e+04
```
The optimum is at the corner of the prescribed grid (smallest alpha 0.01, smallest p_diffuse 0.1). The coat's true
alpha is 0.0025: the format allows down to 0.001, but the search grid you specified starts at 0.01, so the fit wants a
narrower lobe than it may try. Extending the grid below 0.01 is a one-line change in `proxy_fit.py` (`ALPHAS`) if you
want it; I left it as specified. The second moment is 5.9e4 (the estimator variance on the specular peak is huge
either way; the dump's peak values reach 6.5e5). Not a bug, a limit of the v1 constants-only proxy.

## Files

- committed here: `sphere_compare_cull_w32_latent64.png` (ground truth | model, 4x error below), `real_train.py`
  (the exact training/evaluation driver), `make_probe_quad.py`
- local only: dumps, checkpoints, `.nnm` files (`D:/Github/Blender-RND/real/`)
- the proxy-block `.nnm` used for slice 1 testing: `real/cull_w32/model_proxy.nnm`
