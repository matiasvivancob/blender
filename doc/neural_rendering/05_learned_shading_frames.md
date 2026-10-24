# Learned shading frames for the neural BSDF (spec, version 1)

Status: spec, 2026-10-24. Source: Zeltner et al. 2024, "Real-Time Neural Appearance
Models" (ACM TOG 43(3) 33), section 4.2 and Figure 4. In their ablation (Table 2) this
one change cut mean FLIP from 0.196 to 0.082 at the same decoder size, the largest single
gain in the paper. Our first real material sits at 23% held-out relative error against 6%
on the synthetic one (`results/2026-10-24-real-material/`), so this is the next accuracy
step after slice 3.

## 1. What changes

Today the decoder input is `[bsdf_dirs(wi, wo) (12), z (C)]`, with `wi`, `wo` in the
closure's local frame and `z` the bilinear latent at the shading point's UV.

With learned frames, one linear layer maps the latent `z` to `K` frames, each a normal
`n_k` and a tangent `t_k`. Both directions are expressed in every frame, and those `6K`
values replace the 12 `bsdf_dirs` features. The MLP sees
`[frames(z, wi, wo) (6K), z (C)]`. With `K = 2` the encoded width stays 12, so the
decoder topology, `input_dim`, the weight layout and the kernel MLP are unchanged.

Why it helps: an MLP can only multiply inputs by fixed weights, so rotating the
directions by a spatially varying frame (normal maps, layered lobes tilted per texel) costs
it most of its capacity. The frame layer gives it that product for 12 x (C + 1) weights.

## 2. The frame layer (normative)

Inputs: `z` (C floats, the latent after bilinear lookup), `wi`, `wo` (unit, local frame,
`z > 0`, as for `bsdf_dirs`).

```
f = Wf^T z + bf                         # Wf: C x 6K, bf: 6K; fp16 weights, fp32 math
for k in 0..K-1:
    n = f[6k + 0 .. 6k + 3]
    t = f[6k + 3 .. 6k + 6]
    n = n / |n|            if |n|^2 > 1e-12 else (0, 0, 1)
    t = t / |t|            if |t|^2 > 1e-12 else (1, 0, 0)
    b = cross(n, t)
    b = b / |b|            if |b|^2 > 1e-12 else any unit vector orthogonal to n
                           (use make_orthonormals(n) in the kernel, the same
                           branchless construction in numpy and torch)
    out[6k + 0..3] = (dot(wi, t), dot(wi, b), dot(wi, n))
    out[6k + 3..6] = (dot(wo, t), dot(wo, b), dot(wo, n))
```

`n` and `t` are not orthogonalised (Zeltner section 9 found this more stable); only `b`
is derived. No activation. The encoded vector is `out[0..6K)` followed by `z`, so the MLP
input order is `[frames, latent]`, matching the existing `[encoded, latent]` rule.

Initialisation for training: `Wf = 0`, and `bf` set so every frame starts as the identity
(`n = (0, 0, 1)`, `t = (1, 0, 0)`). At step 0 the frame features then equal
`(wi, wo)` repeated K times, a valid starting point that the optimiser rotates away from.

## 3. File format (`.nnm`, version stays 1)

- **New encoding id 2, `learned_frames`.** Requires `latent_channels >= 1` and
  `input_dim - latent_channels == 6 * frames_k`. The caller passes `wi, wo` (6 floats),
  exactly as for `bsdf_dirs`, so the closure code does not change.
- **Header fields in the reserved area:**

  | Offset | Size | Type | Field | Constraint |
  |---:|---:|---|---|---|
  | 184 | 4 | u32 | `frames_k` | 0 unless encoding 2; then 1..4 (v1 writers use 2) |
  | 188 | 4 | u32 | `frames_offset` | byte offset of the frame layer from `weight_offset`, 0 unless encoding 2 |
  | 192 | 64 | | reserved | zero |

- **Frame layer storage:** after the last MLP layer inside the weight blob, same packing
  as an MLP layer (section 4): `C + 1` rows of `pad8(6K)` fp16 values, input-major, bias
  row last, padding zero. `weight_size` includes it; `layer_offset[]` does not list it
  (it is not an MLP layer), `frames_offset` points at it.
- **Test vectors:** unchanged rule; the caller part is `wi, wo, u, v` as for `bsdf_dirs`
  with a latent. Writers must include at least 8 vectors so the loader self-check covers
  the frame layer.
- Readers that do not know encoding 2 already reject the file (`encoding > 1`), so old
  builds fail loudly rather than misdecode.

## 4. Who changes what

| Part | File | Test that proves it |
|---|---|---|
| Spec | `nnm_format.md` sections 3, 4, 7 | review |
| Writer, reader, validation | `tools/nnm/format.py` | round trip; reject `frames_k` 0 or 5, wrong width, nonzero padding, missing latent |
| Reference decoder | `tools/nnm/reference.py` (`encode_learned_frames`, float32 and float64 paths) | identity frames give the `bsdf_dirs`-free `(wi, wo)` repeat; rotation equivariance check |
| Kernel | `kernel/util/mlp.h`: `neural_encode_learned_frames`, dispatch in `neural_model_eval` | standalone parity test gains encoding-2 models (all widths and activations), CPU g++/clang |
| Loader | `scene/neural.cpp`: accept encoding 2, validate the new fields and the frame block | `SceneNeural` gtest: embedded encoding-2 models load, corrupted copies rejected |
| Closure | `scene/shader_nodes.cpp` accepts encoding 2 next to `bsdf_dirs` | slice-2 gates unchanged |
| Trainer | `tools/train/model.py` frame module with the section 2 init; `export_nnm.py` writes it | `crosscheck_torch.py` extended: torch frames vs numpy reference, exported `.nnm` decodes to the torch prediction |
| Experiment | probe material, 2x32, culling, paper schedule, `K = 2` vs the current model | held-out and sphere metrics as in `results/2026-10-24-real-material/` |

The cloud session takes the rows it can test without a GPU: format, reference decoder,
kernel `mlp.h` and the standalone parity test. The local agent takes the loader, closure,
trainer, export, gtests and the experiment, after slice 3.

## 5. Cost

Per evaluation: a C x 12 matrix (96 FMAs for C = 8), two normalisations of `n` and `t`,
one cross product and 12 dot products, against about 1,800 FMAs for the 2x32 MLP: under
10% more. Weights: 108 fp16 values. The latent lookup is unchanged.

## 6. Acceptance

Same culled 2x32 at 64x64 on the probe material, seed 0, paper schedule, with and
without frames. Adopt frames if the sphere relative L1 and the held-out relative error
both improve; report FLIP if available. If held-out error improves by less than 20%
relative, report and stop: the next levers are the analytic specular lobe outside the
network and a per-texel proxy head.
