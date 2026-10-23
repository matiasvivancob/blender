# BSDF dump format, version 1

A dump is one `.npz` file (`numpy.savez_compressed`, no pickled objects) holding N records, each a surface
point, a pair of directions, and what `SHADER_EVAL_BSDF` returned there. It is the input of `train.py` and
`evaluate.py`. Writer and reader: `dataset.py` (`save_dump`, `load_dump`, `filter_records`). The records come
from the `engine.eval_bsdf` wrapper in `intern/cycles/blender/addon/engine.py` (driver: `blender_dump.py`) or,
for tests, from `make_synthetic_dump.py`.

## Arrays

| key | dtype, shape | meaning |
|---|---|---|
| `format_version` | int32, () | 1 |
| `prims` | int32, (N,) | Triangle index in the evaluated mesh as Cycles sees it (after modifiers and triangulation). Input to `eval_bsdf`. |
| `uv` | float32, (N,2) | Barycentric `(u, v)` in that triangle, Cycles convention `P = (1-u-v) v0 + u v1 + v v2`. Input to `eval_bsdf`. |
| `wi` | float32, (N,3) | Direction from the surface towards the viewer (Cycles `ShaderData.wi`). Unit length. |
| `wo` | float32, (N,3) | Direction from the surface towards the light (the sampled direction in path tracing). Unit length. |
| `frame_local` | bool, () | True: `wi`, `wo` are in the local shading frame (z along the shading normal, x along dPdu projected on the tangent plane, y = z x x). This is `flags` bit 0 of `eval_bsdf`. Training requires True. |
| `out` | float32, (N,8) | Result of `eval_bsdf`, see below. |
| `tex_uv` | float32, (N,2), optional | Texture coordinates of the record in [0,1]^2: the position in the latent grid. `blender_dump.py` computes it from the mesh UV layer as the barycentric interpolation of the triangle's UVs. Required for training unless `meta.uv_kind == "texture"` (then `uv` is read as texture coordinates; the synthetic dump does this). |
| `meta_json` | str, () | JSON object, see below. |

`out` columns (same as `eval_bsdf`):

| column | meaning |
|---|---|
| 0:3 | BSDF value RGB **including the cosine of `wo`**, closure weights included: `f(wi, wo) * cos(theta_o)`. This is the training target. |
| 3 | PDF of sampling `wo` with the material's own sampling. Not used in training; kept for the later proxy-lobe fit. |
| 4:7 | Albedo RGB (diffuse + glossy + transmission). Not used in training; kept for the later proxy-lobe fit and for furnace tests. |
| 7 | Flags as a bit mask, 0 = valid. 1 = no BSDF closure at this point, 2 = `wi` on the backfacing side of the geometric normal, 4 = invalid input. Any non-zero value drops the record. (The wrapper docstring in `engine.py` is the definition; the loader only tests for non-zero.) |

All float inputs must be finite. Extra keys are ignored by the loader.

## Metadata (`meta_json`)

All optional; the loader never fails on missing metadata. Written by `blender_dump.py`:

```
object_name, material_name, blender_version, sample_count (records requested), uv_kind ("texture" if `uv` already
is texture space), uv_layer, seed, sampler {peak_frac, grazing_frac}, eval_seconds, created, kind
("training" | "sphere_queries"), sphere_size (sphere queries only), texture_res/texture_seed (synthetic only)
```

## What the loader drops (`filter_records`)

The report dict lists the count per reason; reasons overlap.

- `out[:,7] != 0` (flagged by Cycles),
- any non-finite value in `wi`, `wo`, `tex_uv` or `out`,
- `| |wi| - 1 | > 1e-2` or the same for `wo`,
- `wi.z < 1e-3` or `wo.z < 1e-3` (outside the decoder's domain, the upper hemisphere),
- `tex_uv` outside [0, 1] by more than 1e-4.

Negative BSDF values are clamped to 0 (counted as `negative_clamped`). Surviving directions are renormalised and
`tex_uv` is clipped to [0, 1]. A dump with `frame_local == False` is rejected, and so is a dump with neither
`tex_uv` nor `uv_kind == "texture"`.

## Conventions the network depends on

- The decoder input is `bsdf_dirs(wi, wo)` plus the latent, with `wi` = viewer, `wo` = light, exactly the
  argument order of `intern/cycles/kernel/util/mlp.h`. Most BSDFs are reciprocal, but
  the encoding is not symmetric (`wd` is `wi` rotated), so do not swap them.
- The cosine in the target is of `wo`. The kernel closure must therefore return `decoder_output` directly as the
  eval value (cosine included), as `01_cycles_neural_closures_spec.md` section 4 says.
- Texel `(0, 0)` of the latent grid sits at `tex_uv = (0, 0)` and texel `(W-1, H-1)` at `(1, 1)` (corner-to-corner,
  `nnm_format.md` section 5). Row index = v, column index = u. Flip V here, not in the model, if the engine's
  texture lookup uses the opposite convention.

## Sphere queries

`sphere.py` generates the records of a head-on sphere under two point lights (the picture of
`experiments/neural_material_train/out/sphere_compare.png`): `python3 sphere.py q.npz` writes them with
`out = 0`, `blender_dump.py --fill q.npz` fills `out` through Cycles, and `evaluate.py --sphere-dump` renders the
ground truth and the `.nnm` side by side. `evaluate.py` regenerates the inputs and refuses a file whose
`wi`/`wo`/`tex_uv` differ.

## Size

Per record: 4 + 8 + 8 + 12 + 12 + 32 = 76 bytes uncompressed (`prims`, `uv`, `tex_uv`, `wi`, `wo`, `out`). The
trainer keeps 44 bytes per valid record on the GPU (`tex_uv`, `wi`, `wo`, target), so 8M records are about 350 MB.
Write dumps in shards of about 1M records with different `--seed` and pass them all to `train.py --dump`.
