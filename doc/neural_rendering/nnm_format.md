# `.nnm` neural model file format, version 1

A `.nnm` file holds one small MLP decoder (optionally with a texture of latent
vectors) plus a block of test vectors so that any loader or kernel can
self-check against the reference decoder. It is the "File format" item of
`01_cycles_neural_closures_spec.md` section 1 and the input of
`intern/cycles/kernel/util/mlp.h`.

Reference implementations (they must agree with this text; where they differ
this text is the bug report):

- Python reader, writer and reference decoder: `tools/nnm/` (`format.py`,
  `reference.py`).
- Kernel decoder: `intern/cycles/kernel/util/mlp.h`.

## 1. Conventions

- Little-endian everywhere. `u32`/`u64` are unsigned, `f32` is IEEE binary32,
  `f16` is IEEE binary16 (round-to-nearest-even when written).
- All offsets are in bytes from the start of the file. All sizes are in bytes.
- Reserved fields and padding bytes are written as zero and ignored by readers,
  except that a reader may reject a file whose reserved fields are non-zero.
- `pad8(n) = (n + 7) & ~7`.

## 2. File layout

```
offset 0                    header                (256 bytes)
offset weight_offset        weight blob           (weight_size bytes)
offset latent_offset        latent blob           (latent_size bytes; absent if 0)
offset testvec_offset       test-vector block     (testvec_size bytes; absent if 0)
offset file_size            end of file
```

A writer emits the blocks in this order. `weight_offset` is 256. Each following
block starts at the previous block's end rounded up to a multiple of 64, the gap
being zero bytes. A reader must use the offsets in the header, not assume this
order, and must reject any block that extends past `file_size` or overlaps the
header or another block.

## 3. Header (256 bytes)

| Offset | Size | Type | Field | Constraint / meaning |
|---:|---:|---|---|---|
| 0 | 4 | char[4] | `magic` | ASCII `NNM1` |
| 4 | 4 | u32 | `version` | 1 |
| 8 | 4 | u32 | `header_size` | 256 |
| 12 | 4 | u32 | `flags` | bit 0 `FLAG_PROXY`: the proxy block of section 3a is present. Other bits 0; readers reject unknown bits |
| 16 | 4 | u32 | `input_dim` | 1..64. Width of the MLP input vector = encoded features + latent channels |
| 20 | 4 | u32 | `output_dim` | 1..16 |
| 24 | 4 | u32 | `num_hidden` | 1..4 hidden layers |
| 28 | 4 | u32 | `hidden_width` | 16, 32 or 64 |
| 32 | 4 | u32 | `activation_id` | 0 ReLU, 1 LeakyReLU(eps), 2 LeakySmeLU(beta, eps) |
| 36 | 4 | u32 | `output_transform_id` | 0 none, 1 exp, 2 softplus |
| 40 | 4 | u32 | `input_encoding_id` | 0 raw, 1 `bsdf_dirs`, 2 `learned_frames` |
| 44 | 4 | u32 | `latent_channels` | 0..16. 0 means no latent |
| 48 | 4 | f32 | `act_beta` | finite, > 0 if activation 2, else 0 |
| 52 | 4 | f32 | `act_eps` | finite, in [0, 1) if activation 1 or 2, else 0 |
| 56 | 4 | u32 | `latent_width` | 0 if no latent, else 1..16384 |
| 60 | 4 | u32 | `latent_height` | 0 if no latent, else 1..16384 |
| 64 | 4 | u32 | `latent_levels` | 0 if no latent, else 1..(floor(log2(max(w,h)))+1) |
| 68 | 4 | u32 | `testvec_count` | 0..64 |
| 72 | 8 | u64 | `weight_offset` | 256 |
| 80 | 8 | u64 | `weight_size` | sum of the layer sizes of section 4 |
| 88 | 8 | u64 | `latent_offset` | 0 if no latent, else a multiple of 64 |
| 96 | 8 | u64 | `latent_size` | 0 if no latent, else see section 5 |
| 104 | 8 | u64 | `testvec_offset` | 0 if `testvec_count` is 0, else a multiple of 64 |
| 112 | 8 | u64 | `testvec_size` | `testvec_count * (in + out) * 4`, see section 6 |
| 120 | 4 | u32 | `testvec_in_dim` | see section 6 |
| 124 | 4 | u32 | `testvec_out_dim` | `output_dim` |
| 128 | 4 | u32 | `num_layers` | `num_hidden + 1` |
| 132 | 20 | u32[5] | `layer_offset[5]` | byte offset of each layer from `weight_offset`; entries `>= num_layers` are 0 |
| 152 | 8 | u64 | `file_size` | exact size of the file |
| 160 | 24 | f32[6] | `proxy` | section 3a if `flags` bit 0, else zero |
| 184 | 4 | u32 | `frames_k` | 0 unless `input_encoding_id` is 2; then 1..4 (version 1 writers use 2). Section 7a |
| 188 | 4 | u32 | `frames_offset` | 0 unless encoding 2; then the byte offset of the frame layer from `weight_offset` (section 4) |
| 192 | 64 | | reserved | zero |

### 3a. Proxy block (flag bit 0)

Per-material constants for the Cycles neural closure, version 1 of the closure
(`01_cycles_neural_closures_spec.md` section 4). Six little-endian f32 at offset 160:

| Offset | Field | Range | Use in the closure |
|---:|---|---|---|
| 160 | `alpha_x` | [1e-3, 1] | GGX roughness of the sampling proxy, tangent direction |
| 164 | `alpha_y` | [1e-3, 1] | same, bitangent direction (equal to `alpha_x` for isotropic) |
| 168 | `p_diffuse` | [0.1, 1] | probability of the cosine lobe in the proxy mixture |
| 172 | `albedo_r` | [0, 1] | material albedo, constant over the surface in v1 |
| 176 | `albedo_g` | [0, 1] | |
| 180 | `albedo_b` | [0, 1] | |

The proxy only affects noise, never the expected value, as long as the closure
reports the pdf of the procedure it actually samples (MIS stays unbiased). The
albedo feeds `bsdf_albedo` (closure selection weight and the denoising-albedo
pass), so in v1 those passes see a constant colour per material. A later
version replaces the block with a per-texel head network; it will use a new
flag bit, not reinterpret this one. Files without the flag stay valid.

Enumerations:

| `activation_id` | Name | `act(x)` |
|---:|---|---|
| 0 | ReLU | `x > 0 ? x : 0` |
| 1 | LeakyReLU | `x > 0 ? x : eps * x` |
| 2 | LeakySmeLU | `eps * x + (1 - eps) * g(x)` with `g` below |

`g(x) = s*s*(0.25/beta) + max(x - beta, 0)` with `s = clamp(x + beta, 0, 2*beta)`.
Equivalently `g = 0` for `x <= -beta`, `g = x` for `x >= beta`, and
`(x + beta)^2 / (4 beta)` between. This is the reconstruction used by
`experiments/neural_material_train/train.py` (`act_fwd`); confirm against the
Bitterli paper before relying on it for anything but parity.

| `output_transform_id` | Name | `y = T(x)` applied to each of the `output_dim` outputs |
|---:|---|---|
| 0 | none | `x` |
| 1 | exp | `exp(clamp(x, -20, 14))` (the clamp matches the training forward pass) |
| 2 | softplus | `log(1 + exp(x))`, computed as `max(x, 0) + log1p(exp(-|x|))` (`mlp.h`: `neural_softplus`) |

| `input_encoding_id` | Name | Caller passes | MLP input (before latent) |
|---:|---|---|---|
| 0 | raw | `input_dim - latent_channels` floats (at least 1) | the same floats |
| 1 | bsdf_dirs | `wi`, `wo`: 6 floats, unit vectors in the local shading frame, z up, both z > 0 | 12 floats, section 7 |
| 2 | learned_frames | `wi`, `wo`: 6 floats, as for `bsdf_dirs` | `6 * frames_k` floats, section 7a; needs `latent_channels >= 1` |

For `bsdf_dirs`, `input_dim - latent_channels` must be exactly 12. For `learned_frames` it must be
exactly `6 * frames_k` (12 for the usual `frames_k = 2`).

The MLP input vector is `[encoded features (E floats), latent texel (latent_channels floats)]`,
in that order, `E = input_dim - latent_channels`. (This matches the training
code: `W1d` for the direction features, then `W1z` for the latent. Note that
`01_cycles_neural_closures_spec.md` section 4 writes `[z, wi_local, ...]` for a
different, closure-level input; the `.nnm` order above is the normative one for
the decoder.)

## 4. Weight blob

The network has `num_layers = num_hidden + 1` linear layers. Layer `l` maps
`in_l` to `out_l`:

| Layer | `in_l` | `out_l` |
|---|---|---|
| 0 | `input_dim` | `hidden_width` |
| 1 .. num_hidden-1 | `hidden_width` | `hidden_width` |
| num_hidden (last) | `hidden_width` | `output_dim` |

Let `P_l = pad8(out_l)`. Layer `l` is stored as `(in_l + 1)` rows of `P_l` f16
values:

```
rows 0 .. in_l-1 :  W[i][o]   for o in 0 .. P_l-1    (input-major: one row per input i)
row  in_l        :  b[o]      for o in 0 .. P_l-1    (bias, one separate row)
```

Columns `o >= out_l` are zero padding. Size of layer `l` is
`(in_l + 1) * P_l * 2` bytes, always a multiple of 16. Layers follow one
another with no gap, in order; `layer_offset[l]` is the sum of the sizes of the
layers before `l`, and `weight_size` is the sum of all layer sizes (plus the frame layer if there is one, below).

**Frame layer (encoding 2 only).** Stored after the last MLP layer, with no gap, packed like
an MLP layer: `C + 1` rows of `pad8(6K)` f16 values with `C = latent_channels`, `K = frames_k`;
rows `0 .. C-1` hold `Wf[c][o]` for `o` in `0 .. 6K-1` (input-major, one row per latent
channel), row `C` holds the bias `bf[o]`; columns `o >= 6K` are zero padding. Its size is
`(C + 1) * pad8(6K) * 2` bytes (always a multiple of 16) and is included in `weight_size`.
It is not an MLP layer: `num_layers` and `layer_offset[]` do not count or list it.
`frames_offset` (header) is the sum of the sizes of the MLP layers, i.e. the first byte after
the last MLP layer, relative to `weight_offset`. Files without encoding 2 contain no frame
layer and are laid out exactly as before.

Forward pass, normative (`x` and all accumulators are f32, weights are promoted
from f16 exactly):

```
a = encoded_input                                 # input_dim floats
for l in 0 .. num_layers-1:
    acc[o] = b_l[o]                                # o in 0 .. out_l-1
    for i in 0 .. in_l-1:                          # ascending i
        for o in 0 .. out_l-1:
            acc[o] = fma(a[i], W_l[i][o], acc[o])  # fused multiply-add, f32
    if l < num_layers-1: a = act(acc)  else: y = T(acc)
```

The loop order (bias first, then inputs in ascending order, one fused
multiply-add per product) is part of the format so that implementations agree to
the last bit when they use hardware FMA. An implementation without FMA, or with
a different order, agrees to about 1e-6 relative on well-conditioned outputs and
must be validated against the test vectors with a tolerance (section 6).

## 5. Latent blob

Present iff `latent_channels > 0`. A pyramid of `latent_levels` images stored
one after another, level 0 first. Level `k` has width `max(1, latent_width >> k)`
and height `max(1, latent_height >> k)`. Each image is f16, row-major (row 0
first, left to right), channels interleaved per texel, no padding:

```
index(k, x, y, c) = level_base[k] + ((y * width_k + x) * latent_channels + c)     # in f16 units
```

`latent_size` = total bytes of all levels. Version 1 decoders sample level 0
only; the other levels are stored for later use and need to be present only if
`latent_levels > 1`.

Bilinear lookup of level 0 at `(u, v)`, texels on a corner-to-corner grid (u = 0
is the centre of texel 0, u = 1 the centre of the last texel; this is the
convention of `experiments/neural_material_train/material.py`). All in f32:

```
u = clamp(u, 0, 1);  v = clamp(v, 0, 1)
x = u * (W - 1);     y = v * (H - 1)
ix = clamp(int(floor(x)), 0, max(W - 2, 0));  fx = x - float(ix)
iy = clamp(int(floor(y)), 0, max(H - 2, 0));  fy = y - float(iy)
ix1 = min(ix + 1, W - 1);  iy1 = min(iy + 1, H - 1)
w00 = (1 - fx) * (1 - fy);  w10 = fx * (1 - fy);  w01 = (1 - fx) * fy;  w11 = fx * fy
z[c] = fma(t11[c], w11, fma(t01[c], w01, fma(t10[c], w10, t00[c] * w00)))
```

where `tXY` is the texel at `(ix + X', iy + Y')` with `X' = X ? ix1 : ix`
(similarly `Y'`), promoted to f32.

## 6. Test-vector block

Present iff `testvec_count > 0`. `testvec_count` records of
`testvec_in_dim + testvec_out_dim` f32 each, back to back:

```
record = [ caller inputs (caller_dim floats),  u, v  (only if latent_channels > 0),  expected outputs (output_dim floats) ]
```

`caller_dim` is 6 for `bsdf_dirs` and `learned_frames` and `input_dim - latent_channels` for `raw`.
`testvec_in_dim = caller_dim + (latent_channels > 0 ? 2 : 0)`,
`testvec_out_dim = output_dim`. `u`, `v` are in [0, 1].

The expected outputs are what the Python reference decoder
(`tools/nnm/reference.py`) produces for the stored f16 weights, i.e. the final
outputs after the output transform. A loader self-check decodes every record
(encode, latent lookup, forward pass) and compares with
`|got - expected| <= 1e-4 * |expected| + 1e-5`.

## 7. `bsdf_dirs` encoding

Input `wi`, `wo` (unit, z > 0). Output 12 floats, all f32:

```
h  = wi + wo;  h = h / sqrt(dot(h, h))             # (0, 0, 1) if dot(h, h) == 0
v  = (hy, -hx, 0)                                  # axis of the shortest arc h -> +z
k  = (v . wi) / max(1 + hz, 1e-6)
wd = wi * hz + (v x wi) + k * v                    # wi rotated by the shortest arc taking h to +z
out = [wi.x wi.y wi.z  wo.x wo.y wo.z  h.x h.y h.z  wd.x wd.y wd.z]
```

Expanded with `v = (hy, -hx, 0)`, which is what both implementations compute:

```
v x wi = (-hx * wiz,  -hy * wiz,  hx * wix + hy * wiy)
v . wi = hy * wix - hx * wiy
wd     = (wix * hz + (-hx * wiz) + k * hy,
          wiy * hz + (-hy * wiz) + k * (-hx),
          wiz * hz + (hx * wix + hy * wiy))
```

`wd` is the rotation `R(h -> z)` applied to `wi` (Rodrigues form with
`|v|^2 = (1 - hz)(1 + hz)` simplified); `tools/nnm/selftest.py` checks that
identity independently.

## 7a. `learned_frames` encoding

Source: `05_learned_shading_frames.md` (Zeltner et al. 2024, section 4.2). Inputs: `z`
(`C = latent_channels` floats, the level 0 bilinear latent sample at `(u, v)`, section 5),
and `wi`, `wo` (unit, local frame, z > 0). The latent lookup therefore happens **before** the
encoding. `K = frames_k`, frame layer `Wf` (`C x 6K`) and `bf` (`6K`) as in section 4, f16
promoted to f32 exactly:

```
f = bf;  for c ascending: f[o] = fma(z[c], Wf[c][o], f[o])      # same order as an MLP layer, f32
for k in 0 .. K-1:
    n = f[6k + 0 .. 6k + 3]                                      # normal
    t = f[6k + 3 .. 6k + 6]                                      # tangent
    n = n / sqrt(dot(n, n))   if dot(n, n) > 1e-12 else (0, 0, 1)
    t = t / sqrt(dot(t, t))   if dot(t, t) > 1e-12 else (1, 0, 0)
    b = cross(n, t)
    b = b / sqrt(dot(b, b))   if dot(b, b) > 1e-12 else a        # a from make_orthonormals(n, &a, &bb)
    out[6k + 0 .. 6k + 3] = (dot(wi, t), dot(wi, b), dot(wi, n))
    out[6k + 3 .. 6k + 6] = (dot(wo, t), dot(wo, b), dot(wo, n))
```

`dot(p, q) = (p.x*q.x + p.y*q.y) + p.z*q.z` and `cross(n, t) = (n.y*t.z - n.z*t.y,
n.z*t.x - n.x*t.z, n.x*t.y - n.y*t.x)`, plain f32 operations. `n` and `t` are not
orthogonalised; only `b` is derived. There is no activation. The fallback `a` is the first output
of `make_orthonormals` in `intern/cycles/util/math_float3.h`, which has the construction
`a = (n.z - n.y, n.x - n.z, n.y - n.x)` if `n.x != n.y || n.x != n.z`, else
`a = (n.z - n.y, n.x + n.z, -n.y - n.x)`, followed by `a = a / sqrt(dot(a, a))`; it is a unit
vector orthogonal to `n`.

The MLP input is `[out (6K floats), z (C floats)]`, the usual `[encoded, latent]` order. With
identity frames (`Wf = 0`, `bf` = `(0, 0, 1, 1, 0, 0)` per frame, i.e. `n = z axis`, `t = x axis`),
`b = (0, 1, 0)` and the output is `(wi, wo)` repeated `K` times.

## 8. Validation rules a reader must enforce

1. magic, version, `header_size`, `file_size` equal the actual size.
2. All enumerations known; widths, depths and dimensions in range as in section 3.
3. `bsdf_dirs` implies `input_dim - latent_channels == 12`; `raw` implies it is >= 1.
   `learned_frames` implies `latent_channels >= 1`, `1 <= frames_k <= 4` and
   `input_dim - latent_channels == 6 * frames_k`. For encodings 0 and 1, `frames_k` and
   `frames_offset` (bytes 184..192) are zero. For encoding 2, `frames_offset` equals the sum
   of the MLP layer sizes, and `weight_size` equals that sum plus the frame layer size
   `(latent_channels + 1) * pad8(6 * frames_k) * 2`.
4. `latent_*` fields all zero iff `latent_channels == 0`.
5. `num_layers == num_hidden + 1`; `layer_offset[]` and `weight_size` equal the
   values computed from section 4.
6. `latent_size`, `testvec_size`, `testvec_in_dim`, `testvec_out_dim` equal the
   computed values; all blocks inside the file, aligned, non-overlapping.
7. All f16 weights (including the frame layer) and latents finite (no Inf or NaN). Padding
   columns of every layer and of the frame layer are zero, and the reserved header bytes
   192..256 are zero (the proxy block bytes 160..184 are zero unless `FLAG_PROXY`).

## 9. Runtime mapping

`KernelNeuralModel` (`mlp.h`) carries `input_dim`, `output_dim`, `num_hidden`,
`width`, activation id and parameters, output transform, encoding, latent
descriptor, and two offsets counted in f16 elements into global arrays:
`weight_offset` (the weight blob of this model within the shared weight array)
and `latent_offset` (level 0 of its latent blob within the shared latent array).
Layer offsets are not stored per model; they are recomputed from the dimensions
with `neural_layer_offset()`, the same rule as section 4. Test vectors are not
uploaded.

For encoding 2 the kernel also carries `frames_k` (it reuses the formerly unused last int of the
64-byte struct). The frame layer offset is not stored: like the layer offsets it is recomputed,
`neural_frames_offset()` is the end of the last MLP layer.
