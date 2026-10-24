/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Scalar reference implementation of the small MLP decoders described by `.nnm` files
 * (doc/neural_rendering/nnm_format.md, 01_cycles_neural_closures_spec.md section 2).
 *
 * This is the CPU fallback and the parity reference for the batched GPU stage. It deliberately
 * does not depend on any kernel globals: the caller passes the weight and latent arrays
 * explicitly, so the header compiles standalone (only util/ headers are included) as well as
 * inside the kernel.
 *
 * Numerics, normative and mirrored by doc/neural_rendering/tools/nnm/reference.py:
 * - Weights and latents are fp16 in memory, promoted to fp32 (exact), all arithmetic is fp32.
 * - Weights are input-major `W[in][out_pad8]` so the inner loop is an axpy over the outputs; the
 *   bias is a separate row after the weights of each layer. Accumulation order: bias first, then
 *   inputs in ascending order, one fused multiply-add per product.
 * - The multiply-add is explicit (`neural_fma`): GCC treats `-ffp-contract=on` as off, so plain
 *   `a * b + c` does not become an FMA there. */

#include "util/defines.h"
#include "util/half.h"
#include "util/math.h"
#include "util/types.h"

CCL_NAMESPACE_BEGIN

#define NEURAL_MAX_INPUT 64
#define NEURAL_MAX_OUTPUT 16
#define NEURAL_MAX_LATENT_CHANNELS 16
#define NEURAL_MAX_HIDDEN 4
#define NEURAL_MAX_FRAMES 4

enum NeuralActivation {
  NEURAL_ACTIVATION_RELU = 0,
  NEURAL_ACTIVATION_LEAKY_RELU = 1,
  NEURAL_ACTIVATION_LEAKY_SMELU = 2,
};

enum NeuralOutputTransform {
  NEURAL_OUTPUT_NONE = 0,
  NEURAL_OUTPUT_EXP = 1,
  NEURAL_OUTPUT_SOFTPLUS = 2,
};

enum NeuralInputEncoding {
  NEURAL_ENCODING_RAW = 0,
  NEURAL_ENCODING_BSDF_DIRS = 1,
  NEURAL_ENCODING_LEARNED_FRAMES = 2,
};

/* Runtime description of one model, filled from a `.nnm` header. Offsets count fp16 elements into
 * the shared weight and latent arrays (the weight blob and the level 0 latent image of this
 * model). Plain POD so that an array of these can become a `KERNEL_DATA_ARRAY` later. */
struct ccl_align(16) KernelNeuralModel {
  int input_dim; /* MLP input width: encoded features + latent channels. */
  int output_dim;
  int num_hidden; /* 1..4 */
  int width;      /* 16, 32 or 64 */

  int activation; /* NeuralActivation */
  float act_beta;
  float act_eps;
  int output_transform; /* NeuralOutputTransform */

  int encoding; /* NeuralInputEncoding */
  int latent_channels; /* 0 means no latent. */
  int latent_width;
  int latent_height;

  int latent_levels;
  uint weight_offset; /* In fp16 elements. */
  uint latent_offset; /* In fp16 elements, level 0. */
  int frames_k; /* Learned frames K (1..4) for NEURAL_ENCODING_LEARNED_FRAMES, else 0. */
};
static_assert(sizeof(KernelNeuralModel) == 64, "KernelNeuralModel size/alignment mismatch");

#if defined(__KERNEL_GPU__)
#  define NEURAL_UNROLL _Pragma("unroll")
#else
#  define NEURAL_UNROLL
#endif

ccl_device_inline int neural_pad8(const int n)
{
  return (n + 7) & ~7;
}

/* Fused multiply-add where the hardware has it. Without FMA hardware (baseline SSE2 kernels)
 * fmaf would be a slow library call, so fall back to a separate multiply and add; results then
 * differ from the reference in the last bits only. */
ccl_device_inline float neural_fma(const float a, const float b, const float c)
{
#if defined(__KERNEL_METAL__)
  return metal::fma(a, b, c);
#elif defined(__KERNEL_ONEAPI__)
  return sycl::fma(a, b, c);
#elif defined(__KERNEL_CUDA__) || defined(__KERNEL_HIP__) || defined(__FMA__) || \
    defined(__AVX2__) || defined(__ARM_FEATURE_FMA) || defined(_M_ARM64)
  return fmaf(a, b, c);
#else
  return a * b + c;
#endif
}

/* ------------------------------------------------------------------------------------------
 * Layer layout helpers (nnm_format.md section 4).
 * ------------------------------------------------------------------------------------------ */

/* Offset in fp16 elements of linear layer `layer` (0 .. num_hidden) relative to
 * `weight_offset`. `layer == num_hidden + 1` gives the total number of elements. */
ccl_device_inline uint neural_layer_offset(const ccl_global KernelNeuralModel &m, const int layer)
{
  uint offset = 0;
  for (int l = 0; l < layer; l++) {
    const int in = (l == 0) ? m.input_dim : m.width;
    const int out = (l == m.num_hidden) ? m.output_dim : m.width;
    offset += uint((in + 1) * neural_pad8(out));
  }
  return offset;
}

/* Offset in fp16 elements, relative to `weight_offset`, of the learned frame layer (nnm_format.md
 * section 4: right after the last MLP layer, included in the weight blob). There is no room for a
 * separate `frames_offset` in the 64-byte KernelNeuralModel, and the format fixes it to the end of
 * the MLP layers, so it is recomputed like the layer offsets. Only meaningful if frames_k > 0. */
ccl_device_inline uint neural_frames_offset(const ccl_global KernelNeuralModel &m)
{
  return neural_layer_offset(m, m.num_hidden + 1);
}

/* ------------------------------------------------------------------------------------------
 * Activations and output transforms.
 * ------------------------------------------------------------------------------------------ */

/* LeakySmeLU: `eps * x + (1 - eps) * g(x)` where `g` is 0 below -beta, x above beta and the
 * quadratic `(x + beta)^2 / (4 beta)` in between. `c4 = 0.25 / beta`, `omeps = 1 - eps`. */
ccl_device_inline float neural_activation_smelu(
    const float x, const float beta, const float c4, const float eps, const float omeps)
{
  const float s = min(max(x + beta, 0.0f), 2.0f * beta);
  const float g = neural_fma(s * s, c4, max(x - beta, 0.0f));
  return neural_fma(eps, x, omeps * g);
}

ccl_device_inline float neural_activation(const float x, const int id, const float beta, const float eps)
{
  switch (id) {
    case NEURAL_ACTIVATION_LEAKY_RELU:
      return (x > 0.0f) ? x : eps * x;
    case NEURAL_ACTIVATION_LEAKY_SMELU:
      return neural_activation_smelu(x, beta, 0.25f / beta, eps, 1.0f - eps);
    case NEURAL_ACTIVATION_RELU:
    default:
      return (x > 0.0f) ? x : 0.0f;
  }
}

/* Whole-layer version: the activation id is a loop invariant, so switch outside the loop. */
template<int N>
ccl_device_inline void neural_activation_layer(ccl_private float *a,
                                               const int id,
                                               const float beta,
                                               const float eps)
{
  switch (id) {
    case NEURAL_ACTIVATION_LEAKY_RELU: {
      NEURAL_UNROLL
      for (int o = 0; o < N; o++) {
        a[o] = (a[o] > 0.0f) ? a[o] : eps * a[o];
      }
      break;
    }
    case NEURAL_ACTIVATION_LEAKY_SMELU: {
      const float c4 = 0.25f / beta;
      const float omeps = 1.0f - eps;
      NEURAL_UNROLL
      for (int o = 0; o < N; o++) {
        a[o] = neural_activation_smelu(a[o], beta, c4, eps, omeps);
      }
      break;
    }
    case NEURAL_ACTIVATION_RELU:
    default: {
      NEURAL_UNROLL
      for (int o = 0; o < N; o++) {
        a[o] = (a[o] > 0.0f) ? a[o] : 0.0f;
      }
      break;
    }
  }
}

/* log(1 + e) for e in [0, 1], without log1pf (not available on every backend) and without
 * Kahan's `(1 + e) - 1` trick, which value-unsafe float modes (MSVC /fp:fast, -ffast-math) may
 * fold to `e`, returning 0 for e below float epsilon. Uses log1p(e) = 2 atanh(t),
 * t = e / (2 + e) in [0, 1/3], as an odd series in t truncated after t^15 / 15: the first
 * dropped term is below t^17 / 17 <= 9^-8 / 17 ~ 1.4e-9 of the result. Only a division and a
 * polynomial, so it is accurate in relative terms down to denormals under any float mode. */
ccl_device_inline float neural_log1p_unit(const float e)
{
  const float t = e / (2.0f + e);
  const float t2 = t * t;
  float p = 1.0f / 15.0f;
  p = p * t2 + 1.0f / 13.0f;
  p = p * t2 + 1.0f / 11.0f;
  p = p * t2 + 1.0f / 9.0f;
  p = p * t2 + 1.0f / 7.0f;
  p = p * t2 + 1.0f / 5.0f;
  p = p * t2 + 1.0f / 3.0f;
  p = p * t2 + 1.0f;
  return 2.0f * t * p;
}

/* softplus(x) = log(1 + exp(x)) = max(x, 0) + log1p(exp(-|x|)), so the log1p argument is in
 * (0, 1] for every x and there is no overflow branch. */
ccl_device_inline float neural_softplus(const float x)
{
  return fmaxf(x, 0.0f) + neural_log1p_unit(expf(-fabsf(x)));
}

ccl_device_inline float neural_output_transform(const float x, const int id)
{
  switch (id) {
    case NEURAL_OUTPUT_EXP:
      return expf(clamp(x, -20.0f, 14.0f));
    case NEURAL_OUTPUT_SOFTPLUS:
      return neural_softplus(x);
    case NEURAL_OUTPUT_NONE:
    default:
      return x;
  }
}

/* ------------------------------------------------------------------------------------------
 * MLP evaluation.
 * ------------------------------------------------------------------------------------------ */

/* y[0..N_OUT) = bias + sum_i x[i] * W[i][0..N_OUT), `w` points at the layer (input-major rows of
 * N_OUT fp16 values, then the bias row). */
template<int N_OUT>
ccl_device_inline void neural_linear(ccl_global const half *w,
                                     ccl_private const float *x,
                                     const int n_in,
                                     ccl_private float *y)
{
  ccl_global const half *bias = w + n_in * N_OUT;
  NEURAL_UNROLL
  for (int o = 0; o < N_OUT; o++) {
    y[o] = half_to_float(bias[o]);
  }
  for (int i = 0; i < n_in; i++) {
    const float xi = x[i];
    ccl_global const half *row = w + i * N_OUT;
    NEURAL_UNROLL
    for (int o = 0; o < N_OUT; o++) {
      y[o] = neural_fma(xi, half_to_float(row[o]), y[o]);
    }
  }
}

/* Evaluate the decoder on an already encoded input `in[m.input_dim]`, writes `out[m.output_dim]`
 * (output transform applied). `weights` is the shared fp16 weight array, `m.weight_offset` is
 * added to it. */
template<int W>
ccl_device_noinline void neural_mlp_eval(ccl_global const half *weights,
                                         const ccl_global KernelNeuralModel &m,
                                         ccl_private const float *in,
                                         ccl_private float *out)
{
  static_assert(W == 16 || W == 32 || W == 64, "neural_mlp_eval: unsupported hidden width");

  float buf0[W];
  float buf1[W];
  ccl_private float *cur = buf0;
  ccl_private float *nxt = buf1;

  ccl_global const half *w = weights + m.weight_offset;

  /* First hidden layer: input_dim -> W. */
  neural_linear<W>(w, in, m.input_dim, cur);
  w += (m.input_dim + 1) * W;
  neural_activation_layer<W>(cur, m.activation, m.act_beta, m.act_eps);

  /* Remaining hidden layers: W -> W. */
  for (int l = 1; l < m.num_hidden; l++) {
    neural_linear<W>(w, cur, W, nxt);
    w += (W + 1) * W;
    neural_activation_layer<W>(nxt, m.activation, m.act_beta, m.act_eps);
    ccl_private float *t = cur;
    cur = nxt;
    nxt = t;
  }

  /* Output layer: W -> output_dim, stored padded to a multiple of 8 columns. */
  float y[NEURAL_MAX_OUTPUT];
  if (neural_pad8(m.output_dim) <= 8) {
    neural_linear<8>(w, cur, W, y);
  }
  else {
    neural_linear<NEURAL_MAX_OUTPUT>(w, cur, W, y);
  }

  for (int o = 0; o < m.output_dim; o++) {
    out[o] = neural_output_transform(y[o], m.output_transform);
  }
}

ccl_device void neural_mlp_eval_dispatch(ccl_global const half *weights,
                                         const ccl_global KernelNeuralModel &m,
                                         ccl_private const float *in,
                                         ccl_private float *out)
{
  switch (m.width) {
    case 16:
      neural_mlp_eval<16>(weights, m, in, out);
      break;
    case 32:
      neural_mlp_eval<32>(weights, m, in, out);
      break;
    case 64:
      neural_mlp_eval<64>(weights, m, in, out);
      break;
    default:
      for (int o = 0; o < m.output_dim; o++) {
        out[o] = 0.0f;
      }
      break;
  }
}

/* ------------------------------------------------------------------------------------------
 * Input encoding (nnm_format.md section 7).
 * ------------------------------------------------------------------------------------------ */

/* 12 floats: wi, wo, half vector h, and wd' = wi rotated by the shortest arc taking h to +z,
 * wd' = wi * hz + (v x wi) + (v . wi) v / (1 + hz) with v = (hy, -hx, 0). The directions are unit
 * vectors in the local shading frame (z up, both z > 0). */
ccl_device void neural_encode_bsdf_dirs(const float3 wi, const float3 wo, ccl_private float *out)
{
  const float wix = wi.x, wiy = wi.y, wiz = wi.z;

  float hx = wix + wo.x;
  float hy = wiy + wo.y;
  float hz = wiz + wo.z;
  const float l2 = hx * hx + hy * hy + hz * hz;
  if (l2 > 0.0f) {
    const float inv = 1.0f / sqrtf(l2);
    hx *= inv;
    hy *= inv;
    hz *= inv;
  }
  else {
    hx = 0.0f;
    hy = 0.0f;
    hz = 1.0f;
  }

  /* v = (hy, -hx, 0): v x wi = (-hx * wiz, -hy * wiz, hx * wix + hy * wiy), v . wi = hy * wix - hx * wiy. */
  const float k = (hy * wix - hx * wiy) / max(1.0f + hz, 1e-6f);
  const float wdx = wix * hz + (-hx * wiz) + k * hy;
  const float wdy = wiy * hz + (-hy * wiz) + k * (-hx);
  const float wdz = wiz * hz + (hx * wix + hy * wiy);

  out[0] = wix;
  out[1] = wiy;
  out[2] = wiz;
  out[3] = wo.x;
  out[4] = wo.y;
  out[5] = wo.z;
  out[6] = hx;
  out[7] = hy;
  out[8] = hz;
  out[9] = wdx;
  out[10] = wdy;
  out[11] = wdz;
}

/* ------------------------------------------------------------------------------------------
 * Learned shading frames (nnm_format.md section 7, 05_learned_shading_frames.md section 2).
 * ------------------------------------------------------------------------------------------ */

/* Normalise (x, y, z) in place by division with sqrtf of the plain squared length, or replace it
 * with (fx, fy, fz) when the squared length is <= 1e-12. */
ccl_device_inline void neural_frame_normalize(ccl_private float *x,
                                              ccl_private float *y,
                                              ccl_private float *z,
                                              const float fx,
                                              const float fy,
                                              const float fz)
{
  const float l2 = (*x * *x + *y * *y) + *z * *z;
  if (l2 > 1e-12f) {
    const float len = sqrtf(l2);
    *x = *x / len;
    *y = *y / len;
    *z = *z / len;
  }
  else {
    *x = fx;
    *y = fy;
    *z = fz;
  }
}

template<int N_PAD>
ccl_device_inline void neural_encode_learned_frames_n(ccl_global const half *frame_layer,
                                                      const int latent_channels,
                                                      const int frames_k,
                                                      const float3 wi,
                                                      const float3 wo,
                                                      ccl_private const float *z,
                                                      ccl_private float *out)
{
  /* f = bf + sum_c z[c] * Wf[c], same axpy/fma order as an MLP layer. */
  float f[N_PAD];
  neural_linear<N_PAD>(frame_layer, z, latent_channels, f);

  for (int k = 0; k < frames_k; k++) {
    float nx = f[6 * k + 0], ny = f[6 * k + 1], nz = f[6 * k + 2];
    float tx = f[6 * k + 3], ty = f[6 * k + 4], tz = f[6 * k + 5];
    neural_frame_normalize(&nx, &ny, &nz, 0.0f, 0.0f, 1.0f);
    neural_frame_normalize(&tx, &ty, &tz, 1.0f, 0.0f, 0.0f);

    /* b = n x t, normalised; if (almost) parallel use a unit vector orthogonal to n: the first
     * output `a` of make_orthonormals(n, &a, &b) (util/math_float3.h). */
    float bx = ny * tz - nz * ty;
    float by = nz * tx - nx * tz;
    float bz = nx * ty - ny * tx;
    const float l2b = (bx * bx + by * by) + bz * bz;
    if (l2b > 1e-12f) {
      const float len = sqrtf(l2b);
      bx = bx / len;
      by = by / len;
      bz = bz / len;
    }
    else {
      /* Same construction as make_orthonormals() in util/math_float3.h (first output `a`), written
       * out so that this header does not depend on float3 normalize(), whose SSE4.2 intrinsics
       * do not compile in the generic standalone build. */
      if (nx != ny || nx != nz) {
        bx = nz - ny;
        by = nx - nz;
        bz = ny - nx;
      }
      else {
        bx = nz - ny;
        by = nx + nz;
        bz = -ny - nx;
      }
      const float len = sqrtf((bx * bx + by * by) + bz * bz);
      bx = bx / len;
      by = by / len;
      bz = bz / len;
    }

    ccl_private float *o = out + 6 * k;
    o[0] = (wi.x * tx + wi.y * ty) + wi.z * tz;
    o[1] = (wi.x * bx + wi.y * by) + wi.z * bz;
    o[2] = (wi.x * nx + wi.y * ny) + wi.z * nz;
    o[3] = (wo.x * tx + wo.y * ty) + wo.z * tz;
    o[4] = (wo.x * bx + wo.y * by) + wo.z * bz;
    o[5] = (wo.x * nx + wo.y * ny) + wo.z * nz;
  }
}

/* Writes `out[6 * m.frames_k]`: the directions wi, wo (unit, local frame) expressed in K frames
 * derived from the latent sample `z[m.latent_channels]`. The frame layer is read from
 * `weights + m.weight_offset + neural_frames_offset(m)`. */
ccl_device void neural_encode_learned_frames(ccl_global const half *weights,
                                             const ccl_global KernelNeuralModel &m,
                                             const float3 wi,
                                             const float3 wo,
                                             ccl_private const float *z,
                                             ccl_private float *out)
{
  ccl_global const half *frame_layer = weights + m.weight_offset + neural_frames_offset(m);
  /* 6K padded to a multiple of 8: 8 (K = 1), 16 (K = 2), 24 (K = 3, 4). */
  const int n_pad = neural_pad8(6 * m.frames_k);
  if (n_pad <= 8) {
    neural_encode_learned_frames_n<8>(frame_layer, m.latent_channels, m.frames_k, wi, wo, z, out);
  }
  else if (n_pad <= 16) {
    neural_encode_learned_frames_n<16>(frame_layer, m.latent_channels, m.frames_k, wi, wo, z, out);
  }
  else {
    neural_encode_learned_frames_n<24>(frame_layer, m.latent_channels, m.frames_k, wi, wo, z, out);
  }
}

/* ------------------------------------------------------------------------------------------
 * Latent texture (nnm_format.md section 5).
 * ------------------------------------------------------------------------------------------ */

/* Bilinear lookup of level 0 into `out[m.latent_channels]`. Texels sit on a corner-to-corner grid
 * (u = 0 is texel 0, u = 1 is the last texel), coordinates are clamped to [0, 1]. */
ccl_device void neural_latent_sample_bilinear(ccl_global const half *latents,
                                              const ccl_global KernelNeuralModel &m,
                                              const float u,
                                              const float v,
                                              ccl_private float *out)
{
  const int width = m.latent_width;
  const int height = m.latent_height;
  const int channels = m.latent_channels;
  ccl_global const half *tex = latents + m.latent_offset;

  const float x = clamp(u, 0.0f, 1.0f) * float(width - 1);
  const float y = clamp(v, 0.0f, 1.0f) * float(height - 1);

  const int ix = clamp(int(floorf(x)), 0, max(width - 2, 0));
  const int iy = clamp(int(floorf(y)), 0, max(height - 2, 0));
  const float fx = x - float(ix);
  const float fy = y - float(iy);
  const int ix1 = min(ix + 1, width - 1);
  const int iy1 = min(iy + 1, height - 1);

  const float w00 = (1.0f - fx) * (1.0f - fy);
  const float w10 = fx * (1.0f - fy);
  const float w01 = (1.0f - fx) * fy;
  const float w11 = fx * fy;

  ccl_global const half *t00 = tex + (iy * width + ix) * channels;
  ccl_global const half *t10 = tex + (iy * width + ix1) * channels;
  ccl_global const half *t01 = tex + (iy1 * width + ix) * channels;
  ccl_global const half *t11 = tex + (iy1 * width + ix1) * channels;

  for (int c = 0; c < channels; c++) {
    float z = half_to_float(t00[c]) * w00;
    z = neural_fma(half_to_float(t10[c]), w10, z);
    z = neural_fma(half_to_float(t01[c]), w01, z);
    z = neural_fma(half_to_float(t11[c]), w11, z);
    out[c] = z;
  }
}

/* ------------------------------------------------------------------------------------------
 * Full pipeline: caller input -> encoding -> latent lookup -> MLP.
 * ------------------------------------------------------------------------------------------ */

/* `caller` holds 6 floats (wi, wo) for the bsdf_dirs and learned_frames encodings, or `input_dim - latent_channels`
 * floats for the raw encoding. `u`, `v` are ignored when the model has no latent. */
ccl_device void neural_model_eval(ccl_global const half *weights,
                                  ccl_global const half *latents,
                                  const ccl_global KernelNeuralModel &m,
                                  ccl_private const float *caller,
                                  const float u,
                                  const float v,
                                  ccl_private float *out)
{
  float x[NEURAL_MAX_INPUT];
  const int encoded_dim = m.input_dim - m.latent_channels;

  if (m.encoding == NEURAL_ENCODING_LEARNED_FRAMES) {
    /* The frames depend on the latent sample: look it up first (into its final slot), then encode. */
    neural_latent_sample_bilinear(latents, m, u, v, x + encoded_dim);
    neural_encode_learned_frames(weights,
                                 m,
                                 make_float3(caller[0], caller[1], caller[2]),
                                 make_float3(caller[3], caller[4], caller[5]),
                                 x + encoded_dim,
                                 x);
  }
  else {
    if (m.encoding == NEURAL_ENCODING_BSDF_DIRS) {
      neural_encode_bsdf_dirs(make_float3(caller[0], caller[1], caller[2]),
                              make_float3(caller[3], caller[4], caller[5]),
                              x);
    }
    else {
      for (int i = 0; i < encoded_dim; i++) {
        x[i] = caller[i];
      }
    }

    if (m.latent_channels > 0) {
      neural_latent_sample_bilinear(latents, m, u, v, x + encoded_dim);
    }
  }

  neural_mlp_eval_dispatch(weights, m, x, out);
}

CCL_NAMESPACE_END
