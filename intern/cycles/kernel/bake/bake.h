/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/globals.h"

#include "kernel/camera/projection.h"
#include "kernel/integrator/displacement_shader.h"
#include "kernel/integrator/state.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/integrator/volume_shader.h"

#include "kernel/geom/object.h"
#include "kernel/geom/shader_data.h"

#include "kernel/util/colorspace.h"

#include "util/types_rgbe.h"

CCL_NAMESPACE_BEGIN

ccl_device void kernel_displace_evaluate(KernelGlobals kg,
                                         const ccl_global KernelShaderEvalInput *input,
                                         ccl_global float *output,
                                         ccl_global uint *cache_miss,
                                         const int offset)
{
  /* Setup shader data. */
  const KernelShaderEvalInput in = input[offset];

  ShaderData sd;
  shader_setup_from_displace(kg, &sd, in.object, in.prim, in.u, in.v);

  /* Evaluate displacement shader. */
  ConstIntegratorBakeState state;
  const float3 P = sd.P;
  displacement_shader_eval(kg, state, &sd);
  float3 D = sd.P - P;
  if (sd.runtime_flag & SR_CACHE_MISS) {
    *cache_miss = true;
  }
  object_inverse_dir_transform(kg, &sd, &D);

#ifdef __KERNEL_DEBUG_NAN__
  if (!isfinite_safe(D)) {
    kernel_assert(!"Cycles displacement with non-finite value detected");
  }
#endif

  /* Ensure finite displacement, preventing BVH from becoming degenerate and avoiding possible
   * traversal issues caused by non-finite math. */
  D = ensure_finite(D);

  /* Write output. */
  output[offset * 3 + 0] = D.x;
  output[offset * 3 + 1] = D.y;
  output[offset * 3 + 2] = D.z;
}

ccl_device void kernel_background_evaluate(KernelGlobals kg,
                                           const ccl_global KernelShaderEvalInput *input,
                                           ccl_global float *output,
                                           ccl_global uint *cache_miss,
                                           const int offset)
{
  /* Setup ray */
  const KernelShaderEvalInput in = input[offset];
  const float3 ray_P = zero_float3();
  const float3 ray_D = equirectangular_to_direction(in.u, in.v);
  const float ray_time = 0.5f;

  /* Compute ray differential from resolution passed via object and prim fields. */
  const float du = 1.0f / in.object;
  const float dv = 1.0f / in.prim;
  const float3 ray_D_du = equirectangular_to_direction(in.u + du, in.v);
  const float3 ray_D_dv = equirectangular_to_direction(in.u, in.v + dv);
  const float ray_dD = 0.5f * (len(ray_D_du - ray_D) + len(ray_D_dv - ray_D));

  /* Setup shader data. */
  ShaderData sd;
  shader_setup_from_background(kg, &sd, ray_P, ray_D, ray_dD, ray_time);

  /* Evaluate shader.
   * This is being evaluated for all BSDFs, so path flag does not contain a specific type.
   * However, we want to flag the ray visibility to ignore the sun in the background map. */
  ConstIntegratorBakeState state;
  const uint32_t path_flag = PATH_RAY_EMISSION | PATH_RAY_IMPORTANCE_BAKE;
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_LIGHT &
                      ~(KERNEL_FEATURE_NODE_RAYTRACE | KERNEL_FEATURE_NODE_LIGHT_PATH)>(
      kg, state, &sd, nullptr, PATH_RAY_VISIBILITY_NONE, path_flag);
  if (sd.runtime_flag & SR_CACHE_MISS) {
    *cache_miss = true;
  }

  Spectrum color = surface_shader_background(&sd);

#ifdef __KERNEL_DEBUG_NAN__
  if (!isfinite_safe(color)) {
    kernel_assert(!"Cycles background with non-finite value detected");
  }
#endif

  /* Ensure finite color, avoiding possible numerical instabilities in the path tracing kernels. */
  color = ensure_finite(color);

  const float3 color_rgb = spectrum_to_rgb(color);

  /* Write output. */
  output[offset * 3 + 0] = color_rgb.x;
  output[offset * 3 + 1] = color_rgb.y;
  output[offset * 3 + 2] = color_rgb.z;
}

ccl_device void kernel_curve_shadow_transparency_evaluate(
    KernelGlobals kg,
    const ccl_global KernelShaderEvalInput *input,
    ccl_global float *output,
    ccl_global uint *cache_miss,
    const int offset)
{
#ifdef __HAIR__
  /* Setup shader data. */
  const KernelShaderEvalInput in = input[offset];

  ShaderData sd;
  shader_setup_from_curve(kg, &sd, in.object, in.prim, __float_as_int(in.v), in.u);

  /* Evaluate transparency. */
  ConstIntegratorBakeState state;
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW &
                      ~(KERNEL_FEATURE_NODE_RAYTRACE | KERNEL_FEATURE_NODE_LIGHT_PATH)>(
      kg, state, &sd, nullptr, PATH_RAY_VISIBILITY_SHADOW, PATH_RAY_FLAG_NONE);

  if (sd.runtime_flag & SR_CACHE_MISS) {
    *cache_miss = true;
  }

  /* Write output. */
  output[offset] = rgb_to_rgbe(saturate(spectrum_to_rgb(surface_shader_transparency(&sd)))).f;
#endif
}

ccl_device void kernel_volume_density_evaluate(KernelGlobals kg,
                                               ccl_global const KernelShaderEvalInput *input,
                                               ccl_global float *output,
                                               ccl_global uint *cache_miss,
                                               const int offset)
{
#ifdef __VOLUME__
  if (input[offset * 2 + 1].object == SHADER_NONE) {
    return;
  }

  KernelShaderEvalInput in = input[offset * 2];

  /* Setup ray. */
  Ray ray;
  ray.P = make_float3(__int_as_float(in.prim), in.u, in.v);
  ray.D = zero_float3();
  ray.tmin = 0.0f;
  /* Motion blur is ignored when computing the extrema of the density, but we also don't expect the
   * value to change a lot in one frame. */
  ray.time = 0.5f;

  /* Setup shader data. */
  ShaderData sd;
  shader_setup_from_volume(&sd, &ray, in.object);
  sd.runtime_flag = SR_IS_VOLUME_SHADER_EVAL;
  sd.shader_flag = 0;
  /* For stochastic texture sampling. */
  sd.lcg_state = lcg_state_init(offset, 0, 0, 0x15b4f88d);

  /* Evaluate extinction and emission without allocating closures. */
  sd.num_closure_left = 0;
  /* Evaluate density for camera ray because it usually makes the most visual impact. For shaders
   * that depends on ray types, the extrema are estimated on the fly. */
  /* TODO(weizhen): Volume invisible to camera ray might appear noisy. We can at least build a
   * separate octree for shadow ray. */
  const PathRayVisibility path_visibility = PATH_RAY_VISIBILITY_CAMERA;
  const uint32_t path_flag = PATH_RAY_FLAG_NONE;

  /* Setup volume stack entry. */
  in = input[offset * 2 + 1];
  const int shader = in.object;
  const VolumeStack entry = {sd.object, shader};

  const float3 voxel_size = make_float3(__int_as_float(in.prim), in.u, in.v);
  Extrema<float> extrema = {FLT_MAX, -FLT_MAX};
  /* For heterogeneous volume, we take 16 samples per grid;
   * for homogeneous volume, only 1 sample is needed. */
  const int num_samples = volume_is_homogeneous(kg, entry) ? 1 : 16;

  const bool need_transformation = !(kernel_data_fetch(object_flag, sd.object) &
                                     SD_OBJECT_TRANSFORM_APPLIED);
  const Transform tfm = need_transformation ?
                            object_fetch_transform(kg, sd.object, OBJECT_TRANSFORM) :
                            Transform();
  for (int sample = 0; sample < num_samples; sample++) {
    /* Blue noise indexing. The sequence length is the number of samples. */
    const uint3 index = make_uint3(sample + offset * num_samples, 0, 0xffffffff);

    /* Sample a random position inside the voxel. */
    const float3 rand_p = sobol_burley_sample_3D(
        index.x, PRNG_BAKE_VOLUME_DENSITY_EVAL, index.y, index.z);
    sd.P = ray.P + rand_p * voxel_size;
    if (need_transformation) {
      /* Convert to world space. */
      sd.P = transform_point(&tfm, sd.P);
    }
    sd.closure_transparent_extinction = zero_float3();
    sd.closure_emission_background = zero_float3();

    /* Evaluate volume coefficients. */
    ConstIntegratorBakeState state;
    volume_shader_eval_entry<false,
                             KERNEL_FEATURE_NODE_MASK_VOLUME & ~KERNEL_FEATURE_NODE_LIGHT_PATH>(
        kg, state, &sd, entry, path_visibility, path_flag);

    if (sd.runtime_flag & SR_CACHE_MISS) {
      /* Note we keep rendering other samples so we find all cache misses in one go. */
      *cache_miss = true;
    }

    const float sigma = reduce_max(sd.closure_transparent_extinction);
    const float emission = reduce_max(sd.closure_emission_background);

    extrema = merge(extrema, fmaxf(sigma, emission));
  }

  /* Write output. */
  const float scale = object_volume_density(kg, sd.object);
  output[offset * 2 + 0] = extrema.min / scale;
  output[offset * 2 + 1] = extrema.max / scale;
#endif
}

/* Evaluate the BSDF of a surface material at user supplied positions and directions. Used to
 * generate training data for neural materials.
 *
 * Each evaluation consumes 3 consecutive KernelShaderEvalInput records, and writes 8 floats:
 *
 *   input[3 * i + 0]: object = Cycles object index, prim = global triangle index (mesh
 *                     prim_offset + triangle index in the mesh), u, v = barycentric coordinates
 *                     (same convention as displacement evaluation).
 *   input[3 * i + 1]: wi.x, wi.y (object, prim; stored as float bit patterns), wi.z (u), and the
 *                     integer KernelBsdfEvalInputFlag bit mask (v; stored as integer bits).
 *   input[3 * i + 2]: wo.x, wo.y (object, prim; float bit patterns), wo.z (u), unused (v).
 *
 *   output[8 * i + 0..2]: BSDF value for all closures summed, with closure weights and the cosine
 *                         term |dot(wo, N)| included, as RGB.
 *   output[8 * i + 3]:    PDF of sampling wo from the closure mixture (solid angle measure).
 *   output[8 * i + 4..6]: Albedo, the sum of the diffuse, glossy and transmission albedo, as RGB.
 *   output[8 * i + 7]:    KernelBsdfEvalOutputFlag bit mask stored as a float (0 when valid).
 *
 * Both directions point away from the surface: wi towards the viewer (this is ShaderData::wi,
 * the negated ray direction), wo towards the light (or the next path vertex). They are in world
 * space, or in the local shading frame if KERNEL_BSDF_EVAL_LOCAL_FRAME is set. The local frame has
 * z along the unflipped shading normal, x along dPdu projected on the tangent plane, and y = z x x.
 * If wi is below the geometric normal the shading point is treated as backfacing, as it would be
 * for a camera ray hitting the back of the surface. */
ccl_device void kernel_bsdf_evaluate(KernelGlobals kg,
                                     ccl_global const KernelShaderEvalInput *input,
                                     ccl_global float *output,
                                     ccl_global uint *cache_miss,
                                     const int offset)
{
  const KernelShaderEvalInput in = input[offset * 3 + 0];
  const KernelShaderEvalInput in_wi = input[offset * 3 + 1];
  const KernelShaderEvalInput in_wo = input[offset * 3 + 2];

  ccl_global float *out = output + offset * 8;
  for (int i = 0; i < 8; i++) {
    out[i] = 0.0f;
  }

  if (in.object < 0 || in.prim < 0) {
    out[7] = float(KERNEL_BSDF_EVAL_INVALID);
    return;
  }

  const int flags = __float_as_int(in_wi.v);
  float3 wi = make_float3(__int_as_float(in_wi.object), __int_as_float(in_wi.prim), in_wi.u);
  float3 wo = make_float3(__int_as_float(in_wo.object), __int_as_float(in_wo.prim), in_wo.u);

  if (!isfinite_safe(wi) || !isfinite_safe(wo) || is_zero(wi) || is_zero(wo)) {
    out[7] = float(KERNEL_BSDF_EVAL_INVALID);
    return;
  }
  wi = normalize(wi);
  wo = normalize(wo);

  /* Setup shader data. No incoming direction is passed here since it would be interpreted as
   * object space, and the backfacing check needs the frame built below. */
  float3 P;
  float3 Ng;
  int shader;
  triangle_point_normal(kg, in.object, in.prim, in.u, in.v, &P, &Ng, &shader);

  const bool object_space = !(kernel_data_fetch(object_flag, in.object) &
                              SD_OBJECT_TRANSFORM_APPLIED);

  ShaderData sd;
  shader_setup_from_sample(kg,
                           &sd,
                           P,
                           Ng,
                           zero_float3(),
                           shader,
                           in.object,
                           in.prim,
                           in.u,
                           in.v,
                           0.0f,
                           0.5f,
                           object_space,
                           false);

  /* Transform directions from the local shading frame to world space. At this point N, dPdu and
   * dPdv are not flipped yet, so the frame does not depend on which side wi is on. */
  if (flags & KERNEL_BSDF_EVAL_LOCAL_FRAME) {
    const float3 N = sd.N;
    float3 tangent = zero_float3();
#ifdef __DPDU__
    tangent = sd.dPdu;
#endif
    if (flags & KERNEL_BSDF_EVAL_UV_TANGENT_FRAME) {
      tangent = neural_uv_tangent(kg, &sd, find_attribute(kg, &sd, ATTR_STD_UV));
    }
    const float3 T = neural_frame_tangent(N, tangent);
    const float3 B = cross(N, T);

    wi = normalize(wi.x * T + wi.y * B + wi.z * N);
    wo = normalize(wo.x * T + wo.y * B + wo.z * N);
  }

  /* Incoming direction and backfacing test, same as shader_setup_from_ray(). */
  sd.wi = wi;
  float out_flags = 0.0f;
  if (dot(sd.Ng, sd.wi) < 0.0f) {
    sd.runtime_flag |= SR_BACKFACING;
    sd.Ng = -sd.Ng;
    sd.N = -sd.N;
#ifdef __DPDU__
    sd.dPdu = -sd.dPdu;
    sd.dPdv = -sd.dPdv;
#endif
    out_flags += float(KERNEL_BSDF_EVAL_BACKFACING);
  }

#ifdef __SPECTRAL__
  /* Wavelength is not sampled, shaders that depend on it see the middle of the range. */
  sd.rand_wavelength = 0.5f;
#endif

  /* Evaluate surface shader. This is being evaluated like a primary camera ray hit. AOV nodes
   * need an integrator state to write to a render buffer, so they are not supported. Closure
   * filtering and glossy blur (surface_shader_prepare_closures) are deliberately not applied, to
   * get the unmodified BSDF. */
  ConstIntegratorBakeState state;
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE &
                      ~(KERNEL_FEATURE_NODE_RAYTRACE | KERNEL_FEATURE_NODE_LIGHT_PATH |
                        KERNEL_FEATURE_NODE_AOV)>(
      kg, state, &sd, nullptr, PATH_RAY_VISIBILITY_CAMERA, PATH_RAY_FLAG_NONE);

  if (sd.runtime_flag & SR_CACHE_MISS) {
    /* The whole evaluation is repeated after the missing textures are loaded. */
    *cache_miss = true;
    return;
  }

  bool has_bsdf = false;
  for (int i = 0; i < sd.num_closure; i++) {
    if (CLOSURE_IS_BSDF(sd.closure[i].type)) {
      has_bsdf = true;
      break;
    }
  }

  if (!has_bsdf) {
    out[7] = out_flags + float(KERNEL_BSDF_EVAL_NO_BSDF);
    return;
  }

  /* Evaluate BSDF. This is surface_shader_bsdf_eval() without the path guiding part, which needs
   * an integrator state that does not exist here. */
  BsdfEval eval;
  bsdf_eval_init(&eval, zero_spectrum());
  float avg_roughness_squared = 0.0f;
  const float pdf = _surface_shader_bsdf_eval_mis(
      kg, &sd, wo, nullptr, &eval, 0.0f, 0.0f, 0, 0.0f, avg_roughness_squared);

  const float3 bsdf_rgb = ensure_finite(spectrum_to_rgb(bsdf_eval_sum(&eval)));

  const Spectrum albedo = surface_shader_diffuse(kg, &sd) + surface_shader_glossy(kg, &sd) +
                          surface_shader_transmission(kg, &sd);
  const float3 albedo_rgb = ensure_finite(spectrum_to_rgb(albedo));

  out[0] = bsdf_rgb.x;
  out[1] = bsdf_rgb.y;
  out[2] = bsdf_rgb.z;
  out[3] = ensure_finite(pdf);
  out[4] = albedo_rgb.x;
  out[5] = albedo_rgb.y;
  out[6] = albedo_rgb.z;
  out[7] = out_flags;
}

CCL_NAMESPACE_END
