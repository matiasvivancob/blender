/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/closure/alloc.h"
#include "kernel/geom/triangle.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/closure/bsdf_util.h"
#include "kernel/sample/mapping.h"
#include "kernel/util/colorspace.h"
#include "kernel/util/mlp.h"

CCL_NAMESPACE_BEGIN

/* Neural BSDF (doc/neural_rendering/01_cycles_neural_closures_spec.md section 4, version 1).
 *
 * The material is an MLP decoder loaded from a `.nnm` file (kernel/util/mlp.h). It takes the
 * incoming and outgoing directions in the local shading frame plus a latent looked up at the
 * shading point's UV, and returns f(wi, wo) * cos(theta_o) as RGB. The network is trained on that
 * product, so it is returned as is, without another cosine.
 *
 * The local frame is the one of SHADER_EVAL_BSDF, which produced the training data: z is the
 * shading normal, x the projected texture-U tangent, y = z cross x.
 *
 * Sampling uses a per-material proxy stored in the `.nnm` proxy block: a mixture of a cosine lobe
 * (probability `p_diffuse`) and a GGX visible-normal reflection lobe (`alpha`). The reported pdf is
 * the pdf of exactly that procedure, so the proxy changes noise, never the expected value.
 *
 * `alpha` and `p_diffuse` are copied into the closure so the roughness/label queries need no
 * kernel globals; the albedo is looked up when asked for. */

struct NeuralBsdf {
  SHADER_CLOSURE_BASE;

  /* x axis of the local frame; the frame is (T, N x T, N). */
  float3 T;
  uint model;
  float u, v;
  float alpha;
  float p_diffuse;
};

static_assert(sizeof(ShaderClosure) >= sizeof(NeuralBsdf), "NeuralBsdf is too large!");

ccl_device_inline const ccl_global KernelNeuralModel *bsdf_neural_model(KernelGlobals kg,
                                                                        const uint model)
{
  return (const ccl_global KernelNeuralModel *)(kernel_data_array(neural_models) +
                                                size_t(model) *
                                                    (sizeof(KernelNeuralModel) / sizeof(uint)));
}

ccl_device_inline float3 bsdf_neural_albedo_rgb(KernelGlobals kg, const uint model)
{
  const ccl_global float *p = kernel_data_array(neural_proxy) + size_t(model) * NEURAL_PROXY_STRIDE;
  return make_float3(p[NEURAL_PROXY_ALBEDO], p[NEURAL_PROXY_ALBEDO + 1], p[NEURAL_PROXY_ALBEDO + 2]);
}

ccl_device_inline Spectrum bsdf_neural_albedo(KernelGlobals kg,
                                              const ccl_private ShaderClosure *sc)
{
  const ccl_private NeuralBsdf *bsdf = (const ccl_private NeuralBsdf *)sc;
  return rgb_to_spectrum(bsdf_neural_albedo_rgb(kg, bsdf->model));
}

ccl_device void bsdf_neural_setup(KernelGlobals kg,
                                  ccl_private ShaderData *sd,
                                  const Spectrum weight,
                                  const float3 N,
                                  const uint model,
                                  const float2 uv,
                                  const AttributeDescriptor uv_desc)
{
  ccl_private NeuralBsdf *bsdf = (ccl_private NeuralBsdf *)bsdf_alloc(
      sd, sizeof(NeuralBsdf), weight);
  if (!bsdf) {
    return;
  }

  const ccl_global float *proxy = kernel_data_array(neural_proxy) +
                                  size_t(model) * NEURAL_PROXY_STRIDE;

  bsdf->N = N;
  bsdf->type = CLOSURE_BSDF_NEURAL_ID;
  bsdf->model = model;
  bsdf->u = uv.x;
  bsdf->v = uv.y;
  bsdf->alpha = proxy[NEURAL_PROXY_ALPHA];
  bsdf->p_diffuse = proxy[NEURAL_PROXY_P_DIFFUSE];

  bsdf->T = neural_frame_tangent(N, neural_uv_tangent(kg, sd, uv_desc));

  /* The sample weight must match the albedo reported by bsdf_albedo(), see the note there. */
  bsdf->sample_weight *= saturatef(average(bsdf_neural_albedo_rgb(kg, model)));

  sd->runtime_flag |= (SR_BSDF | SR_BSDF_HAS_EVAL);
}

/* World to local frame of the closure. */
ccl_device_inline float3 bsdf_neural_to_local(const ccl_private NeuralBsdf *bsdf, const float3 w)
{
  const float3 B = cross(bsdf->N, bsdf->T);
  return make_float3(dot(w, bsdf->T), dot(w, B), dot(w, bsdf->N));
}

ccl_device_inline float3 bsdf_neural_to_world(const ccl_private NeuralBsdf *bsdf, const float3 w)
{
  const float3 B = cross(bsdf->N, bsdf->T);
  return w.x * bsdf->T + w.y * B + w.z * bsdf->N;
}

/* GGX visible-normal reflection pdf (what bsdf_microfacet_eval reports for an isotropic GGX
 * reflection with no Fresnel) mixed with the cosine lobe. Directions are in the local frame and
 * both above the horizon. */
ccl_device_inline float bsdf_neural_pdf(const float3 wi,
                                        const float3 wo,
                                        const float alpha,
                                        const float p_diffuse)
{
  const float alpha2 = alpha * alpha;
  const float3 H = normalize(wi + wo);

  const float D = bsdf_D<MicrofacetType::GGX>(alpha2, H.z);
  const float lambda_i = bsdf_lambda<MicrofacetType::GGX>(alpha2, wi.z);
  const float pdf_glossy = D * 0.25f / (wi.z * (1.0f + lambda_i));

  return p_diffuse * wo.z * M_1_PI_F + (1.0f - p_diffuse) * pdf_glossy;
}

/* f * cos(theta_o) from the network. */
ccl_device_inline Spectrum bsdf_neural_f(KernelGlobals kg,
                                         const ccl_private NeuralBsdf *bsdf,
                                         const float3 wi,
                                         const float3 wo)
{
  const ccl_global KernelNeuralModel *m = bsdf_neural_model(kg, bsdf->model);

  float caller[6] = {wi.x, wi.y, wi.z, wo.x, wo.y, wo.z};
  float out[NEURAL_MAX_OUTPUT];
  neural_model_eval(kernel_data_array(neural_weights),
                    kernel_data_array(neural_latents),
                    *m,
                    caller,
                    bsdf->u,
                    bsdf->v,
                    out);

  const float3 rgb = max(ensure_finite(make_float3(out[0], out[1], out[2])), zero_float3());
  return rgb_to_spectrum(rgb);
}

ccl_device Spectrum bsdf_neural_eval(KernelGlobals kg,
                                     const ccl_private ShaderClosure *sc,
                                     const float3 wi,
                                     const float3 wo,
                                     ccl_private float *pdf)
{
  const ccl_private NeuralBsdf *bsdf = (const ccl_private NeuralBsdf *)sc;

  const float3 wi_local = bsdf_neural_to_local(bsdf, wi);
  const float3 wo_local = bsdf_neural_to_local(bsdf, wo);

  if (!(wi_local.z > 0.0f) || !(wo_local.z > 0.0f)) {
    *pdf = 0.0f;
    return zero_spectrum();
  }

  *pdf = bsdf_neural_pdf(wi_local, wo_local, bsdf->alpha, bsdf->p_diffuse);
  return bsdf_neural_f(kg, bsdf, wi_local, wo_local);
}

ccl_device int bsdf_neural_sample(KernelGlobals kg,
                                  const ccl_private ShaderClosure *sc,
                                  const float3 wi,
                                  const float3 rand,
                                  ccl_private Spectrum *eval,
                                  ccl_private float3 *wo,
                                  ccl_private float *pdf,
                                  ccl_private float2 *sampled_roughness,
                                  ccl_private float *eta)
{
  const ccl_private NeuralBsdf *bsdf = (const ccl_private NeuralBsdf *)sc;

  *eval = zero_spectrum();
  *pdf = 0.0f;
  *sampled_roughness = make_float2(bsdf->alpha, bsdf->alpha);
  *eta = 1.0f;

  const float3 wi_local = bsdf_neural_to_local(bsdf, wi);
  if (!(wi_local.z > 0.0f)) {
    return LABEL_NONE;
  }

  /* Pick a lobe with rand.z; the direction uses only rand.xy. */
  float3 wo_local;
  if (rand.z < bsdf->p_diffuse) {
    float unused_pdf;
    sample_cos_hemisphere(make_float3(0.0f, 0.0f, 1.0f), make_float2(rand), &wo_local, &unused_pdf);
  }
  else {
    const float3 H = microfacet_ggx_sample_vndf(wi_local, bsdf->alpha, bsdf->alpha, make_float2(rand));
    wo_local = 2.0f * dot(wi_local, H) * H - wi_local;
  }

  if (!(wo_local.z > 0.0f)) {
    return LABEL_NONE;
  }

  *pdf = bsdf_neural_pdf(wi_local, wo_local, bsdf->alpha, bsdf->p_diffuse);
  if (!(*pdf > 0.0f)) {
    *pdf = 0.0f;
    return LABEL_NONE;
  }

  *eval = bsdf_neural_f(kg, bsdf, wi_local, wo_local);
  *wo = bsdf_neural_to_world(bsdf, wo_local);

  return LABEL_REFLECT | ((bsdf->p_diffuse > 0.5f) ? LABEL_DIFFUSE : LABEL_GLOSSY);
}

/* Filter Glossy widens only the sampling proxy, not the network lobe: it reduces noise but does not
 * soften the neural highlight. */
ccl_device_inline void bsdf_neural_blur(ccl_private ShaderClosure *sc, const float roughness)
{
  ccl_private NeuralBsdf *bsdf = (ccl_private NeuralBsdf *)sc;
  bsdf->alpha = fmaxf(roughness, bsdf->alpha);
}

CCL_NAMESPACE_END
