/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Shared triangle UV derivative and frame math. Preserve the determinant sign on mirrored UVs. */
ccl_device_inline float3 neural_uv_tangent_from_edges(const float3 e1,
                                                       const float3 e2,
                                                       const float2 d1,
                                                       const float2 d2,
                                                       const float3 fallback)
{
  const float det = d1.x * d2.y - d2.x * d1.y;
  if (!(fabsf(det) > 1e-20f)) {
    return fallback;
  }
  return (e1 * d2.y - e2 * d1.y) / det;
}

ccl_device_inline float3 neural_frame_tangent(const float3 N, const float3 tangent)
{
  float3 T = tangent - N * dot(tangent, N);
  const float length_squared = len_squared(T);
  if (length_squared > 1e-20f && isfinite_safe(T)) {
    return T * inversesqrtf(length_squared);
  }
  float3 B;
  make_orthonormals(N, &T, &B);
  return T;
}

CCL_NAMESPACE_END
