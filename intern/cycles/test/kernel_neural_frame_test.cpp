/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>
#include "kernel/closure/bsdf_neural_frame.h"

CCL_NAMESPACE_BEGIN

TEST(KernelNeuralFrame, UVRotationAndMirroring)
{
  const float3 e1 = make_float3(1, 0, 0), e2 = make_float3(0, 1, 0);
  const float3 fallback = make_float3(0, 0, 1);
  const float3 normal = make_float3(0, 0, 1);
  const auto check = [&](const float2 d1, const float2 d2, const float3 expected) {
    const float3 t = neural_frame_tangent(normal,
        neural_uv_tangent_from_edges(e1, e2, d1, d2, fallback));
    EXPECT_NEAR(t.x, expected.x, 1e-6f);
    EXPECT_NEAR(t.y, expected.y, 1e-6f);
    EXPECT_NEAR(t.z, expected.z, 1e-6f);
  };
  check(make_float2(1, 0), make_float2(0, 1), e1);
  check(make_float2(0, 1), make_float2(-1, 0), -e2);
  check(make_float2(-1, 0), make_float2(0, 1), -e1);
}

TEST(KernelNeuralFrame, BothQuadTrianglesAgree)
{
  const float3 e1 = make_float3(1, 0, 0), e2 = make_float3(1, 1, 0);
  const float3 t = neural_uv_tangent_from_edges(e1, e2,
      make_float2(1, 0), make_float2(1, 1), zero_float3());
  const float3 u = neural_uv_tangent_from_edges(e2, make_float3(0, 1, 0),
      make_float2(1, 1), make_float2(0, 1), zero_float3());
  EXPECT_NEAR(len(t - u), 0.0f, 1e-6f);
  EXPECT_NEAR(len(t - e1), 0.0f, 1e-6f);
}

TEST(KernelNeuralFrame, DegenerateUVAndNormalProjection)
{
  const float3 fallback = make_float3(1, 0, 0);
  const float3 t = neural_uv_tangent_from_edges(make_float3(1, 0, 0), make_float3(0, 1, 0),
      zero_float2(), zero_float2(), fallback);
  EXPECT_NEAR(len(t - fallback), 0.0f, 1e-6f);
  const float3 N = make_float3(0, 0, 1);
  const float3 projected = neural_frame_tangent(N, make_float3(2, 0, 4));
  EXPECT_NEAR(len(projected - fallback), 0.0f, 1e-6f);
  const float3 degenerate = neural_frame_tangent(N, N);
  EXPECT_NEAR(dot(degenerate, N), 0.0f, 1e-6f);
  EXPECT_NEAR(len(degenerate), 1.0f, 1e-6f);
}

CCL_NAMESPACE_END
