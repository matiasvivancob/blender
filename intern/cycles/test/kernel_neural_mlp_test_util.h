/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Minimal `.nnm` parser and test-vector checker shared by kernel_neural_mlp_test.cpp and the
 * standalone parity test in doc/neural_rendering/tools/nnm/. Test code only: uses the STL. */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "kernel/util/mlp.h"

CCL_NAMESPACE_BEGIN

struct NnmTestModel {
  KernelNeuralModel km;
  /* The shared arrays, with some junk in front so that a wrong `weight_offset` / `latent_offset`
   * would be noticed. */
  std::vector<half> weights;
  std::vector<half> latents;
  int caller_dim = 0;
  int tv_count = 0;
  int tv_in_dim = 0;
  int tv_out_dim = 0;
  std::vector<float> tv; /* tv_count records of (tv_in_dim + tv_out_dim) floats. */
  uint64_t weight_blob_bytes = 0;
  uint64_t layer_offset_bytes[5] = {};
};

namespace nnm_detail {

static constexpr uint16_t k_junk_half = 0x7e01; /* A NaN. */
static constexpr int k_weight_prefix = 24;
static constexpr int k_latent_prefix = 40;

inline uint32_t rd32(const uint8_t *d, size_t off)
{
  return uint32_t(d[off]) | (uint32_t(d[off + 1]) << 8) | (uint32_t(d[off + 2]) << 16) |
         (uint32_t(d[off + 3]) << 24);
}

inline uint64_t rd64(const uint8_t *d, size_t off)
{
  return uint64_t(rd32(d, off)) | (uint64_t(rd32(d, off + 4)) << 32);
}

inline float rdf(const uint8_t *d, size_t off)
{
  const uint32_t u = rd32(d, off);
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline std::vector<half> rd_halves(const uint8_t *d, size_t off, size_t count, int prefix)
{
  std::vector<half> out(prefix + count, half(k_junk_half));
  for (size_t i = 0; i < count; i++) {
    out[prefix + i] = half(uint16_t(d[off + 2 * i] | (d[off + 2 * i + 1] << 8)));
  }
  return out;
}

}  // namespace nnm_detail

/* Parses and validates `data` as a `.nnm` file. Returns false and sets `err` on failure. */
inline bool nnm_parse(const uint8_t *d, const size_t size, NnmTestModel &out, std::string &err)
{
  using namespace nnm_detail;
#define NNM_REQUIRE(cond, msg) \
  if (!(cond)) { \
    err = msg; \
    return false; \
  }

  NNM_REQUIRE(size >= 256, "file shorter than the header");
  NNM_REQUIRE(std::memcmp(d, "NNM1", 4) == 0, "bad magic");
  NNM_REQUIRE(rd32(d, 4) == 1, "unsupported version");
  NNM_REQUIRE(rd32(d, 8) == 256, "bad header size");
  NNM_REQUIRE(rd32(d, 12) == 0, "unknown flags");

  KernelNeuralModel &km = out.km;
  std::memset(&km, 0, sizeof(km));
  km.input_dim = int(rd32(d, 16));
  km.output_dim = int(rd32(d, 20));
  km.num_hidden = int(rd32(d, 24));
  km.width = int(rd32(d, 28));
  km.activation = int(rd32(d, 32));
  km.output_transform = int(rd32(d, 36));
  km.encoding = int(rd32(d, 40));
  km.latent_channels = int(rd32(d, 44));
  km.act_beta = rdf(d, 48);
  km.act_eps = rdf(d, 52);
  km.latent_width = int(rd32(d, 56));
  km.latent_height = int(rd32(d, 60));
  km.latent_levels = int(rd32(d, 64));
  out.tv_count = int(rd32(d, 68));
  const uint64_t w_off = rd64(d, 72), w_size = rd64(d, 80);
  const uint64_t l_off = rd64(d, 88), l_size = rd64(d, 96);
  const uint64_t t_off = rd64(d, 104), t_size = rd64(d, 112);
  out.tv_in_dim = int(rd32(d, 120));
  out.tv_out_dim = int(rd32(d, 124));
  const uint32_t num_layers = rd32(d, 128);
  for (int i = 0; i < 5; i++) {
    out.layer_offset_bytes[i] = rd32(d, 132 + 4 * i);
  }
  NNM_REQUIRE(rd64(d, 152) == size, "file_size mismatch");

  NNM_REQUIRE(km.input_dim >= 1 && km.input_dim <= NEURAL_MAX_INPUT, "input_dim out of range");
  NNM_REQUIRE(km.output_dim >= 1 && km.output_dim <= NEURAL_MAX_OUTPUT, "output_dim out of range");
  NNM_REQUIRE(km.num_hidden >= 1 && km.num_hidden <= NEURAL_MAX_HIDDEN, "num_hidden out of range");
  NNM_REQUIRE(km.width == 16 || km.width == 32 || km.width == 64, "bad hidden width");
  NNM_REQUIRE(km.activation >= 0 && km.activation <= 2, "bad activation id");
  NNM_REQUIRE(km.output_transform >= 0 && km.output_transform <= 2, "bad output transform id");
  NNM_REQUIRE(km.encoding >= 0 && km.encoding <= 1, "bad encoding id");
  NNM_REQUIRE(km.latent_channels >= 0 && km.latent_channels <= NEURAL_MAX_LATENT_CHANNELS,
              "bad latent channel count");
  const int encoded_dim = km.input_dim - km.latent_channels;
  NNM_REQUIRE(encoded_dim >= 1, "no room for encoded features");
  if (km.encoding == NEURAL_ENCODING_BSDF_DIRS) {
    NNM_REQUIRE(encoded_dim == 12, "bsdf_dirs needs 12 encoded features");
  }
  out.caller_dim = (km.encoding == NEURAL_ENCODING_BSDF_DIRS) ? 6 : encoded_dim;
  NNM_REQUIRE(num_layers == uint32_t(km.num_hidden + 1), "num_layers mismatch");

  /* Weight blob: layer table must agree with the layout rule used by the kernel. */
  km.weight_offset = uint(k_weight_prefix);
  for (int l = 0; l < km.num_hidden + 1; l++) {
    NNM_REQUIRE(out.layer_offset_bytes[l] == 2ull * neural_layer_offset(km, l), "layer offset table mismatch");
  }
  NNM_REQUIRE(w_size == 2ull * neural_layer_offset(km, km.num_hidden + 1), "weight_size mismatch");
  NNM_REQUIRE(w_off == 256 && w_off + w_size <= size, "weight blob out of bounds");
  out.weight_blob_bytes = w_size;
  out.weights = rd_halves(d, size_t(w_off), size_t(w_size / 2), k_weight_prefix);

  if (km.latent_channels > 0) {
    NNM_REQUIRE(km.latent_width >= 1 && km.latent_height >= 1 && km.latent_levels >= 1, "bad latent descriptor");
    uint64_t expect = 0;
    for (int k = 0; k < km.latent_levels; k++) {
      const uint64_t w = std::max(1, km.latent_width >> k), h = std::max(1, km.latent_height >> k);
      expect += w * h * km.latent_channels * 2;
    }
    NNM_REQUIRE(l_size == expect, "latent_size mismatch");
    NNM_REQUIRE(l_off % 64 == 0 && l_off + l_size <= size, "latent blob out of bounds");
    km.latent_offset = uint(k_latent_prefix);
    out.latents = rd_halves(d, size_t(l_off), size_t(l_size / 2), k_latent_prefix);
  }
  else {
    NNM_REQUIRE(km.latent_width == 0 && km.latent_height == 0 && km.latent_levels == 0 && l_size == 0,
                "latent fields must be zero");
    out.latents.assign(1, half(k_junk_half));
  }

  if (out.tv_count > 0) {
    NNM_REQUIRE(out.tv_in_dim == out.caller_dim + (km.latent_channels > 0 ? 2 : 0), "testvec_in_dim mismatch");
    NNM_REQUIRE(out.tv_out_dim == km.output_dim, "testvec_out_dim mismatch");
    NNM_REQUIRE(t_size == uint64_t(out.tv_count) * (out.tv_in_dim + out.tv_out_dim) * 4, "testvec_size mismatch");
    NNM_REQUIRE(t_off % 64 == 0 && t_off + t_size <= size, "testvec block out of bounds");
    out.tv.resize(size_t(t_size / 4));
    for (size_t i = 0; i < out.tv.size(); i++) {
      out.tv[i] = rdf(d, size_t(t_off) + 4 * i);
    }
  }
#undef NNM_REQUIRE
  return true;
}

/* Runs every embedded test vector through the kernel decoder. A value passes when
 * |got - expected| <= rtol * |expected| + atol. `max_abs_err` / `max_rel_err` report the worst
 * deviation (relative error uses max(|expected|, atol) as denominator). */
inline bool nnm_check_test_vectors(const NnmTestModel &t,
                                   const double rtol,
                                   const double atol,
                                   double &max_abs_err,
                                   double &max_rel_err,
                                   std::string &err)
{
  max_abs_err = 0.0;
  max_rel_err = 0.0;
  bool ok = true;
  const int rec = t.tv_in_dim + t.tv_out_dim;
  for (int n = 0; n < t.tv_count; n++) {
    const float *in = &t.tv[size_t(n) * rec];
    const float *expected = in + t.tv_in_dim;
    float got[NEURAL_MAX_OUTPUT];
    const float u = t.km.latent_channels > 0 ? in[t.caller_dim] : 0.0f;
    const float v = t.km.latent_channels > 0 ? in[t.caller_dim + 1] : 0.0f;
    neural_model_eval(t.weights.data(), t.latents.data(), t.km, in, u, v, got);
    for (int o = 0; o < t.tv_out_dim; o++) {
      const double e = expected[o], g = got[o];
      const double abs_err = std::fabs(g - e);
      const double rel_err = abs_err / std::max(std::fabs(e), atol);
      max_abs_err = std::max(max_abs_err, abs_err);
      max_rel_err = std::max(max_rel_err, rel_err);
      if (!(abs_err <= rtol * std::fabs(e) + atol)) { /* Also catches NaN. */
        if (ok) {
          err = "vector " + std::to_string(n) + " output " + std::to_string(o) + ": got " + std::to_string(g) +
                ", expected " + std::to_string(e);
        }
        ok = false;
      }
    }
  }
  return ok;
}

CCL_NAMESPACE_END
