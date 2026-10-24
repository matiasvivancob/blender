/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "scene/neural.h"

#include "kernel_neural_mlp_test_data.h"
#include "kernel_neural_mlp_test_util.h"

/* NeuralFile / NeuralManager (scene/neural.cpp): the C++ loader against complete `.nnm` files written
 * by the Python tools (the embedded models of the kernel test) and against corrupted copies. */

CCL_NAMESPACE_BEGIN

namespace {

vector<uint8_t> model_bytes(const neural_test_data::EmbeddedModel &model)
{
  vector<uint8_t> bytes(model.size_bytes);
  for (size_t i = 0; i < bytes.size(); i++) {
    bytes[i] = uint8_t(model.words[i / 4] >> (8 * (i % 4)));
  }
  return bytes;
}

void set_u32(vector<uint8_t> &b, const size_t offset, const uint32_t value)
{
  std::memcpy(b.data() + offset, &value, sizeof(value));
}

void set_f32(vector<uint8_t> &b, const size_t offset, const float value)
{
  std::memcpy(b.data() + offset, &value, sizeof(value));
}

uint32_t get_u32(const vector<uint8_t> &b, const size_t offset)
{
  uint32_t v;
  std::memcpy(&v, b.data() + offset, sizeof(v));
  return v;
}

uint64_t get_u64(const vector<uint8_t> &b, const size_t offset)
{
  uint64_t v;
  std::memcpy(&v, b.data() + offset, sizeof(v));
  return v;
}

/* A model without latent so that offsets are simple, and one with latent for the latent path. */
const neural_test_data::EmbeddedModel &first_model_with(const bool latent)
{
  for (const neural_test_data::EmbeddedModel &m : neural_test_data::k_models) {
    const vector<uint8_t> b = model_bytes(m);
    if ((get_u32(b, 44) > 0) == latent && get_u32(b, 24) < 4) {
      return m;
    }
  }
  ADD_FAILURE() << "no embedded model with latent = " << latent;
  return neural_test_data::k_models[0];
}

void expect_rejected(const vector<uint8_t> &bytes, const char *fragment)
{
  NeuralFile file;
  EXPECT_FALSE(file.load(bytes));
  EXPECT_NE(file.error.find(fragment), string::npos) << "error was: " << file.error;
  EXPECT_TRUE(file.empty());
}

}  // namespace

TEST(SceneNeural, AllEmbeddedModelsLoadAndMatchTheTestParser)
{
  for (const neural_test_data::EmbeddedModel &model : neural_test_data::k_models) {
    SCOPED_TRACE(model.name);
    const vector<uint8_t> bytes = model_bytes(model);

    NeuralFile file;
    ASSERT_TRUE(file.load(bytes)) << file.error;
    EXPECT_FALSE(file.empty());
    EXPECT_FALSE(file.has_proxy); /* Embedded models predate the proxy block. */

    NnmTestModel t;
    std::string err;
    ASSERT_TRUE(nnm_parse(bytes.data(), bytes.size(), t, err)) << err;
    EXPECT_EQ(file.model.input_dim, t.km.input_dim);
    EXPECT_EQ(file.model.output_dim, t.km.output_dim);
    EXPECT_EQ(file.model.num_hidden, t.km.num_hidden);
    EXPECT_EQ(file.model.width, t.km.width);
    EXPECT_EQ(file.model.activation, t.km.activation);
    EXPECT_EQ(file.model.output_transform, t.km.output_transform);
    EXPECT_EQ(file.model.encoding, t.km.encoding);
    EXPECT_EQ(file.model.latent_channels, t.km.latent_channels);
    EXPECT_EQ(file.model.latent_width, t.km.latent_width);
    EXPECT_EQ(file.model.latent_height, t.km.latent_height);
    EXPECT_EQ(file.model.weight_offset, 0u);
    EXPECT_EQ(file.weights.size(), size_t(get_u64(bytes, 80) / 2));
    if (file.model.latent_channels > 0) {
      EXPECT_EQ(file.model.latent_levels, 1); /* Only level 0 is kept. */
      EXPECT_EQ(file.latents.size(),
                size_t(file.model.latent_width) * file.model.latent_height *
                    file.model.latent_channels);
    }
    else {
      EXPECT_TRUE(file.latents.empty());
    }
  }
}

TEST(SceneNeural, ProxyBlock)
{
  vector<uint8_t> b = model_bytes(first_model_with(false));
  set_u32(b, 12, 1);
  const float proxy[6] = {0.2f, 0.25f, 0.5f, 0.9f, 0.8f, 0.7f};
  for (int i = 0; i < 6; i++) {
    set_f32(b, 160 + 4 * i, proxy[i]);
  }
  NeuralFile file;
  ASSERT_TRUE(file.load(b)) << file.error;
  EXPECT_TRUE(file.has_proxy);
  EXPECT_FLOAT_EQ(file.proxy.alpha_x, 0.2f);
  EXPECT_FLOAT_EQ(file.proxy.alpha_y, 0.25f);
  EXPECT_FLOAT_EQ(file.proxy.p_diffuse, 0.5f);
  EXPECT_FLOAT_EQ(file.proxy.albedo.x, 0.9f);
  EXPECT_FLOAT_EQ(file.proxy.albedo.y, 0.8f);
  EXPECT_FLOAT_EQ(file.proxy.albedo.z, 0.7f);

  /* Each range check of nnm_format.md section 3a. */
  const struct {
    int index;
    float value;
  } bad[] = {{0, 0.0f}, {0, 1.5f}, {1, 0.0f}, {2, 0.05f}, {2, 1.5f}, {3, -0.1f}, {5, 1.1f},
             {4, std::numeric_limits<float>::quiet_NaN()}};
  for (const auto &c : bad) {
    vector<uint8_t> m = b;
    set_f32(m, 160 + 4 * c.index, c.value);
    expect_rejected(m, "proxy block");
  }

  /* Flag set but the block left zero: alpha 0 is out of range. */
  vector<uint8_t> zero = model_bytes(first_model_with(false));
  set_u32(zero, 12, 1);
  expect_rejected(zero, "proxy block value out of range");
}

TEST(SceneNeural, CorruptedFilesAreRejectedWithAReason)
{
  const vector<uint8_t> good = model_bytes(first_model_with(false));
  const vector<uint8_t> good_latent = model_bytes(first_model_with(true));

  vector<uint8_t> b = good;
  b[0] = 'X';
  expect_rejected(b, "bad magic");

  b = good;
  set_u32(b, 4, 2);
  expect_rejected(b, "version");

  b = good;
  set_u32(b, 12, 2);
  expect_rejected(b, "unknown flag");

  b = good;
  b.pop_back();
  expect_rejected(b, "file size");

  b = good;
  b.resize(100);
  expect_rejected(b, "smaller than");

  b = good;
  set_u32(b, 32, 7);
  expect_rejected(b, "out of range");

  b = good;
  set_u32(b, 28, 24); /* hidden_width not in {16, 32, 64}. */
  expect_rejected(b, "out of range");

  b = good;
  set_u32(b, 136, get_u32(b, 136) + 16);
  expect_rejected(b, "layer offset");

  b = good;
  set_u32(b, 132 + 4 * get_u32(b, 128), 16);
  expect_rejected(b, "unused layer offsets");

  b = good;
  b[200] = 1;
  expect_rejected(b, "reserved");

  /* Inf in the first weight. */
  b = good;
  b[256] = 0x00;
  b[257] = 0x7C;
  expect_rejected(b, "Inf or NaN");

  /* An expected test-vector output the decoder cannot reproduce. */
  b = good;
  const uint64_t tv_offset = get_u64(b, 104);
  ASSERT_GT(tv_offset, 0u);
  set_f32(b, tv_offset + 4 * get_u32(b, 120), 123456.0f);
  expect_rejected(b, "test vector");

  /* Latent description inconsistent with the blocks. */
  b = good_latent;
  set_u32(b, 56, get_u32(b, 56) + 1); /* latent_width: size no longer matches the block. */
  expect_rejected(b, "latent");

  b = good_latent;
  set_u32(b, 56, 0); /* latent_width 0 with latent channels present. */
  expect_rejected(b, "latent size");

  b = good_latent;
  set_u32(b, 64, 99); /* more levels than the image can have. */
  expect_rejected(b, "latent_levels");

  /* bsdf_dirs needs exactly 12 encoded inputs: adding a latent channel to a model without one breaks it. */
  b = good;
  if (get_u32(b, 40) == NEURAL_ENCODING_BSDF_DIRS) {
    set_u32(b, 44, 1);
    expect_rejected(b, "encoded inputs");
  }
}

TEST(SceneNeural, ManagerSharesIdenticalFilesAndFreesSlots)
{
  const std::filesystem::path dir = std::filesystem::temp_directory_path();
  const vector<uint8_t> a = model_bytes(first_model_with(false));
  vector<uint8_t> a2 = a;
  a2[256 + 2] ^= 0x01; /* A different but still valid weight. */
  vector<uint8_t> broken = a;
  broken[0] = 'X';

  const auto write = [&](const char *name, const vector<uint8_t> &bytes) {
    const std::string path = (dir / name).string();
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(bytes.data()),
                                                std::streamsize(bytes.size()));
    return path;
  };
  const std::string path_a = write("cycles_neural_test_a.nnm", a);
  const std::string path_a_copy = write("cycles_neural_test_a_copy.nnm", a);
  const std::string path_a2 = write("cycles_neural_test_a2.nnm", a2);
  const std::string path_broken = write("cycles_neural_test_broken.nnm", broken);

  NeuralManager manager;
  const int s0 = manager.add_file(path_a);
  ASSERT_GE(s0, 0);
  EXPECT_EQ(manager.add_file(path_a_copy), s0); /* Same bytes share a slot. */
  const int s1 = manager.add_file(path_a2);
  ASSERT_GE(s1, 0);
  EXPECT_NE(s1, s0);
  ASSERT_NE(manager.get(s0), nullptr);
  EXPECT_FALSE(manager.get(s0)->empty());

  string error;
  EXPECT_EQ(manager.add_file(path_broken, &error), -1);
  EXPECT_NE(error.find("bad magic"), string::npos) << error;
  EXPECT_EQ(manager.add_file((dir / "cycles_neural_test_missing.nnm").string(), &error), -1);
  EXPECT_NE(error.find("cannot read"), string::npos) << error;
  EXPECT_EQ(manager.get(-1), nullptr);
  EXPECT_EQ(manager.get(99), nullptr);

  manager.remove(s0); /* Two users: still alive. */
  EXPECT_NE(manager.get(s0), nullptr);
  manager.remove(s0);
  EXPECT_EQ(manager.get(s0), nullptr);
  EXPECT_NE(manager.get(s1), nullptr);
  EXPECT_EQ(manager.add_file(path_a), s0); /* The freed slot is reused. */

  for (const std::string &p : {path_a, path_a_copy, path_a2, path_broken}) {
    std::remove(p.c_str());
  }
}

CCL_NAMESPACE_END
