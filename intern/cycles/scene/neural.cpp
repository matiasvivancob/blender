/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/neural.h"

#include "device/device.h"
#include "scene/scene.h"

#include "util/algorithm.h"
#include "util/log.h"
#include "util/md5.h"
#include "util/path.h"
#include "util/progress.h"

#include <cstring>

CCL_NAMESPACE_BEGIN

namespace {

constexpr uint32_t NNM_HEADER_SIZE = 256;
constexpr uint32_t NNM_FLAG_PROXY = 1;
constexpr size_t NNM_PROXY_OFFSET = 160;
constexpr size_t NNM_RESERVED_OFFSET = 192;

/* Little-endian reads at a byte offset; every supported host is little-endian. */
struct Bytes {
  const uint8_t *data;

  uint32_t u32(const size_t offset) const
  {
    uint32_t v;
    std::memcpy(&v, data + offset, sizeof(v));
    return v;
  }
  uint64_t u64(const size_t offset) const
  {
    uint64_t v;
    std::memcpy(&v, data + offset, sizeof(v));
    return v;
  }
  float f32(const size_t offset) const
  {
    float v;
    std::memcpy(&v, data + offset, sizeof(v));
    return v;
  }
};

struct Range {
  uint64_t begin, end;
  const char *name;
};

bool all_zero(const uint8_t *p, const size_t size)
{
  for (size_t i = 0; i < size; i++) {
    if (p[i] != 0) {
      return false;
    }
  }
  return true;
}

bool all_finite_half(const uint8_t *p, const size_t count)
{
  for (size_t i = 0; i < count; i++) {
    uint16_t h;
    std::memcpy(&h, p + 2 * i, sizeof(h));
    if ((h & 0x7C00u) == 0x7C00u) { /* Inf or NaN. */
      return false;
    }
  }
  return true;
}

}  // namespace

void NeuralFile::clear()
{
  error.clear();
  model = {};
  has_proxy = false;
  proxy = NeuralProxy();
  weights.clear();
  weights.shrink_to_fit();
  latents.clear();
  latents.shrink_to_fit();
}

bool NeuralFile::load_from_file(const string &filename)
{
  vector<uint8_t> data;
  if (filename.empty() || !path_read_binary(filename, data)) {
    clear();
    error = "cannot read file";
    return false;
  }
  return load(data);
}

bool NeuralFile::load(const vector<uint8_t> &data)
{
  clear();

  const auto fail = [this](const string &message) {
    clear();
    error = message;
    return false;
  };

  const size_t file_size = data.size();
  if (file_size < NNM_HEADER_SIZE) {
    return fail("file is smaller than the 256 byte header");
  }
  const uint8_t *file = data.data();
  const Bytes h{file};

  if (std::memcmp(file, "NNM1", 4) != 0) {
    return fail("bad magic, not a .nnm file");
  }
  if (h.u32(4) != 1) {
    return fail("unsupported .nnm version");
  }
  if (h.u32(8) != NNM_HEADER_SIZE) {
    return fail("unexpected header size");
  }
  const uint32_t flags = h.u32(12);
  if (flags & ~NNM_FLAG_PROXY) {
    return fail("unknown flag bits");
  }
  if (h.u64(152) != file_size) {
    return fail("file size does not match the header");
  }

  const uint32_t input_dim = h.u32(16), output_dim = h.u32(20), num_hidden = h.u32(24);
  const uint32_t width = h.u32(28), activation = h.u32(32), transform = h.u32(36);
  const uint32_t encoding = h.u32(40), latent_channels = h.u32(44);
  const float beta = h.f32(48), eps = h.f32(52);
  const uint32_t latent_width = h.u32(56), latent_height = h.u32(60), latent_levels = h.u32(64);
  const uint32_t testvec_count = h.u32(68);
  const uint64_t weight_offset = h.u64(72), weight_size = h.u64(80);
  const uint64_t latent_offset = h.u64(88), latent_size = h.u64(96);
  const uint64_t testvec_offset = h.u64(104), testvec_size = h.u64(112);
  const uint32_t testvec_in_dim = h.u32(120), testvec_out_dim = h.u32(124);
  const uint32_t num_layers = h.u32(128);
  const uint32_t frames_k = h.u32(184), frames_offset = h.u32(188);

  if (input_dim < 1 || input_dim > NEURAL_MAX_INPUT || output_dim < 1 ||
      output_dim > NEURAL_MAX_OUTPUT || num_hidden < 1 || num_hidden > NEURAL_MAX_HIDDEN ||
      !(width == 16 || width == 32 || width == 64) || activation > 2 || transform > 2 ||
      encoding > 2 || latent_channels > NEURAL_MAX_LATENT_CHANNELS || testvec_count > 64)
  {
    return fail("a dimension or enumeration is out of range");
  }
  if (!std::isfinite(beta) || !std::isfinite(eps) || (activation == 2 && !(beta > 0.0f)) ||
      (activation >= 1 && !(eps >= 0.0f && eps < 1.0f)))
  {
    return fail("invalid activation parameters");
  }
  if (num_layers != num_hidden + 1) {
    return fail("num_layers is not num_hidden + 1");
  }
  if (input_dim <= latent_channels) {
    return fail("input_dim must exceed latent_channels");
  }
  const uint32_t encoded_dim = input_dim - latent_channels;
  if (encoding == NEURAL_ENCODING_BSDF_DIRS && encoded_dim != 12) {
    return fail("bsdf_dirs encoding needs 12 encoded inputs");
  }
  /* Learned shading frames (nnm_format.md section 7a, 05_learned_shading_frames.md). */
  if (encoding == NEURAL_ENCODING_LEARNED_FRAMES) {
    if (latent_channels < 1) {
      return fail("learned_frames encoding needs a latent");
    }
    if (frames_k < 1 || frames_k > NEURAL_MAX_FRAMES) {
      return fail("frames_k out of range");
    }
    if (encoded_dim != 6 * frames_k) {
      return fail("learned_frames needs 6 * frames_k encoded inputs");
    }
  }
  else if (frames_k != 0 || frames_offset != 0) {
    return fail("frames fields must be zero without the learned_frames encoding");
  }

  /* Latent description: all fields zero iff there are no latents. */
  uint64_t expected_latent_size = 0;
  if (latent_channels == 0) {
    if (latent_width || latent_height || latent_levels || latent_offset || latent_size) {
      return fail("latent fields must be zero without latent channels");
    }
  }
  else {
    if (latent_width < 1 || latent_width > 16384 || latent_height < 1 || latent_height > 16384) {
      return fail("latent size out of range");
    }
    uint32_t max_levels = 1;
    for (uint32_t s = max(latent_width, latent_height); s > 1; s >>= 1) {
      max_levels++;
    }
    if (latent_levels < 1 || latent_levels > max_levels) {
      return fail("latent_levels out of range");
    }
    for (uint32_t k = 0; k < latent_levels; k++) {
      const uint64_t w = max(latent_width >> k, 1u), hh = max(latent_height >> k, 1u);
      expected_latent_size += w * hh * latent_channels * 2;
    }
    if (latent_size != expected_latent_size || latent_offset % 64 != 0) {
      return fail("latent block size or alignment is wrong");
    }
    /* The device packs offsets as 32-bit counts of halves. */
    if (latent_size / 2 > UINT32_MAX / 2) {
      return fail("latent block is too large (more than 2^31 values)");
    }
  }

  /* Weight blob layout (nnm_format.md section 4). */
  uint64_t layer_offset = 0;
  for (uint32_t l = 0; l < num_layers; l++) {
    const uint32_t in = (l == 0) ? input_dim : width;
    const uint32_t out = (l == num_hidden) ? output_dim : width;
    if (h.u32(132 + 4 * l) != layer_offset) {
      return fail("layer offset table does not match the topology");
    }
    layer_offset += uint64_t(in + 1) * uint64_t(neural_pad8(int(out))) * 2;
  }
  for (uint32_t l = num_layers; l < 5; l++) {
    if (h.u32(132 + 4 * l) != 0) {
      return fail("unused layer offsets must be zero");
    }
  }
  /* The frame layer follows the last MLP layer inside the weight blob. */
  if (frames_k > 0) {
    if (uint64_t(frames_offset) != layer_offset) {
      return fail("frames_offset does not point right after the last MLP layer");
    }
    layer_offset += uint64_t(latent_channels + 1) * uint64_t(neural_pad8(int(6 * frames_k))) * 2;
  }
  if (weight_offset != NNM_HEADER_SIZE || weight_size != layer_offset) {
    return fail("weight block size or position is wrong");
  }

  /* Test vectors (section 6). */
  const uint32_t caller_dim = (encoding == NEURAL_ENCODING_BSDF_DIRS ||
                               encoding == NEURAL_ENCODING_LEARNED_FRAMES) ?
                                  6 :
                                  encoded_dim;
  const uint32_t expected_tv_in = caller_dim + (latent_channels > 0 ? 2 : 0);
  if (testvec_in_dim != expected_tv_in || testvec_out_dim != output_dim ||
      testvec_size != uint64_t(testvec_count) * (expected_tv_in + output_dim) * 4 ||
      (testvec_count == 0 && testvec_offset != 0) || testvec_offset % 64 != 0)
  {
    return fail("test vector block description is wrong");
  }

  /* All blocks inside the file, not overlapping each other or the header. */
  vector<Range> ranges;
  ranges.push_back({0, NNM_HEADER_SIZE, "header"});
  ranges.push_back({weight_offset, weight_offset + weight_size, "weights"});
  if (latent_channels > 0) {
    ranges.push_back({latent_offset, latent_offset + latent_size, "latents"});
  }
  if (testvec_count > 0) {
    ranges.push_back({testvec_offset, testvec_offset + testvec_size, "test vectors"});
  }
  sort(ranges.begin(), ranges.end(), [](const Range &a, const Range &b) {
    return a.begin < b.begin;
  });
  for (size_t i = 0; i < ranges.size(); i++) {
    /* Compare against the remaining size: begin + size may wrap for a corrupt 64-bit offset. */
    if (ranges[i].begin > file_size || ranges[i].end - ranges[i].begin > file_size - ranges[i].begin ||
        ranges[i].end < ranges[i].begin)
    {
      return fail(string("the ") + ranges[i].name + " block extends past the end of the file");
    }
    if (i > 0 && ranges[i].begin < ranges[i - 1].end) {
      return fail(string("the ") + ranges[i].name + " block overlaps another block");
    }
  }

  /* Proxy block and reserved bytes. */
  if (flags & NNM_FLAG_PROXY) {
    const float ax = h.f32(NNM_PROXY_OFFSET), ay = h.f32(NNM_PROXY_OFFSET + 4);
    const float pd = h.f32(NNM_PROXY_OFFSET + 8);
    const float ar = h.f32(NNM_PROXY_OFFSET + 12), ag = h.f32(NNM_PROXY_OFFSET + 16);
    const float ab = h.f32(NNM_PROXY_OFFSET + 20);
    const float all[6] = {ax, ay, pd, ar, ag, ab};
    for (const float v : all) {
      if (!std::isfinite(v)) {
        return fail("non-finite value in the proxy block");
      }
    }
    if (ax < 1e-3f || ax > 1.0f || ay < 1e-3f || ay > 1.0f || pd < 0.1f || pd > 1.0f ||
        min(ar, min(ag, ab)) < 0.0f || max(ar, max(ag, ab)) > 1.0f)
    {
      return fail("proxy block value out of range");
    }
    has_proxy = true;
    proxy.alpha_x = ax;
    proxy.alpha_y = ay;
    proxy.p_diffuse = pd;
    proxy.albedo = make_float3(ar, ag, ab);
  }
  else if (!all_zero(file + NNM_PROXY_OFFSET, 24)) {
    return fail("proxy bytes are set but the proxy flag is not");
  }
  /* Bytes 184..192 are frames_k and frames_offset, checked above; the rest is reserved. */
  if (!all_zero(file + NNM_RESERVED_OFFSET, NNM_HEADER_SIZE - NNM_RESERVED_OFFSET)) {
    return fail("reserved header bytes are not zero");
  }

  /* Weights and latents must be finite. */
  if (!all_finite_half(file + weight_offset, weight_size / 2) ||
      (latent_channels > 0 && !all_finite_half(file + latent_offset, latent_size / 2)))
  {
    return fail("weights or latents contain Inf or NaN");
  }

  /* Keep the weights and level 0 of the latents. */
  weights.resize(weight_size / 2);
  std::memcpy(weights.data(), file + weight_offset, weight_size);
  if (latent_channels > 0) {
    const size_t level0 = size_t(latent_width) * latent_height * latent_channels;
    latents.resize(level0);
    std::memcpy(latents.data(), file + latent_offset, level0 * 2);
  }

  model = {};
  model.input_dim = int(input_dim);
  model.output_dim = int(output_dim);
  model.num_hidden = int(num_hidden);
  model.width = int(width);
  model.activation = int(activation);
  model.act_beta = beta;
  model.act_eps = eps;
  model.output_transform = int(transform);
  model.encoding = int(encoding);
  model.latent_channels = int(latent_channels);
  model.latent_width = int(latent_width);
  model.latent_height = int(latent_height);
  model.latent_levels = latent_channels > 0 ? 1 : 0;
  model.frames_k = int(frames_k);

  /* Self-check: the kernel decoder must reproduce every embedded test vector
   * (|got - expected| <= 1e-4 |expected| + 1e-5, nnm_format.md section 6). */
  const float *tv = reinterpret_cast<const float *>(file + testvec_offset);
  for (uint32_t i = 0; i < testvec_count; i++) {
    float record[NEURAL_MAX_INPUT + 2 + NEURAL_MAX_OUTPUT];
    std::memcpy(record, tv + size_t(i) * (expected_tv_in + output_dim),
                (expected_tv_in + output_dim) * sizeof(float));
    float out[NEURAL_MAX_OUTPUT];
    const float u = latent_channels > 0 ? record[caller_dim] : 0.0f;
    const float v = latent_channels > 0 ? record[caller_dim + 1] : 0.0f;
    neural_model_eval(weights.data(), latents.data(), model, record, u, v, out);
    for (uint32_t o = 0; o < output_dim; o++) {
      const float expected = record[expected_tv_in + o];
      if (!(fabsf(out[o] - expected) <= 1e-4f * fabsf(expected) + 1e-5f)) {
        return fail("kernel decoder disagrees with embedded test vector " + to_string(i));
      }
    }
  }

  return true;
}

/* Neural Manager */

NeuralManager::NeuralManager() = default;

NeuralManager::~NeuralManager() = default;

int NeuralManager::add_file(const string &filename, string *error)
{
  const auto report = [&](const string &message) {
    LOG_ERROR << "Neural model \"" << filename << "\": " << message;
    if (error) {
      *error = message;
    }
    return -1;
  };

  vector<uint8_t> data;
  if (filename.empty() || !path_read_binary(filename, data)) {
    return report("cannot read file");
  }

  /* MD5Hash takes an int length. */
  MD5Hash md5;
  for (size_t pos = 0; pos < data.size(); pos += size_t(1) << 30) {
    md5.append(data.data() + pos, int(std::min(data.size() - pos, size_t(1) << 30)));
  }
  const string hash = md5.get_hex();

  const thread_scoped_lock lock(mutex_);

  for (size_t slot = 0; slot < slots_.size(); slot++) {
    if (slots_[slot]->users > 0 && slots_[slot]->hash == hash) {
      slots_[slot]->users++;
      return int(slot);
    }
  }

  NeuralFile file;
  if (!file.load(data)) {
    return report(file.error);
  }

  size_t slot = 0;
  while (slot < slots_.size() && slots_[slot]->users > 0) {
    slot++;
  }
  if (slot == slots_.size()) {
    slots_.push_back(make_unique<Slot>());
  }
  slots_[slot]->file = std::move(file);
  slots_[slot]->hash = hash;
  slots_[slot]->users = 1;
  need_update_ = true;
  return int(slot);
}

void NeuralManager::remove(const int slot)
{
  const thread_scoped_lock lock(mutex_);
  if (slot < 0 || slot >= int(slots_.size()) || slots_[slot]->users == 0) {
    return;
  }
  if (--slots_[slot]->users == 0) {
    slots_[slot]->file.clear();
    slots_[slot]->hash.clear();
    need_update_ = true;
  }
}

const NeuralFile *NeuralManager::get(const int slot) const
{
  const thread_scoped_lock lock(mutex_);
  if (slot < 0 || slot >= int(slots_.size()) || slots_[slot]->users == 0 ||
      slots_[slot]->file.empty())
  {
    return nullptr;
  }
  return &slots_[slot]->file;
}

void NeuralManager::device_update(Device *device,
                                  DeviceScene *dscene,
                                  Scene * /*scene*/,
                                  Progress &progress)
{
  if (!need_update()) {
    return;
  }

  device_free(device, dscene);

  const thread_scoped_lock lock(mutex_);

  /* Shrink the table past trailing empty slots (earlier slot numbers stay valid). */
  while (!slots_.empty() && slots_.back()->users == 0) {
    slots_.pop_back();
  }

  size_t total_weights = 0, total_latents = 0;
  for (const unique_ptr<Slot> &slot : slots_) {
    total_weights += slot->file.weights.size();
    total_latents += slot->file.latents.size();
  }
  if (total_weights > UINT32_MAX || total_latents > UINT32_MAX) {
    LOG_ERROR << "Neural models are too large for the device (" << total_weights << " weights, "
              << total_latents << " latent values; limit 2^32); they will render black";
    /* Zeroed records are empty models, which the closure never samples. */
    uint *models = dscene->neural_models.alloc(slots_.size() * sizeof(KernelNeuralModel) /
                                               sizeof(uint));
    std::fill(models, models + slots_.size() * sizeof(KernelNeuralModel) / sizeof(uint), 0u);
    dscene->neural_models.copy_to_device();
    need_update_ = false;
    return;
  }
  LOG_INFO << "Total " << slots_.size() << " neural model slots, " << total_weights
           << " fp16 weights, " << total_latents << " fp16 latents.";

  if (slots_.empty()) {
    need_update_ = false;
    return;
  }

  progress.set_status("Updating Neural Models", "Copying to device");

  constexpr size_t model_uints = sizeof(KernelNeuralModel) / sizeof(uint);
  static_assert(sizeof(KernelNeuralModel) % sizeof(uint) == 0);

  uint *models = dscene->neural_models.alloc(slots_.size() * model_uints);
  half *weights = (total_weights > 0) ? dscene->neural_weights.alloc(total_weights) : nullptr;
  half *latents = (total_latents > 0) ? dscene->neural_latents.alloc(total_latents) : nullptr;

  size_t weight_offset = 0, latent_offset = 0;
  for (size_t i = 0; i < slots_.size(); i++) {
    const NeuralFile &file = slots_[i]->file;
    KernelNeuralModel model = {};
    if (!file.empty()) {
      model = file.model;
      model.weight_offset = uint(weight_offset);
      std::memcpy(weights + weight_offset, file.weights.data(), file.weights.size() * sizeof(half));
      weight_offset += file.weights.size();
      if (!file.latents.empty()) {
        model.latent_offset = uint(latent_offset);
        std::memcpy(latents + latent_offset, file.latents.data(), file.latents.size() * sizeof(half));
        latent_offset += file.latents.size();
      }
    }
    std::memcpy(models + i * model_uints, &model, sizeof(model));
  }

  dscene->neural_models.copy_to_device();
  if (weights) {
    dscene->neural_weights.copy_to_device();
  }
  if (latents) {
    dscene->neural_latents.copy_to_device();
  }

  need_update_ = false;
}

void NeuralManager::device_free(Device * /*device*/, DeviceScene *dscene)
{
  dscene->neural_models.free();
  dscene->neural_weights.free();
  dscene->neural_latents.free();
}

void NeuralManager::tag_update(Scene * /*scene*/)
{
  need_update_ = true;
}

bool NeuralManager::need_update() const
{
  return need_update_;
}

CCL_NAMESPACE_END
