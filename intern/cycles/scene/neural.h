/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Neural models (`.nnm` files, doc/neural_rendering/nnm_format.md) for the neural BSDF.
 *
 * NeuralFile parses and validates one file (nnm_format.md section 8) and checks it against its
 * embedded test vectors with the kernel decoder `kernel/util/mlp.h`, so a file that loads here is
 * decoded identically by the CPU kernel. NeuralManager owns the loaded files, hands out slots to
 * shader nodes (same pattern as the IES slots of LightManager) and packs everything into the
 * `neural_models`, `neural_weights` and `neural_latents` device arrays. */

#include "kernel/util/mlp.h"

#include "util/half.h"
#include "util/string.h"
#include "util/thread.h"
#include "util/types.h"
#include "util/unique_ptr.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

class Device;
class DeviceScene;
class Progress;
class Scene;

/* Per-material constants of the proxy block (nnm_format.md section 3a). */
struct NeuralProxy {
  float alpha_x = 0.2f;
  float alpha_y = 0.2f;
  float p_diffuse = 0.5f;
  float3 albedo = make_float3(0.5f, 0.5f, 0.5f);
};

class NeuralFile {
 public:
  /* Parse `data` (the whole file). On failure returns false and `error` says why; the object is
   * left empty. */
  bool load(const vector<uint8_t> &data);
  bool load_from_file(const string &filename);
  void clear();

  bool empty() const
  {
    return weights.empty();
  }

  string error;

  /* Offsets are zero here; the manager fills them when packing. Only level 0 of the latent
   * pyramid is kept (version 1 decoders sample level 0 only), so `latent_levels` is 1. */
  KernelNeuralModel model = {};
  bool has_proxy = false;
  NeuralProxy proxy;

  vector<half> weights;
  vector<half> latents;
};

class NeuralManager {
 public:
  NeuralManager();
  ~NeuralManager();

  /* Load a model and return its slot, or -1 if the file cannot be read or is invalid (the reason
   * is logged, and returned in `error` when given). Identical files share a slot. */
  int add_file(const string &filename, string *error = nullptr);
  void remove(const int slot);

  /* The file of a slot, or nullptr for an invalid or freed slot. */
  const NeuralFile *get(const int slot) const;

  void device_update(Device *device, DeviceScene *dscene, Scene *scene, Progress &progress);
  void device_free(Device *device, DeviceScene *dscene);

  void tag_update(Scene *scene);
  bool need_update() const;

 private:
  struct Slot {
    NeuralFile file;
    string hash;
    int users = 0;
  };

  vector<unique_ptr<Slot>> slots_;
  mutable thread_mutex mutex_;
  bool need_update_ = true;
};

CCL_NAMESPACE_END
