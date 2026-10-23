/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Standalone parity check: runs the Cycles kernel MLP decoder (intern/cycles/kernel/util/mlp.h)
 * on .nnm files written by the Python package and compares with the embedded test vectors, which
 * were produced by the numpy reference decoder. No gtest, no Cycles library: only the header-only
 * util/ headers of Cycles are used, see build_standalone_test.sh for the required defines.
 *
 *   standalone_test model1.nnm [model2.nnm ...]
 *
 * Prints PASS/FAIL per model; the exit code is non-zero if any model fails. */

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "kernel/util/mlp.h"

#include "test/kernel_neural_mlp_test_util.h"

using namespace ccl;

static bool read_file(const char *path, std::vector<uint8_t> &data)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s model.nnm [...]\n", argv[0]);
    return 2;
  }

#if defined(__FMA__)
  const char *fma_mode = "hardware fmaf";
#else
  const char *fma_mode = "a*b+c (no FMA)";
#endif
  std::printf("kernel mlp.h standalone parity test, multiply-add: %s\n", fma_mode);

  const double rtol = 1e-4, atol = 1e-5;
  int failures = 0, count = 0;
  double worst_abs = 0.0, worst_rel = 0.0;

  for (int a = 1; a < argc; a++) {
    const std::string name = argv[a];
    std::vector<uint8_t> bytes;
    NnmTestModel t;
    std::string err;
    double max_abs = 0.0, max_rel = 0.0;

    bool ok = read_file(argv[a], bytes);
    if (!ok) {
      err = "cannot read file";
    }
    ok = ok && nnm_parse(bytes.data(), bytes.size(), t, err);
    ok = ok && nnm_check_test_vectors(t, rtol, atol, max_abs, max_rel, err);

    /* The harness itself must be able to fail: corrupt one weight and expect a mismatch. */
    bool harness_ok = true;
    if (ok && t.tv_count > 0) {
      NnmTestModel broken = t;
      const size_t idx = nnm_detail::k_weight_prefix;
      broken.weights[idx] = float_to_half(half_to_float(broken.weights[idx]) + 0.5f);
      double dummy_abs, dummy_rel;
      std::string dummy_err;
      harness_ok = !nnm_check_test_vectors(broken, rtol, atol, dummy_abs, dummy_rel, dummy_err);
      if (!harness_ok) {
        err = "harness did not notice a corrupted weight";
      }
    }

    count++;
    if (ok && harness_ok) {
      std::printf("PASS %s  (width %d, hidden %d, vectors %d, max abs err %.2e, max rel err %.2e)\n",
                  name.c_str(), t.km.width, t.km.num_hidden, t.tv_count, max_abs, max_rel);
      worst_abs = std::max(worst_abs, max_abs);
      worst_rel = std::max(worst_rel, max_rel);
    }
    else {
      std::printf("FAIL %s: %s\n", name.c_str(), err.c_str());
      failures++;
    }
  }

  std::printf("%s: %d/%d models passed (worst abs err %.2e, worst rel err %.2e)\n",
              failures ? "FAIL" : "PASS", count - failures, count, worst_abs, worst_rel);
  return failures ? 1 : 0;
}
