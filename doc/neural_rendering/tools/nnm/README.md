# nnm: tools for the `.nnm` neural model format

Format spec: [`../../nnm_format.md`](../../nnm_format.md). numpy only, no torch.

| File | Purpose |
|---|---|
| `format.py` | `NnmModel` dataclass, `write_nnm`, `read_nnm` (full validation), layout helpers |
| `reference.py` | numpy reference decoder: `bsdf_dirs` encoding, bilinear latent lookup, fp16 weights promoted to fp32, fp32 axpy accumulation with emulated FMA, activations, output transforms. Also a float64 matmul decoder used as an independent check |
| `export_numpy.py` | `build_model(layers, width, ...)` quantises float weights to fp16 and generates the test-vector block; CLI for `.npz` weights |
| `selftest.py` | builds 27 random models (3 widths x 3 activations x 3 output transforms, cycling six structural variants), round-trips them, checks the reference decoder reproduces the embedded vectors, negative tests, encoding and activation math |
| `gen_test_header.py` | regenerates `intern/cycles/test/kernel_neural_mlp_test_data.h` (models embedded for the gtest) |
| `standalone_test.cpp`, `build_standalone_test.sh` | compile `intern/cycles/kernel/util/mlp.h` with g++ and clang++ and run it on the selftest models |

Run from `doc/neural_rendering/tools`:

```
python3 -m nnm.selftest                       # Python only
python3 -m nnm.selftest --out /tmp/models     # keep the .nnm files
python3 -m nnm.gen_test_header                # refresh the gtest data header
nnm/build_standalone_test.sh                  # selftest + C++ parity, g++ and clang++
```

Exporting your own weights:

```python
from nnm.export_numpy import build_model
from nnm.format import write_nnm, ENC_BSDF_DIRS, ACT_LEAKY_SMELU, OUT_EXP
# layers: [(W0[in, out], b0), (W1, b1), (W2, b2)] float32, input-major; W0 has 12 + C inputs
m = build_model(layers, width=32, activation=ACT_LEAKY_SMELU, beta=1.0, eps=0.05,
                output_transform=OUT_EXP, encoding=ENC_BSDF_DIRS,
                latent=latent_hwc, latent_levels=1)   # latent (H, W, C) or None
write_nnm(m, "decoder.nnm")
```

Training code stores `W[in][out]` already (`x @ W + b`), so no transpose is needed.

## Build notes for the standalone test

- Cycles `util/` headers need C++20 (`std::popcount`), so the script uses `-std=c++20`.
- The only compat defines needed are `CCL_NAMESPACE_BEGIN=namespace ccl {` and
  `CCL_NAMESPACE_END=}`; no stub headers. The harness shares its parser with the gtest
  (`intern/cycles/test/kernel_neural_mlp_test_util.h`).
- Each compiler is run in an AVX2+FMA+F16C configuration (hardware `fmaf`, as the Cycles AVX2
  kernel) and a generic x86-64 configuration (`a*b+c`, the SSE2 kernel path).

## Parity expectations

With hardware FMA the kernel and the reference follow the same operation order, so the layers
agree except for `expf`/`logf` differences between libm and numpy (a few ulp) and
compiler contraction in the input encoding. Observed worst error on the 27 models:
about 2e-6 absolute. The loaders compare with `|got - expected| <= 1e-4 |expected| + 1e-5`.
