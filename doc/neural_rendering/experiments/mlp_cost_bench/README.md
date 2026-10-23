# Neural BSDF decoder vs. analytic BSDF: CPU cost in Cycles-style kernel C++

Question: if Cycles evaluated a small neural BSDF decoder in plain kernel-style C++ (no STL,
no allocation, fixed-size loops, float math, `ccl_device_inline` functions, AVX2), how
expensive is one evaluation compared with the analytic BSDFs it already has?

Nothing under `/home/user/blender` was modified. The analytic reference follows
`intern/cycles/kernel/closure/bsdf_microfacet.h` (`bsdf_D`, `bsdf_lambda`,
`bsdf_microfacet_eval`) and `bsdf_diffuse.h`; the macros (`ccl_device_inline` etc.) and
`fast_expf` mirror `intern/cycles/util/defines.h` and `util/math_fast.h`.

Machine: Intel Xeon (Cascade Lake, AVX2 + AVX-512, 4 vCPUs, 2.8 GHz base, ~3.2 GHz measured),
g++ 13.3, clang++ 18.1. All numbers are single-thread. The two Cycles-flag builds use 5M evals instead of 10M. Measured AVX2 FMA peak of one core:
89.6 GFLOP/s (8-wide, 12 independent chains); scalar FMA peak 11-13 GFLOP/s.

## Files

| file | purpose |
|---|---|
| `bench.cpp` | everything: kernel-style analytic BSDF + MLP decoder (part 1), harness (part 2) |
| `build.sh` | builds the 6 compiler/flag combinations into `bin/` |
| `run_all.sh` | runs the whole matrix sequentially (about 40 min), logs into `results/` |
| `make_table.py` | turns `results/*.log` into `results/tables.md` |
| `results/` | raw logs (`<build>.log`, `<build>_extra.log`), `verify_*.txt`, `tables.md` |
| `diag/` | minimal repro of the clang gather finding |

## Rebuild and run

```
./build.sh                      # about 1.5 minutes, 6 binaries in bin/
./bin/bench_gcc_fast --verify   # correctness checks (see below), a few seconds
QUICK=1 ./run_all.sh            # smoke test: 1M evals, 1 rep, about 3 minutes
./run_all.sh                    # full matrix: 10M evals (5M for the Cycles-flag builds), best of 3
python3 make_table.py           # regenerate results/tables.md
```

Single binary, ad hoc: `./bin/bench_gcc_fast --n 1e7 --reps 3 --only h32x2_in28,an_ggx_lambert`
(`--only a,b` substring filter, `--skip`, `--n64` separate N for the 64x3 net, `--no-in12`,
`--no-peak`). Do not run two benchmarks at the same time; they are single-thread and sensitive to
memory bandwidth. Compile by hand: `g++ -std=c++17 -O3 -march=native -ffast-math -DBUILD_NAME='"x"' bench.cpp -o x`.

Builds (`build.sh`):

| name | flags |
|---|---|
| `gcc_fast` | `g++ -O3 -march=native -ffast-math` |
| `gcc_strict` | `g++ -O3 -march=native` |
| `clang_fast` | `clang++ -O3 -march=native -ffast-math` |
| `gcc_fast_novec` | `g++ -O3 -march=native -ffast-math -fno-tree-vectorize` |
| `gcc_cycles` | `g++ -O3 -mavx2 -mfma -mlzcnt -mbmi -mbmi2 -mf16c` + Cycles' math flags (`-fno-trapping-math -fno-math-errno -fno-signed-zeros -ffp-contract=on -freciprocal-math -fno-signaling-nans -fno-rounding-math -mfpmath=sse`) |
| `clang_cycles` | same as above with clang |

The last two exist because Cycles' own CMake (`intern/cycles/CMakeLists.txt`) does not use
`-ffast-math` ("the BVH code relies on NaNs") and builds the AVX2 kernel with `-mavx2 -mfma`, not
`-march=native`. They are the closest thing to what a Cycles build would produce.

## What is measured

* 10M random (wi, wo) pairs (cosine-distributed upper hemisphere, cos >= 0.02), 10M random latents
  (uniform [-1,1]^16) and 10M random analytic parameters (roughness, F0 rgb, albedo rgb), all
  pre-generated (about 1.2 GB per layout, larger than L3, so inputs stream from DRAM).
  AoS arrays feed the single-eval variants, an SoA copy (blocks of 8) feeds the batched ones;
  both hold the same numbers.
* Results are accumulated (float partial sums flushed to double every 4096 evals) and passed through an empty
  `asm volatile` per eval, so nothing is dead-code-eliminated and the compiler cannot vectorize
  across evals in the single-eval variants (that is what one path vertex looks like).
* Best of 3 repetitions of the whole 10M loop; `worst` is also logged. Variants under 250 ns
  are also re-run on a 2048-eval working set that stays in L2 ("cache-resident"), because at
  5-10 ns/eval the DRAM stream (about 11 ns/eval for the bytes of one MLP eval on this VM,
  `stream_baseline`) starts to matter. The analytic and `part_*` variants got a second pass
  (7 reps, 5M) to filter VM noise; `make_table.py` keeps the minimum of both passes.
* The 64x3 net uses 3M evals (not 10M) in the non-primary builds to keep the run time sane.

### Analytic reference (`analytic_eval`)

Isotropic GGX reflection (D, Smith lambda G, Schlick F with f90 = 1, alpha = roughness^2) plus
Lambert, layered: the diffuse lobe is weighted by (1 - F_avg(cos_NI)) (1 - F_avg(cos_NO)). Returns
BSDF * cos(wo) like Cycles' `eval`. About 110 flops, including 3 sqrt and 6 divisions. Variants:
value only (headline), value + pdf (what Cycles' eval returns, adds a lobe-weighted pdf, +24%),
and the same source on 8 lanes.

### Neural decoder (`NeuralMlp<IN, HID, OUT, LAYERS>`)

IN = 12 (wi, wo, h, shortest-arc difference vector wd' = wi*hz + (v x wi) + (v.wi) v / (1+hz),
v = (hy, -hx, 0)) + 16 latent = 28; IN = 12 is the no-latent version (the task text says "8-float
version", but 12 floats and IN=12 are what it specifies, so that is what is built). `LAYERS` is the
number of hidden layers; hidden layers use LeakySmeLU (beta = 0.1, eps = 0.01, written with |.| so it
is branch-free; checked against the piecewise definition and for C1 continuity), then a linear
HID -> 3 layer and `fast_expf` (Cycles' own polynomial exp). Weights are random He-initialised
floats in one flat `const float *` array, bias after each weight matrix; timing does not depend on
their values (no denormals).
Weights: 2x16 = 787 floats (3.1 KB), 2x32 = 2083 (8.1 KB), 3x64 = 10371 (40.5 KB, does not fit L1).

Evaluation strategies (names in the tables):

| tag | what |
|---|---|
| `rm` | one eval, row-major `W[out][in]` exactly as specified, plain loops |
| `im` | one eval, input-major `W[in][out]`, plain loops: the compiler vectorizes over outputs and needs no reassociation, so it also works without `-ffast-math` |
| `imv` | one eval, input-major padded to 8 outputs, hand-written `float8` (AVX2 intrinsics wrapped like Cycles' `float8`), accumulators split so >= 8 FMA chains are in flight: the best a careful kernel author can do for one path vertex |
| `b8n` | 8 evals batched in SoA, plain loops, compiler vectorizes across evals, one output neuron at a time |
| `b8t` | like `b8n` but 8 output neurons accumulated together (register tile) |
| `b8v` | 8 evals batched, explicit `float8`, 8-output register tile: a hand-vectorized "wavefront neural stage" |

Checks (`--verify`, logs in `results/verify_*.txt`, all builds pass): activation vs piecewise definition,
`wd'.z == wi.h` and |wd'| = 1, GGX/Lambert directional albedo by Monte Carlo (GGX 0.998/0.886/0.377 for
roughness 0.25/0.5/1.0; Lambert 0.95 = 1 - coat loss), and every MLP variant vs a double-precision
reference (max relative error 1.4e-5, i.e. `fast_expf`'s error).

## FLOP model

Per eval, `matmul FLOPs = 2 x MACs` (bias adds ignored). "All est." adds 11 flops per hidden neuron
for LeakySmeLU, 40 for the encoding and 3 x 15 for the exps.

| config | IN | MACs | matmul FLOPs | all-in est. |
|---|---:|---:|---:|---:|
| 2x16 | 28 | 752 | 1504 | 1941 |
| 2x32 | 28 | 2016 | 4032 | 4821 |
| 3x64 | 28 | 10176 | 20352 | 22549 |
| 2x16 | 12 | 496 | 992 | 1429 |
| 2x32 | 12 | 1504 | 3008 | 3797 |
| 3x64 | 12 | 9152 | 18304 | 20501 |
| analytic GGX+Lambert | - | - | ~110 (3 sqrt, 6 div) | |

GFLOP/s = matmul FLOPs / ns per eval.

## Results (g++ -O3 -march=native -ffast-math, IN=28)

Ratios: single-eval rows are vs the single-eval analytic (19.4 ns); batched rows are vs the
8-batched explicit-float8 analytic (5.9 ns streaming, 3.7 ns cache-resident; the second ratio uses that).

| config | variant | ns/eval (best of 3) | ratio vs analytic | matmul FLOPs | GFLOP/s |
|---|---|---:|---:|---:|---:|
| analytic GGX+Lambert | single, value | 19.4 | 1x | ~110 | ~5.7 |
| analytic GGX+Lambert | single, value + pdf | 24.0 | 1.2x | | |
| analytic GGX+Lambert | 8-batched, float8 | 5.9 (3.7 cache) | | | |
| 2x16 | rm (as specified) | 281 | 14.5x | 1504 | 5.3 |
| 2x16 | best single (imv) | 94 | 4.9x | 1504 | 16.0 |
| 2x16 | batched b8v | 40 | 6.8x / 10.3x | 1504 | 37.3 |
| 2x32 | rm (as specified) | 584 | 30x | 4032 | 6.9 |
| 2x32 | best single (imv; im 167) | 158 | 8.2x | 4032 | 25.4 |
| 2x32 | batched b8v | 91.5 | 15.4x / 23.2x | 4032 | 44.1 |
| 3x64 | rm (as specified) | 1846 | 95x | 20352 | 11.0 |
| 3x64 | best single (im; imv 712) | 662 | 34x | 20352 | 30.7 |
| 3x64 | batched b8v | 392 | 66x / 106x | 20352 | 51.9 |

Without the latent (IN = 12): 2x16 imv 84 ns (4.4x), b8v 27 ns; 2x32 imv 111 ns (5.7x), b8v 76 ns;
3x64 im 622 ns, b8v 367 ns. Complete tables for every build, with worst-of-3, cache-resident
numbers, % of peak and all-in GFLOP/s: `results/tables.md`.

Same 2x32 (IN=28) decoder across builds, ns/eval (ratio vs that build's own single-eval analytic):

| variant | gcc_fast | gcc_strict | clang_fast | gcc_cycles | clang_cycles |
|---|---:|---:|---:|---:|---:|
| analytic single | 19.4 | 27.7 | 20.6 | 28.9 | 25.7 |
| rm (as specified) | 584 (30x) | 1682 (61x) | 4349 (211x) | 1708 (59x) | 695 (27x) |
| im | 167 (9x) | 208 (8x) | 4699 (228x) | 233 (8x) | 206 (8x) |
| imv | 158 (8x) | 159 (6x) | 155 (8x) | 159 (5x) | 146 (6x) |
| b8n | 227 (38x) | 1357 (237x) | 255 (46x) | 1761 (283x) | 264 (42x) |
| b8v | 91 (15x) | 86 (15x) | 93 (17x) | 106 (17x) | 108 (17x) |

`-fno-tree-vectorize` (HID=32, IN=28, gcc_fast_novec): rm 1947 ns, im 997, b8n 578, b8t 790; the
explicit-vector `imv` (146) and `b8v` (86) are unaffected (intrinsics), analytic 18.8 ns single.
So auto-vectorization is worth 3.3x for `rm`, 6x for `im`, 2.5x for `b8n`, 1.6x for `b8t`.

## Compiler findings that matter for a real implementation

* **Row-major dot products only vectorize with reassociation.** Without `-ffast-math`
  (and Cycles does not use it) GCC 13 vectorizes the multiplies and then adds the lanes in
  program order: a serial chain of 28 dependent adds per output (about 2.4 GFLOP/s, 3% of peak).
  With `-ffast-math` the horizontal-sum code gives 5-11 GFLOP/s. An input-major layout
  (`im`) vectorizes over outputs and needs no reassociation: 3x better with fast-math, 8x better without.
* **clang -march=native (Skylake-family tuning) is pathological for `rm`/`im`.** It vectorizes the
  loop across output neurons with `vgatherdps` (see `diag/`), which is 6-23x slower than the same code
  compiled with `-mavx2 -mfma` (2x32: rm 695 ns in clang_cycles vs 4349 ns in clang_fast; im 206 vs 4699 ns).
* **GCC 13 treats `-ffp-contract=on` as `off`.** Cycles' CMake passes `-ffp-contract=on`; with GCC 13 plain
  `a*b+c` loops therefore get no FMA (the gcc_cycles peak microbenchmark halves to 40 GFLOP/s).
  Clang does fuse. Explicit `madd`/`_mm256_fmadd_ps` is immune.
* **Plain `(a < b) ? b : a` min/max is compiled to compare-and-branch by GCC** unless `-ffinite-math-only` is on.
  A branchy activation on data-dependent signs mispredicts constantly (1.6x slower on a single 28x32 dense layer in a side experiment).
  The activation is therefore written with `fabsf`.
* **Register tiling in plain loops backfires**: `b8t` is 5-17x slower than the explicit-`float8` `b8v` in every build,
  and 2-6x slower than the untiled `b8n` wherever `b8n` is vectorized (gcc_fast, clang). The GCC asm shows it vectorizing
  across the 8 output neurons (strided weight loads, shuffles) instead of across the 8 evals. Only the explicit-`float8`
  version (`b8v`) delivers.
* GCC without `-ffast-math` did not vectorize the lane loops of the analytic BSDF (probably `fmaxf`/select NaN
  semantics; not investigated): gcc_strict/gcc_cycles `an_ggx_lambert_b8` are 32-37 ns, slower than the scalar eval;
  clang and the explicit `float8` version get 6 ns.
* The single-eval hand-vectorized kernel (`imv`) is robust: 146-159 ns for 2x32 in all six builds.

## Interpretation (CPU)

Single path vertex, best case (hand-written SIMD, input-major weights): 2x16 is 3.8-5.5x the analytic
layered eval, 2x32 is 5.5-8.2x, 3x64 is 25-34x. A 2x32 decoder is therefore not within 5x of an
analytic GGX+Lambert on CPU in any build (closest 5.5x against the slowest analytic baseline; 8.2x in the fast-math build);
written the way the task specifies (row-major loops), it is 27-211x. Adding the 16-float latent costs about 40% on
the 2x32 net (111 -> 158 ns); the encoding (about 9 ns) and the three exps (about 10-14 ns) are a fixed
20 ns that matters only for 2x16.
Batching 8 evals with explicit SIMD makes the MLP 1.7x faster than the best single-eval kernel
(158 -> 91.5 ns, 28% -> 49% of AVX2 peak) but makes the analytic BSDF 3.3-5x faster (19.4 -> 5.9/3.7 ns), so the
ratio gets worse (15-23x for 2x32): the MLP is FMA-throughput bound, the analytic eval is a latency-bound chain of
divides and square roots that SIMD parallelizes almost for free. Compiler-vectorized batching is not a free
lunch: plain-loop batching (`b8n`) beat the row-major single-eval code in the fast-math builds (2.6x for 2x32) but never beat
the hand-written single-eval kernel (1.1-11x slower).

## GPU extrapolation (rough, assumptions stated)

Assumptions: fp32 weights; a GPU with 30 TFLOP/s fp32 peak (about 15 T FMA lane-ops/s) as a stand-in for a
current mid/high-end card; the fraction of FMA peak a CPU variant reaches (measured above) is
assumed to indicate the achievable fraction for the analogous GPU strategy. Instruction-count ratio
analytic : 2x32 MLP is about 170 : 2900 lane-ops (110 flops + divides/sqrt expanded, vs 2016 FMA + about 900 other),
i.e. about 17x before any efficiency loss.

Reference: an analytic eval (about 170 lane-ops) at 30% of 15 T lane-ops/s is about 38 ps, i.e. 26 G evals/s.

* Per-thread scalar, weights read per thread (what `rm`/`im` map to; divergent materials in a warp): the weight
  stream is 4 B per MAC, L1 supplies roughly a quarter of the FMA rate, so 5-10% of peak: 1.5-3 TFLOP/s,
  1.6-3.2 ns per 2x32 eval of aggregate throughput (0.3-0.6 G evals/s), about 40-85x an analytic eval.
* Per-thread scalar, warp-uniform weights (paths sorted/binned by decoder, weights broadcast from constant/L1,
  what `b8v` models): 30-55% of peak, about 9-16 TFLOP/s: 0.3-0.5 ns per eval (2-3.4 G evals/s), about 8-17x the
  analytic eval (17x at equal efficiency, the pure instruction-count ratio). Sorting by material is required to get there.
* Warp-cooperative (a warp holds 32 evals as a batch, weights staged once in shared memory or registers, matmuls as
  mma tiles on tensor cores in fp16/tf32): matmul leaves the FP32 pipe; the remaining activation, encoding and exp
  work (about 16% of the FLOPs of a 2x32 net) is about 50 ps on the FP32 pipe at 50% efficiency, and the tensor-core
  matmul is of similar size, so roughly 8-19 G evals/s, optimistically within about 1.5-3x of the analytic eval.
  This is the only strategy that plausibly gets near "within 5x"; it needs a wavefront stage that gathers evals per
  decoder, and fp16 accuracy checks. Untested guess, not a measurement.
* At 1080p, 1 spp, about 5 bounces x 2 evals per vertex = 20M evals: 1-2.5 ms (cooperative), 6-10 ms (scalar, uniform
  weights), 32-65 ms (scalar, divergent) per spp for the decoder alone, against under 1 ms for an analytic eval.

## Not covered

Latent code fetch/interpolation (16 floats from a texture, 4 x 64 B per bilinear fetch), pdf and importance
sampling for the decoder (an eval-only comparison), spectral rendering, several lobes per material, weights
coming from L2/L3/DRAM when many different decoders are live (hot here: 8 KB for 2x32), fp16/int8 CPU paths
(AVX512-VNNI, AMX), and any measurement on a GPU. The VM is shared, so run-to-run noise of 5-10% is normal
(`worst` column); the analytic single-eval times in particular varied from 26 to 42 ns between passes in
one build (clang_cycles) before the second pass was added.
