# Experiment results

Both experiments ran on a CPU-only cloud VM (Xeon Cascade Lake, 4 vCPUs,
about 3.2 GHz, 15 GB RAM, g++ 13.3, clang++ 18.1, Python 3.13, numpy 2.5).
PyTorch could not be installed through the VM's proxy, so the training
experiment is pure numpy with hand-written backprop (gradients verified
against finite differences). Code to reproduce is under `experiments/`.

## A. Cost of a kernel-style MLP decoder versus an analytic BSDF

Code: `experiments/mlp_cost_bench/` (`bench.cpp`, `build.sh`, `run_all.sh`,
`make_table.py`; full tables in `results/tables.md`).

Question: if Cycles evaluated a small neural BSDF decoder in plain
kernel-style C++ (no STL, fixed loops, float math, AVX2), how expensive is one
evaluation compared with the analytic BSDFs Cycles already uses?

**Short answer:** a 2x32 decoder is not within 5x of an analytic layered BSDF
on this CPU. Hand-vectorized, one path vertex: 5.5 to 8.2x. Written as naive
row-major loops: 27 to 211x. Batched 8-wide with explicit SIMD against an
equally vectorized analytic eval: 15 to 23x. Only the 2x16 net gets near 5x.

Table (g++ -O3 -march=native -ffast-math, IN = 12 encoding + 16 latent, best of
3 over 10M evals):

| config | variant | ns/eval | ratio vs analytic | FLOPs (matmul / all) | GFLOP/s |
|---|---|---:|---:|---|---:|
| analytic GGX+Lambert | single eval, value | 19.4 | 1x | ~110 (3 sqrt, 6 div) | ~5.7 |
| analytic GGX+Lambert | single eval, value + pdf | 24.0 | 1.2x | | |
| analytic GGX+Lambert | 8-batched, explicit float8 | 5.9 / 3.7 cache | | | |
| 2x16 | row-major loops | 281 | 14.5x | 1504 / 1941 | 5.3 |
| 2x16 | best single eval (hand float8) | 94 | 4.9x | | 16.0 |
| 2x16 | batched-8 (explicit float8) | 40 | 6.8x / 10.3x | | 37.3 |
| 2x32 | row-major loops | 584 | 30x | 4032 / 4821 | 6.9 |
| 2x32 | best single eval (hand float8) | 158 | 8.2x | | 25.4 |
| 2x32 | batched-8 (explicit float8) | 91.5 | 15.4x / 23.2x | | 44.1 |
| 3x64 | row-major loops | 1846 | 95x | 20352 / 22549 | 11.0 |
| 3x64 | best single eval | 662 | 34x | | 30.7 |
| 3x64 | batched-8 (explicit float8) | 392 | 66x / 106x | | 51.9 |

Batched ratios are shown as streaming / cache-resident analytic baseline.
Measured AVX2 FMA peak of one core is 89.6 GFLOP/s.

The same 2x32 net across builds, ns/eval (ratio vs that build's own analytic):

| variant | g++ fast | g++ no fast-math | clang fast | g++ Cycles flags | clang Cycles flags |
|---|---:|---:|---:|---:|---:|
| analytic single | 19.4 | 27.7 | 20.6 | 28.9 | 25.7 |
| row-major | 584 (30x) | 1682 (61x) | 4349 (211x) | 1708 (59x) | 695 (27x) |
| input-major, compiler-vectorized | 167 (9x) | 208 (8x) | 4699 (228x) | 233 (8x) | 206 (8x) |
| input-major, hand float8 | 158 (8x) | 159 (6x) | 155 (8x) | 159 (5x) | 146 (6x) |
| batched-8, plain loops | 227 (38x) | 1357 (237x) | 255 (46x) | 1761 (283x) | 264 (42x) |
| batched-8, explicit float8 | 91 (15x) | 86 (15x) | 93 (17x) | 106 (17x) | 108 (17x) |

"Cycles flags" means `-mavx2 -mfma` without `-ffast-math`, which is what
Cycles' CMake uses.

Findings that carry over to the kernel design:

1. Store weights input-major and vectorize over outputs. It works without
   fast-math and is 3 to 8x faster than row-major.
2. Clang with `-march=native` vectorizes the row-major loop with `vgatherdps`
   and becomes 6 to 23x slower than `-mavx2 -mfma` (repro in `diag/`).
3. GCC 13 treats Cycles' `-ffp-contract=on` as off, so plain `a*b+c` loops get
   no FMA; explicit FMA intrinsics avoid this.
4. Batching with explicit 8-wide SIMD makes the MLP 1.7x faster but the
   analytic eval 3.3 to 5x faster, so the ratio gets worse. The MLP is
   FMA-throughput bound; the analytic eval is a latency-bound chain of divides
   and square roots that SIMD parallelizes almost for free.
5. Compiler-only batching is fragile; a batched neural stage needs explicit
   SIMD (Cycles' `float8` style) or a hand-written SoA kernel.
6. The single-eval kernel reaches only 28 to 34% of peak: weights are used once
   per eval and activations form a serial chain.

GPU extrapolation (assumptions: fp32 weights, a 30 TFLOP/s fp32 card, the
CPU's achieved fraction of FMA peak carries to the analogous GPU strategy; an
analytic eval about 170 lane-ops at 30% efficiency, a 2x32 eval about 2900
lane-ops):

- Per-thread scalar, divergent weights: about 5 to 10% of peak, 0.3 to 0.6 G
  evals/s, 40 to 85x an analytic eval.
- Per-thread scalar, warp-uniform weights (needs sorting by decoder): about
  30 to 55% of peak, 2 to 3.4 G evals/s, 8 to 17x.
- Warp-cooperative on tensor cores: roughly 8 to 19 G evals/s, optimistically
  1.5 to 3x the analytic eval. Untested guess.
- Per spp at 1080p (about 20M evals): 1 to 2.5 ms cooperative, 6 to 10 ms
  scalar with uniform weights, 32 to 65 ms scalar with divergent weights.

Conclusion: only a batched, warp-cooperative stage can plausibly approach
parity with analytic BSDFs. Not covered: latent texture fetch, pdf and
sampling, fp16/int8 paths, weight cache misses with many live decoders, any
real GPU measurement.

## B. Training a tiny neural BRDF decoder with the Bitterli 2026 recipe

Code: `experiments/neural_material_train/` (`material.py`, `train.py`,
`experiments.py`, `render.py`, `analysis.py`; results in `out/`). Reproduce in
about 15 minutes: `python3 experiments.py run 2400 && python3 experiments.py report && python3 render.py && python3 analysis.py`.

Setup: ground truth is an analytic two-layer BRDF (dielectric GGX coat, F0
0.04, over GGX metal/dielectric plus Lambert; Smith G, Schlick Fresnel, base
attenuated by (1 - Fc(wi))(1 - Fc(wo))) with per-texel parameters from a 64x64
procedural texture (smooth noise plus an 8x8 checker). Model: 64x64x8 bilinear
latent plus a 2-hidden-layer decoder (width 16 or 32), exp() output,
LeakySmeLU, power loss n=3, inputs wi, wo, h and the shortest-arc difference
vector. Multi-instance training: K=64 instances in lock-step on a shared batch,
culled 64 > 16 > 4 > 1 at phase ends, batch grown 128 > 512 > 2048 > 8192 so
K x batch is constant; total network evaluations equal the single-instance
baseline at 19.7M per run.

Scaled down from the paper: batch 8192 instead of 64k; N = 2400 steps (600 per
phase); one shared Adam LR of 0.04 with cosine decay; 8 seeds per cell; the
LeakySmeLU definition is a reconstruction (`0.05*x + 0.95*SmeLU(x)`, beta=1);
validation over-represents grazing angles (about 38% above 75 degrees).

Variance (8 seeds, final held-out power-loss L1, lower is better):

| method | mean | std | min | max | wall per run (s) |
|---|---|---|---|---|---|
| 2x16 baseline | 0.1176 | 0.0149 | 0.0949 | 0.1314 | 81 |
| 2x16 Bitterli culling | 0.1126 | 0.0052 | 0.1045 | 0.1184 | 97 |
| 2x32 baseline | 0.0766 | 0.0074 | 0.0617 | 0.0871 | 90 |
| 2x32 Bitterli culling | 0.0853 | 0.0033 | 0.0808 | 0.0909 | 110 |

Std ratio Bitterli/baseline: 0.35 for 2x16, 0.45 for 2x32. F-test: 2x16
significant (p about 0.01), 2x32 borderline (p about 0.02 one-sided).
Bootstrap 90% interval of the std ratio: 0.17 to 0.82 for 2x16, 0.20 to 1.08
for 2x32.

Accuracy (mean over 8 seeds, best seed in brackets, held-out set):

| model | method | pow-loss | rel L1 (all) | rel L1 (top 1%) | rel L1 (grazing > 75) | decoder params |
|---|---|---|---|---|---|---|
| 2x16 | baseline | 0.1176 (0.0949) | 0.2405 (0.2103) | 0.3486 (0.3066) | 0.2611 | 659 |
| 2x16 | Bitterli | 0.1126 (0.1045) | 0.2574 (0.2408) | 0.3753 (0.3533) | 0.2785 | 659 |
| 2x32 | baseline | 0.0766 (0.0617) | 0.1843 (0.1659) | 0.2780 (0.2471) | 0.2013 | 1827 |
| 2x32 | Bitterli | 0.0853 (0.0808) | 0.2183 (0.2090) | 0.3257 (0.3101) | 0.2368 | 1827 |

Ablations (2x16, baseline schedule, 4 seeds):

| variant | pow-loss mean (std) | rel L1 | rel L1 top 1% |
|---|---|---|---|
| power n=3 + LeakySmeLU | 0.1226 (0.0066) | 0.2464 | 0.3547 |
| log(1+x) + LeakySmeLU | 0.1707 (0.0157) | 0.2639 | 0.3980 |
| power n=3 + ReLU | 0.1238 (0.0158) | 0.2267 | 0.3122 |

Where it fails (rel L1 by subset): coat roughness below 0.2: 0.455 (2x16),
0.404 (2x32); gold checker: 0.26 / 0.22; 85 to 90 degree grazing: 0.29 / 0.23.
Peaks are systematically too low and too wide: the top 0.1% of targets come
out at 55% (2x16) and 64% (2x32) of their energy.

Interpretation:

1. At equal network evaluations, culling cuts run-to-run std by 2.2 to 2.9x.
2. It does not improve the mean at this scale; at 2x32 it is about 11% worse
   on pow-loss. Most of the variance reduction comes from losing the lucky
   tail. Likely cause (not isolated): with only 2400 steps the survivor spends
   3/4 of its steps at 128 to 2048 samples per instance, which updates the
   4096-texel latent grid sparsely. The recipe is meant for long runs at 64k
   batch.
3. 2x32 is clearly better than 2x16. Both reproduce the checker, coat haze and
   highlights in the sphere render (`out/sphere_compare.png`: ground truth |
   2x16 | 2x32, 4x amplified error below), but neither is production quality
   at this training length.
4. Power loss looks mildly better than log(1+x). LeakySmeLU vs ReLU is
   inconclusive; ReLU was better on accuracy, LeakySmeLU had lower seed std.
5. Caveats: unconverged training, one texture and one sampling distribution,
   reconstructed hyper-parameters, grazing over-weighted in validation.

Next steps on the GPU box: rerun with PyTorch at the paper's schedule (64k
batch, 100k steps), add an analytic specular lobe outside the network and let
the network fit the residual, and train from `SHADER_EVAL_BSDF` dumps of real
Cycles materials instead of the synthetic ground truth.
