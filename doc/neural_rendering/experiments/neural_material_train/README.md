# Neural-material feasibility check (numpy, CPU) - tiny decoder + Bitterli-style multi-instance training

**NOTE: PyTorch could NOT be installed here.** Attempt 1 (`pip install --user torch --index-url https://download.pytorch.org/whl/cpu`)
failed: the proxy returns 403 for download.pytorch.org. Attempt 2 (plain `pip install torch` from PyPI) resolves, but only the CUDA
build (~4 GB of nvidia-* wheels at 1-4 MB/s) is offered, which cannot finish inside the time budget and is useless on CPU. So everything
is **pure numpy with hand-written backprop + Adam** (gradients verified against finite differences for all parameters, both losses,
both activations). Python 3.13, numpy 2.5, 4 cores (4 single-thread worker processes).

## Files
- `material.py`  - ground truth: procedural 64x64 texture (smooth value noise + 8x8 checker) -> per-texel (base color, base roughness,
  metallic, coat roughness); two-layer BRDF = dielectric GGX coat (F0 0.04) over GGX metal/dielectric + Lambert, Smith G1, Schlick Fresnel,
  base attenuated by (1-Fc(wi))(1-Fc(wo)). Target = f * cos(theta_i).
- `train.py`     - model + trainer (latent grid 64x64x8 bilinear, decoder MLP 2xW, exp() output, LeakySmeLU, power loss n=3, inputs
  wi/wo/h/wd'), K-instance lock-step training with shared batch, cull 64>16>4>1 with batch 128>512>2048>8192. CLI for a single run.
- `experiments.py` - runs all experiments in a 4-process pool (`run [steps]`) and writes the tables (`report`).
- `render.py`    - 128x128 sphere, two point lights, GT vs best 2x16 / 2x32 model (PNG). `analysis.py` - error breakdown by angle / brightness / material.
- `lr_sweep.py`  - 800-step LR sanity sweep used to pick the shared Adam LR (0.04, cosine-decayed to 2%).
- `out/results.md`, `out/results.json`, `out/error_breakdown.md`, `out/progress.log`, `out/models/*.{npz,json}` (weights + per-run metrics/phase diagnostics).
- PNGs: `out/sphere_compare.png` (top: GT | 2x16 | 2x32; bottom: 4x amplified |error|), `out/sphere_gt.png`, `out/sphere_2x16.png`, `out/sphere_2x32.png` (512x512 each).

Reproduce: `python3 experiments.py run 2400 && python3 experiments.py report && python3 render.py && python3 analysis.py`
(about 15 min on 4 cores). Single run: `python3 train.py --mode bitterli --width 16 --seed 0 --steps 2400 --lr 0.04`.

## What was scaled down (honest list)
- Batch 64k -> 8192; schedule 64@128 -> 16@512 -> 4@2048 -> 1@8192 (K x batch = 8192 in every phase, so total network evaluations are
  identical to the baseline: 2400 steps x 8192 = 19.7M evaluations per run).
- N = **2400 steps** (600 per phase), not 4000: time budget. This is a short-training regime; the models are not converged.
- One shared Adam LR 0.04 (cosine to 2%), chosen from an 800-step sweep in which the highest LRs tested were still improving; same for both methods.
- 8 seeds per cell; ablations only 4 seeds, 2x16, baseline schedule, same seeds 0-3 as the main runs.
- Cull score = mean per-instance loss on the fresh (not-yet-trained-on) batches over the last quarter of each phase.
- LeakySmeLU is my reconstruction: `a*x + (1-a)*SmeLU(x)`, SmeLU(beta=1) from Shamir et al., leak a=0.05 (C1 continuous). The paper's exact
  definition/hyper-parameters were not available to me. Optimizer: Adam (0.9, 0.999), same LR for latents and decoder.
- Validation: fixed 65,536 held-out (uv, wi, wo): uv continuous (never on the training stream), directions = 50% uniform solid angle + 50% "peak-targeted"
  (wo = reflect(wi, h), h ~ GGX with alpha log-uniform in [0.01, 0.5]). The same mixture is used for training. Because it is uniform in solid angle,
  ~38% of validation samples have theta_i or theta_o > 75 deg (grazing is over-represented relative to typical renders).
- Metrics: "pow-loss" = held-out L1 of M_pow(pred) vs M_pow(target), n=3 (the variance-study metric, used for *all* variants so they are comparable);
  "rel L1" = sum|pred-target| / sum(target) (linear space, dominated by bright values); "top 1%" = same on the 1% brightest targets (mean RGB).

## Results

### Variance over 8 seeds (final held-out pow-loss; lower is better)
| method | seeds | mean | std | min | max | mean wall/run (s)* |
|---|---|---|---|---|---|---|
| 2x16 baseline (1 inst, bs 8192) | 8 | 0.1176 | 0.0149 | 0.0949 | 0.1314 | 81 |
| 2x16 Bitterli (64>16>4>1) | 8 | 0.1126 | 0.0052 | 0.1045 | 0.1184 | 97 |
| 2x32 baseline | 8 | 0.0766 | 0.0074 | 0.0617 | 0.0871 | 90 |
| 2x32 Bitterli | 8 | 0.0853 | 0.0033 | 0.0808 | 0.0909 | 110 |

Std ratio Bitterli/baseline: 2x16 = 0.35, 2x32 = 0.45 (variance F ratios 8.15 and 5.05 with df 7,7; one-sided F critical values 3.79 @5%, 6.99 @1%,
i.e. 2x16 is significant (p ~ 0.01), 2x32 borderline (p ~ 0.02 one-sided / ~0.05 two-sided); the crude bootstrap 90% interval of the ratio is 0.17-0.82 for
2x16 and 0.20-1.08 for 2x32). With n=8 the std estimates themselves are good to only about +-25-30%.
*Wall = per run, with 4 runs in parallel on 4 cores, single-threaded numpy (contention inflates absolute values; total for all 40 runs: 892 s).
The Bitterli run costs ~20% more wall time at identical network evaluations (many small matrices, dense Adam over 64 latent grids).

### Accuracy, 2x16 vs 2x32 (mean over 8 seeds, best seed in brackets; held-out set)
| model | method | pow-loss | rel L1 (all) | rel L1 (top 1% brightest) | rel L1 (grazing >75 deg) | median per-sample rel err | decoder params |
|---|---|---|---|---|---|---|---|
| 2x16 | baseline | 0.1176 (0.0949) | 0.2405 (0.2103) | 0.3486 (0.3066) | 0.2611 (0.2292) | 0.1457 (0.1131) | 659 |
| 2x16 | Bitterli | 0.1126 (0.1045) | 0.2574 (0.2408) | 0.3753 (0.3533) | 0.2785 (0.2610) | 0.1314 (0.1208) | 659 |
| 2x32 | baseline | 0.0766 (0.0617) | 0.1843 (0.1659) | 0.2780 (0.2471) | 0.2013 (0.1801) | 0.0857 (0.0625) | 1827 |
| 2x32 | Bitterli | 0.0853 (0.0808) | 0.2183 (0.2090) | 0.3257 (0.3101) | 0.2368 (0.2257) | 0.0922 (0.0829) | 1827 |

(plus 64x64x8 = 32,768 latent floats per model). The best models by held-out pow-loss are rendered: 2x16 = `baseline_w16_pow_lsmelu_s5`, 2x32 = `baseline_w32_pow_lsmelu_s6`
(chosen on the validation set among 16 runs per width, so slightly optimistic). Sphere-image relative L1 (linear): 0.118 (2x16), 0.061 (2x32).

### Where the best models fail (`out/error_breakdown.md`; rel L1 and predicted/target energy ratio)
| subset | 2x16 | 2x32 |
|---|---|---|
| coat roughness < 0.2 (6.6% of samples; sharp coat lobes) | 0.455 / 0.65 | 0.404 / 0.72 |
| coat roughness 0.2-0.4 (74%) | 0.085 / 0.98 | 0.046 / 0.99 |
| gold (metallic>0.5) texels | 0.262 / 0.82 | 0.218 / 0.86 |
| dielectric base texels | 0.100 / 0.99 | 0.061 / 0.99 |
| max(theta_i,theta_o) 30-60 deg | 0.089 / 0.99 | 0.070 / 0.98 |
| max(theta_i,theta_o) 85-90 deg | 0.291 / 0.80 | 0.226 / 0.86 |
| top 1% brightest | 0.321 / 0.75 | 0.269 / 0.80 |
| top 0.1% brightest | 0.480 / 0.55 | 0.413 / 0.64 |

### Ablations (2x16, baseline schedule, seeds 0-3, same held-out metrics)
| variant | pow-loss mean (std) | rel L1 | rel L1 top 1% | log(1+x) val loss |
|---|---|---|---|---|
| power n=3 + LeakySmeLU (default) | 0.1226 (0.0066) | 0.2464 | 0.3547 | 0.0329 |
| log(1+x) loss + LeakySmeLU | 0.1707 (0.0157) | 0.2639 | 0.3980 | 0.0343 |
| power n=3 + ReLU | 0.1238 (0.0158) | 0.2267 | 0.3122 | 0.0350 |

### Interpretation
1. At equal network evaluations the culling recipe does reduce run-to-run variance in this setting: std 0.0149 -> 0.0052 (2x16) and 0.0074 -> 0.0033 (2x32), significant for 2x16 and borderline for 2x32 with only 8 seeds.
2. It did not improve the average result, though: 2x16 pow-loss mean is ~4% better (within the noise) but rel L1 is ~7% worse, and 2x32 is ~11% worse on pow-loss and ~18% worse on rel L1; the best baseline seeds (0.062 / 0.095) beat every Bitterli run. The variance reduction here comes largely from clipping the lucky tail, and at 2x32 the worst Bitterli run (0.0909) is also worse than the worst baseline run (0.0871).
3. Likely reason (not isolated by an experiment): with only 2400 steps the survivor spends 3/4 of its steps at 128-2048 samples per instance, where the 4096-texel latent grid is updated very sparsely and noisily, so it arrives at the full-batch phase behind a baseline that used full batches throughout; the recipe was designed for long runs at 64k batch and larger networks, which I could not reproduce on CPU. At this scale I would call the claim "variance reduction: yes, accuracy gain: no".
4. The 64 phase-1 instances differ by only ~8% (min 0.368 to max 0.431 selection loss, 2x16), so selection at that point is a noisy predictor of final quality; I did not measure the rank correlation between early score and final loss.
5. 2x16 vs 2x32: 2x32 is clearly more accurate (baseline pow-loss 0.077 vs 0.118, rel L1 0.18 vs 0.24, top-1% 0.28 vs 0.35). Both reproduce the checker, coat haze and highlights visibly well in the sphere render, but neither is "production quality".
6. 2x16 is adequate for the smooth part of the material (coat roughness >= 0.2: 8-10% rel L1; dielectric base ~10%) and marginal otherwise; it fails on the sharp coat lobes (rel L1 0.45 for coat roughness < 0.2, peak energy under-estimated by ~35%), on the gold checker (0.26), and at extreme grazing angles (85-90 deg: 0.29). Peaks are systematically too low/too wide (top 0.1% targets reproduced at 55% / 64% of their energy for 2x16 / 2x32), exactly what a 659/1827-parameter decoder with a 8-float bilinear latent should do with alpha~0.01 lobes.
7. Power loss (n=3) vs log(1+x): the log-trained model is worse on pow-loss (expected, different objective), but also slightly worse on rel L1 (0.264 vs 0.246) and top 1% (0.398 vs 0.355) and no better on its own log metric (0.0343 vs 0.0329), so there is mild support for the power loss (4 seeds only).
8. LeakySmeLU vs ReLU: equal mean pow-loss (0.1226 vs 0.1238); ReLU was actually better on rel L1 (0.227 vs 0.246) and top 1% (0.312 vs 0.355), while LeakySmeLU had a lower seed std (0.0066 vs 0.0158). With 4 seeds this is inconclusive; I see no evidence at this scale that it improves accuracy.
9. Caveats: short, unconverged training; one texture and one sampling distribution; hyper-parameters (LR, LeakySmeLU constants, selection score) are mine; 8 seeds; the "paper" recipe was implemented from the task description only; validation over-weights grazing angles.
