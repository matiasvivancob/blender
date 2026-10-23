# Training pipeline for neural materials

Dump a Cycles material with `SHADER_EVAL_BSDF`, fit a latent grid plus a small MLP decoder in PyTorch, export
a `.nnm` file (`../../nnm_format.md`), verify it against the numpy reference decoder, and evaluate it.

**Status of this directory.** It was written on a machine without PyTorch and without Blender. Everything that
does not need them runs and is tested (`./run_all_synthetic.sh`). `model.py`, `train.py` and
`crosscheck_torch.py` were reviewed by reading, byte-compiled, and the forward pass, loss and Adam math were run
against the numpy prototype through a throwaway numpy stand-in for the torch subset used; they have never run
on real torch. `blender_dump.py` has never run in Blender and the `eval_bsdf` wrapper it calls has not been built.
**On the workstation, run `python3 crosscheck_torch.py` before anything else.**

```
Blender (eval_bsdf)            torch                         numpy only
blender_dump.py ──► dump.npz ──► train.py ──► model_final.npz/.pt ──► export_nnm.py ──► material.nnm
                       ▲                                                      │
make_synthetic_dump.py ┘ (no Blender)                evaluate.py ◄────────────┘  (nnm.reference decoder)
```

| File | Needs | Purpose |
|---|---|---|
| `dump_format.md`, `dataset.py` | numpy | dump container, loader, validation, record filter, train/val split |
| `sampling.py` | numpy | texel, direction and mesh-UV sampling shared by synthetic and Blender dumps |
| `make_synthetic_dump.py` | numpy | dump from the analytic layered BRDF of `experiments/neural_material_train/material.py` |
| `blender_dump.py` | Blender (untested) | writes a dump through `engine.eval_bsdf` |
| `model.py` | torch | latent grid (K instances), decoder, encoding, loss; optional tiny-cuda-nn decoder (untested) |
| `train.py` | torch | training, culling, validation, checkpoints, CSV log, export |
| `export_nnm.py` | numpy (torch only for `.pt`) | checkpoint to `.nnm` with test vectors, re-read and verify |
| `evaluate.py`, `sphere.py`, `metrics.py`, `png_io.py` | numpy | held-out metrics, error map, sphere render |
| `encoding_np.py`, `np_model.py` | numpy | numpy encoding and training-model forward; config and checkpoint-array format |
| `schedule.py` | none | culling and LR schedules |
| `numpy_train_small.py` | numpy | the prototype's numpy trainer fed from a dump (smoke test and numerical oracle) |
| `test_torchfree.py`, `run_all_synthetic.sh` | numpy | tests and the end-to-end smoke test |
| `crosscheck_torch.py` | torch | compares the torch code with the torch-free references; run first on the workstation |

## 0. Smoke test without Blender or torch

```
./run_all_synthetic.sh            # about 15 s; prints the output directory
```

It runs the unit tests, makes a 200k-record synthetic dump, trains the prototype's numpy model for 400 steps,
exports `.nnm`, verifies it, and evaluates it (held-out records and a sphere render). Expected: held-out relative
L1 of about 0.56 for the 400-step numpy model against 1.0 for the untrained one, sphere error 0.29 against 2.3.
These models are far from converged; the point is that every stage's file format and arithmetic agree.

## 1. Produce a dump from Blender

`intern/cycles/blender/addon/engine.py:eval_bsdf(engine, depsgraph, obj, prims, uvs, wi, wo, flags, out)`
evaluates the closures of the object's surface shader at `n` records `{prim, barycentric uv, wi, wo}` and fills
`out (n, 8)` with `[f*cos(wo) RGB, pdf, albedo RGB, flags]`. Its docstring has the engine-subclass recipe;
`blender_dump.py` implements it plus the sampling below:

```
blender -b scene.blend --python blender_dump.py -E BSDF_DUMP -f 1 -- --object Plane --n 4000000 --out dump_a.npz --seed 0
```

Time a `--n 100000` run first, then size the real one. Format: `dump_format.md`. Before trusting a dump run
`--probe` and check that Cycles' triangle index is the index into the evaluated mesh's `loop_triangles` (assumed,
not verified), and compare a few records with a hand-built render of the same material.

**Sampling strategy** (`sampling.py`, one distribution per record, all in the local frame):

1. Texel: uniform over the used UV area (`uniform_uv_on_mesh`: triangle by UV area, uniform barycentric), so that
   every latent texel is visited equally often. The `tex_uv` of the record indexes the latent grid; `uv` and
   `prims` are what Cycles needs.
2. `wo` (light): cosine-weighted hemisphere, the distribution a path tracer draws from; 10% of records instead
   uniform in cos(theta) with theta > 75 degrees (`--grazing-frac`), because renders of curved objects hit grazing
   angles far more often than a cosine density suggests and the metrics report them separately.
3. `wi` (viewer): a mixture, 50% (`--peak-frac`) GGX-peak targeted, 40% uniform in solid angle, 10% grazing.
   Peak targeting reflects the sampled `wo` about a GGX half vector (alpha log-uniform in [0.01, 0.5]), which puts
   `wi` inside the specular lobe for any roughness in that range. Without it most records sit in the diffuse
   floor and the highlights, which carry the energy and the error, are barely seen. (The numpy experiment found the
   brightest 0.1% of targets at 55 to 64% of their energy with this recipe; more peak coverage and a bigger net
   are the levers.)
4. Directions are float32 unit vectors with z >= 0.005.

**How many records.** The latent grid has H x W texels and the decoder sees each through bilinear weights, so the
data must cover texels x angles. Rules of thumb, to be replaced by measurement: 500 records per texel for a first
look (64 x 64 grid: 2M), 2000 or more for quality (8M); 256 x 256 grids are out of reach with a fixed dump, use a
lower latent resolution. The paper streams fresh samples (64k batch x 100k steps = 6.5G visits); with a dump of
8M that is about 800 epochs, so watch the gap between `train_loss` and `val_loss` in the CSV and add shards
(different `--seed`, repeat `--dump a.npz b.npz`) when it opens. Keep `--n` per Blender call at 1 to 4M.

## 2. Train

Train on the synthetic dump first, on the workstation, to check the torch code and get reference numbers:

```
python3 crosscheck_torch.py                                  # CPU, ~1 min; every line must say PASS
python3 make_synthetic_dump.py synthetic.npz --n 2000000 --sphere synthetic_sphere.npz
python3 train.py --dump synthetic.npz --out runs/syn_cull  --steps 20000                 # Bitterli culling, 64 -> 1
python3 train.py --dump synthetic.npz --out runs/syn_base  --steps 20000 --no-culling    # baseline
```

then a real dump:

```
python3 train.py --dump dump_a.npz dump_b.npz --out runs/mat1 --steps 100000 --width 32 --hidden 2 \
    --latent-channels 8 --latent-res 128 --lr 0.02
python3 train.py --dump dump_a.npz dump_b.npz --out runs/mat1_base --steps 100000 --no-culling   # compare
python3 train.py --dump dump_a.npz dump_b.npz --out runs/mat1 --resume runs/mat1/ckpt_last.pt    # after an interruption
```

What it does (`train.py` docstring has the details):

- Model: latent grid `H x W x C` (C = 4..16, bilinear, corner-to-corner) concatenated after the 12 encoded
  direction features, decoder MLP 1..4 hidden layers of 16/32/64, LeakySmeLU(`--beta`, `--eps`) (or `relu`,
  `leaky_relu`), output `exp(clamp(x, -20, 14))` with a straight-through gradient at the clamp.
- Loss: power loss n = 3 on f*cos targets. Adam, cosine LR from `--lr` to `--lr * 0.02`.
  The prototype used 0.04 for 2400 steps; for 100k steps start at 0.02 and sweep {0.005, 0.01, 0.02, 0.04}
  on a short run.
- Culling (default): K = 64 instances train in lock-step on one shared batch, in 4 equal phases. At the end of a
  phase the instances are ranked on a fixed held-out subset (`--eval-size`) and the best K/4 survive, the batch
  grows by 4: 64@1024, 16@4096, 4@16384, 1@65536 (K x batch constant, so the number of network evaluations equals
  the baseline's). `--instances`, `--phases`, `--batch-total` change the shape, `--schedule "K:batch,..."` sets it
  explicitly. The numpy experiment saw 2 to 3 times lower seed-to-seed spread and no better mean at 2400 steps
  (`04_experiment_results.md` section B); whether that changes at 100k steps and 64k batch is the experiment to
  run. `--no-culling` is the baseline.
- Validation: a fixed split by record (`--val-frac 0.05`, at most `--val-max`, `--split-seed 0`, independent of
  `--seed`, so runs are comparable).
- Outputs in `--out`: `train_log.csv` (step, phase, K, batch, lr, train loss min/mean over instances, held-out
  loss min/mean every `--eval-every`), `phases.json` (per-phase held-out loss spread and the kept scores),
  `config.json`, `ckpt_last.pt` (resume), `model_final.pt` and `model_final.npz` (weights for export),
  `final_metrics.json` (held-out relative L1 overall, brightest 1%, grazing, peak energy), `model.nnm`.
- The final step checks that the arrays that go to the exporter reproduce the torch forward (numpy float64,
  p99 relative error < 1e-3) and aborts the export otherwise.
- Records live on the GPU as float32 (44 bytes each); `--max-records` caps them. `--device cpu` works for tests.
- `--backend tcnn`: decoder on tiny-cuda-nn, ReLU only, one instance, no biases (emulated, see `model.py`),
  parameter layout assumed. **Untested.** If the layout assumption is wrong the export self-check fails
  instead of writing a wrong model. Use the torch backend unless training speed is the bottleneck.

## 3. Export

`train.py` exports at the end. By hand, from either file type:

```
python3 export_nnm.py runs/mat1/model_final.npz material.nnm                  # no torch needed
python3 export_nnm.py runs/mat1/model_final.pt  material.nnm
python3 export_nnm.py runs/mat1/ckpt_last.pt    material.nnm --instance 0     # a culling run stopped early
```

It quantises weights and latents to fp16, writes the activation, the exp transform and the `bsdf_dirs` encoding
into the header, embeds 32 test vectors computed by `nnm.reference`, re-reads the file, and fails (exit 1) unless
(1) the file parses with full validation, (2) every test vector matches `nnm.reference` to the format's
tolerance, (3) the float64 matmul decoder agrees, (4) on 4096 random inputs the fp16 `.nnm` decode matches the
unquantised float64 forward of the training model with p99 relative error below 5%. The fp16 error
observed on the numpy-trained 2x16 model is 0.1% median, 0.6% p99. Interchange layout of the `.npz`:
`latent (H,W,C)`, `W0..Wn (in,out)` input-major with the 12 direction rows first and the latent rows after,
`b0..bn`, `config_json`.

## 4. Verify and evaluate

```
python3 evaluate.py material.nnm --dump dump_a.npz dump_b.npz --out eval/mat1 [--sphere-dump sphere_dump.npz]
python3 -m nnm.selftest                         # from ..: the format and reference decoder themselves
python3 -m nnm.gen_test_header                  # regenerate the kernel gtest data if the model is to be a golden vector
```

`evaluate.py` decodes with `nnm.reference` (fp16 weights, the arithmetic of the kernel's `mlp.h`) on the validation
records of the dump (same split flags as `train.py`; `--split all` for every record; add
`--train-max-records` if training used `--max-records`) and writes `metrics.json` (relative L1 overall, brightest
1%, grazing > 75 degrees, power loss, energy of the top 0.1%), `error_map.npy` (per-texel-bin relative L1) and with
`--sphere-dump` `sphere_compare.png` (ground truth | model, 4x error below). Sphere ground truth:
`python3 sphere.py q.npz` then `blender -b ... blender_dump.py ... -- --object Plane --fill q.npz --out sphere_dump.npz`.
Final acceptance of a real material is not a number from here: it is the furnace test and the MIS-consistency
render in `tools/buildbox/` once the kernel closure exists.

## 5. Verifying the torch code (what was and was not done)

Done here, without torch: the numpy encoding agrees with `nnm.reference` (2e-6) and with an independent float64
Rodrigues rotation (5e-5); the numpy forward of the exported arrays (`np_model.forward_np`) matches the numpy
prototype's own `forward` at 2.5e-6 relative; an `.nnm` exported from a prototype-trained model decodes to the
prototype's forward within 0.2% p99 (0.3% max), the fp16 quantisation error; a permuted first layer (latent rows
first) or a wrong `beta` makes the exporter's verification fail. `train.Adam`, `model.power_loss` and
`NeuralMaterial.forward/select` were executed through a throwaway numpy stand-in for the torch subset they use
(not part of this directory) and match the prototype's Adam (exactly), `loss_and_gout` (4e-8) and the numpy
forward (1e-6). That checks shapes, indexing and arithmetic, not autograd or real torch semantics.

Not done: any run on real torch. What `crosscheck_torch.py` settles on the workstation, in order: torch encoding
and bilinear lookup versus numpy; activations; power-loss value and gradient versus the prototype's analytic
gradient (including the straight-through clamp); Adam including `select`; model forward versus `np_model`, and the
exported `.nnm` decoded by `nnm.reference` versus the torch prediction for three activations, K = 3, 3 hidden
layers; `select()`; and a tiny `train.py` run with culling 4 -> 1, a stop/resume cycle, export and verify.
Points to look at if it fails: `torch.bmm` of the (K,B,in) activations with the (K,in,out) weights; the `lat[:, idx]` gather and its
backward; `torch.randperm(..., generator=)` on a CUDA generator; `Generator(device=...)`; `torch.load(...,
weights_only=True)` on the resume checkpoint (plain dicts, lists, tensors only).

Cross-check recipe to run by hand on any trained model: `export_nnm.py` already compares the `.nnm` decode with
`np_model.forward_np`, and `train.py` compares `forward_np` with the torch forward; the two together tie the
`.nnm` to the torch function. For the kernel, `nnm/build_standalone_test.sh` runs the C++ decoder on exported models.

## 6. Open items

- The LeakySmeLU definition and `eps = 0.05`, `beta = 1` are a reconstruction (`nnm_format.md` section 3).
- The prims-index assumption and the UV convention (V flip) of `blender_dump.py` need a real Blender run.
- One material per model; the latent grid is indexed by the UV layer, so texture-space tiling or UV overlaps are
  not representable. Anisotropy is out of scope (the local frame has no tangent direction in the features).
- Peaks: the numpy results under-fit sharp specular peaks. Next levers: more `--peak-frac`, width 64, an analytic
  specular lobe kept outside the network with the network fitting the residual (needs a `pdf`/albedo-based proxy
  fit, which is why those columns are kept in the dump).
- `pdf` and `albedo` columns are carried but unused by training.
