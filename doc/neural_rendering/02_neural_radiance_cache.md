# Neural Radiance Cache in Cycles: feasibility and design

Written against HEAD `afb395bd`. Paths are relative to `intern/cycles`. All cost
numbers are estimates; nothing was built or benchmarked. OpenPGL headers were
not available, so OpenPGL claims come from Cycles call sites only.

Reference: Müller, Rousselle, Novák, Keller 2021, "Real-time Neural Radiance
Caching for Path Tracing". With the NVIDIA-only decision in `00_synthesis.md`,
the network (multiresolution hash grid plus 4x64 MLP) and its online training
should be implemented with tiny-cuda-nn rather than hand-written kernels.

## Bottom line

- Feasible. The wavefront design has clean seams, and the guiding hooks already
  mark almost every site that needs a training record.
- It is a biased feature, and under Cycles defaults it gives little or no gain.
  Defaults are 12 total bounces, 4 diffuse, 4 glossy, 0 volume
  (`blender/addon/properties.py:800-830`), so most paths end after about 5
  vertices.
- Gains appear with raised bounce limits, high-albedo interiors, SSS, and
  multi-scatter volumes.
- MVP (surface only, opt-in, biased, host-trained): about 20 human
  engineer-weeks; with tcnn and agents, 4 to 6 weeks dominated by validation.

## 1. Integrator state machine

Kernels (enum `kernel/types.h:1807-1826`): `init_from_camera`,
`init_from_bake`, `intersect_closest`, `intersect_shadow`,
`intersect_subsurface`, `intersect_volume_stack`, `intersect_dedicated_light`,
`intersect_mnee`, `shade_background`, `shade_light_nee/forward`,
`shade_surface(+_raytrace)`, `shade_volume(+_ray_marching)`, `shade_shadow`,
`shade_dedicated_light`; `shadow_path_mnee_pending` is a marker.

The CPU megakernel is a switch (`kernel/integrator/megakernel.h:60-110`). The
GPU wavefront picks the most-queued kernel
(`integrator/path_trace_work_gpu.cpp:431, 554-646`).

State layout:

- Path state is `kernel/integrator/state_template.h`: `flag` :41,
  `mis_ray_pdf` :52, `throughput` :65, `pass_*_weight` :69-75,
  `denoising_feature_throughput` :81, `guiding` struct :135.
- Shadow path state is `shadow_state_template.h`.
- Members are gated by `KernelFeatureRequest`, so a `KERNEL_FEATURE_NRC` bit
  (next free after GSPLATS, `features.h:86`) costs nothing when off.
- GPU SoA allocation is `path_trace_work_gpu.cpp:117-233`.
- `KernelIntegrator` members go in `kernel/data_template.h`, next to the
  guiding block at ~:229-241.

**(a) Termination.** Russian roulette runs in `integrator_intersect_terminate`
(`intersect_closest.h:29-87`), using `path_state_continuation_probability`
(`path_state.h:279-307`), which is min(sqrt(max throughput), 1). It runs
BEFORE the shader is evaluated, so albedo, roughness and normal are not known
there. The `throughput /= p` division happens in `integrate_surface_terminate`
(`shade_surface.h:658`, called at :840). Bounce caps live in `path_state_next`
(`path_state.h:112-244`). Volumes have their own roulette
(`shade_volume.h:1273`). So NRC must decide inside `shade_surface` or a new
kernel, not in `intersect_closest`.

**(b) Accumulation.** There is no per-path radiance variable. Every
contribution is premultiplied by absolute throughput and atomically added to
the render buffer. Sinks: `film_write_surface_emission`
(`kernel/film/light_passes.h:680`), `film_write_background` :630,
`film_write_volume_emission` :661, `film_write_direct_light` :489 (called at
`shade_shadow.h:293-294`, using the shadow path's own throughput). All go
through `film_clamp_light` :126, then `film_write_combined_pass` :344, then
atomics in `kernel/film/write.h:58-104`. Pass splits use the first-bounce
`pass_diffuse_weight` and `pass_glossy_weight` (set at bounce 0,
`shade_surface.h:602-612`) plus a bounce-count test for direct vs indirect.

**(c) Query site, `integrate_surface` (`shade_surface.h:767`).** Order of
calls: shader eval :795, prepare_closures :823, holdout :826, emission :832,
roulette :840, data passes :847, denoising features :851, direct light :867, AO
:877, bounce :882. Insert the query after :851. Emission at the terminal vertex
is already written with the correct MIS weight against the previous vertex's
NEE (via `mis_ray_pdf`, `kernel/light/sample.h:480-607`), so the cache must
predict reflected radiance L_o minus L_e, not total radiance; otherwise light
is double-counted. Write throughput x L_r through a new film function cloned
from `film_write_surface_emission`: clamp, combined pass, and
diffuse/glossy/transmission-indirect through the existing weights; skip
lightgroups; then terminate the path.

On GPU, make the query its own kernel (`DEVICE_KERNEL_INTEGRATOR_NRC_QUERY`,
reached via `integrator_path_next`): coherent control flow, its own register
budget, no bloat in `shade_surface` (HIP is already fragile there,
`shade_surface.h:316-320`). A new `DeviceKernel` touches about 17 files and 31
references (compare `SHADE_DEDICATED_LIGHT`). This is the same stage the
synthesis proposes for neural BSDF evaluation; share it.

Feature extractors already exist: `surface_shader_diffuse/glossy/transmission`
(`surface_shader.h:1073-1115`), `surface_shader_average_normal`, the
average-roughness helper (~:30-48), `closure_albedo` (`closure/bsdf.h:732`).
The spread criterion needs a new state float, updated at
`shade_surface.h:602-629` from `sd->ray_length`, `bsdf_pdf` and N dot wi.

**(d) Suffix flag.** Add `PATH_RAY_NRC_TRAIN` in a free bit above bit 27, which
is SPECTRAL (`types.h:303`). Choose it by hash at `path_state_init_integrator`
(`path_state.h:42`). Add `nrc.record_head` (index of the last record in a
global atomically-allocated buffer) and `nrc.spread` to the path state. Copy
the record index into the shadow path where the guiding fields are copied
(`shade_surface.h:301-308`), so `shade_shadow.h:293` can atomically add the NEE
term; this is exactly the guiding pattern (`path_segment` pointer plus
`unlit_throughput`). Gotcha: kernels are re-executed after a texture cache miss
(`state_flow.h:100`; returns at `shade_surface.h:800` and :869). Allocate
records only after the last such return, or make writes idempotent.

## 2. Online training

Hook points:

- Host: `PathTrace::render_pipeline` (`path_trace.cpp:184-256`). Guiding
  already does `guiding_prepare_structures` (:208), `path_trace` (:218),
  `guiding_update_structures` (:227). Add `nrc_prepare/update` the same way.
  Add a `RenderWork.nrc.train` flag like `volume_guiding_denoise`
  (`render_scheduler.cpp:383`). Use `set_limit_samples_per_update` to bound
  staleness (guiding uses 4, `path_trace.cpp:1576`).
- Intra-call GPU hook: the loop in `PathTraceWorkGPU::render_samples`
  (:365-430). A sync point already exists at `update_queue_counter_and_cache`
  (:341).
- CPU: `render_samples` is a `parallel_for` over pixels
  (`path_trace_work_cpu.cpp:72-110`), and the megakernel runs a whole path per
  thread. Per-thread record storage pushed to a global buffer at path end, as
  in `guiding_push_sample_data_to_global_storage` (:411).
- Cache invalidation: `Session::update_scene` uses `scene->need_reset(false)`,
  which excludes the camera (`scene.cpp:497`, `session.cpp:706-728`). The cache
  therefore survives camera moves and drops on edits.
  `set_guiding_params(params, reset)` at `session.cpp:350-353` is the template.

Options evaluated:

- (i) Hand-written train kernels (forward, backward, Adam) in the kernel
  language. Portable, but the 64x64 weight-gradient reduction needs tiled
  shared-memory reductions on every backend. 6 to 8 human weeks. **Superseded
  by tcnn for this fork.**
- (ii) Train on the CPU from a downloaded record buffer. Records about 48 B
  each, so 64k records is about 3 MB; weight upload about 17 MB fp16 grid,
  about 1.5 ms over PCIe. About 50 to 150 ms per 64k-record step (estimate).
  Fine for offline, marginal for viewport. **Fallback.**
- (iii) Vendor libraries. tiny-cuda-nn (BSD-3) is the natural choice on
  NVIDIA; it was written for NRC. Precedent for NVIDIA-only code paths:
  `denoiser_optix.cpp`, `denoiser_dlss.cpp` (`WITH_DLSS`). **Chosen.**

Recommendation: inference and training through tcnn on the NRC query stage
and in the scheduler hook; CPU fallback (ii) only for correctness tests.

## 3. OpenPGL reuse

Guiding is CPU-only: `device.info.has_guiding` is set only by the CPU device
(`device/cpu/device_impl.cpp:313-315`), `__PATH_GUIDING__` code is gated on it,
and there is nothing in `path_trace_work_gpu.cpp`.

The `guiding_record_*` call sites are the reusable asset. Definitions are in
`kernel/integrator/guiding.h`; callers:

- surface segment: `shade_surface.h:810`
- bounce: `shade_surface.h:631`
- emission: `shade_surface.h:157`
- background: `shade_background.h:154` and :241
- light hits: `shade_light.h:26` and :71
- NEE: `shade_shadow.h:293`
- volume: `shade_volume.h:1491, 1763, 2396, 2688, 2764, 2820, 2834`
- BSSRDF: `subsurface.h:150`, `subsurface_disk.h:207`, `subsurface_random_walk.h:485`
- roulette probability: `intersect_closest.h:64`

Together they carry position, normal, wo, scattering weight, Le plus MIS
weight, NEE divided by unlit throughput, and roulette probability. That is what
a training target needs, and they already handle light/shadow linking, volumes
and SSS. The shadow-catcher exclusion at `intersect_closest.h:122-127` must be
copied.

Plan: refactor `guiding.h` into a generic vertex-recorder interface with two
sinks, OpenPGL and NRC. Not reusable: OpenPGL's storage (CPU C++ library,
directional scalar-weight samples); NRC needs per-vertex RGB targets from a
different backward recursion, so its own segment struct and backward pass.

Guiding trains only for the first `training_samples` iterations (default 128,
`scene/integrator.cpp:104`; `path_trace.cpp:1561`) and then freezes. Freezing
NRC after a warm-up has a real benefit: later samples are independent given a
fixed cache.

## 4. Interactions

- Adaptive sampling: needs care. The half-sample error estimator
  (`light_passes.h:175-199`, `adaptive_sampling.h:29-92`) assumes independent
  samples; cache error is correlated, so pixels stop early.
- Denoising albedo/normal: needs care. Written at `shade_surface.h:851`. Do
  not terminate while `PATH_RAY_DENOISING_FEATURES` is set (cleared at
  `denoising_passes.h:248`).
- Denoisers on cache-correlated noise: needs care. Cache error looks like
  signal and gives splotches.
- Diffuse/glossy/transmission/volume direct and indirect passes: works with
  care. Cached radiance is always bounce >= 1, so indirect. Terminate only at
  bounce >= 1.
- Emission, environment and data passes: works.
- Lightgroups: incompatible unless the cache has 3 outputs per group.
- Shadow catcher: incompatible (`shadow_catcher.h`, `intersect_closest.h:89-163`).
- AO bounces: incompatible (`path_state_ao_bounce`).
- Light Path node: incompatible (`KERNEL_FEATURE_NODE_LIGHT_PATH`, `features.h:28`).
- Motion blur: incompatible unless time is an input.
- Dispersion (`PATH_RAY_SPECTRAL`): incompatible; the cache is RGB.
- Bake: disable.
- Bounce limits: needs care; per-type caps are path history. Feed remaining
  diffuse/glossy budget as inputs, or require high limits.
- Light/shadow linking: works (depends on receiver object, so on position).
- MNEE and SSS: needs care. Volumes: second feature set, phase 2.
- Viewport: best case. Cache persists across camera moves and resolution
  changes.
- Final render: needs the convergence story in section 6.

## 5. Budget

- Hash grid: 16 levels x 2^18 x 2 fp16 = 16.8 MB inference copy. With fp32
  master, grads and Adam m/v, training memory is about 150 MB.
- Query cost: about 128 grid gathers plus about 16.6k MLP FMAs, roughly 35
  kFLOP.
- A Cycles bounce is about 4 kernel visits (intersect_closest, shade_surface
  with SVM/closures/light tree, intersect_shadow, shade_shadow), 10 to 50k
  divergent instructions. On GPU a coherent query kernel is about 0.3 to 1
  bounce; on CPU about 1 to 1.5 us, 0.5 to 1.5 bounces.
- Training: about 3x inference per record, about 7 GFLOP per 64k-record step:
  a few ms on a GPU with tcnn.
- Pays off when the replaced tail averages more than about 2 to 3 bounces:
  albedo >= 0.8 interiors, SSS, thick scattering volumes, raised bounce caps.
- Does not pay: default caps, outdoor sun/sky, direct-light-dominated product
  shots, caustics (MNEE is better), mirrors and glossy where the (x, wo) cache
  is too blurry, per-frame animation needing a cold retrain.

## 6. Bias control

- Mode A, biased: opt-in, off by default, labeled like Filter Glossy and
  clamping. Truncation at max_bounce is already a bias in Cycles.
- Mode B, unbiased, cheap: Russian roulette with the cache as control variate.
  At an eligible vertex, with probability 1-q terminate and write T x Lc;
  otherwise write T x Lc x (1 - 1/q) (negative) and continue with T/q. The
  expectation is exact. Reuses the roulette pattern in
  `integrate_surface_terminate`. Variance falls only if cache error is below
  about 10%. Negative per-sample contributions need care with denoisers and
  adaptive sampling.
- Mode C: use the cache to drive importance-aware roulette and splitting.
  Unbiased, but the query has to happen before shading.
- Viewport-only use is possible, but then the viewport will not match the
  final render.
- Freeze the network after warm-up so later samples are i.i.d.

## 7. Plan (human engineer-weeks; see `00_synthesis.md` for agent recalibration)

1. Spike: dump records, fit offline in PyTorch, pick features. 2
2. Generic vertex recorder (refactor `guiding.h`), `KERNEL_FEATURE_NRC`, state
   fields, `data_template.h` and scene gating. 1.5
3. Hash grid + MLP forward (tcnn on CUDA; scalar CPU reference), with CPU/GPU
   parity tests. 2
4. Spread criterion, `NRC_QUERY` kernel, film write, gating. 3
5. Records: suffix selection, shadow-path NEE attribution, MIS-correct target,
   backward accumulation, bootstrap. 3
6. Trainer (tcnn Adam, double-buffered upload), scheduler hooks, reset logic. 3
7. Python/UI/sync (`properties.py`, `sync.cpp:509-529` pattern),
   incompatibility warnings. 1
8. Validation: bias metrics, regression scenes
   (`tests/python/cycles_render_tests.py`), cross-device parity. 3

MVP about 18 to 22. Phase 2: volumes and SSS 3, Mode B 2.

Top 5 risks:

1. **Quality and expectations.** Error does not shrink with samples, light
   leaks on thin geometry, denoiser splotches. Users expect Cycles to converge.
2. **Silent target errors.** Asynchronous NEE, MIS, linking, volume, SSS and
   MNEE paths all feed the target. OpenPGL needed `PATH_GUIDING_LEVEL` gating
   to get this right; any bug is invisible bias.
3. **Performance and portability.** Extra state shrinks the concurrent path
   count; register pressure and compile time grow.
4. **Training stability and determinism.** HDR fireflies in targets, fp16,
   GPU atomics work against Cycles' seeded reproducibility.
5. **Value and scope.** Defaults give no benefit; many features unsupported.
