# Synthesis: what to build and in what order

## Conclusion

Build one thing in Cycles: a **batched neural evaluation stage** in the
wavefront integrator, NVIDIA-only, and make every neural feature a client of it.
Do not build neural closures that run inline inside the shading kernel.

The reason is measured, not guessed. A 2x32 decoder MLP evaluated per path
vertex costs 8 to 17x an analytic layered BSDF even when every lane in a warp
shares the same weights, and 40 to 85x when lanes diverge, which is Cycles'
normal state (see `04_experiment_results.md`). Only a stage that gathers
queries into a buffer, sorts them by model, and evaluates them
warp-cooperatively on tensor cores can plausibly get within a small factor of
an analytic evaluation (extrapolated 1.5 to 3x, unmeasured).

Going NVIDIA-only is what makes that stage cheap to build:

- **tiny-cuda-nn becomes the engine.** BSD-3 licensed, so this fork can ship
  it. Its fused MLP and multiresolution hash-grid kernels operate on exactly
  what the stage produces: a device buffer of queries. The stage becomes
  "gather, call tcnn forward, scatter". Online training for a radiance cache is
  tcnn's original purpose. This removes hand-written matrix code for five
  backends from the plan.
- **Both target cards do warp-cooperative fp16 matmul.** `mma.sync` fp16 is
  sm_80 and up, so the 3090 (sm_86) and the 4070 Ti Super (sm_89) both run it
  natively. OptiX cooperative vectors (OptiX 9.0+) are API-available on both,
  but hardware acceleration is documented by NVIDIA for Ada and Blackwell, so
  the 3090 likely takes a slower path there. Verify against the OptiX 9 release
  notes before relying on it.
- **Build iteration gets fast.** Cycles compiles seven CUDA architectures by
  default. Restricting `CYCLES_CUDA_BINARIES_ARCH` to `sm_86;sm_89` removes most
  kernel compile time. HIP, Metal and oneAPI parity drop off the plan, as does
  the register-pressure concern on sm_5x and sm_6x budgets.

What stays vendor-neutral is a plain-ALU scalar fallback on CPU, kept for unit
tests and parity checks against the PyTorch reference, not for production.

## Why this architecture and not the obvious one

The obvious design is a `CLOSURE_BSDF_NEURAL_ID` whose `bsdf_eval` runs the MLP
inline. `01_cycles_neural_closures_spec.md` specifies that design in full
because its pieces (weights transport, closure struct, proxy-lobe importance
sampling, albedo for passes) are needed anyway. But three facts from the code
investigation make inline evaluation the wrong place to run the network:

1. Cycles' shading kernels run as plain CUDA kernels even under OptiX. Only the
   `intersect_*` kernels and `SHADE_SURFACE_RAYTRACE` are OptiX raygens
   (`intern/cycles/device/kernel.cpp:30-39`, `device/optix/queue.cpp:39,57`).
   OptiX cooperative vectors are therefore not reachable from surface shading
   without a new OptiX module and pipeline entry.
2. Shader sorting groups lanes by shader index, not by closure
   (`kernel/integrator/state_flow.h:162-199`), and `integrate_surface` has
   early-outs and stochastic closure picks. Warps are not convergent, so
   warp-synchronous matrix instructions cannot be issued from that control
   flow.
3. Even with warp-uniform weights, scalar per-thread evaluation is
   FMA-throughput bound at roughly 30 to 55% of peak and lands at 8 to 17x an
   analytic BSDF. The analytic BSDF is latency bound and vectorizes almost for
   free.

A dedicated stage fixes all three: it is a separate `DeviceKernel` (precedent:
`SHADE_SURFACE_RAYTRACE`, about 55 touch points; `SHADE_DEDICATED_LIGHT`, about
17 files), it runs over a compact query buffer so warps are full and uniform,
and it can call tcnn or hand-written `mma.sync` code.

## Clients of the stage, in order of value

1. **Neural BSDF** (Zeltner et al. 2024 style: per-texel latent + 2x32
   decoder). Closure stores the latent and proxy-lobe parameters at shading
   time, defers the network evaluation to the stage, importance-samples with a
   fitted GGX proxy plus MIS. Needs `SHADER_EVAL_BSDF` to produce training
   data from existing Cycles materials.
2. **Neural Radiance Cache** (Müller et al. 2021). Query after the denoising
   features are written in `shade_surface.h`; records via a generic vertex
   recorder refactored out of the OpenPGL guiding calls; training step in the
   render scheduler using tcnn. Biased unless used as a control variate in
   Russian roulette. Pays off only on interiors, SSS and volumes with raised
   bounce limits, and in the viewport. See `02_neural_radiance_cache.md`.
3. **Neural SSS** (Vicini et al. 2019 style, modeled on the disk sampler).
   One global parametric model, no per-material training. Training data
   generation is the dominant cost.
4. **Neural texture decode** (NTC style). Economics are worse than the BSDF
   case because it runs several times per vertex. Only via the stage, and only
   after the above exist. NTC "inference on load" is reachable today as a
   Python add-on for disk savings.

Not in scope for now:

- 8DNA-style neural assets: four 128-wide MLPs plus a normalizing flow per
  bounce, 20 to 100x heavier than anything above. Prototype in Mitsuba 3 if
  wanted; do not touch the kernel for it yet.
- The Xu et al. generalizable GI embedding: a 42M-parameter transformer, 208 ms
  encode and 368 ms per 512x512 frame, diffuse-only, indoor-only, code not yet
  released. If it lands, the natural target is an out-of-process irradiance
  bake into EEVEE light probe volumes, not a renderer.
- Custom OIDN denoiser weights: superseded by OIDN 3 (temporal denoising,
  upscaling). DLSS Ray Reconstruction already exists in this tree
  (`intern/cycles/integrator/denoiser_dlss.cpp`, `WITH_DLSS`).
- Diffusion post-processing ("imagining detail"): belongs in a Python render
  handler that ships depth, normal and albedo passes to an external service as
  conditioning. Never in the kernel.

## EEVEE, in parallel

A **decoder node** whose network outputs Principled BSDF parameters (albedo,
roughness, metallic, normal, AO) needs no changes to EEVEE's G-buffer or
lighting and is the right first step. A true neural BRDF would need a new
G-buffer closure plus LTC, probe and ray-tracing support and is not worth it.
Blender's pinned glslang already supports `GL_NV_cooperative_vector`, but the
Vulkan device is created in GHOST without it; enabling it is a contained change
(`intern/ghost/intern/GHOST_ContextVK.cc`, `vk_device.cc`, `vk_backend.cc`)
plus a storage-buffer vehicle in material codegen. Cooperative matrix is
unusable in EEVEE's fragment-shader material passes. See
`03_eevee_gpu_module.md`.

## Ordered plan for agent workers

Each phase is the test harness for the next. Do not reorder.

1. **Infrastructure.** Build box with both cards.
   `CYCLES_CUDA_BINARIES_ARCH=sm_86;sm_89`, ccache, headless render tests
   (`tests/python/cycles_render_tests.py` pattern), an image-diff script,
   PyTorch and tiny-cuda-nn installed. This is the schedule lever.
2. **`SHADER_EVAL_BSDF` plus Python probe.** Dumps BSDF values, pdf and albedo
   for any Cycles material at chosen (object, prim, uv, wi, wo). Spec in
   `01_cycles_neural_closures_spec.md` section 6. Validates the build loop and
   feeds all material training.
3. **Neural stage skeleton.** New `DeviceKernel` with a query buffer, a model
   table, a scalar CPU fallback, and a tcnn forward on CUDA. Unit-tested
   against PyTorch golden vectors (there are no kernel math tests in
   `intern/cycles/test` today; `kernel_camera_projection_test.cpp` is the
   pattern). This phase deserves the most review attention.
4. **First client: neural BSDF closure.** Furnace test and MIS-consistency
   render are mandatory gates; the failure mode is a plausible-looking biased
   image.
5. **Second client: NRC** with tcnn online training. Viewport first, opt-in,
   control-variate mode as a stretch goal.
6. **EEVEE decoder node**, plain ALU first, then Vulkan cooperative vectors.

## Effort: human estimates versus agent-driven work

The specs quote human engineer-weeks. For agent-driven work those are wrong in
a specific way: code writing collapses 5 to 10x; build cycles, hardware
validation, training wall-clock and silent-bias hunting do not collapse at
all.

| Phase | Human estimate | Agent-driven, one GPU box, working build |
|---|---|---|
| `SHADER_EVAL_BSDF` + Python probe | 2 weeks | 2 to 3 days |
| Weights manager, node, sync | 2 weeks | 2 to 3 days |
| `mlp.h` + neural closure + tests | 3 weeks | 1 week, dominated by parity and MIS debugging |
| Neural stage + tcnn integration | 4 to 6 weeks | 1.5 to 2 weeks |
| EEVEE decoder node (plain ALU) | 7 to 9 weeks | 1 to 2 weeks |
| NRC surface-only MVP | 20 weeks | 4 to 6 weeks, mostly target validation |
| OptiX cooperative vectors module | 3 weeks | 1 week plus SDK and driver setup |
| Production-quality trained material | n/a | GPU hours per material; the pipeline is the work |

Budget in build-and-render iterations, not engineer-weeks. A full Blender build
is 30 to 60 minutes cold; an incremental Cycles kernel edit recompiles CUDA
kernels for 5 to 15 minutes per architecture without ccache.

## Training findings that constrain the design

From `04_experiment_results.md` (numpy, CPU, 2400 steps, so a floor on
quality):

- 2x32 decoders clearly beat 2x16 (relative L1 0.18 vs 0.24). Use 2x32 as the
  minimum.
- Both under-fit sharp specular peaks: the brightest 0.1% of targets reach 55
  to 64% of their energy. Peak handling needs either a larger net, a better
  encoding, or an analytic specular lobe kept outside the network (the 8DNA
  paper's direct/indirect separation is the same idea).
- The Bitterli culling schedule cut seed-to-seed variance 2 to 3x at equal
  compute but did not improve the mean in this short run. Treat it as a
  reliability tool for long GPU runs at 64k batch, not as an accuracy gain.
- Power loss (n=3) beat log(1+x). LeakySmeLU vs ReLU was inconclusive.
