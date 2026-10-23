# Neural BSDF and neural SSS in the Cycles kernel: implementation spec

Written against HEAD `afb395bd` (5.4 alpha, after the 5.3 merge). Nothing here
was built or run; there was no GPU or OptiX SDK on the investigating machine.
Everything about OptiX, Metal 4, HIP and SYCL matrix APIs is from memory and
must be checked against SDK headers. All Cycles paths and line numbers are
relative to `intern/cycles` unless noted.

Read together with `00_synthesis.md`: the synthesis moves the network
evaluation out of the closure and into a batched stage. The closure, weight
transport, proxy sampling and `SHADER_EVAL_BSDF` parts of this spec are needed
either way.

Verified locations:

- Closure IDs and the `CLOSURE_IS_*` macros are in `kernel/svm/types.h:499-600`.
- In `kernel/closure/bsdf.h`: `bsdf_get_specular_roughness_squared` :29,
  `bsdf_sample` :173, `bsdf_roughness_eta` :333, `bsdf_label` :442,
  `bsdf_eval` :554, `bsdf_blur` :665, `bsdf_albedo` :697.
- `ShaderEvalType` enum: `integrator/shader_eval.h:18-23`.
  `DEVICE_KERNEL_SHADER_EVAL_*`: `kernel/types.h:1843-1846`.

## 1. Weight and latent upload

**File format.** Define a `.nnm` file: header (topology, dims, activation id
with beta and epsilon, input-encoding id, output transform, latent
resolution/levels/channels, test vectors) plus raw fp16 tensors.

**Blender datablock.** Blender has no "neural material" datablock. For v1 put a
filepath on the node, using IES as the precedent:

- DNA `NodeShaderTexIES`: `source/blender/makesdna/DNA_node_types.h:2946`.
- RNA: `source/blender/makesrna/intern/rna_nodetree.cc:4148` and `:6862`.
- Free/copy: `source/blender/blenkernel/intern/node.cc:466`.
- Cycles sync: `blender/shader.cpp:1128`, using `blender_absolute_path`.

A real ID type (DNA, RNA, versioning, relink) would add 3 to 6 weeks. Not
recommended.

**Blender node.**

- Add `SH_NODE_BSDF_NEURAL 726` to
  `source/blender/blenkernel/BKE_node_legacy_types.hh` (last existing ID is
  725, line 144).
- Create `nodes/shader/nodes/node_shader_bsdf_neural.cc` by copying
  `node_shader_bsdf_sheen.cc`.
- Register in `node_shader_register.cc:40`, `node_shader_register.hh:36` and
  `nodes/shader/CMakeLists.txt:37`.
- EEVEE gets no `gpu_fn`, so the node is Cycles-only, gated by
  `object_cycles_shader_nodes_poll`.

**Cycles node.**

- Add `NeuralBsdfNode : BsdfNode`, copied from `SheenBsdfNode`
  (`scene/shader_nodes.h:642`, `shader_nodes.cpp:2596-2640`). `BsdfNode::compile`
  is at :2195.
- Add the typed SVM payload `SVMNodeNeuralBsdfData` next to
  `SVMNodeSimpleBsdfData` in `kernel/svm/node_types.h:~1050`.
- Add the sync branch next to `blender/shader.cpp:775`.

**Scene manager.** Add `NeuralManager` (`scene/neural.{h,cpp}`) by cloning the
IES pattern:

- Slots with hash and refcount: `LightManager::add_ies` (`light.cpp:1551`),
  `remove_ies` (:1592), `IESSlot` (`light.h:245`).
- The node calls `get_slot()` from `compile()` (`shader_nodes.cpp:1458-1477`).
- Upload packs an offset table plus blobs, as in `device_update_ies`
  (`light.cpp:1611-1657`).
- Add `NeuralFile::load/packed_size/pack` modelled on `util/ies.h:12-26`.
- Hook into Scene in `scene.cpp`: constructor :59-70, `device_free` :141-153,
  `device_update` after `shader_manager->device_update_pre` (:215) and near
  `image_manager->device_update` (:336), `need_update` check :490.

**Device storage.**

- In `scene/devicescene.h:~89` (constructor at `devicescene.cpp:58`), add
  `device_vector<half> neural_weights` and `neural_latents`, plus
  `device_vector<uint> neural_models`. Use `MEM_GLOBAL`, name equal to the
  array name (`device/memory.h:402`).
- Add matching `KERNEL_DATA_ARRAY(half, neural_weights)` lines at
  `kernel/data_arrays.h:86`.
- No backend code is needed; every backend expands `data_arrays.h`:
  `cpu/kernel.cpp:74-80`, `cuda/globals.h:29-31` with
  `cuda/device_impl.cpp:748-770`, `hip/device_impl.cpp:642-649`,
  `hiprt/device_impl.cpp:381-391`, `optix/globals.h:40-42`,
  `metal/globals.h:19-21` with `metal/device_impl.mm:1035-1041`,
  `oneapi/globals.h:28-30` with `oneapi/device_impl.cpp:618-665`.
  `device_type_traits<half>` exists (`memory.h:195`).

**Latents.**

- v1: a plain `half` array with manual bilinear sampling (one 16-byte load per
  texel for 8 channels). No ImageManager dependency, identical on every backend.
- v2, for huge or streamed latents, use the image system: an `ImageLoader`
  subclass (`scene/image_loader.h:33`), `image_manager->add_image(...)` as at
  `blender/shader.cpp:~950`, pass `handle.kernel_id()` into the SVM node as in
  `shader_nodes.cpp:465-475`, fetch with `kernel_image_interp`
  (`kernel/device/gpu/image.h:95`). Images are RGBA-only
  (`types_image.h:35-48`), so 8 channels means 2 images per level. They need
  Non-Color and alpha-ignore, and inherit the `SR_CACHE_MISS` retry behaviour.

**Backend constraints.**

- CUDA, OptiX and HIP: each array is an 8-byte pointer in `__constant__
  kernel_params`.
- OptiX with cooperative vectors needs weights converted to the
  inferencing-optimal layout with 64-byte alignment: a second device blob built
  in `OptiXDevice` (hook near `device_impl.cpp:2023`).
- Metal: arrays go in the `KernelParamsMetal` argument buffer, kept resident
  through a residency set (`device_impl.mm:174-186`).

## 2. `kernel/util/mlp.h`

**Weights.** fp16 in memory, fp32 accumulation (fp16 to fp32 is exact, keeps
backends close). **Input-major layout `W[in][out_pad8]`**, so the inner loop is
an axpy over outputs: vectorizes on CPU, and on GPU it is warp-uniform
broadcast `float4` loads. The benchmark in `04_experiment_results.md` showed
row-major loops are 3 to 8x slower and only vectorize with fast-math.

The `neural_models` table per model holds: dims and per-layer offsets;
activation id, beta and epsilon; input-encoding id and output transform (exp,
softplus or none); latent offset, width, height, levels and channels.

**Activation (LeakySmeLU).** Reconstructed from the Bitterli 2026 description
(quadratic transition for |x| < beta, slope epsilon outside); confirm against
the paper:

- `g(x) = 0` for `x <= -beta`, `x` for `x >= beta`, `(x+beta)^2/(4 beta)` between.
- `act(x) = eps * x + (1 - eps) * g(x)`. C1, slope eps on the left, 1 on the right.

**API.**

- `template<int W, int L> ccl_device_noinline void neural_mlp_eval(KernelGlobals kg, const ccl_global KernelNeuralModel &m, ccl_private const float *in, ccl_private float *out)`.
- Instantiate widths 16, 32 and 64 and select with a switch, so loop bounds are
  constants and fully unrolled.
- `float4` accumulators with constant indexing so they stay in registers.
- Access weights through `kernel_data_array(neural_weights)`.
- Input encoding fixed per model id; must match training exactly.

**Register-pressure risk** (estimates, not measured):

| Backend | Budget | Source |
|---|---|---|
| CUDA sm_70 to 12.x | 168 regs, 384 threads | `kernel/device/cuda/config.h:36-45` |
| CUDA sm_5x/6x | 48 to 64 regs, 256 threads | `config.h:16-26` |
| HIP | 1024 threads, macro value 64 | `kernel/device/hip/config.h:21-22` |

A 64-wide layer needs about 64 live accumulators plus about 16 temporaries. The
kernel already keeps a per-thread `ShaderData` with up to 64 closures of 80
bytes (`kernel/types.h:869`, `shade_surface.h:775`), so it is already spilling.
32-wide fits everywhere. Mitigations: `ccl_device_noinline` to isolate callee
register allocation (not on OptiX modules, whose `compat.h` force-inlines),
half2 accumulation, and gating everything behind a new `KERNEL_FEATURE_NEURAL`
(`1ULL<<33`, after GSPLATS at `kernel/features.h:~64`) and `__NEURAL__`, set in
`ShaderManager::get_kernel_features` near `scene/shader.cpp:648`. Measure with
`WITH_CYCLES_DEBUG` (adds ptxas `-v`, `kernel/device/cuda/CMakeLists.txt:193`).

Cost estimate: a 64x64x3-hidden network is about 16K MAC per eval, about 2
evals per bounce (NEE plus sampled direction), so about 32K FMA per path per
bounce. At 2M paths that is about 64 GFMA: roughly 1.6 ms ideal and 6 to 8 ms
realistic on an RTX 4090 for a scalar path. A 32-wide network is about 4x
cheaper. The measured CPU numbers in `04_experiment_results.md` are worse than
this estimate and are the reason the synthesis moves evaluation to a batched
stage.

## 3. Matrix-hardware options

**Shader sorting and coalescing.** The sort key is the shader index plus a
path-index partition (`state_flow.h:162-199`); partitions are about 65536 paths
when the scene has fewer than 300 shaders (`cuda/queue.cpp:51`,
`metal/queue.mm:290`, `oneapi/queue.cpp:61`). Sorting applies only to
`SHADE_SURFACE` and `SHADE_SURFACE_RAYTRACE` (`path_trace_work_gpu.cpp:1347`,
`:647-707`). Lanes in a warp mostly share a shader, hence a model, assuming one
neural node per shader. Lanes are not convergent though: the key is the shader,
not the closure (a Mix Shader with a texture factor changes closure sets per
lane; weight cutoff in `closure/alloc.h`); `surface_shader_bsdf_bssrdf_pick`
(`surface_shader.h:441`) makes lanes sample different closures;
`integrate_surface` has early-outs before direct light
(`shade_surface.h:~810-836`); one warp of straddle per shader bucket per
partition.

Every warp-synchronous matrix instruction therefore cannot be called safely
from today's control flow. Three ways out:

1. A per-thread API that tolerates divergence. Only OptiX cooperative vectors
   qualify.
2. A wavefront "neural eval" stage with a new `DeviceKernel` and new state
   fields. `SHADE_SURFACE_RAYTRACE` has about 55 touch points, which is the
   cost precedent. 4 to 6 human weeks. **This is the chosen design.**
3. Compact lanes with ballot or `__match_any_sync` so the full warp
   participates, with zero-fed inactive lanes. Fragile.

**(a) OptiX cooperative vectors (`optixCoopVec*`): reachable with caveats.**

- Nothing in CMake enforces a minimum OptiX version (`FindOptiX.cmake:38-46`
  only reads it). Buildbot pins 9.0.0
  (`build_files/buildbot/config/blender_version.cmake:6`). Code needs at least
  7.7 (`optixModuleCreateWithTasks`, `device/optix/device_impl.cpp:223`). The
  add-on requires NVIDIA driver 580 (`blender/addon/properties.py:1957`).
- Developer builds need an `OPTIX_VERSION >= 90000` guard and a CMake define.
- The OptiX kernels are separate compilation units:
  `kernel/device/optix/CMakeLists.txt:43-47` builds `kernel.cu`,
  `kernel_mnee.cu` and `kernel_shader_raytrace.cu` as separate modules.
- The SVM `shade_*` kernels are not in any OptiX module. They run as plain CUDA
  kernels (`device/optix/queue.cpp:39,57`). Only `intersect_*` and
  `SHADE_SURFACE_RAYTRACE` are raygens (`device/kernel.cpp:30-39`).
  `__KERNEL_OPTIX__` is defined in `optix/compat.h:10`.
- Template to copy: `kernel_shader_raytrace.cu`, which includes
  `shade_surface.h` and is loaded only when the feature is on
  (`device_impl.cpp:282, 737-740`). A neural OptiX variant needs a new module
  `kernel_optix_neural.cu` plus `PG_RGEN_SHADE_SURFACE_NEURAL` plus dispatch in
  `queue.cpp:116-122`. About 300 to 400 lines.
- Arch trap: `foreach(_arch compute_50 compute_75)` at
  `optix/CMakeLists.txt:47` picks compute_50 first. The neural module needs
  compute_75 or newer. Both pinned toolkits (CUDA 12.9.1 and 13.4.1) are above
  the 12.8 cooperative vectors are believed to require.
- SSS is easier: `intersect_subsurface` is already a raygen
  (`optix/kernel.cu:~38`), so cooperative vectors work there without a new
  pipeline.
- The API is per-thread, tolerates divergence, and is fastest when active
  lanes share a matrix, which sorting approximates.

**(b) CUDA `mma.sync` / wmma via inline PTX: reachable in `cuda/kernel.cu`.**
Also serves OptiX devices, since they run shading as CUDA kernels. CUDA kernels
are compiled for sm_50, 52, 60, 70, 75, 86, 120 plus compute_75 PTX
(`CMakeLists.txt:679-681`). `mma.m16n8k16` needs sm_80, so it needs `#if
__CUDA_ARCH__ >= 800` with a scalar fallback. For this fork, restrict the arch
list to sm_86 and sm_89 and the guard becomes moot. Needs convergent warps, so
it depends on the stage.

**(c) HIP WMMA: partial.** RDNA3 and RDNA4 only, via clang builtins. Out of
scope for this fork.

**(d) Metal: partial.** `simdgroup_matrix` is reachable today under
`__KERNEL_METAL_APPLE__`; Metal 4 `tensor_ops` needs MSL 4.0 and Cycles caps at
`MTLLanguageVersion3_2` (`device/metal/device_impl.mm:560-568`). Out of scope.

**(e) oneAPI `joint_matrix`: experimental.** Out of scope.

**CPU.** AVX2 and F16C are available in the baseline and `kernel_avx2.cpp`
variants (`kernel/device/cpu/CMakeLists.txt:16,42`). The input-major axpy form
auto-vectorizes, but note from the benchmark: Cycles compiles without
`-ffast-math`, and GCC 13 treats `-ffp-contract=on` as off, so plain `a*b+c`
loops get no FMA; use explicit FMA intrinsics or Cycles' `float8` type.

## 4. Neural BSDF closure

**Closure ID.** Insert `CLOSURE_BSDF_NEURAL_ID` after
`CLOSURE_BSDF_GLOSSY_TOON_ID` (`kernel/svm/types.h:524`). That is inside
`CLOSURE_IS_BSDF` (:570) and `CLOSURE_IS_BSDF_GLOSSY` (:573), outside the
microfacet range. Light passes will count all neural energy as glossy
(`bsdf_eval_accum`, `film/light_passes.h:64`); documented limitation.

**Struct** (new file `kernel/closure/bsdf_neural.h`):
`NeuralBsdf { SHADER_CLOSURE_BASE; float3 T; float alpha_x, alpha_y, p_diffuse; uint model; ccl_private NeuralData *data; }`.
`data` points to a `closure_alloc_extra` slot (`closure/alloc.h:~40`) holding
latent `z[8..16]` as half, the albedo and a frame rotation; same pattern as
`MicrofacetBsdf::fresnel` (`bsdf_microfacet.h:106-123`). Add
`static_assert(sizeof(ShaderClosure) >= sizeof(NeuralBsdf))`.

**Setup and SVM wiring.** `bsdf_neural_setup(kg, sd, weight, N, T, model, uv)`
using `bsdf_alloc` (`closure/alloc.h:~65`): sample the latent pyramid; run a
tiny head MLP `z -> {albedo_rgb, alpha_x, alpha_y, p_d}` once per shading
point; clamp alpha to at least 1e-3 and `p_d` to at least 0.1; set
`sample_weight *= average(albedo)` as the NOTE at `bsdf.h:704-708` requires;
set `sd->runtime_flag |= SR_BSDF | SR_BSDF_HAS_EVAL` (not
`SR_BSDF_HAS_TRANSMISSION`). In `kernel/svm/closure.h` add a case in
`svm_node_closure_bsdf` (:165, Sheen case at :813) and an offset case in
`svm_node_closure_bsdf_skip` (:29).

**eval** (`bsdf_neural_eval`, case in `bsdf_eval` at `bsdf.h:554`): build the
frame from `make_orthonormals_tangent(N, T)`; reject `cosNI <= 0` or `cosNO <=
0`; input `[z, wi_local, wo_local, encoding]`, MLP outputs RGB; return
`rgb_to_spectrum(f) * cosNO` (cosine included, as `bsdf_diffuse_eval`,
`bsdf_diffuse.h:38-49`; `rgb_to_spectrum` at `kernel/util/colorspace.h:40`).
pdf is `p_d * cosNO / pi + (1 - p_d) * pdf_ggx`; for `pdf_ggx` fill a
stack-local `MicrofacetBsdf` proxy with `fresnel_type = NONE` and call
`bsdf_microfacet_eval<MicrofacetType::GGX>` (`bsdf_microfacet.h:836`), keeping
only its pdf. In the staged design, `eval` enqueues the query and the stage
writes the value back; the pdf is computed inline since it only depends on the
proxy.

**sample** (`bsdf_neural_sample`, case in `bsdf_sample` at `bsdf.h:173`): pick
a lobe with `rand.z` and rescale; diffuse lobe `sample_cos_hemisphere`; glossy
lobe `microfacet_ggx_sample_vndf` (`bsdf_microfacet.h:304`) and reflect;
evaluate the network at the sampled `wo`; return the mixture pdf,
`*sampled_roughness = (alpha_x, alpha_y)`, `*eta = 1`. Label `LABEL_REFLECT |
(p_d > 0.5 ? LABEL_DIFFUSE : LABEL_GLOSSY)`, deterministic per closure because
`bsdf_label` (`bsdf.h:442`) is asserted against the sample label. Also add cases
to `bsdf_roughness_eta` (:333) and `bsdf_blur` (:665).

**MIS.** `_surface_shader_bsdf_eval_mis` (`surface_shader.h:277`) needs no
change, provided the reported pdf is the real pdf of the proxy sampling
procedure. The cosine lobe floor (`p_d >= 0.1`) keeps full-hemisphere support.

**Passes and light tree.** `bsdf_albedo` (`bsdf.h:697`) returns the stored
albedo; it feeds `closure_albedo` for the denoising-albedo pass
(`film/denoising_passes.h:100`) and the diffuse/glossy colour passes
(`surface_shader.h:1073-1100`). `bsdf_get_specular_roughness_squared`
(`bsdf.h:29`) returns `alpha_x * alpha_y`. The light tree reads only
`SR_BSDF_HAS_TRANSMISSION` (`light/tree.h:756`); the shadow catcher has no
closure references. Train the albedo head jointly, targeting the integral of
f cos over the hemisphere, clamped to at most 1.

## 5. Neural SSS

**Where the random walk is entered.** `integrate_surface_bsdf_bssrdf_bounce`
(`shade_surface.h:497-516`) calls `subsurface_bounce` (`subsurface.h:58-118`),
which sets `PATH_RAY_SUBSURFACE_RANDOM_WALK` and writes the `subsurface.*`
state (`state_template.h:113-123`). The path goes to `INTERSECT_SUBSURFACE`
(`integrator_shade_surface_next_kernel`, `shade_surface.h:~908`);
`integrator_intersect_subsurface` (`intersect_subsurface.h:11-23`) calls
`subsurface_scatter` (`subsurface.h:~150`), which dispatches to
`subsurface_random_walk` (`subsurface_random_walk.h:202-460`) or
`subsurface_disk` (`subsurface_disk.h:36-215`). On success it rewrites the ray
at the exit point and queues `SHADE_SURFACE` sorted by the exit shader;
`subsurface_shader_data_setup` (`subsurface.h:~128`) installs a Diffuse BSDF at
the exit.

**What to replace.** Model `subsurface_neural()` on `subsurface_disk`, not on
the random walk: it is the only existing probe-ray exit sampler with multi-hit
importance resampling. Keep the 3-axis MIS (:75-124), `scene_intersect_local`,
the hit-resampling loop (:150-190) and the exit handling. Swap two pieces:
`bssrdf_sample` (radius sampling) becomes a sampler for a mixture of 2D
Gaussians in the tangent plane; `subsurface_disk_eval` (:30-36) becomes
`network_value / mixture_pdf`. Network input: probe-ray distances (K of about 8
to 16 via `scene_intersect_local`), `albedo`, `radius`, anisotropy `g`, `ior`,
local `wi` (`Bssrdf` fields at `closure/bssrdf.h:14-24`). Outputs: mixture
parameters plus an RGB throughput scale. This is a reading of Vicini 2019 from
memory; verify against the paper. The model is parametric in the material, so
one global model ships with Blender.

**Minimal change list.**

1. Add `CLOSURE_BSSRDF_NEURAL_ID` after `kernel/svm/types.h:548`. Widen
   `CLOSURE_IS_BSDF_OR_BSSRDF` (:591) and `CLOSURE_IS_BSSRDF` (:593).
2. Closure-setup case beside `RANDOM_WALK` (`svm/closure.h:1057-1077`) calling
   `bssrdf_setup` (`bssrdf.h:288`).
3. New `PATH_RAY_SUBSURFACE_NEURAL` bit (`types.h:258-262`). Branch in
   `subsurface_bounce` and `subsurface_scatter`.
4. New file `kernel/integrator/subsurface_neural.h`.
5. Host node or a Principled `subsurface_method` enum value.
6. Optionally sort `INTERSECT_SUBSURFACE` by shader (`kernel_uses_sorting`,
   `path_trace_work_gpu.cpp:1347`; `alloc_integrator_sorting` :258-300), about
   40 lines.

Training-data generation (volumetric slab and shape simulation) and training
are the largest cost, 6 to 10 human weeks, not included below.

## 6. `SHADER_EVAL_BSDF` training-data mode

**Input encoding.** `KernelShaderEvalInput` is 16 bytes (`int object; int
prim; float u, v`, `kernel/types.h:1771`). Use stride 3, as volume density does
with stride 2 (`kernel/bake/bake.h:145`, `octree.cpp:235-238`). Record 0 is
`{object, prim, u, v}`; records 1 and 2 carry `wi`, `wo` and a flags word via
`__float_as_int`. One flag means "directions are in the local frame".

**Kernel `kernel_bsdf_evaluate`** (in `kernel/bake/bake.h` after :143): set up
with `shader_setup_from_sample(..., I = -wi, ...)` (`geom/shader_data.h:258-290`
template) so backfacing is correct; call `surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE & ~(RAYTRACE | LIGHT_PATH)>`
with `PATH_RAY_VISIBILITY_CAMERA` (`surface_shader.h:1254`); call
`surface_shader_bsdf_eval(kg, state, &sd, wo, &ev, SHADER_USE_MIS, r)` (:389).
Output 8 channels: `bsdf_eval_sum` RGB (includes cosine), pdf, and albedo RGB
from `surface_shader_diffuse/glossy/transmission`.

**Registration points** (verified by grep for `SHADER_EVAL_VOLUME_DENSITY`):

- `kernel/types.h:1846`: new enum value after it.
- `device/kernel.cpp:21-28` (`device_kernel_has_shading`) and :120-126 (name).
- `kernel/device/gpu/kernel.h:~1007`: copy of the volume-density kernel. CUDA
  and HIP pick it up by enum (`device/cuda/kernel.cpp:16`).
- CPU: `kernel/device/cpu/kernel_arch.h:73-90`, `kernel_arch_impl.h:80-160`,
  `device/cpu/kernel.h:40-43`, `device/cpu/kernel.cpp:24-27`.
- oneAPI: `kernel/device/oneapi/kernel.cpp:626`, `device/oneapi/device_impl.cpp:1397`.
- Metal: range check at `device/metal/kernel.mm:426`.
- Non-OSL OptiX needs nothing (runs through the CUDA queue). OSL OptiX needs
  about 30 more lines; skip.
- `integrator/shader_eval.h:18-23` and `shader_eval.cpp` (switches at :110-122
  and :138-148).

**Python exposure.** A session needs a `RenderEngine` pointer:
`_cycles.create` takes `engine.as_pointer()` (`blender/addon/engine.py:59-75`).
Add `BlenderSession::eval_bsdf`, modelled on `BlenderSession::bake`
(`blender/session.cpp:668-767`): sync, `scene->update(progress)`, run
`ShaderEval::eval`; do not call `session->start()`. Add `eval_bsdf_func` in
`blender/python.cpp` near `bake_func` (:316) and the methods table (:934); pass
numpy buffers as `ctypes.data` integer pointers (file has no `Py_buffer`
precedent). Add the `engine.py` wrapper near :115. Practical driver: a thin
add-on `RenderEngine` subclass whose `render()` calls `engine.create`,
`_cycles.sync` and `_cycles.eval_bsdf`, run with `blender -b -E <engine> -f 1`.
Whether Cycles' session setup accepts a non-Cycles engine pointer is unverified
and needs a half-day spike.

**Estimate.** About 400 to 450 lines excluding tests.

## 7. Plan and risks (human engineer-weeks; see `00_synthesis.md` for the agent recalibration)

| Phase | Work | Weeks |
|---|---|---|
| P0 | Ramp-up, `.nnm` format, PyTorch reference and golden vectors | 1.0 |
| P1 | `SHADER_EVAL_BSDF` and the Python probe | 2.0 |
| P2 | `NeuralManager`, arrays, Blender and Cycles node, sync | 2.0 |
| P3 | `mlp.h`, neural closure, tests (CPU unit test modelled on `test/kernel_camera_projection_test.cpp`; furnace and MIS-consistency renders) | 3.0 |
| P4 | Register and perf tuning, kernel-feature gating | 1.5 |
| P5 | Neural SSS, assuming a trained model exists | 3.0 |
| P6 | OptiX coopvec variant (new module, raygen, dispatch) | 3.0 |
| P7 | Wavefront neural-eval stage, then matrix paths | 4 to 6 for the stage, plus 2 to 4 per backend |

Top 5 risks:

1. **Fidelity and numerics.** The kernel must reproduce the training inference.
   Fast-math is on in the GPU builds (nvcc `--use_fast_math` in
   `cuda/CMakeLists.txt:187`). Half-precision weights, `exp` and `softplus` can
   drift. Clamp outputs with `ensure_finite`.
2. **Register pressure and occupancy.** The generic CUDA kernels are not
   feature-specialized, so scenes without neural materials may pay for it.
   64-wide networks risk spills.
3. **Tensor cores need the wavefront stage.** Warp-cooperative matrix
   instructions cannot be called from the divergent shading code.
4. **Estimator correctness.** The network is not energy-conserving or
   reciprocal, and the albedo head must match `sample_weight`. The neural SSS
   exit density must be a true normalized pdf under multi-hit resampling.
5. **Toolchain and scope.** OptiX 9.0 and driver R570+ for cooperative vectors;
   its PTX is built for compute_50 by default. The Blender-core DNA/RNA/node
   work is Cycles-only until EEVEE gets its own node implementation.
