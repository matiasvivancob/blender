# EEVEE and the GPU module: running small neural networks in material shaders

Written against HEAD `afb395bd`. Paths are relative to `source/blender` unless
they start with `intern/` or `build_files/`. Read-only investigation; nothing
built.

## Verdict

- A plain-ALU decoder node is feasible in EEVEE with no changes to the closure
  pipeline. The network outputs Principled parameters.
- Cooperative vectors are feasible only on NVIDIA Vulkan, and only after
  changes in GHOST, the Vulkan backend and material codegen. Cooperative
  matrix is a poor fit, because every EEVEE material pass is a fragment
  shader.
- A Python add-on can prototype the decoder today. It can only bake results
  into an Image. It cannot feed materials or use cooperative hardware.

## 1. Material codegen and the weight vehicle

**How a node becomes a function call.** A node's `ntype.gpu_fn` calls
`GPU_stack_link(mat, node, "fn", in, out, extra links)`. Example:
`nodes/shader/nodes/node_shader_tex_image.cc:42` and :113-118 pass
`GPU_image(...)` and `GPU_kernel_globals()`. `GPU_stack_link`
(`gpu/intern/gpu_node_graph.cc:1061`, impl :987) looks the function up in the
`g_functions` library (`gpu_shader_dependency.cc:638`) and builds a `GPUNode`.
`GPUCodegen::node_serialize` (`gpu_codegen.cc:339-489`) prints
`fn(tmpN, node_tree.uN, sampN, kg, ...);` and records which library file is
needed. A `[[node]]` function can only take these types
(`gpu_shader_dependency.cc:157-190`): float/vec/mat3/mat4, int, bool,
sampler1DArray/2DArray/2D/3D, Closure, KernelGlobals and ShadingData, at most
39 parameters (`gpu_material_library.hh:18`). Buffers and arrays are not
accepted.

**What codegen can bind** (`gpu_codegen.cc:257-337`): samplers only, assigned
slots at `Frequency::BATCH`; the NodeTree UBO at slot 0
(`GPU_NODE_TREE_UBO_SLOT`, `GPU_uniform_buffer.hh:62`) and the UniformAttrs
UBO at slot 2. There is no storage-buffer support in material codegen.

**How existing nodes pass data.**

- Image Texture (`GPU_image`, `gpu_node_graph.cc:888`) creates a
  `GPUMaterialTexture` (`gpu/GPU_material.hh:363`), bound in
  `PassBase::material_set` (`draw/intern/draw_pass.hh:1130-1193`), in
  `compositor/intern/shader_operation.cc:94-100,1202` and in
  `blenkernel/intern/image.cc:3120`.
- Color ramp and curves (`node_shader_color_ramp.cc:99`): `GPU_color_band`
  (`gpu_node_graph.cc:938`) copies 257 RGBA values into the material's builder
  (`gpu_material.cc:475`), which becomes a 1D-array RGBA16F texture
  (`gpu_material.cc:505`) bound at `draw_pass.hh:1179`. The sky texture does
  the same with RGBA32F at a fixed size (`gpu_material.cc:446,525`).
- Uniforms: `GPU_uniform` fills the NodeTree UBO. Two hazards: if the UBO is
  larger than `GPU_max_uniform_buffer_size()` it silently stays null
  (`gpu_uniform_buffer.cc:331`); and `gpu_node_graph_optimize`
  (`gpu_node_graph.cc:1299`) bakes every uniform into shader-source constants.
  Weights must never go through the uniform path.

**Ranking for 2k to 50k weights.**

1. **Texture, following the color-band/sky pattern.** New `GPUNodeLinkType`
   and builder (RGBA32F, read with `texelFetch`). Works on GL, Vulkan and
   Metal; EEVEE's `SlotAllocator` (`eevee_shader.cc:595-690`) allocates sampler
   slots automatically. Cannot feed cooperative vectors, which read buffer
   memory. About 1 human week.
2. **Storage buffer.** Needed for cooperative vectors and large nets. Needs a
   link type, `info.storage_buf(slot, Qualifier::read, ..., Frequency::BATCH)`
   in `generate_resources`, and binding in `material_set`. `SlotAllocator`
   only manages samplers, so storage-buffer slots must be reserved by hand.
3. **UBO.** Only a 2x16 net (about 3 KB) would fit comfortably. Not recommended.
4. **EEVEE-owned weight atlas.** A `[[resource_table]] [[condition(...)]]`
   entry in `KernelGlobals` (`eevee_nodetree_lib.bsl.hh:141-176`), the way
   `use_raycast` and `use_ambient_occlusion` work, driven by `GPU_MATFLAG_*`
   compile constants (`eevee_shader.cc:~1040-1090`). The node passes an int
   offset. No codegen changes and a storage buffer directly, but EEVEE-only.

**Latent texture.** Use `GPU_image()` unchanged, with hardware bilinear for the
latent interpolation. Float images set to Non-Color with channel-packed alpha.
More than 4 latent channels need several images, each costing a sampler.

## 2. Shader library and the BSL dialect

**Registering `gpu_shader_material_neural.bsl.hh`.** Add to `GLSL_SRC` in
`gpu/CMakeLists.txt` (material block :621-742); add an `#include` in
`gpu/shaders/nodes/bsl_shader_linting.cc` (pattern at :92, :142). The filename
must start with `gpu_shader_material_`; function names must be globally unique
(`gpu_shader_dependency.cc:~318-335`). At build time `glsl_to_c`
(`build_files/cmake/macros.cmake:1143`) runs `shader_tool`, which emits lowered
source, a metadata `.hh` and an `.info` file; `datatoc` embeds the result and
`glsl_gpu_source_list.h` (`gpu/CMakeLists.txt:854-885`) registers it. The
node's C++ side is a `bke::bNodeType` with `gpu_fn`, plus DNA/RNA/declaration.

**Dialect constraints.** C++-like source lowered by `shader_tool`
(`shader_tool/processor.hh:227-350`) to GLSL or MSL. Supports templates,
namespaces, classes, enums, `[[unroll]]`, `[[unroll_n(N)]]`,
`[[static_branch]]`, `[[storage(slot,qual)]]`, `[[resource_table]]`, `[[node]]`
and `[[capacity]]` (`gpu/shaders/compat/gpu_shader_cxx_attribute.hh:100-170`).
Builtins come only from C++ stubs in `gpu_shader_cxx_builtin.hh`: `fma`,
`packHalf2x16`, `unpackHalf2x16` exist (:73, :242-247); there is no subgroup,
coopmat, coopvec or float16 support, and `half` is `#define`d to `float`
(`gpu_shader_compat_glsl.glsl:93`). Any new builtin needs a C++ stub,
`gpu_shader_compat_glsl.glsl` and the MSL compat files. Unverified: whether
constant arrays declared inside function bodies lower correctly on both GLSL
and MSL.

## 3. EEVEE closures and lighting

**What EEVEE does with a closure.** The material pass `surf_deferred` is a
fragment shader with early-z (`eevee_surf_deferred.bsl.hh:81-82`).
`nodetree_surface` (:127) picks up to 3 closures stochastically through
`closure_bins` and `Reservoir` (`eevee_nodetree_closures_lib.bsl.hh:13`).
`ClosureUndetermined` is `{color, N, type, vec4 data}`
(`gpu/shaders/gpu_shader_codegen_lib.glsl:110`); `ClosureType` ids are 4-bit,
15 is the last used (:83). These are packed into the G-buffer:
`GBUFFER_LAYER_MAX=3`, a 4-bit mode (11 of 16 used,
`eevee_gbuffer_types.bsl.hh:~55-76`), 10-10-10-2 UNORM layers. Lighting is a
separate fullscreen **fragment** pass, `light_eval_frag`
(`eevee_deferred_eval.bsl.hh:114-174`): per closure `closure_light_new`
(`eevee_closure.bsl.hh:101`), then per light an LTC evaluation
(`eevee_light_eval.bsl.hh:44-58`). Probes, ray tracing and denoising also
depend on closure type and roughness (`eevee_closure.bsl.hh:11,32,64`). A
neural BRDF has no LTC fit, no directional-albedo LUT, no pdf or sampler for
ray tracing, and no lightprobe lobe.

**Options.**

- (i) **Decoder only.** The network outputs albedo, roughness, F0/metallic,
  normal and AO into the existing Principled BSDF. Nothing changes in the
  G-buffer, lighting, ray tracing or SSS. NTC decode is exactly this.
- (ii) **True neural BRDF.** A new closure carrying the latent in the G-buffer
  is large work with precision problems (10-bit UNORM, 3 bins). A cheaper
  route exists in this tree: the new lighting nodes (Light Iteration, Light
  Evaluation, Shadow Raycast, Light Accumulation; `GPU_MATFLAG_LIGHTING`) route
  the material to the hybrid forward-lit pass (`eevee_pipeline.hh:289-291`,
  `eevee_surf_hybrid.bsl.hh`). A neural-BRDF `[[node]]` evaluated per light
  inside that loop needs no new closure, at the cost of forward-style shading
  with direct light only.

**Recommendation:** ship (i) first. Treat (ii) via lighting nodes as a later
experiment. Do not add a G-buffer closure.

## 4. Vulkan and Metal

**(i) Enabling extensions is not a local change.** The device is created in
GHOST (`intern/ghost/intern/GHOST_ContextVK.cc`): optional-extension list at
:1924-1950; feature chain in `create_device` (:558; features :630; chain
linked :813), following the `VK_KHR_ray_query` pattern at :806.
`VKDevice::init` takes the handles (`vk_device.cc:~160`). `volk` already
exposes `vkConvertCooperativeVectorMatrixNV` and
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`
(`extern/volk/volk.h:1202,1644`); Vulkan headers are 1.4.341. Needed changes:
GHOST extension and feature struct; a flag in `VKExtensions`
(`vk_device.hh:~34`); detection in `detect_workarounds` (`vk_backend.cc:502`);
a properties query (`vk_device.cc:201`) and a `GCaps` entry
(`vk_backend.cc:865`); a GLSL patch (`vk_device.cc:291-333`). Ray query is the
complete template: `GCaps.ray_query_support` (`vk_backend.cc:877`),
`BuiltinBits::RAY_QUERY` (`gpu_shader_create_info.hh:513,1446`), a Vulkan patch
with `#version 460`, `#extension` and `#define GPU_EXT_RAY_QUERY`, a Metal
define (`mtl_shader.mm:269-306`), skipping unsupported static shaders
(`gpu_shader_create_info.cc:873`). Instance and VMA API version is 1.1
(`vk_backend.cc:246`, `GHOST_ContextVK.cc:325`); both extensions only require
1.1 plus `VK_KHR_get_physical_device_properties2`. No fp16 features
(`shaderFloat16`, 16-bit storage) are enabled today.

**(ii) The compiler path supports both extensions.** Shaders go GLSL to SPIR-V
at runtime through shaderc (`vk_shader_compiler.cc:~330-420`), target Vulkan
1.1/SPIR-V 1.3 or 1.2/1.4 when available (:362-368), `-O performance` runs
spirv-opt. `build_files/build_environment/cmake/versions.cmake:1339-1352` pins
shaderc v2025.4 and glslang `d213562e` (at least glslang 15.4.0), whose
`Versions.cpp` registers both `GL_KHR_cooperative_matrix` and
`GL_NV_cooperative_vector` (:272-273). The NV extension does not predefine a
macro, so gate on a Blender-defined macro. Unverified: whether the bundled
spirv-opt handles these ops; `do_optimize=false` is the escape hatch (:~371).

**(iii) Fragment versus compute.** Every material-evaluating EEVEE pass is
`[[fragment]]`: deferred, forward, hybrid, depth, shadow, capture, world and
volume (`eevee_surf_*.bsl.hh`). Only support passes (ray tracing, denoise,
culling, film) are compute. Cooperative vector is specified for all stages
subject to `cooperativeVectorSupportedStages`, driver 570+. Cooperative matrix
needs subgroup-uniform execution, which does not hold in a divergent G-buffer
fragment shader; using it would mean a compute pre-pass decoding into a tile
cache (NTC's "inference on feedback").

**Metal.** Shader options reach at most `MTLLanguageVersion3_1`
(`mtl_shader.mm:213-240`); no Metal 4 or tensor-op path. Minimum macOS is 13.0
(`platform_apple_xcode.cmake:161`). Out of scope for this fork.

## 5. Python prototype

**Reachable today** (`python/gpu/`): `GPUShaderCreateInfo.compute_source`,
`local_group_size`, `storage_buf`, `image`, `sampler`, `typedef_source`,
`define`; `GPUStorageBuf(data)` with `shader.storage_block`; `GPUTexture`
1D/2D/3D and arrays in RGBA16F/RGBA32F; `gpu.compute.dispatch`; `gpu.init()`
for background use (`gpu_py_api.cc:44`). `doc/python_api/examples/gpu.11.py`
is a working compute-to-texture example. An add-on can load latents and
weights, dispatch an ALU MLP, call `GPUTexture.read()` and write the result
into an Image through the buffer protocol (`gpu_py_buffer.cc:678`). This is
NTC "inference on load" and works for both EEVEE and Cycles.

**Not reachable:** cooperative vector and matrix (device created without
them); feeding a GPUTexture or storage buffer into a material node (EEVEE and
Cycles only see Image datablocks); per-sample decoding inside material
shaders.

## 6. Plan, effort and risks (human weeks, about +/-40%)

1. Format and Python reference decoder with golden outputs: 1.5.
2. Node, DNA/RNA, UI and versioning: 1.5. Sockets: UV in; Color, Roughness,
   Metallic, Normal, AO, Alpha out.
3. GPU-module texture vehicle: 1. New link type and builder, binding in
   `material_set` and the compositor.
4. BSL ALU MLP node: 1.5. Latent taps (manual 4-tap bilinear or stochastic
   texture filtering), activations, `unpackHalf2x16` fp16 weights.
5. EEVEE integration and render tests: 1. `tests/python/eevee_render_tests.py`.
6. Cooperative vector on Vulkan: 3 to 4. GHOST extension and features; backend
   flags, properties, `GCaps`; GLSL patch and builtin bit; host-side
   `vkConvertCooperativeVectorMatrixNV`; storage-buffer plumbing with EEVEE
   slot allocation; spirv-opt validation and hardware testing.

**ALU fallback.** Step 4 is the permanent default; cooperative vector is a
capability-gated variant chosen at codegen time.

**Speedup of cooperative vectors.** The NVIDIA RTXNTC README states a 2 to 4x
inference throughput gain on Ada and Blackwell over optimal non-extension
implementations. No cited number exists for a 2x32 MLP versus naive fp32
GLSL. Estimate for a 2x32 net: about 1.8k MAC per sample, so about 0.2 to 0.5
ms ideal at 1080p; realistic plain-ALU cost around 1 to 3 ms on a discrete
GPU.

**Top 5 risks.**

1. **Filtering.** NTC decode is unfiltered point data; NVIDIA pairs it with
   stochastic texture filtering. EEVEE relies on hardware `textureGrad` and
   anisotropic filtering (`gpu_shader_material_tex_image.bsl.hh:~66-80`).
2. **Shader compile time and size.** An unrolled MLP multiplied across about 8
   EEVEE pipeline variants and the geometry types.
3. **Vendor reach and testability.** Cooperative vector is NVIDIA-only; CI
   lacks the hardware; parity between ALU and cooperative-vector results is not
   guaranteed.
4. **Plumbing and data.** Resources are bound in three places; UBOs fail
   silently when oversized; SSBO slot allocation is unmanaged in EEVEE;
   embedded weights and latents grow `.blend` size and undo memory.
5. **Scope.** New BSL builtins need module-team sign-off upstream; a new DNA
   node type needs EEVEE/Cycles parity; training stays external.
