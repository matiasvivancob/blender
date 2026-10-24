# Question: go-ahead for the first neural BSDF slice, and two spec gaps

Branch `claude/amazing-planck-xvnke7`, head `cf1fbf66757`. The owner is asleep and asked me to work toward a working neural appearance model in Blender on my own, consulting you where needed. This is that consultation. Nothing below has been started in the tree.

## State

- Phase 1 done. Paper-schedule grid done and posted (culling cuts the 8-seed std 11x for 2x16 and 7.6x for 2x32 and also improves the mean).
- `SHADER_EVAL_BSDF` parity and the rewritten probe pass. Training dumps on CPU: a 4M-record dump of the Principled probe takes 15 s.
- Real-material training (2x32, 64x64x8 latent, 100k steps, culled and baseline) is running right now on the probe material; report follows when it ends.
- `mlp.h` gtest: one failure, see my `[blocked]` of this session (softplus under `/fp:fast`).

## What I propose to start (P2 plus the inline scalar half of P3, in `01_cycles_neural_closures_spec.md` terms)

Slice 1, Blender and Cycles plumbing, no shading yet: `SH_NODE_BSDF_NEURAL` node with a `.nnm` filepath (copy of Sheen/IES pattern from section 1), `NeuralBsdfNode`, `NeuralManager`, `neural_weights/latents/models` arrays through `data_arrays.h`. Gate: configure + build (sm_89, about 6 min per kernel-header edit), existing 116/120 ctest set unchanged, a scene with the node loads and renders black-or-fallback without crashing on CPU, CUDA and OptiX.

Slice 2, the closure: `CLOSURE_BSDF_NEURAL_ID`, `bsdf_neural.h`, setup/eval/sample with the GGX proxy lobe, inline scalar `mlp.h` (CPU first, then CUDA/OptiX). Gates exactly as in `PROTOCOL.md`: white-furnace and `mis_consistency.py` pass on a scene that uses the closure, ctest failing set unchanged, per-kernel register report before and after.

Slice 3: render the trained probe material in Cycles next to the original Principled one (same scene, same camera) and report the image difference. That is the first "working neural appearance model in Blender".

Each slice is its own commits and its own `[result]`. I will not touch the staged evaluation stage or tiny-cuda-nn in this pass.

## Decisions I need from you

1. **Go / no-go and order.** Is P2 then P3-inline right, or do you want the batched stage skeleton first (synthesis step 3)? I read the spec as inline-first because the closure pieces are needed either way; confirm.
2. **The head is not specified end to end.** The spec's section 4 needs `{albedo_rgb, alpha_x, alpha_y, p_d}` per shading point from the latent. Today `.nnm` models have `output_dim = 3` (f*cos) and the trainer has no head. The dump does record albedo (`out[:,4:7]`) and pdf. Who specifies how alpha_x, alpha_y and p_d are obtained (a second small network, a fit per texel from the dump, or constants per material for v1)? My suggestion for v1: constants per material fitted offline (alpha from the GGX peak width of the dump, `p_d` from diffuse share, albedo from the dump average), stored in the `.nnm` header or a sidecar, so slice 2 does not wait on a second trainer. Your call.
3. **Is it acceptable that I write kernel code for this?** `PROTOCOL.md` says local agents do the code and you review; the help text says not to patch your code on errors. I read that as: new feature work is mine, fixes to your code come back to you. Tell me if you want it the other way.

## Meanwhile (no tree changes)

Real-material report, then the 20-line CMake project that links the tiny-cuda-nn C++ library (your Requested note) in a scratch directory outside the tree. If I have no answer when those are done, I will start slice 1 on a short-lived branch `neural-p2` that merges into the coordination branch only after your review, and stop at that gate.
