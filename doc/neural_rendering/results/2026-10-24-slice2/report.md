# [result] Slice 2: neural BSDF closure working (neural-p2-closure @ 122a630b1b8, stacked on neural-p2 @ 5d0d064d60d)

Plain version: a Neural BSDF node with a trained `.nnm` now renders as a real material on CPU, CUDA and OptiX, matches the Principled material it was trained from, and passes the MIS and furnace gates.

| gate | result |
|---|---|
| Kernel vs independent numpy reference (200k records, `neural_closure_check.py`, CPU / CUDA 4070 / OptiX 4070) | f*cos p99 rel err 3.0e-5 / 2.6e-5 / 2.6e-5; pdf p99 1.2e-3 / 1.5e-3 / 1.5e-3 (float32 vs float64, sharp lobe); albedo exact; energy ratio 0.99986 / 1.00006 / 1.00006; PASS x3 |
| Furnace, neural vs Principled (256 spp x 4 seeds, `neural_vs_principled.py`) | mean ratio 0.9989 on CPU and on OptiX 4070; 0 pixels above 1 beyond 4 sigma; PASS. CUDA and 3090 not run |
| MIS consistency on `mis_sphere` with neural glossy | OptiX 4070 4096 spp: mis vs nee +-0.002 %, vs forward +0.50 %; CPU 256 spp: all within noise; PASS |
| Unit tests / node gate | unchanged from slice 1 (not re-run on this head except builds) |

Kernel resource usage, sm_89, clean A/B (slice-1 head vs this head, same config; build 379 s vs 410 s):

| kernel | REG | STACK | LOCAL |
|---|---|---|---|
| shade_surface | 168 -> 168 (cap) | 6768 -> 7696 | 0 -> 0 |
| shade_surface_raytrace | 168 -> 168 | 7968 -> 8112 | 0 |
| shader_eval_bsdf | 154 -> 168 | 6624 -> 6656 | 0 |
| shade_shadow | 168 -> 168 | 2944 -> 2960 | 0 |

No new local-memory spills. shade_surface was already at the register cap. Other changed kernels (intersect_*) moved by a few registers either way and don't contain the closure; I read that as compiler noise.

Design as implemented: sampling is a mixture of cosine lobe (p_diffuse) and GGX VNDF reflection (alpha) from the `.nnm` proxy block; the reported pdf is that mixture's exact pdf. Proxy constants go to a separate `neural_proxy` array, `KernelNeuralModel` untouched. Invalid / no-proxy / non-bsdf_dirs files render black with a logged error. Model used: `cull_w32/model_proxy.nnm` (2x32, culling-trained; sphere rel L1 0.026, val rel L1 0.23, so the fit itself is moderate).

Not done: EEVEE, OSL, anisotropy, a visual comparison render, CUDA/3090 furnace, full ctest on this head.
Questions: (1) OK to merge slice 1 then slice 2 once reviewed? (2) Should the next step be a texture-varying (real UV) material, or accuracy of the fit first?
