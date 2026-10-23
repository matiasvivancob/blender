# Neural rendering in Blender: research and design notes

This directory holds the research, feasibility analysis, design specs and
experiments for adding neural rendering capabilities to this Blender fork.
Target hardware is NVIDIA only (RTX 3090, sm_86, and RTX 4070 Ti Super, sm_89).

Read in this order:

| File | What it is |
|---|---|
| `00_synthesis.md` | Conclusion, architecture decision, ordered plan, effort recalibration for agent-driven work. Start here. |
| `01_cycles_neural_closures_spec.md` | File-level spec: weights transport, `mlp.h`, neural BSDF closure, neural SSS, `SHADER_EVAL_BSDF` training-data mode, per-backend matrix hardware options. |
| `02_neural_radiance_cache.md` | File-level design for a Neural Radiance Cache in Cycles: integrator hooks, online training, OpenPGL reuse, interactions, bias control. |
| `03_eevee_gpu_module.md` | How EEVEE and the GPU module could run small networks in materials: codegen, weight vehicles, Vulkan cooperative vector/matrix, Metal, Python prototype. |
| `04_experiment_results.md` | Results of the two experiments run so far: kernel-style MLP cost benchmark, and tiny neural BRDF training with the Bitterli 2026 recipe. |
| `PROTOCOL.md` | How local agents and the cloud session coordinate through this repository and its pull request. |
| `STATUS.md` | Living status board. Updated by whoever does work. |
| `experiments/` | Re-runnable code for the experiments in `04_experiment_results.md`. |

Source papers (SIGGRAPH 2026 conference track):

- Wu, Lu, Xu, Hašan, Ramamoorthi. *8DNA: 8D Neural Asset Light Transport by Distribution Learning.* https://research.nvidia.com/labs/rtr/publication/wu20268dna/
- Bitterli, Clarberg, Cummings, Lefohn, Marschner, Novák, Rousselle, Weidlich, Zeltner. *Taming optimization variance in compact neural shading networks.* https://research.nvidia.com/labs/rtr/publication/bitterli2026taming/
- Xu, Varma T, Wang, Li, Wu, Wronski, Ramamoorthi, Salvi. *A Generalizable Light Transport 3D Embedding for Global Illumination.* https://research.nvidia.com/labs/rtr/publication/xu2026generalizable/

Tree state these notes were written against: HEAD `afb395bd` (Blender 5.4 alpha
with the 5.3 release branch merged). Line numbers in the specs refer to that
commit and will drift.

Nothing in these notes was built or rendered. The experiments ran on a CPU-only
machine. Every performance claim about GPUs is an extrapolation and is marked as
such.
