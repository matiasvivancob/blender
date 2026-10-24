# Local frame from the UV tangent (spec, fixes slice 2 and the training dumps)

Status: spec, 2026-10-24, after slice 3 (`results/2026-10-24-slice3/`). Prerequisite for
learned shading frames (`05_learned_shading_frames.md`) and for any anisotropic material.

## 1. The problem

The neural BSDF and `SHADER_EVAL_BSDF` both build the local frame as
`x = dPdu` projected on the tangent plane, `z = N`, `y = z x x`
(`kernel/bake/bake.h`, `kernel/closure/bsdf_neural.h`). For triangles, Cycles' `dPdu` is
**not** the texture U direction: `triangle_dPdudv` (`kernel/geom/triangle.h`) returns
`p1 - p0`, an edge of the triangle in its vertex order. On the probe plane (16x16 quads,
two triangles each) the frame's rotation about the normal therefore changes from one
triangle to the next, while the latent is indexed by texture UV.

The network then sees the same texel under several azimuthal frames and has to learn
invariance to that rotation, which it does imperfectly. The slice-3 residual shows it:
the difference image has a triangle pattern at the mesh's quad size (about 16 cells
across), much coarser than the 64x64 latent texels. This is a defect of the frame
definition in `01_cycles_neural_closures_spec.md` section 6, not of the agent's
implementation, which followed the spec. An anisotropic material would be wrong, not
only noisier, because its anisotropy axis would jump between triangles.

## 2. Definition (normative)

The local x axis is the **texture-space tangent** of the UV map that indexes the latent:
the direction of increasing U on the surface, per triangle, from the triangle's corner
positions `p0, p1, p2` and corner UVs `t0, t1, t2`:

```
e1 = p1 - p0, e2 = p2 - p0           (object space, transformed like dPdu)
d1 = t1 - t0, d2 = t2 - t0           (UV)
det = d1.x * d2.y - d2.x * d1.y
T_uv = (e1 * d2.y - e2 * d1.y) / det  if |det| > 1e-20, else fall back to dPdu
x = normalize(T_uv - N * dot(T_uv, N)); fall back to make_orthonormals(N) if degenerate
y = cross(N, x)    (unchanged convention: z = N, y = z x x)
```

This is constant across a triangle and continuous across triangles wherever the UV map is
continuous and not mirrored. It is what normal maps call the per-face UV tangent; it is
not the MikkTSpace per-vertex tangent (no smoothing), which keeps the computation local to
one triangle and identical in the bake and closure paths.

Mirrored UV islands (`det < 0`) keep the sign: `T_uv` then points along increasing U as
seen in the mirrored island, which is consistent with how the latent was sampled there.

## 3. Implementation

1. **One kernel helper** in `kernel/geom/triangle.h` or a new
   `kernel/closure/bsdf_neural_frame.h`:
   `float3 neural_uv_tangent(KernelGlobals kg, const ShaderData *sd, AttributeDescriptor uv_desc)`.
   It reads the three corner UVs with `attribute_data_fetch_3` (the corner path of
   `triangle_attribute`), the three positions as `triangle_dPdudv` does, applies the
   object transform the same way `sd->dPdu` gets it, and returns `T_uv` per section 2.
   Non-triangle primitives (curves, points) and a missing UV attribute return `sd->dPdu`.
2. **Closure:** the Neural BSDF node requests the UV map attribute it uses for the latent
   (`attributes->add(ATTR_STD_UV)` or the named map, as texture nodes do) and passes its
   attribute id in `SVMNodeNeuralBsdfData`; `bsdf_neural_setup` resolves the descriptor
   with `find_attribute` and builds `T` from `neural_uv_tangent`.
3. **`SHADER_EVAL_BSDF`:** `kernel_bsdf_evaluate` builds the local frame with the same
   helper, for the object's default UV map (`ATTR_STD_UV`). The bake-side frame and the
   closure frame must be the same function call; nothing else may compute it.
4. **Version marker:** add `KERNEL_BSDF_EVAL_UV_TANGENT_FRAME` (bit 1) to the eval_bsdf
   input flags, set by `blender_dump.py`, so a dump records which frame it used; write it
   into the dump's meta (`frame: "uv_tangent"`). `train.py` refuses dumps without it once
   this lands, so old dumps cannot silently train a model for the new frame.
5. **Tests:** a unit test of `neural_uv_tangent` on hand-built triangles (rotated UVs,
   mirrored UVs, degenerate UVs). Re-run `--probe` and the parity run (the albedo probe is
   frame-independent, so add one record set whose `f` depends on azimuth: a Principled
   with anisotropy 0.8 on the probe plane must give the same value on both triangles of a
   quad for the same UV and local directions).

## 4. Retraining and acceptance

All existing `.nnm` models were trained in the old frame and must be retrained from new
dumps. Acceptance on the probe material, same 2x32 culled model and schedule as slice 3:

- the triangle pattern is gone from the slice-3 residual (inspect the x8 difference), and
- mean raw RGB relative error drops below the slice-3 value of 1.94% (CPU), with the
  held-out relative error reported next to the old 23%.

Only then start the learned-frames work of `05_learned_shading_frames.md`; frames learned
on top of an inconsistent base frame would inherit the same defect.
