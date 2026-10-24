# eval_bsdf probe (rewritten `--probe`, 8fd96af2): PASS on both scenes, CPU device

Both runs print `PROBE PASS`. Max abs albedo error 1.2e-07 against the expected UV-encoded
color (tolerance 1e-3), 0 flagged records, so the prim-index and barycentric assumptions hold
for quads and for an n-gon.

Executable: preserved first cold build, `D:/Github/Blender-RND/install/baseline-first-build/blender.exe`
(renderer source unchanged since `da3963af977`). Device pinned with
`results/2026-10-23-eval-bsdf-parity/pin_device.py`, `NR_DEVICE=CPU`; readback
`NR_DEVICE CPU NONE ['CPU']` in both runs. `blender_dump.py` used unchanged at `82a3ea65`.

## a. `neural_material_probe.blend`, object `ProbePlane`

```text
blender.exe -b scenes/neural_material_probe.blend --python-exit-code 1 --python pin_device.py
    --python doc/neural_rendering/tools/train/blender_dump.py -E BSDF_DUMP -f 1 -- --object ProbePlane --probe
```
exit 0
```text
probe n=256 flagged=0 max_abs_err=1.192e-07 median_abs_err=1.490e-08 tol=0.001
  worst i=28 expect=[0.89647, 0.64036, 0.5] got=[0.89647, 0.64036, 0.5] flag=0
  worst i=89 expect=[0.60379, 0.66358, 0.5] got=[0.60379, 0.66358, 0.5] flag=0
  worst i=122 expect=[0.48605, 0.85872, 0.5] got=[0.48605, 0.85872, 0.5] flag=0
  worst i=109 expect=[0.65669, 0.65287, 0.5] got=[0.65669, 0.65287, 0.5] flag=0
  worst i=102 expect=[0.77923, 0.71823, 0.5] got=[0.77923, 0.71823, 0.5] flag=0
PROBE PASS
```

## b. `probe_ngon.blend`, object `ProbeNgon`

Built by `make_probe_ngon.py` (this directory; the `.blend` is not committed): default cube,
`subdivide_edges(cuts=1, grid fill)`, the four top quads dissolved into one 8-vertex n-gon
(asserted: 20 quads + 1 octagon), Smart UV Project, plus a camera. Same command with
`scenes/probe_ngon.blend` and `--object ProbeNgon`; exit 0
```text
probe n=256 flagged=0 max_abs_err=1.192e-07 median_abs_err=1.490e-08 tol=0.001
  worst i=168 expect=[0.6013, 0.66695, 0.5] got=[0.6013, 0.66695, 0.5] flag=0
  worst i=12 expect=[0.40357, 0.68161, 0.5] got=[0.40357, 0.68161, 0.5] flag=0
  worst i=27 expect=[0.57176, 0.78775, 0.5] got=[0.57176, 0.78775, 0.5] flag=0
  worst i=15 expect=[0.19344, 0.88973, 0.5] got=[0.19344, 0.88973, 0.5] flag=0
  worst i=52 expect=[0.63313, 0.64956, 0.5] got=[0.63313, 0.64956, 0.5] flag=0
PROBE PASS
```

N-gon coverage, by re-running the same sampler with the same seed (`ngon_coverage.py`, this
directory): the octagon triangulates into 6 of the mesh's 46 loop triangles, and **55 of the 256
probe records land on it**, so the n-gon case is exercised, not just the quads.

## One harness note (no change made)

The first ngon attempt had no camera. `-f 1` then logs `ERROR No camera found in scene "Scene"`,
never calls the engine, prints neither PASS nor FAIL, and **Blender still exits 0**, even with
`--python-exit-code 1`. Anything that automates `--probe` or a dump should require the
`PROBE PASS` line (or the `wrote <file>` line) rather than trust the exit code. Fixed on my side by
adding a camera to the scene builder.

Training dumps from now on use the CPU device, as instructed.
