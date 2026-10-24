# Blocked: `KernelNeuralMlp.OutputTransforms` fails (softplus returns 0 for x < -16.7)

Branch `claude/amazing-planck-xvnke7`, source `31644222ef0`. First time the `mlp.h` gtest has run on a real compiler (MSVC 19.44.35228, Release, `/fp:fast`).

Command, from `D:/Github/Blender-RND/build_windows_neural`:
```text
ctest -R "^cycles$" --output-on-failure
```
Result: 223 of 224 gtest cases pass, including `EmbeddedModelsMatchPythonReference`,
`EmbeddedModelsCoverWidthsActivationsTransforms`, `LayerOffsets`, `EncodeBsdfDirs`, `Activations`,
`LatentBilinear`, `DispatchRejectsUnsupportedWidth`. One fails:
```text
[ RUN      ] KernelNeuralMlp.OutputTransforms
...kernel_neural_mlp_test.cpp(164): error: The difference between neural_output_transform(x, NEURAL_OUTPUT_SOFTPLUS) and sp is 4.2483542552915889e-18, which exceeds 2e-6 * sp + 1e-30, where
neural_output_transform(x, NEURAL_OUTPUT_SOFTPLUS) evaluates to 0,
sp evaluates to 4.2483542552915889e-18, and
2e-6 * sp + 1e-30 evaluates to 8.4967095105831767e-24.
x = -40
[  FAILED  ] KernelNeuralMlp.OutputTransforms (10 ms)
```
Only the SOFTPLUS assertion at line 164 fails (NONE and EXP pass for every x). 354 of 801 grid
points fail, x from -40.0 to -3.9. Full table (x, kernel, exact, relative error):
`doc/neural_rendering/results/2026-10-24-mlp-gtest/gtest_cycles_tail.txt`.

| x | kernel | exact | relative error |
|---|---|---|---|
| -40 to -16.7 (234 values) | exactly 0 | 4e-18 to 5e-8 | -100 % |
| -16.6 | 1.19e-07 | 6.18e-08 | +93 % |
| -14.8 | 3.58e-07 | 3.74e-07 | -4.3 % |
| -13.0 | 2.265e-06 | 2.260e-06 | +0.2 % |
| -9.4 | 8.273e-05 | 8.272e-05 | +8.7e-05 |
| -5.6 | 3.691e-03 | 3.691e-03 | +2.2e-06 |
| -3.9 | 2.004e-02 | 2.004e-02 | -2.6e-06 |

## Diagnosis (read-only; I changed nothing)

`neural_log1p` in `kernel/util/mlp.h` (line ~187) uses Kahan's trick: `u = 1+e; d = u-1;
d == 0 ? e : logf(u) * (e/d)`. That only works if `u - 1.0f` is evaluated in float. Here
`intern/cycles/CMakeLists.txt:149` builds MSVC with `/fp:fast`, which may simplify `(1+e)-1` to `e`.
Then `d == e`, never 0, and the result is `logf(u) * 1`, which is `logf(1.0f) = 0` once `e < ~6e-8`
(x < ~-16.6) and badly quantised just above it. The numbers match that exactly: zeros end where
`1+e` rounds to 1, a +93 % overshoot at the first representable step, then error decaying with e.
This is a hypothesis from the numbers and the flag, not confirmed by looking at the assembly.
The -3.9 to -5.6 failures (~2.5e-6) are a separate, small thing: float32 rounding of `log(u)` for
`u` near 1 slightly exceeds the test's 2e-6 relative tolerance.

## Impact

None today: the exported models use `NEURAL_OUTPUT_EXP` (passes). Softplus is only affected for
very negative outputs, where the true value is below 5e-8. It would still matter if a model with a
softplus head is added, and the same pattern may hit other backends (CUDA `--use_fast_math` does
not reassociate by default, so likely fine there; untested, the gtest only builds the CPU path).

## Suggested fix (yours to apply; I will not edit kernel code)

Avoid the `u - 1` trick. For `e < ~1e-3` use the series `e * (1 - 0.5f*e)` (error < 5e-7 relative);
otherwise `logf(1.0f + e)`. Or call `log1pf` where available and keep the series only for backends
without it. Then loosen or leave the test tolerance: 2e-6 is near float32 resolution for
`logf(1+e)` at e = 0.02.

## Also noted

- `ctest -R "^cycles_"` (the 120-subject baseline) does not include this suite; test #35 `cycles`
  is the unit-test binary, which the baseline never ran. This failure does not change the accepted
  116/120 figure.
- I am continuing with the real-material pipeline (CPU dump, train, export, evaluate), which does
  not depend on this.
