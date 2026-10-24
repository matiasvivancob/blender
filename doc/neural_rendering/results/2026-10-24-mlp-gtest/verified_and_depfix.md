# mlp.h softplus fix verified; and a build-system bug that hid header changes from Ninja

## 1. The softplus fix (`9f7c3f74`) works

Rebuilt from the pulled source and re-ran the unit-test binary:

```text
ctest -R "^cycles$" --output-on-failure   ->   100% tests passed, 1 tests passed out of 1
```
All 8 `KernelNeuralMlp` cases pass, including `OutputTransforms` (it failed at 354 grid points before).
CPU only: the gtest does not exercise the CUDA path.

## 2. Why my first rebuild did nothing: Ninja was not tracking C++ header dependencies

After pulling `9f7c3f74` I ran `ninja cycles_test` and got `ninja: no work to do`, with the old
test binary still failing. The cause is in this box's build, not in the tree:

- This MSVC install has only the Spanish message catalogue (`bin/Hostx64/x64/3082`, no `1033`), so
  `VSLANG=1033` has no effect and `cl /showIncludes` prints `Nota: inclusión del archivo: <path>`.
- CMake detects that prefix correctly and writes it to `msvc_deps_prefix`.
- sccache (`WITH_WINDOWS_SCCACHE=ON`) re-emits the compiler output **without the non-ASCII byte**:
  `Nota: inclusin del archivo: `. Verified directly: plain `cl` with `chcp 65001` gives
  `inclusi\xc3\xb3n`, `sccache cl` gives `inclusin`. The lines never match, so Ninja records no deps.
- Evidence: `ninja -t deps | grep "#deps 0"`: 5199 objects had zero dependencies, among them 199 of 199
  under `intern/cycles` and the test object. `chcp 65001` alone does not fix it (the lossy step is in sccache).

Consequences for the numbers already reported: the nvcc cubins are unaffected (their custom commands
list the headers), so the Phase 1 kernel-edit timings stand. But every C++ object depending on an edited
header, including the CPU kernel, the Blender session code and the unit tests, was silently not rebuilt.
Anything run from `bin/` after a header edit could have used stale CPU code. Nothing in this session's
earlier results is affected (no header was edited between builds), but kernel work cannot start on this.

## 3. Fix (local build configuration + one README note, no renderer code)

- `tools/buildbox/showincludes_fix.cmake` (new, 5 lines) sets `CMAKE_{C,CXX,}_CL_SHOWINCLUDES_PREFIX` to
  the text sccache produces. Used with `-DCMAKE_PROJECT_INCLUDE=...` at configure time. Edit the string
  for another language.
- Reconfigured, deleted the 199 `intern/cycles` objects once so they record dependencies, rebuilt
  (`ninja install`, 93 s).
- Result: 9 of 199 `intern/cycles` objects still show `#deps 0` (was 199); the test object now lists 69
  dependencies including `mlp.h`.
- Dry-run proof (`iter_tools.py edit add` on `mlp.h`, `ninja -n install`, then `edit remove`; file restored
  byte for byte): ninja now plans to rebuild `kernel_neural_mlp_test.cpp.obj`, relink `cycles_test.exe`, and
  install. Before: nothing.
- README section 5, Step 3, documents the symptom, the cause and the check.
- The other ~5000 objects outside `intern/cycles` still have `#deps 0`; deleting `build/**/*.obj` once, or a
  fresh build directory, fixes them. I did not do that (Cycles is what we edit).

## 4. Suggested follow-up for you

Phase 1's "about 6 minutes per header edit" measured only the cubin step. With dependencies now tracked, a
`bsdf.h` edit also rebuilds CPU kernel objects and relinks Blender. I will re-time edit/undo/touch on
`bsdf.h` in the corrected build and post the new numbers.
