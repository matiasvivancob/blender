# Windows configuration milestone

Source: `D:/Github/Blender-RND/blender`, branch `claude/amazing-planck-xvnke7`, renderer/configuration source at `da3963af977` (subsequent milestone commits are documentation only).

| Check | Executed result |
|---|---|
| `make_update.py --no-blender` | exit 0; Windows libraries fetched |
| Library LFS files | 1260 hydrated (`*`), no pointers remaining |
| `check_env.py --strict` in training venv | exit 0 under the updated checker (tiny-cuda-nn optional) |
| CMake configure | exit 0; MSVC 19.44.35228.0 from VS 2022 |
| CUDA | 13.3; generated for both `sm_86;sm_89` |
| OptiX | 9.1.0; PTX compute_75 |
| `WITH_WINDOWS_SCCACHE` | ON; fresh project-specific cache, 10G limit |
| Build | started `ninja -j 12 install`, BelowNormal priority; **not yet passed** |
| Free D: space before build | about 59.6 decimal GB |
| Float comparator self-test | 37 checks, 0 failed |
| Fake-Blender stage tests | 32 checks, 0 failed (not real renders) |
| Fake scene-builder tests | 17 checks, 0 failed (not real scenes) |
| torch crosscheck | 28/28 PASS, previously posted |

Build logs/timing on the workstation:
- `D:/Github/Blender-RND/configure.log`
- `D:/Github/Blender-RND/build_cold.log`
- `D:/Github/Blender-RND/build_cold_timing.json` (written when build exits)
- `D:/Github/Blender-RND/configure.cmd` and `build.cmd` hold the exact invocations.

## tiny-cuda-nn: failed, not silently counted as installed

The cloud's `cece812a058` makes this optional. Initial native attempt used tiny-cuda-nn `3daa6e5b8b8e0fa17971a1a20acfbb3754494b38`, VS 2022, CUDA 13.3, torch 2.14.1+cu130, and `TCNN_CUDA_ARCHITECTURES=86,89`.

First error (`tcnn_install.log`):
```text
c10/util/StringUtil.h(171): error C7555: el uso de los inicializadores designados requiere al menos "/std:c++20"
```

A bounded retry changed both `cpp_standard = 17` assignments to `cpp_standard = 20` in the external checkout's `bindings/torch/setup.py`. This is the only local dependency-source patch; it is NOT copied into Blender. Retry also failed (`tcnn_install_cpp20.log`):
```text
tiny-cuda-nn/object.h(195): error: no instance of constructor "std::unique_ptr<_Ty, _Dx>::unique_ptr [with _Ty=tcnn::CudaRtcKernel, _Dx=std::default_delete<tcnn::CudaRtcKernel>]" matches the argument list
```

No WSL2 fallback attempted: following the updated request, the Blender build takes priority. This does not satisfy the original tiny-cuda-nn import criterion; it remains an explicit deferred dependency.

## Remaining acceptance gates

Cold build completion, cold/warm-cache and incremental timings, both-card real renders, ctest baseline, furnace/MIS, real `eval_bsdf` CPU/CUDA parity, full training experiment, and OptiX 9 release-note qualification remain open. Installed OptiX 9.1 programming guide says cooperative vectors support all RTX devices and exposes `OPTIX_DEVICE_PROPERTY_COOP_VEC`; this alone is not a measurement or an OptiX 9.0 hardware-acceleration comparison.
