# Phase 1 build gates complete

Source/coordination head before this report: `18e5647b04e377ac22058d929ae72527a95f2248`.

| build case | wall time (s) |
|---|---:|
| cold_empty_cache | 2778.96 |
| cold_warm_cache | 2081.99 |
| parallel_two_arch_edit | 4758.55 |
| serial_two_arch_edit | 736.44 |
| serial_two_arch_undo | 676.63 |
| serial_two_arch_touch | 669.91 |
| serial_sm89_edit | 374.67 |
| serial_sm89_undo | 353.56 |
| serial_sm89_touch | 347.95 |
| parallel_two_arch_undo | interrupted by coordinator; invalid timing |
| parallel_two_arch_touch | not run; superseded by serial protocol |

Step-1 evidence (captured during undo; original slow edit had already finished):
```json
{
  "total_RAM_KB": 100573948,
  "free_RAM_KB": 59337388,
  "memory_activity": {
    "PagesInputPersec": 0,
    "PagesOutputPersec": 0,
    "PageReadsPersec": 0,
    "PageWritesPersec": 0
  },
  "cpu": {
    "CurrentClockSpeed": 3701,
    "MaxClockSpeed": 3701,
    "LoadPercentage": 100
  },
  "consumers": [
    {
      "Id": 88968,
      "ProcessName": "ptxas",
      "CPU": 19277.53125,
      "CPU_seconds_in_5s": 54.4375
    },
    {
      "Id": 106560,
      "ProcessName": "ptxas",
      "CPU": 18654.84375,
      "CPU_seconds_in_5s": 54.078125
    },
    {
      "Id": 39192,
      "ProcessName": "explorer",
      "CPU": 8407.09375,
      "CPU_seconds_in_5s": 1.640625
    },
    {
      "Id": 39280,
      "ProcessName": "ChatGPT",
      "CPU": 1080.328125,
      "CPU_seconds_in_5s": 0.359375
    },
    {
      "Id": 46776,
      "ProcessName": "python",
      "CPU": 14337.109375,
      "CPU_seconds_in_5s": 0.296875
    },
    {
      "Id": 22412,
      "ProcessName": "FanControl",
      "CPU": 15047.875,
      "CPU_seconds_in_5s": 0.234375
    },
    {
      "Id": 102260,
      "ProcessName": "powershell",
      "CPU": 0.828125,
      "CPU_seconds_in_5s": 0.21875
    },
    {
      "Id": 73748,
      "ProcessName": "qbittorrent",
      "CPU": 873.75,
      "CPU_seconds_in_5s": 0.203125
    },
    {
      "Id": 60720,
      "ProcessName": "screenbridge",
      "CPU": 125.5625,
      "CPU_seconds_in_5s": 0.171875
    },
    {
      "Id": 66720,
      "ProcessName": "ChatGPT",
      "CPU": 1181.640625,
      "CPU_seconds_in_5s": 0.140625
    },
    {
      "Id": 26540,
      "ProcessName": "chrome",
      "CPU": 11681.09375,
      "CPU_seconds_in_5s": 0.125
    },
    {
      "Id": 32352,
      "ProcessName": "RazerAppEngine",
      "CPU": 18969.75,
      "CPU_seconds_in_5s": 0.125
    }
  ],
  "ninja": [
    {
      "Id": 55632,
      "StartTime": "/Date(1792795982473)/",
      "ElapsedSeconds": 1782.7556828
    }
  ]
}
```

Slowest Ninja steps:
```text
  4672.7 s  intern/cycles/kernel/device/cuda/kernel_sm_89.cubin
  4672.7 s  D:/Github/Blender-RND/build_windows_neural/intern/cycles/kernel/device/cuda/kernel_sm_89.cubin
  4666.3 s  intern/cycles/kernel/device/cuda/kernel_sm_86.cubin
  4666.3 s  D:/Github/Blender-RND/build_windows_neural/intern/cycles/kernel/device/cuda/kernel_sm_86.cubin
  1742.6 s  intern/cycles/kernel/device/cuda/kernel_sm_86.cubin
  1742.6 s  D:/Github/Blender-RND/build_windows_neural/intern/cycles/kernel/device/cuda/kernel_sm_86.cubin
  1708.3 s  intern/cycles/kernel/device/cuda/kernel_sm_89.cubin
  1708.3 s  D:/Github/Blender-RND/build_windows_neural/intern/cycles/kernel/device/cuda/kernel_sm_89.cubin
   130.8 s  intern/libmv/CMakeFiles/libmv_tracks_test.dir/libmv/autotrack/tracks_test.cc.obj
   129.2 s  intern/libmv/CMakeFiles/libmv_predict_tracks_test.dir/libmv/autotrack/predict_tracks_test.cc.obj
   121.1 s  intern/libmv/CMakeFiles/libmv_convolve_test.dir/libmv/image/convolve_test.cc.obj
   116.2 s  intern/atomic/CMakeFiles/atomic_test.dir/tests/atomic_test.cc.obj
   106.4 s  bin/tests/libmv_resection_test.exe
    93.6 s  bin/tests/libmv_triangulation_test.exe
    93.4 s  source/blender/blendthumb/CMakeFiles/BlendThumb.dir/src/blendthumb_win32.cc.obj

```

Protocol deviation: the original parallel edit completed in 4758.55 s before the latest instruction was acted on. The coordinator mistakenly terminated the subsequent undo at about 1783 s, before its own deadline; it is invalid, not a passed measurement. The serial retry cases above replace the remaining parallel measurements. The existing build directory was reused to avoid another complete tree. Restore both GPU architectures with `cmake -S D:/Github/Blender-RND/blender -B D:/Github/Blender-RND/build_windows_neural "-DCYCLES_CUDA_BINARIES_ARCH=sm_86;sm_89"` then `ninja install`. Do not treat leftover installed sm_86 artifacts as a newly coherent two-architecture build.

All completed builds exited 0, VS 2022 x64, Ninja -j 12, BelowNormal priority. Warm-cache cold rebuild used an empty build directory and retained project sccache; the initial working runtime and SHA-256 manifest were preserved under `D:/Github/Blender-RND/install/baseline-first-build` and `timing-evidence/baseline_manifest.json`. The one-line header edit was removed exactly before the touch-only case. The additional undo rebuild has its own local timing JSON but is not substituted for touch-only.

Blender lists both GPUs under CUDA and OptiX; all four furnace cases passed. CTest: 116/120 accepted baseline, three motion-blur subjects plus the OptiX sparse-volume shadow-catcher mismatch, dispositioned in STATUS.md. PyTorch crosscheck: 28/28 PASS. tiny-cuda-nn torch bindings remain unavailable and deferred by the cloud decision; no WSL2.

Local commands: `configure.cmd`, `build.cmd`, `measure_remaining_builds.py` in `D:/Github/Blender-RND`. Raw build logs and sccache statistics: `build_cold.log`, `timing-evidence/*.log`. This closes the requested build gates, not neural-stage implementation, BSDF parity, or the training experiment.
