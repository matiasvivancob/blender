# Incremental CUDA compile duration: request guidance

Branch `claude/amazing-planck-xvnke7`, source HEAD `94139f7d5bfd8b656f6223e24364289f223b203f` plus the prescribed temporary iteration-timing comment in `intern/cycles/kernel/closure/bsdf.h`. No kernel logic edits.

Command: `cmd.exe /d /c D:/Github/Blender-RND/build.cmd`, which enters VS 2022 x64 then executes `ninja -C D:/Github/Blender-RND/build_windows_neural -j 12 install` at BelowNormal priority.

Warm-cache cold rebuild completed in 2081.9875097 s (initial cold: 2778.9637193 s). The header-edit incremental rebuild now spends over 74 minutes in the two concurrent ptxas processes. CPU time continues increasing: sm_89 PID 63564 about 48090 CPU seconds, sm_86 PID 65192 about 46066 CPU seconds. Each uses `-split-compile=0`; nvcc commands also have `--split-compile=0`, `WITH_CYCLES_DEBUG`, and `--ptxas-options=-v`.

No compiler error or failing Ninja target has been reported. Exact log tail:
```text
[vcvarsall.bat] Environment initialized for: 'x64'
```
Ninja buffers the custom-command output until completion. The compiler processes are active, not shown deadlocked. Raw log: `D:/Github/Blender-RND/timing-evidence/incremental_edit.log`. Completed timing: `timing-evidence/cold_warm_cache.json`.

Already checked: process ownership/command lines and repeated CPU growth; no restart, kill, compiler flag changes, or kernel edits. Training remains queued behind completion to avoid CPU/build contention. Undo and touch measurements are still pending.

Is this duration expected with split-compile=0 and two simultaneous architectures, or should the remaining timing protocol change? I will preserve this ongoing measurement and not restart it without direction. Reporting now rather than spending further diagnostic time; the earlier escalation should have happened sooner.
