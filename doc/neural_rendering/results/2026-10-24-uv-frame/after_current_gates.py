"""Queue reviewed-head gates after the current owned validation exits; no polling or process kills."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys

W = Path(__file__).resolve().parent
STATUS = W / 'review_fix_queue.json'

def save(**fields):
    STATUS.write_text(json.dumps(fields, indent=2))


if __name__ == '__main__':
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.WaitForSingleObject.restype = wintypes.DWORD
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    pid = int(sys.argv[1])
    save(status='queued', waiting_for_pid=pid, next='reviewed-head build, unit, furnace, MIS, regression')
    print('QUEUED: waiting for owned validation PID', pid, flush=True)
    handle = k.OpenProcess(0x00100000, False, pid)
    if handle:
        try:
            assert k.WaitForSingleObject(handle, 0xFFFFFFFF) == 0
        finally:
            k.CloseHandle(handle)
    elif ctypes.get_last_error() != 87:
        raise ctypes.WinError(ctypes.get_last_error())
    prior = json.loads(Path('D:/Github/Blender-RND/real/uv_frame_p4/final_gates/status.json').read_text())
    assert prior['status'] == 'complete', prior
    save(status='running', source_head='80e28b275d6', current='reviewed-head qualification')
    print('START reviewed-head qualification', flush=True)
    env = dict(os.environ, NR_GATE_DIR='D:/Github/Blender-RND/real/uv_frame_p4/review_fix_gates')
    result = subprocess.run([sys.executable, '-u', str(W/'finish_gates.py')], env=env)
    save(status='complete' if result.returncode == 0 else 'failed', source_head='80e28b275d6', exit=result.returncode)
    raise SystemExit(result.returncode)
