"""Tracked fail-fast build -> real frame checks -> probe retraining -> matched renders.
No remote writes, source scene saves, or changes to existing GUI processes.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

W = Path(__file__).resolve().parent
R = Path('D:/Github/Blender-RND')
S = R / 'blender'
T = S / 'doc/neural_rendering/tools'
B = R / 'install/neural-p4/blender.exe'
PY = R / 'venv-neural/Scripts/python.exe'
O = R / 'real/uv_frame_p4'
O.mkdir(parents=True, exist_ok=True)
state = {'status': 'running', 'stages': [], 'source_branch': 'neural-p4'}
env = dict(os.environ, PYTHONUNBUFFERED='1')
for key, leaf in [('BLENDER_USER_CONFIG', 'config'), ('BLENDER_USER_SCRIPTS', 'scripts'),
                  ('BLENDER_USER_EXTENSIONS', 'extensions')]:
    d = O / leaf
    d.mkdir(exist_ok=True)
    env[key] = str(d)


def save():
    (W / 'pipeline_status.json').write_text(json.dumps(state, indent=2))


def run(name, cmd, *, cwd=S, extra=None, marker=None):
    row = {'name': name, 'command': list(map(str, cmd)), 'status': 'running'}
    state['stages'].append(row)
    state['current'] = name
    save()
    print('START', name, flush=True)
    start = time.perf_counter()
    log = W / (name + '.log')
    row['log'] = str(log)
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.run(list(map(str, cmd)), cwd=cwd, env=dict(env, **(extra or {})),
                           stdout=f, stderr=subprocess.STDOUT,
                           creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    text = log.read_text(encoding='utf-8', errors='replace')
    row.update(exit=p.returncode, seconds=time.perf_counter() - start)
    row['status'] = 'passed' if p.returncode == 0 and (marker is None or marker in text) else 'failed'
    save()
    print('END', name, row['status'], row['seconds'], flush=True)
    if row['status'] != 'passed':
        state['status'] = 'failed'
        save()
        print(text[-6000:], flush=True)
        raise RuntimeError(f'{name} failed: {log}')


def blender(scene=None):
    return [B, '-b', '--factory-startup', '--disable-autoexec', *([scene] if scene else []),
            '--python-exit-code', '1']


def dump(name, obj, scene, *args, n=None):
    cmd = blender(scene) + ['--python', T / 'buildbox/pin_device.py', '--python', T / 'train/blender_dump.py',
                            '-E', 'BSDF_DUMP', '-f', '1', '--', '--object', obj, *args]
    run(name, cmd, extra={'NR_DEVICE': 'CPU'}, marker='PROBE PASS' if '--probe' in args else 'wrote ')


if __name__ == '__main__':
    try:
        run('configure', ['cmd.exe', '/d', '/c', R / 'reconfigure_iteration.cmd',
                          '-DCYCLES_CUDA_BINARIES_ARCH=sm_89',
                          '-DCMAKE_PROJECT_INCLUDE=D:/Github/Blender-RND/p4_runtime.cmake',
                          '-DCMAKE_INSTALL_PREFIX=D:/Github/Blender-RND/install/neural-p4'])
        run('build', ['cmd.exe', '/d', '/c', R / 'ninja_target.cmd', 'install'])
        run('unit', ['ctest', '-R', '^cycles$', '--output-on-failure'], cwd=R / 'build_windows_neural',
            marker='100% tests passed')
        run('depcheck', [PY, T / 'buildbox/iter_tools.py', 'depcheck', R / 'build_windows_neural'],
            marker='depcheck PASS')
        for dev in ['CPU', 'CUDA', 'OPTIX']:
            run('anisotropy_' + dev, blender() + ['--python', W / 'check_anisotropy.py',
                '-E', 'UV_FRAME_CHECK', '-f', '1'], extra={'NR_DEVICE': dev, 'NR_GPU': '4070',
                'NR_FRAME_OUT': str(W / ('anisotropy_' + dev + '.json'))}, marker='UV FRAME ANISOTROPY PASS')
        scene = R / 'scenes/neural_material_probe.blend'
        dump('probe', 'ProbePlane', scene, '--probe')
        for seed in [0, 1]:
            dump('dump_' + str(seed), 'ProbePlane', scene, '--n', '4000000', '--seed', str(seed),
                 '--out', str(O / ('train_' + str(seed) + '.npz')))
        dump('sphere_dump', 'ProbeQuad', R / 'scenes/probe_quad.blend', '--fill',
             str(R / 'real/sphere_q.npz'), '--out', str(O / 'sphere_dump.npz'))
        train = O / 'cull_w32'
        dumps = [O / 'train_0.npz', O / 'train_1.npz']
        run('train', [PY, T / 'train/train.py', '--dump', *dumps, '--out', train, '--device', 'cuda:0',
            '--seed', '0', '--width', '32', '--hidden', '2', '--latent-channels', '8', '--latent-res', '64',
            '--steps', '100000', '--batch-total', '65536', '--lr', '0.02'], cwd=T / 'train',
            extra={'CUDA_VISIBLE_DEVICES': 'GPU-96322c27-d5ad-69a1-22ff-f9edbbed80ba'})
        run('export', [PY, T / 'train/export_nnm.py', train / 'model_final.npz', train / 'model_proxy.nnm',
                       '--dump', *dumps], cwd=T / 'train')
        run('evaluate', [PY, T / 'train/evaluate.py', train / 'model_proxy.nnm', '--dump', *dumps,
                        '--out', train / 'evaluation', '--sphere-dump', O / 'sphere_dump.npz'], cwd=T / 'train')
        for dev in ['CPU', 'OPTIX']:
            run('compare_' + dev, blender(scene) + ['--log-level', '1', '--python',
                T / 'buildbox/neural_vs_principled_render.py', '--', O / ('compare_' + dev)],
                extra={'NR_DEVICE': dev, 'NR_GPU': '4070', 'NR_MODEL': str(train / 'model_proxy.nnm'),
                       'NR_RES': '512', 'NR_SPP': '1024'}, marker='RENDERCMP {')
        state['status'] = 'complete'
        save()
        print('PIPELINE COMPLETE', flush=True)
    except BaseException as e:
        state['status'] = 'failed'
        state['error'] = repr(e)
        save()
        raise
