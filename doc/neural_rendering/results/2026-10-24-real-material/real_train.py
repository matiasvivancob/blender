"""Real-material training: Principled probe (metallic base, roughness texture, clear coat 0.05) from CPU eval_bsdf dumps."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path('D:/Github/Blender-RND')
TOOLS = ROOT / 'blender/doc/neural_rendering/tools/train'
PY = ROOT / 'venv-neural/Scripts/python.exe'
GPU = {'4070': 'GPU-96322c27-d5ad-69a1-22ff-f9edbbed80ba', '3090': 'GPU-9abcf4a4-7659-8da2-d759-d900bfc2e5db'}
name, card, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
out = ROOT / 'real' / name
env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU[card], PYTHONUNBUFFERED='1')


def run(tag, args):
    t = time.perf_counter()
    with (ROOT / 'real' / f'{name}_{tag}.log').open('w', encoding='utf-8') as log:
        r = subprocess.run([str(PY), *map(str, args)], cwd=TOOLS, env=env, stdout=log, stderr=subprocess.STDOUT)
    s = time.perf_counter() - t
    assert r.returncode == 0, f'{tag} failed, see real/{name}_{tag}.log'
    return s


train_s = run('train', ['train.py', '--dump', ROOT / 'real/train_a.npz', ROOT / 'real/train_b.npz', '--out', out,
                        '--device', 'cuda:0', '--seed', '0', '--width', '32', '--hidden', '2', '--latent-channels', '8',
                        '--latent-res', '64', '--steps', '100000', '--batch-total', '65536', '--lr', '0.02', *extra])
eval_s = run('eval', ['evaluate.py', out / 'model.nnm', '--dump', ROOT / 'real/train_a.npz', ROOT / 'real/train_b.npz',
                      '--out', out / 'evaluation', '--sphere-dump', ROOT / 'real/sphere_dump.npz'])
m = json.loads((out / 'evaluation/metrics.json').read_text())
res = {'name': name, 'card': card, 'train_wall_s': train_s, 'eval_wall_s': eval_s, 'extra_args': extra, 'metrics': m}
(out / 'result.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res), flush=True)
