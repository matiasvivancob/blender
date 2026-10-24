"""Regenerate the slice-3 contact sheet and sampling-time diagnostics from retained outputs.

Run with a Python containing numpy and Pillow:
  python make_comparison.py D:/Github/Blender-RND/real/slice3_20261024 <output directory>
The contact uses one shared exposure (-2 EV), Reinhard RGB, then sRGB; not the original display transform.
"""
import json
from pathlib import Path
import re
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def display(rgb):
    x = np.maximum(rgb, 0) * 0.25
    x = x / (1 + x)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.uint8(np.clip(x, 0, 1) * 255 + 0.5)


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    canvas = Image.new('RGB', (1536, 1140), '#161616')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), 'Original / trained Neural BSDF / absolute difference x8 | shared display: -2 EV, Reinhard, sRGB', font=font)
    diagnostics = {}
    for row, device in enumerate(('CPU', 'OPTIX')):
        d = np.load(root / device / 'beauties.npz')
        p, n = d['principled'], d['neural']
        assert p.shape == n.shape == (512, 512, 3) and np.isfinite(p).all() and np.isfinite(n).all()
        assert p.max() > 1e-3, 'empty native image'
        y = 60 + 540 * row
        for col, (title, image) in enumerate((('Principled', p), ('Neural', n), ('abs difference x8', 8 * np.abs(p - n)))):
            draw.text((12 + 512 * col, y - 24), device + ': ' + title, font=font)
            canvas.paste(Image.fromarray(display(np.flipud(image))), (512 * col, y))
        text = (root / (device + '.log')).read_text(encoding='utf-8', errors='replace')
        completions = re.findall(r'Rendered (\d+) samples in ([\d.]+) seconds[ \t]*$', text, re.MULTILINE)
        tags = re.findall(r'RENDERCMP RENDER (\S+) requested_samples (\d+) seconds ([\d.]+)', text)
        assert len(completions) == len(tags) == 13, (device, 'missing render completions', len(completions), len(tags))
        matches = [(got, seconds, tag, requested, wall)
                   for (got, seconds), (tag, requested, wall) in zip(completions, tags)]
        assert all(int(got) == int(requested) for got, _, _, requested, _ in matches)
        values = {}
        for got, seconds, tag, requested, wall in matches:
            values.setdefault(tag, []).append({'samples': int(got), 'render_seconds': float(seconds), 'wall_seconds': float(wall)})
        diagnostics[device] = {'renders': values, 'note': 'sampling-time slopes from terminal Cycles Rendered N samples lines; diagnostic, not controlled equal-quality speedup'}
        for label in ('principled', 'neural'):
            lo = min(v['render_seconds'] for v in values[label + '_lo'])
            hi = min(v['render_seconds'] for v in values[label + '_hi'])
            assert hi > lo and hi / lo > 2, 'invalid time scaling'
            diagnostics[device][label + '_sampling_ms_per_spp'] = 1000 * (hi - lo) / (512 - 64)
        diagnostics[device]['sampling_cost_ratio'] = diagnostics[device]['neural_sampling_ms_per_spp'] / diagnostics[device]['principled_sampling_ms_per_spp']
    canvas.save(out / 'comparison.png')
    (out / 'sampling_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps({dev: {k: v for k, v in d.items() if k != 'renders'} for dev, d in diagnostics.items()}, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
