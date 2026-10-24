"""Regenerate native/old/new UV-frame contact from retained arrays. Requires numpy and Pillow."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def display(x):
    x = np.maximum(x, 0) * .25
    x = x / (1 + x)
    x = np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)
    return Image.fromarray(np.uint8(np.clip(np.flipud(x), 0, 1) * 255 + .5))


def main(root, out):
    old = np.load(root / 'real/slice3_20261024/CPU/beauties.npz')
    new = np.load(root / 'real/uv_frame_p4/compare_CPU/beauties.npz')
    native, a, b, repeat = old['principled'], old['neural'], new['neural'], old['principled_repeat']
    assert np.array_equal(native, new['principled']), 'native image changed'
    assert all(np.isfinite(x).all() and x.shape == (512, 512, 3) for x in (native, a, b, repeat))
    panels = [('Native Principled', native), ('Old neural (edge frame)', a), ('New neural (UV frame)', b),
              ('Native repeat difference x8', 8 * abs(native - repeat)),
              ('Old neural difference x8', 8 * abs(native - a)), ('New neural difference x8', 8 * abs(native - b))]
    canvas = Image.new('RGB', (1536, 1128), '#161616')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    for i, (label, image) in enumerate(panels):
        x, y = i % 3 * 512, i // 3 * 552 + 40
        draw.text((x + 8, y - 27), label, font=font, fill='white')
        canvas.paste(display(image), (x, y))
    canvas.save(out)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
