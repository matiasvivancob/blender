# SPDX-License-Identifier: Apache-2.0
"""Dependency-free PNG writer (8-bit RGB), so evaluation does not need PIL."""
import struct
import zlib

import numpy as np


def write_png(path, img):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    h, w, c = img.shape
    if c != 3:
        raise ValueError("expected an (H, W, 3) uint8 image")
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def read_png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


def upscale(img, k):
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)
