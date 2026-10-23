# SPDX-License-Identifier: Apache-2.0
"""Dataclasses and reader/writer for the .nnm file format, version 1.

The byte layout is specified in doc/neural_rendering/nnm_format.md. Everything
here is numpy only.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

MAGIC = b"NNM1"
VERSION = 1
HEADER_SIZE = 256
BLOCK_ALIGN = 64

# Activation ids.
ACT_RELU, ACT_LEAKY_RELU, ACT_LEAKY_SMELU = 0, 1, 2
# Output transform ids.
OUT_NONE, OUT_EXP, OUT_SOFTPLUS = 0, 1, 2
# Input encoding ids.
ENC_RAW, ENC_BSDF_DIRS = 0, 1

BSDF_DIRS_CALLER_DIM = 6
BSDF_DIRS_ENCODED_DIM = 12

MAX_INPUT_DIM = 64
MAX_OUTPUT_DIM = 16
MAX_HIDDEN = 4
MAX_LATENT_CHANNELS = 16
MAX_LATENT_SIZE = 16384
MAX_TESTVEC = 64
WIDTHS = (16, 32, 64)

# magic, version, header_size, flags,
# input_dim, output_dim, num_hidden, hidden_width, activation, out_transform, encoding, latent_channels,
# beta, eps, latent_w, latent_h, latent_levels, testvec_count,
# weight_offset, weight_size, latent_offset, latent_size, testvec_offset, testvec_size,
# testvec_in_dim, testvec_out_dim, num_layers, layer_offset[5], file_size, reserved
_HEADER = struct.Struct("<4s3I8I2f4I6Q2I1I5IQ96x")
assert _HEADER.size == HEADER_SIZE


class NnmFormatError(ValueError):
    """Raised for files or models that violate the .nnm specification."""


def pad8(n: int) -> int:
    return (n + 7) & ~7


def align64(n: int) -> int:
    return (n + BLOCK_ALIGN - 1) & ~(BLOCK_ALIGN - 1)


@dataclass
class NnmModel:
    """In-memory .nnm model. Weights and latents are already float16."""

    input_dim: int
    output_dim: int
    num_hidden: int
    width: int
    layers: List[Tuple[np.ndarray, np.ndarray]]  # (W[in, out] f16, b[out] f16) per linear layer
    activation: int = ACT_RELU
    act_beta: float = 0.0
    act_eps: float = 0.0
    output_transform: int = OUT_NONE
    encoding: int = ENC_RAW
    latent_channels: int = 0
    latent_levels: List[np.ndarray] = field(default_factory=list)  # each (h, w, C) f16, level 0 first
    tv_inputs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    tv_outputs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))

    # -- derived quantities ------------------------------------------------
    @property
    def num_layers(self) -> int:
        return self.num_hidden + 1

    @property
    def encoded_dim(self) -> int:
        return self.input_dim - self.latent_channels

    @property
    def caller_dim(self) -> int:
        return BSDF_DIRS_CALLER_DIM if self.encoding == ENC_BSDF_DIRS else self.encoded_dim

    @property
    def tv_in_dim(self) -> int:
        return self.caller_dim + (2 if self.latent_channels > 0 else 0)

    @property
    def latent_width(self) -> int:
        return self.latent_levels[0].shape[1] if self.latent_levels else 0

    @property
    def latent_height(self) -> int:
        return self.latent_levels[0].shape[0] if self.latent_levels else 0

    def layer_dims(self) -> List[Tuple[int, int]]:
        dims = [(self.input_dim, self.width)]
        dims += [(self.width, self.width)] * (self.num_hidden - 1)
        dims.append((self.width, self.output_dim))
        return dims


def layer_sizes(model: NnmModel) -> List[int]:
    """Byte size of every layer in the weight blob."""
    return [(i + 1) * pad8(o) * 2 for i, o in model.layer_dims()]


def layer_offsets(model: NnmModel) -> List[int]:
    offs, acc = [], 0
    for s in layer_sizes(model):
        offs.append(acc)
        acc += s
    return offs


def latent_level_shapes(width: int, height: int, levels: int) -> List[Tuple[int, int]]:
    return [(max(1, height >> k), max(1, width >> k)) for k in range(levels)]


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise NnmFormatError(msg)


def validate_model(m: NnmModel) -> None:
    """Check everything that does not need the file bytes."""
    _check(1 <= m.input_dim <= MAX_INPUT_DIM, f"input_dim {m.input_dim} out of range")
    _check(1 <= m.output_dim <= MAX_OUTPUT_DIM, f"output_dim {m.output_dim} out of range")
    _check(1 <= m.num_hidden <= MAX_HIDDEN, f"num_hidden {m.num_hidden} out of range")
    _check(m.width in WIDTHS, f"hidden width {m.width} not in {WIDTHS}")
    _check(m.activation in (ACT_RELU, ACT_LEAKY_RELU, ACT_LEAKY_SMELU), "unknown activation id")
    _check(m.output_transform in (OUT_NONE, OUT_EXP, OUT_SOFTPLUS), "unknown output transform id")
    _check(m.encoding in (ENC_RAW, ENC_BSDF_DIRS), "unknown input encoding id")
    _check(0 <= m.latent_channels <= MAX_LATENT_CHANNELS, "latent_channels out of range")
    _check(m.encoded_dim >= 1, "input_dim must exceed latent_channels")
    if m.encoding == ENC_BSDF_DIRS:
        _check(m.encoded_dim == BSDF_DIRS_ENCODED_DIM, "bsdf_dirs needs input_dim - latent_channels == 12")
    beta, eps = float(m.act_beta), float(m.act_eps)
    _check(np.isfinite(beta) and np.isfinite(eps), "activation parameters must be finite")
    if m.activation == ACT_LEAKY_SMELU:
        _check(beta > 0.0, "LeakySmeLU needs beta > 0")
    if m.activation in (ACT_LEAKY_RELU, ACT_LEAKY_SMELU):
        _check(0.0 <= eps < 1.0, "eps must be in [0, 1)")
    if m.activation == ACT_RELU:
        _check(beta == 0.0 and eps == 0.0, "ReLU takes no parameters (store 0)")
    if m.activation == ACT_LEAKY_RELU:
        _check(beta == 0.0, "LeakyReLU stores beta = 0")

    _check(len(m.layers) == m.num_layers, "wrong number of layers")
    for l, ((w, b), (i, o)) in enumerate(zip(m.layers, m.layer_dims())):
        _check(w.dtype == np.float16 and b.dtype == np.float16, f"layer {l}: weights must be float16")
        _check(w.shape == (i, o), f"layer {l}: weight shape {w.shape}, expected {(i, o)}")
        _check(b.shape == (o,), f"layer {l}: bias shape {b.shape}, expected {(o,)}")
        _check(bool(np.all(np.isfinite(w)) and np.all(np.isfinite(b))), f"layer {l}: non-finite weight")

    if m.latent_channels == 0:
        _check(not m.latent_levels, "latent levels given but latent_channels == 0")
    else:
        _check(len(m.latent_levels) >= 1, "latent_channels > 0 needs at least one level")
        h0, w0 = m.latent_levels[0].shape[:2]
        _check(1 <= w0 <= MAX_LATENT_SIZE and 1 <= h0 <= MAX_LATENT_SIZE, "latent size out of range")
        max_levels = int(np.floor(np.log2(max(w0, h0)))) + 1
        _check(len(m.latent_levels) <= max_levels, "too many latent levels")
        for k, (lv, (hk, wk)) in enumerate(zip(m.latent_levels, latent_level_shapes(w0, h0, len(m.latent_levels)))):
            _check(lv.dtype == np.float16, f"latent level {k} must be float16")
            _check(lv.shape == (hk, wk, m.latent_channels), f"latent level {k} shape {lv.shape}")
            _check(bool(np.all(np.isfinite(lv))), f"latent level {k}: non-finite value")

    n = m.tv_inputs.shape[0]
    _check(0 <= n <= MAX_TESTVEC, "too many test vectors")
    _check(m.tv_outputs.shape[0] == n, "test vector count mismatch")
    if n:
        _check(m.tv_inputs.dtype == np.float32 and m.tv_outputs.dtype == np.float32, "test vectors must be float32")
        _check(m.tv_inputs.shape[1] == m.tv_in_dim, f"test vector input width {m.tv_inputs.shape[1]} != {m.tv_in_dim}")
        _check(m.tv_outputs.shape[1] == m.output_dim, "test vector output width")


def _pack_layer(w: np.ndarray, b: np.ndarray) -> bytes:
    i, o = w.shape
    buf = np.zeros((i + 1, pad8(o)), np.float16)
    buf[:i, :o] = w
    buf[i, :o] = b
    return buf.astype("<f2").tobytes()


def write_nnm(model: NnmModel, path=None) -> bytes:
    """Serialise ``model``. Returns the bytes and also writes them to ``path`` if given."""
    validate_model(model)
    weights = b"".join(_pack_layer(w, b) for w, b in model.layers)
    offs = layer_offsets(model)
    assert len(weights) == sum(layer_sizes(model))

    latents = b""
    if model.latent_channels:
        latents = b"".join(lv.astype("<f2").tobytes() for lv in model.latent_levels)

    n_tv = model.tv_inputs.shape[0]
    testvec = b""
    if n_tv:
        rec = np.concatenate([model.tv_inputs, model.tv_outputs], axis=1).astype("<f4")
        testvec = rec.tobytes()

    pos = HEADER_SIZE
    weight_offset = pos
    pos = align64(pos + len(weights))
    latent_offset = pos if latents else 0
    if latents:
        pos = align64(pos + len(latents))
    testvec_offset = pos if testvec else 0
    if testvec:
        pos += len(testvec)
    file_size = pos

    lay = (offs + [0] * 5)[:5]
    header = _HEADER.pack(
        MAGIC, VERSION, HEADER_SIZE, 0,
        model.input_dim, model.output_dim, model.num_hidden, model.width,
        model.activation, model.output_transform, model.encoding, model.latent_channels,
        float(model.act_beta), float(model.act_eps),
        model.latent_width, model.latent_height, len(model.latent_levels), n_tv,
        weight_offset, len(weights), latent_offset, len(latents), testvec_offset, len(testvec),
        model.tv_in_dim if n_tv else 0, model.output_dim if n_tv else 0,
        model.num_layers, *lay, file_size,
    )
    out = bytearray(file_size)
    out[:HEADER_SIZE] = header
    out[weight_offset:weight_offset + len(weights)] = weights
    if latents:
        out[latent_offset:latent_offset + len(latents)] = latents
    if testvec:
        out[testvec_offset:testvec_offset + len(testvec)] = testvec
    data = bytes(out)
    if path is not None:
        with open(path, "wb") as f:
            f.write(data)
    return data


def read_nnm(src) -> NnmModel:
    """Parse and fully validate a .nnm file. ``src`` is bytes or a path."""
    if isinstance(src, (bytes, bytearray, memoryview)):
        data = bytes(src)
    else:
        with open(src, "rb") as f:
            data = f.read()
    _check(len(data) >= HEADER_SIZE, "file shorter than header")
    f = _HEADER.unpack_from(data, 0)
    (magic, version, header_size, flags,
     input_dim, output_dim, num_hidden, width, act, xform, enc, lat_c,
     beta, eps, lat_w, lat_h, lat_levels, n_tv,
     w_off, w_size, l_off, l_size, t_off, t_size,
     t_in, t_out, num_layers, *rest) = f
    layer_off, file_size = rest[:5], rest[5]
    _check(magic == MAGIC, f"bad magic {magic!r}")
    _check(version == VERSION, f"unsupported version {version}")
    _check(header_size == HEADER_SIZE, "bad header_size")
    _check(flags == 0, "non-zero flags")
    _check(file_size == len(data), f"file_size {file_size} != actual {len(data)}")
    _check(data[160:HEADER_SIZE] == bytes(96), "non-zero reserved header bytes")

    # Build a skeleton model to reuse the dimension rules.
    m = NnmModel(input_dim, output_dim, num_hidden, width, layers=[], activation=act, act_beta=beta,
                 act_eps=eps, output_transform=xform, encoding=enc, latent_channels=lat_c)
    _check(1 <= num_hidden <= MAX_HIDDEN and width in WIDTHS, "bad topology")
    _check(num_layers == num_hidden + 1, "num_layers != num_hidden + 1")
    sizes = layer_sizes(m)
    offs = layer_offsets(m)
    _check(list(layer_off) == (offs + [0] * 5)[:5], "layer_offset table mismatch")
    _check(w_off == HEADER_SIZE and w_size == sum(sizes), "weight blob offset/size mismatch")

    blocks = [("weights", w_off, w_size)]
    layers = []
    for (i, o), lo in zip(m.layer_dims(), offs):
        base = w_off + lo
        raw = np.frombuffer(data, "<f2", (i + 1) * pad8(o), base).reshape(i + 1, pad8(o))
        _check(not np.any(raw[:, o:]), "non-zero weight padding")
        layers.append((raw[:i, :o].astype(np.float16), raw[i, :o].astype(np.float16)))
    m.layers = layers

    if lat_c == 0:
        _check(lat_w == lat_h == lat_levels == 0 and l_off == 0 and l_size == 0, "latent fields must be zero")
    else:
        _check(1 <= lat_w <= MAX_LATENT_SIZE and 1 <= lat_h <= MAX_LATENT_SIZE, "latent size out of range")
        _check(lat_levels >= 1, "latent_levels >= 1 required")
        shapes = latent_level_shapes(lat_w, lat_h, lat_levels)
        expect = sum(h * w * lat_c * 2 for h, w in shapes)
        _check(l_size == expect, "latent_size mismatch")
        _check(l_off % BLOCK_ALIGN == 0, "latent_offset not aligned")
        pos, lv_list = l_off, []
        for h, w in shapes:
            n = h * w * lat_c
            lv_list.append(np.frombuffer(data, "<f2", n, pos).reshape(h, w, lat_c).astype(np.float16))
            pos += n * 2
        m.latent_levels = lv_list
        blocks.append(("latents", l_off, l_size))

    _check(0 <= n_tv <= MAX_TESTVEC, "testvec_count out of range")
    if n_tv:
        _check(t_in == m.tv_in_dim and t_out == output_dim, "testvec dims mismatch")
        _check(t_size == n_tv * (t_in + t_out) * 4, "testvec_size mismatch")
        _check(t_off % BLOCK_ALIGN == 0, "testvec_offset not aligned")
        rec = np.frombuffer(data, "<f4", n_tv * (t_in + t_out), t_off).reshape(n_tv, t_in + t_out)
        m.tv_inputs = rec[:, :t_in].astype(np.float32)
        m.tv_outputs = rec[:, t_in:].astype(np.float32)
        blocks.append(("testvec", t_off, t_size))
    else:
        _check(t_off == 0 and t_size == 0 and t_in == 0 and t_out == 0, "testvec fields must be zero")

    # Blocks inside the file, after the header, non-overlapping.
    blocks.sort(key=lambda b: b[1])
    end = HEADER_SIZE
    for name, off, size in blocks:
        _check(off >= end, f"block {name} overlaps header or previous block")
        _check(off + size <= len(data), f"block {name} extends past end of file")
        end = off + size

    validate_model(m)
    return m
