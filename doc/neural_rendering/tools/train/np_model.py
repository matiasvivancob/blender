# SPDX-License-Identifier: Apache-2.0
"""Model configuration and a numpy (float64) forward pass of the trained model, no torch.

The array dictionary ("arrays") is the interchange format between training and export::

    latent  (H, W, C)  float32         (absent if latent_channels == 0)
    W0..Wn  (in, out)  float32         input-major, W0 rows = [12 encoded direction features, then C latent]
    b0..bn  (out,)     float32

stored in an .npz together with `config_json` (a 0-d string array holding ModelConfig.to_dict()).
"""
import json
from dataclasses import asdict, dataclass

import numpy as np

import _paths  # noqa: F401
from encoding_np import encode_bsdf_dirs, latent_lookup
from nnm.format import (ACT_LEAKY_RELU, ACT_LEAKY_SMELU, ACT_RELU, ENC_BSDF_DIRS, OUT_EXP)

ACTIVATIONS = {"relu": ACT_RELU, "leaky_relu": ACT_LEAKY_RELU, "leaky_smelu": ACT_LEAKY_SMELU}
LOG_CLAMP = (-20.0, 14.0)            # exp output clamp, nnm_format.md section 3 (output_transform_id 1)
ENC_DIM = 12


@dataclass
class ModelConfig:
    width: int = 32
    num_hidden: int = 2
    latent_channels: int = 8
    latent_height: int = 64
    latent_width: int = 64
    output_dim: int = 3
    activation: str = "leaky_smelu"
    beta: float = 1.0                  # LeakySmeLU only
    eps: float = 0.05                  # LeakyReLU / LeakySmeLU
    loss_power: float = 3.0
    output_transform: str = "exp"      # only exp is supported by the training code
    encoding: str = "bsdf_dirs"        # only bsdf_dirs is supported by the training code

    @property
    def input_dim(self):
        return ENC_DIM + self.latent_channels

    def layer_dims(self):
        d = [(self.input_dim, self.width)] + [(self.width, self.width)] * (self.num_hidden - 1)
        return d + [(self.width, self.output_dim)]

    def nnm_activation(self):
        """(activation_id, beta, eps) with the unused parameters zeroed as the format requires."""
        aid = ACTIVATIONS[self.activation]
        if aid == ACT_RELU:
            return aid, 0.0, 0.0
        if aid == ACT_LEAKY_RELU:
            return aid, 0.0, float(self.eps)
        return aid, float(self.beta), float(self.eps)

    def validate(self):
        assert self.activation in ACTIVATIONS, f"unknown activation {self.activation}"
        assert self.width in (16, 32, 64), "width must be 16, 32 or 64 (nnm format)"
        assert 1 <= self.num_hidden <= 4, "1..4 hidden layers (nnm format)"
        assert 1 <= self.latent_channels <= 16, "1..16 latent channels (nnm format)"
        assert self.output_transform == "exp" and self.encoding == "bsdf_dirs"
        assert 1 <= self.output_dim <= 16
        if self.activation == "leaky_smelu":
            assert self.beta > 0 and 0 <= self.eps < 1
        return self

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        known = {k: d[k] for k in ModelConfig.__dataclass_fields__ if k in d}
        return ModelConfig(**known)


def config_to_json_array(cfg: ModelConfig):
    return np.array(json.dumps(cfg.to_dict(), sort_keys=True))


def save_arrays_npz(path, arrays, cfg: ModelConfig):
    np.savez(path, config_json=config_to_json_array(cfg), **{k: np.asarray(v, np.float32) for k, v in arrays.items()})


def load_arrays_npz(path):
    """-> (arrays dict of float32 numpy arrays, ModelConfig or None)."""
    z = np.load(path, allow_pickle=False)
    cfg = ModelConfig.from_dict(json.loads(str(z["config_json"]))) if "config_json" in z.files else None
    arrays = {k: np.asarray(z[k], np.float32) for k in z.files if k != "config_json"}
    return arrays, cfg


def layers_from_arrays(arrays):
    n = 0
    while f"W{n}" in arrays:
        n += 1
    return [(arrays[f"W{i}"], arrays[f"b{i}"]) for i in range(n)]


def infer_config(arrays, base: ModelConfig = None) -> ModelConfig:
    """Fill the shape-derived fields of a config from the arrays (activation etc. come from `base`)."""
    cfg = ModelConfig(**(base.to_dict() if base else {}))
    layers = layers_from_arrays(arrays)
    cfg.num_hidden = len(layers) - 1
    cfg.width = int(layers[0][0].shape[1])
    cfg.output_dim = int(layers[-1][0].shape[1])
    if "latent" in arrays:
        cfg.latent_height, cfg.latent_width, cfg.latent_channels = (int(s) for s in arrays["latent"].shape)
    return cfg.validate()


def check_arrays(arrays, cfg: ModelConfig):
    """Shape checks of the interchange dict against the config. Raises ValueError."""
    cfg.validate()
    if "latent" not in arrays:
        raise ValueError("arrays has no 'latent'")
    want = (cfg.latent_height, cfg.latent_width, cfg.latent_channels)
    if tuple(arrays["latent"].shape) != want:
        raise ValueError(f"latent shape {arrays['latent'].shape}, config says {want}")
    layers = layers_from_arrays(arrays)
    if len(layers) != cfg.num_hidden + 1:
        raise ValueError(f"{len(layers)} layers, config says {cfg.num_hidden + 1}")
    for l, ((w, b), (i, o)) in enumerate(zip(layers, cfg.layer_dims())):
        if w.shape != (i, o) or b.shape != (o,):
            raise ValueError(f"layer {l}: W {w.shape} b {b.shape}, expected {(i, o)} and {(o,)}")
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(b))):
            raise ValueError(f"layer {l}: non-finite values")


def activation_np(x, cfg: ModelConfig):
    if cfg.activation == "relu":
        return np.maximum(x, 0)
    if cfg.activation == "leaky_relu":
        return np.where(x > 0, x, cfg.eps * x)
    beta, eps = cfg.beta, cfg.eps
    s = np.clip(x + beta, 0.0, 2.0 * beta)
    return eps * x + (1.0 - eps) * (s * s * (0.25 / beta) + np.maximum(x - beta, 0.0))


def forward_np(arrays, cfg: ModelConfig, uv, wi, wo, dtype=np.float64):
    """Training-model forward in numpy: (N,2), (N,3), (N,3) -> (N, output_dim) predictions (after exp + clamp)."""
    enc = encode_bsdf_dirs(wi, wo).astype(dtype)
    z = latent_lookup(np.asarray(arrays["latent"], dtype), uv).astype(dtype)
    a = np.concatenate([enc, z], axis=-1)
    layers = layers_from_arrays(arrays)
    for l, (w, b) in enumerate(layers):
        a = a @ w.astype(dtype) + b.astype(dtype)
        if l < len(layers) - 1:
            a = activation_np(a, cfg)
    return np.exp(np.clip(a, LOG_CLAMP[0], LOG_CLAMP[1]))


def random_arrays(cfg: ModelConfig, seed=0, out_bias=np.log(0.2)):
    """Initialisation identical in distribution to the torch model and the numpy prototype."""
    cfg.validate()
    rng = np.random.default_rng(seed)
    arrays = {"latent": rng.uniform(-0.5, 0.5, (cfg.latent_height, cfg.latent_width, cfg.latent_channels)).astype(np.float32)}
    dims = cfg.layer_dims()
    for l, (i, o) in enumerate(dims):
        last = l == len(dims) - 1
        bound = (0.3 if last else 1.0) * np.sqrt(6.0 / i)
        arrays[f"W{l}"] = rng.uniform(-bound, bound, (i, o)).astype(np.float32)
        arrays[f"b{l}"] = np.full(o, out_bias if last else 0.0, np.float32)
    return arrays
