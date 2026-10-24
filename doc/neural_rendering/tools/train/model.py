# SPDX-License-Identifier: Apache-2.0
"""Torch model of the neural material: latent grid + decoder MLP, K independent instances in one module.

NOT RUNNABLE WHERE THIS WAS WRITTEN (no torch there). Review notes and the cross-check procedure are in
README.md ("Verifying the torch code"); run `python3 crosscheck_torch.py` first on the workstation.

Everything that defines the *deployed* function is specified by nnm_format.md and mirrored here:
  * encoding: `encode_bsdf_dirs` == nnm_format.md section 7 == encoding_np.py == nnm.reference
  * latent lookup: corner-to-corner bilinear, clamped (section 5), `bilinear_corners`
  * MLP input order: [12 encoded features, then the C latent channels] (section 3), so layer 0 of the exported
    file is `W0 = model.W0[i]` unchanged
  * activation: relu / leaky_relu / LeakySmeLU `eps*x + (1-eps)*(s*s*0.25/beta + max(x-beta, 0))`, s = clamp(x+beta, 0, 2 beta)
  * output: exp(clamp(x, -20, 14)); the training forward uses the same clamp (straight-through gradient)

Shapes: K instances, batch B, texels H x W, latent channels C, hidden width Wd.
  latent (K, H, W, C)   W_l (K, in_l, out_l)   b_l (K, 1, out_l)   input uv (B, 2), wi (B, 3), wo (B, 3)
  forward -> (K, B, 3) pre-exp output (clamped); predict -> exp of it.
All K instances see the same batch (Bitterli et al. lock-step training); independent weights and latents.
"""
import math

import torch
import torch.nn as nn

import _paths  # noqa: F401
from np_model import ENC_DIM, LOG_CLAMP, ModelConfig


# ----------------------------------------------------------------------------- encoding and lookup
def encode_bsdf_dirs(wi, wo):
    """wi, wo (..., 3) unit vectors, z > 0 -> (..., 12) = [wi, wo, h, wd]; nnm_format.md section 7."""
    wix, wiy, wiz = wi[..., 0], wi[..., 1], wi[..., 2]
    hx, hy, hz = wi[..., 0] + wo[..., 0], wi[..., 1] + wo[..., 1], wi[..., 2] + wo[..., 2]
    l2 = hx * hx + hy * hy + hz * hz
    ok = l2 > 0
    inv = 1.0 / torch.sqrt(l2.clamp_min(1e-30))             # clamp only avoids inf where `ok` is False
    hx = torch.where(ok, hx * inv, torch.zeros_like(hx))
    hy = torch.where(ok, hy * inv, torch.zeros_like(hy))
    hz = torch.where(ok, hz * inv, torch.ones_like(hz))
    vdw = hy * wix - hx * wiy                               # v = (hy, -hx, 0); v . wi
    k = vdw / (1.0 + hz).clamp_min(1e-6)
    wdx = wix * hz + (-hx * wiz) + k * hy
    wdy = wiy * hz + (-hy * wiz) + k * (-hx)
    wdz = wiz * hz + (hx * wix + hy * wiy)
    return torch.stack([wix, wiy, wiz, wo[..., 0], wo[..., 1], wo[..., 2], hx, hy, hz, wdx, wdy, wdz], dim=-1)


def bilinear_corners(uv, h, w):
    """uv (..., 2) -> flat texel indices (..., 4) [00, 10, 01, 11] (index = y*w + x, int64), weights (..., 4).

    Corner-to-corner grid: u = 0 is the centre of texel 0, u = 1 the centre of texel w-1 (nnm_format.md section 5).
    """
    u = uv[..., 0].clamp(0.0, 1.0)
    v = uv[..., 1].clamp(0.0, 1.0)
    x = u * (w - 1)
    y = v * (h - 1)
    ix = torch.floor(x).long().clamp(0, max(w - 2, 0))
    iy = torch.floor(y).long().clamp(0, max(h - 2, 0))
    fx = x - ix.to(x.dtype)
    fy = y - iy.to(y.dtype)
    ix1 = (ix + 1).clamp(max=w - 1)
    iy1 = (iy + 1).clamp(max=h - 1)
    idx = torch.stack([iy * w + ix, iy * w + ix1, iy1 * w + ix, iy1 * w + ix1], dim=-1)
    wt = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], dim=-1)
    return idx, wt


# ----------------------------------------------------------------------------- activation, loss
def activation(x, kind, beta=1.0, eps=0.05):
    if kind == "relu":
        return torch.relu(x)
    if kind == "leaky_relu":
        return torch.where(x > 0, x, eps * x)
    s = torch.clamp(x + beta, 0.0, 2.0 * beta)
    return eps * x + (1.0 - eps) * (s * s * (0.25 / beta) + torch.clamp(x - beta, min=0.0))


def clamp_straight_through(x, lo=LOG_CLAMP[0], hi=LOG_CLAMP[1]):
    """Forward: clamp(x, lo, hi) (the deployed function). Backward: identity, as in the numpy prototype
    (its backward ignores the clip), so an instance that overshoots the clamp can still recover."""
    return x + (x.clamp(lo, hi) - x).detach()


def power_loss(out, target, n=3.0):
    """Power loss on f*cos targets (Zeltner et al. style, n = 3).

    out    (K, B, 3) pre-exp network output (already clamped), prediction = exp(out)
    target (B, 3) linear f*cos values >= 0
    returns (K,) per-instance mean over B and channels of |M(pred) - M(target)|, M(x) = n (x^(1/n) - 1).
    pred^(1/n) = exp(out / n) is computed directly (no exp followed by pow), which is stable and what the
    numpy prototype does (experiments/neural_material_train/train.py: loss_and_gout).
    """
    pred_p = torch.exp(out / n)
    tgt_p = target.clamp_min(0.0).pow(1.0 / n)
    return (n * (pred_p - tgt_p.unsqueeze(0))).abs().mean(dim=(1, 2))


# ----------------------------------------------------------------------------- model
class NeuralMaterial(nn.Module):
    """K lock-step instances of (latent grid + MLP decoder)."""

    backend = "torch"

    def __init__(self, cfg: ModelConfig, k=1, seed=0, device=None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.k = k
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)                                  # init is identical on every device
        names = []
        lat = torch.rand(k, cfg.latent_height, cfg.latent_width, cfg.latent_channels, generator=g) - 0.5
        self.latent = nn.Parameter(lat)                      # U(-0.5, 0.5) as in the prototype
        names.append("latent")
        dims = cfg.layer_dims()
        for l, (i, o) in enumerate(dims):
            last = l == len(dims) - 1
            bound = (0.3 if last else 1.0) * math.sqrt(6.0 / i)
            w = (torch.rand(k, i, o, generator=g) * 2.0 - 1.0) * bound
            b = torch.full((k, 1, o), math.log(0.2) if last else 0.0)   # initial prediction 0.2
            self.register_parameter(f"W{l}", nn.Parameter(w))
            self.register_parameter(f"b{l}", nn.Parameter(b))
            names += [f"W{l}", f"b{l}"]
        self.param_names = names
        if device is not None:
            self.to(device)

    # parameters in a fixed order (optimizer state is aligned with it)
    def param_list(self):
        return [getattr(self, n) for n in self.param_names]

    def num_decoder_params(self):
        return sum(p[0].numel() for n, p in zip(self.param_names, self.param_list()) if n != "latent")

    def forward(self, uv, wi, wo):
        """uv (B,2), wi (B,3), wo (B,3) float32 on the model's device -> (K, B, 3) clamped pre-exp output."""
        c = self.cfg
        enc = encode_bsdf_dirs(wi, wo)                                        # (B, 12)
        idx, wt = bilinear_corners(uv, c.latent_height, c.latent_width)       # (B, 4)
        lat = self.latent.reshape(self.k, c.latent_height * c.latent_width, c.latent_channels)
        z = (lat[:, idx] * wt[None, :, :, None]).sum(dim=2)                   # (K, B, C)
        a = torch.cat([enc.unsqueeze(0).expand(self.k, -1, -1), z], dim=-1)   # (K, B, 12 + C): features first
        n = len(c.layer_dims())
        for l in range(n):
            a = torch.bmm(a, getattr(self, f"W{l}")) + getattr(self, f"b{l}")      # (K,B,i) @ (K,i,o) + (K,1,o)
            if l < n - 1:
                a = activation(a, c.activation, c.beta, c.eps)
        return clamp_straight_through(a)

    def predict(self, uv, wi, wo):
        return torch.exp(self.forward(uv, wi, wo))

    @torch.no_grad()
    def select(self, keep):
        """Keep only the instances `keep` (1-D index list) of every parameter. Fresh Parameter objects."""
        keep = torch.as_tensor(keep, dtype=torch.long, device=self.latent.device)
        for name in self.param_names:
            setattr(self, name, nn.Parameter(getattr(self, name).detach()[keep].clone()))
        self.k = int(keep.numel())

    @torch.no_grad()
    def to_arrays(self, i=0):
        """Interchange arrays (numpy float32) of instance i: latent (H,W,C), W{l} (in,out), b{l} (out,)."""
        d = {"latent": self.latent[i].detach().cpu().numpy()}
        for l in range(len(self.cfg.layer_dims())):
            d[f"W{l}"] = getattr(self, f"W{l}")[i].detach().cpu().numpy()
            d[f"b{l}"] = getattr(self, f"b{l}")[i, 0].detach().cpu().numpy()
        return d


# ----------------------------------------------------------------------------- tiny-cuda-nn backend
class TcnnMaterial(nn.Module):
    """UNTESTED. Single instance (K = 1), ReLU decoder on tiny-cuda-nn, latent grid in plain torch.

    tcnn networks have no biases. Two exact workarounds keep the exported .nnm equivalent:
      * a constant 1.0 input column (after the encoded features and latent) acts as the first-layer bias;
      * the output offset log(0.2) is added outside the network and exported as the last-layer bias.
    Hidden layers get a zero bias. The parameter layout assumed by `to_arrays` is: for each layer one dense
    row-major (out, in) matrix, concatenated in order, first layer in = padded input width, last layer
    out = output width padded to 16. If that assumption is wrong, train.py's export self-check
    (forward of the torch module versus the numpy forward of the extracted arrays) fails loudly instead of
    writing a wrong model.
    """

    backend = "tcnn"

    def __init__(self, cfg: ModelConfig, seed=0, device="cuda"):
        super().__init__()
        import tinycudann as tcnn                                              # noqa: F401  (workstation only)
        if cfg.activation != "relu":
            raise ValueError("tcnn backend supports --activation relu only")
        self.cfg, self.k = cfg, 1
        self.n_real = ENC_DIM + cfg.latent_channels
        self.in_pad = 16 * math.ceil((self.n_real + 1) / 16)                   # +1: constant-one column
        self.out_pad = 16 * math.ceil(cfg.output_dim / 16)
        torch.manual_seed(seed)
        self.latent = nn.Parameter(torch.rand(1, cfg.latent_height, cfg.latent_width, cfg.latent_channels) - 0.5)
        self.net = tcnn.Network(n_input_dims=self.in_pad, n_output_dims=cfg.output_dim, network_config={
            "otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
            "n_neurons": cfg.width, "n_hidden_layers": cfg.num_hidden})
        self.out_bias = math.log(0.2)
        self.to(device)

    def param_list(self):
        return [self.latent] + list(self.net.parameters())

    def forward(self, uv, wi, wo):
        c = self.cfg
        enc = encode_bsdf_dirs(wi, wo)
        idx, wt = bilinear_corners(uv, c.latent_height, c.latent_width)
        lat = self.latent.reshape(1, c.latent_height * c.latent_width, c.latent_channels)
        z = (lat[:, idx] * wt[None, :, :, None]).sum(dim=2)[0]                 # (B, C)
        ones = torch.ones_like(enc[:, :1])
        pad = torch.zeros(enc.shape[0], self.in_pad - self.n_real - 1, device=enc.device, dtype=enc.dtype)
        x = torch.cat([enc, z, ones, pad], dim=-1)
        y = self.net(x).float() + self.out_bias
        return clamp_straight_through(y).unsqueeze(0)

    def predict(self, uv, wi, wo):
        return torch.exp(self.forward(uv, wi, wo))

    @torch.no_grad()
    def to_arrays(self, i=0):
        import numpy as np
        c = self.cfg
        flat = self.net.params.detach().float().cpu().numpy()
        shapes = [(c.width, self.in_pad)] + [(c.width, c.width)] * (c.num_hidden - 1) + [(self.out_pad, c.width)]
        mats, pos = [], 0
        for r, cc in shapes:
            mats.append(flat[pos:pos + r * cc].reshape(r, cc))
            pos += r * cc
        if pos != flat.size:
            raise RuntimeError(f"tcnn parameter count {flat.size} != assumed layout {pos}")
        d = {"latent": self.latent[0].detach().cpu().numpy()}
        d["W0"] = np.ascontiguousarray(mats[0][:, :self.n_real].T)
        d["b0"] = np.ascontiguousarray(mats[0][:, self.n_real])               # weight of the constant-one column
        for l in range(1, c.num_hidden):
            d[f"W{l}"] = np.ascontiguousarray(mats[l].T)
            d[f"b{l}"] = np.zeros(c.width, np.float32)
        d[f"W{c.num_hidden}"] = np.ascontiguousarray(mats[-1][:c.output_dim].T)
        d[f"b{c.num_hidden}"] = np.full(c.output_dim, self.out_bias, np.float32)
        return d


def make_model(cfg, k=1, backend="torch", seed=0, device="cuda"):
    if backend == "tcnn":
        if k != 1:
            raise ValueError("tcnn backend trains a single instance: use --no-culling")
        return TcnnMaterial(cfg, seed, device)
    return NeuralMaterial(cfg, k, seed, device)
