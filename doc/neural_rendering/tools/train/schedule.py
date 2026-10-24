# SPDX-License-Identifier: Apache-2.0
"""Bitterli multi-instance culling schedule and LR schedule (pure python, no torch)."""
import math


def make_schedule(instances=64, phases=4, batch_total=65536):
    """[(K_i, batch_i)] with K_{i+1} = K_i / c, c = K^(1/(P-1)), batch_i = batch_total / K_i.

    The defaults give 64@1024 -> 16@4096 -> 4@16384 -> 1@65536 (K * batch = 65536 in every phase), i.e. a
    cull factor of 4 at each of the 3 phase ends. K * batch is constant, so every phase costs the same
    number of network evaluations as a single-instance baseline with batch `batch_total`.
    """
    if phases < 1 or instances < 1:
        raise ValueError("instances and phases must be >= 1")
    if phases == 1 or instances == 1:
        return [(1, batch_total)]
    c = instances ** (1.0 / (phases - 1))
    ks = []
    for i in range(phases):
        k = max(1, int(round(instances / (c ** i))))
        if ks and k >= ks[-1]:
            k = max(1, ks[-1] - 1)
        ks.append(k)
    ks[-1] = 1
    return [(k, max(1, batch_total // k)) for k in ks]


def parse_schedule(text):
    """'64:1024,16:4096,4:16384,1:65536' -> [(64,1024),...]. The last K must be 1, K must not increase."""
    out = []
    for part in text.split(","):
        k, b = part.split(":")
        out.append((int(k), int(b)))
    if out[-1][0] != 1 or any(out[i][0] < out[i + 1][0] for i in range(len(out) - 1)):
        raise ValueError("schedule must end with K=1 and K must not increase")
    return out


def phase_steps(total_steps, n_phases):
    """Equal split; the remainder goes to the last phase."""
    base = total_steps // n_phases
    steps = [base] * n_phases
    steps[-1] += total_steps - base * n_phases
    return steps


def lr_at(step, total, lr0, end_frac=0.02):
    """Cosine decay from lr0 to lr0 * end_frac over `total` steps."""
    c = 0.5 * (1.0 + math.cos(math.pi * min(step, total) / max(total, 1)))
    return lr0 * (end_frac + (1.0 - end_frac) * c)
