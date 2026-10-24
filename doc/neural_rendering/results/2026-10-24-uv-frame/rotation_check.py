"""Paired 45-degree azimuth diagnostic of the existing exported numpy decoder."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

TOOLS = Path("D:/Github/Blender-RND/blender/doc/neural_rendering/tools")
sys.path.insert(0, str(TOOLS))
from nnm.format import read_nnm
from nnm.reference import decode, decode_test_inputs
from train.sampling import sample_directions

MODEL = Path("D:/Github/Blender-RND/real/cull_w32/model_proxy.nnm")
OUTPUT = Path(__file__).with_suffix(".json")
N = 65536
SEED = 45
FLOORS = (1e-6, 1e-5, 1e-4, 1e-3)


def rotate_z(directions, degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    out = directions.copy()
    out[:, 0] = c * directions[:, 0] - s * directions[:, 1]
    out[:, 1] = s * directions[:, 0] + c * directions[:, 1]
    return out


def stats(values):
    return {"median": float(np.median(values)), "mean": float(np.mean(values)),
            "p99": float(np.quantile(values, 0.99))}


def main():
    model_bytes = MODEL.read_bytes()
    model = read_nnm(model_bytes)
    assert model.caller_dim == 6 and model.output_dim == 3
    assert model.latent_channels > 0
    rng = np.random.default_rng(SEED)
    uv = rng.random((N, 2)).astype(np.float32)
    wi, wo = sample_directions(rng, N)
    caller = np.concatenate((wi, wo), axis=1).astype(np.float32)
    rotated = np.concatenate((rotate_z(wi, 45), rotate_z(wo, 45)), axis=1).astype(np.float32)

    for start in (0, 3):
        a, b = caller[:, start:start + 3], rotated[:, start:start + 3]
        np.testing.assert_array_equal(a[:, 2], b[:, 2])
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1, atol=2e-7, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(b, axis=1), 1, atol=2e-7, rtol=0)
        np.testing.assert_allclose(rotate_z(b, -45), a, atol=2e-7, rtol=0)
    np.testing.assert_allclose(np.sum(caller[:, :3] * caller[:, 3:], axis=1),
                               np.sum(rotated[:, :3] * rotated[:, 3:], axis=1),
                               atol=3e-7, rtol=0)
    np.testing.assert_array_equal(rotate_z(caller[:, :3], 0), caller[:, :3])
    if len(model.tv_inputs):
        np.testing.assert_allclose(decode_test_inputs(model, model.tv_inputs), model.tv_outputs,
                                   atol=1e-5, rtol=1e-4)

    baseline = np.empty((N, 3), np.float32)
    changed = np.empty_like(baseline)
    uv_before = uv.copy()
    for first in range(0, N, 4096):
        sl = slice(first, first + 4096)
        baseline[sl] = decode(model, caller[sl], uv[sl])
        changed[sl] = decode(model, rotated[sl], uv[sl])
    np.testing.assert_array_equal(uv, uv_before)
    np.testing.assert_array_equal(decode(model, caller[:16], uv[:16]), baseline[:16])
    assert np.isfinite(baseline).all() and np.isfinite(changed).all()
    assert (baseline >= 0).all() and (changed >= 0).all()

    a, b = baseline.astype(np.float64), changed.astype(np.float64)
    error = np.abs(b - a)
    intensity = a.mean(axis=1)
    assert a.sum() > 0 and b.sum() > 0
    relative = []
    for floor in FLOORS:
        relative.append({
            "floor_fcos_per_channel": floor,
            "record_denominator_floored_count": int((intensity < floor).sum()),
            "record_denominator_floored_fraction": float((intensity < floor).mean()),
            "channel_denominator_floored_fraction": float((a < floor).mean()),
            "record_relative_l1": stats(error.mean(axis=1) / np.maximum(intensity, floor)),
            "channel_relative_absolute": stats(error / np.maximum(a, floor)),
        })
    report = {
        "diagnostic": "same-UV simultaneous +45-degree z rotation of wi and wo",
        "interpretation": "An isotropic target is invariant at fixed UV. Nonzero differences diagnose the exported model's azimuthal dependence; they do not prove attribution of rendered image errors to a UV-frame defect.",
        "model": str(MODEL),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "decoder": str(TOOLS / "nnm/reference.py"),
        "decoder_signature": "decode(model, caller, uv=None); caller=[wi.xyz, wo.xyz]",
        "decoder_sha256": hashlib.sha256((TOOLS / "nnm/reference.py").read_bytes()).hexdigest(),
        "sampler": str(TOOLS / "train/sampling.py"),
        "sampler_sha256": hashlib.sha256((TOOLS / "train/sampling.py").read_bytes()).hexdigest(),
        "python": sys.executable,
        "numpy_version": np.__version__,
        "model_shape": {"width": model.width, "hidden_layers": model.num_hidden,
                        "encoding_id": model.encoding, "latent_channels": model.latent_channels,
                        "latent_height": model.latent_height, "latent_width": model.latent_width},
        "sampling": {"n_pairs": N, "seed": SEED, "uv": "uniform [0,1]^2; identical in each pair",
                     "directions": "sample_directions defaults: peak_frac=0.5, grazing_frac=0.1, grazing_deg=75",
                     "rotation_degrees": 45, "rotation_axis": "+z", "batch_size": 4096},
        "quantity": "RGB f*cos(wo); decoder output used directly, no extra cosine",
        "relative_error_definition": "record: mean_RGB(abs(rotated-baseline))/max(mean_RGB(baseline),floor); channel: abs(rotated-baseline)/max(baseline,floor). All relative values are fractions, not percent.",
        "relative_by_floor": relative,
        "absolute_fcos_error": {"record_mean_rgb": stats(error.mean(axis=1)),
                                "channel": stats(error)},
        "sampled_energy": {
            "definition": "Unweighted totals under the stated mixed sampler; not a hemispherical integral or furnace test.",
            "baseline_mean_rgb": a.mean(axis=0).tolist(),
            "rotated_mean_rgb": b.mean(axis=0).tolist(),
            "baseline_sum_rgb": a.sum(axis=0).tolist(),
            "rotated_sum_rgb": b.sum(axis=0).tolist(),
            "rotated_over_baseline_rgb": (b.sum(axis=0) / a.sum(axis=0)).tolist(),
            "rotated_over_baseline_total": float(b.sum() / a.sum()),
            "signed_relative_total_change": float((b.sum() - a.sum()) / a.sum()),
            "energy_weighted_relative_l1": float(error.sum() / a.sum()),
        },
        "self_checks": {"unit_directions_z_and_pair_dot_preserved": True,
                        "rotation_inverse_and_zero": True, "same_uv_unmodified": True,
                        "embedded_vectors_checked": int(len(model.tv_inputs)),
                        "repeat_decode_exact": True, "finite_nonnegative_outputs": True},
        "limitations": "One existing exported model, one seed, sampled UV/directions; no source BSDF evaluation, renderer, training or causal attribution test.",
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    OUTPUT.write_text(text, encoding="utf-8")
    saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert saved["sampling"]["n_pairs"] == N and saved["model_sha256"] == report["model_sha256"]
    print(text, end="")


if __name__ == "__main__":
    main()
