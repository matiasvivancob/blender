#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Torch-free tests of the pipeline pieces: encoding, lookup, schedule, dump format, filters, sampling,
export round trip, numpy forward against the numpy prototype, metrics, sphere records. Exit status 1 on failure.

    python3 test_torchfree.py
"""
import itertools
import os
import sys
import tempfile
import traceback

import numpy as np

import _paths  # noqa: F401
import dataset
import encoding_np
import evaluate
import export_nnm
import make_synthetic_dump
import metrics
import np_model
import numpy_train_small
import png_io
import sampling
import schedule
import sphere
from nnm import reference as ref
from nnm.format import read_nnm

T = _paths.load_proto()
FAILS = []


def test(fn):
    try:
        fn()
        print(f"[PASS] {fn.__name__}")
    except Exception:
        FAILS.append(fn.__name__)
        print(f"[FAIL] {fn.__name__}\n{traceback.format_exc()}")
    return fn


def hemi(rng, n, cmin=0.005):
    ct = rng.uniform(cmin, 1.0, n)
    st = np.sqrt(1 - ct * ct)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1).astype(np.float32)


@test
def encoding_matches_reference_and_float64():
    rng = np.random.default_rng(0)
    wi, wo = hemi(rng, 20000), hemi(rng, 20000)
    wo[:50] = wi[:50]
    a = encoding_np.encode_bsdf_dirs(wi, wo)
    assert a.shape == (20000, 12) and a.dtype == np.float32
    assert np.abs(a - ref.encode_bsdf_dirs(wi, wo)).max() < 2e-6
    assert np.abs(a - encoding_np.encode_bsdf_dirs_f64(wi, wo)).max() < 5e-5
    assert np.abs(encoding_np.encode_bsdf_dirs(wi.reshape(2, 10000, 3), wo.reshape(2, 10000, 3)).reshape(-1, 12) - a).max() == 0
    # wd must be wi rotated so that h maps to +z: |wd| = 1 and wd . z = wi . h
    wd, h = a[:, 9:12], a[:, 6:9]
    assert np.abs(np.linalg.norm(wd, axis=1) - 1).max() < 2e-6
    assert np.abs(wd[:, 2] - np.sum(a[:, 0:3] * h, 1)).max() < 3e-6


@test
def latent_lookup_matches_reference():
    rng = np.random.default_rng(1)
    for (H, W) in ((7, 5), (1, 1), (4, 1), (64, 64)):
        lat = rng.uniform(-1, 1, (H, W, 3)).astype(np.float16)
        uv = rng.random((2000, 2)).astype(np.float32)
        uv[:4] = [[0, 0], [1, 1], [0, 1], [1, 0]]
        got = encoding_np.latent_lookup(lat.astype(np.float32), uv)
        want = ref.latent_sample_bilinear(lat, uv[:, 0], uv[:, 1])
        assert np.abs(got - want).max() < 2e-6, (H, W)
    # corner-to-corner: u = 0 / 1 hit the first / last texel exactly
    lat = rng.random((5, 6, 2)).astype(np.float32)
    g = encoding_np.latent_lookup(lat, np.array([[0, 0], [1, 1]], np.float32))
    assert np.allclose(g[0], lat[0, 0]) and np.allclose(g[1], lat[-1, -1])


@test
def schedule_defaults_and_errors():
    s = schedule.make_schedule()
    assert s == [(64, 1024), (16, 4096), (4, 16384), (1, 65536)], s
    assert all(k * b == 65536 for k, b in s)
    s2 = schedule.make_schedule(16, 3, 4096)
    assert [k for k, _ in s2] == [16, 4, 1] and all(k * b == 4096 for k, b in s2), s2
    assert schedule.make_schedule(1, 4, 1000) == [(1, 1000)] and schedule.make_schedule(8, 1, 10) == [(1, 10)]
    assert schedule.parse_schedule("4:64,1:256") == [(4, 64), (1, 256)]
    for bad in ("4:64,2:128", "1:64,4:16"):
        try:
            schedule.parse_schedule(bad)
            raise AssertionError("accepted " + bad)
        except ValueError:
            pass
    assert schedule.phase_steps(10, 4) == [2, 2, 2, 4] and sum(schedule.phase_steps(100000, 4)) == 100000
    assert abs(schedule.lr_at(0, 100, 1.0) - 1.0) < 1e-12 and abs(schedule.lr_at(100, 100, 1.0) - 0.02) < 1e-12
    assert 0.02 < schedule.lr_at(50, 100, 1.0) < 1.0


@test
def dump_roundtrip_validation_and_filter():
    d, _ = make_synthetic_dump.make(5000, seed=3, invalid_frac=0.05)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "d.npz")
        dataset.save_dump(p, **d)
        L = dataset.load_dump(p)
        assert L.n == 5000 and L.frame_local is True and L.meta["material_name"].startswith("analytic")
        assert np.array_equal(L.wi, d["wi"]) and L.out.dtype == np.float32 and L.prims.dtype == np.int32
        rs, rep = dataset.filter_records(L)
        n_flag = int((d["out"][:, 7] != 0).sum())
        assert rep["flagged"] == n_flag > 0 and rep["kept"] == 5000 - n_flag and rs.n == rep["kept"], rep
        assert (rs.wi[:, 2] > 0).all() and (rs.wo[:, 2] > 0).all() and (rs.target >= 0).all()
        # extra bad records the loader must drop: NaN output, non-unit direction, below horizon, uv out of range
        d2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        d2["out"][10:13, 7] = 0
        d2["out"][10, 0] = np.nan
        d2["wi"][11] *= 2.0
        d2["wo"][12, 2] = -0.5
        d2["wo"][12] /= np.linalg.norm(d2["wo"][12])
        d2["tex_uv"][13] = [1.5, 0.2]
        d2["out"][13, 7] = 0
        dataset.save_dump(os.path.join(td, "d2.npz"), **d2)
        rs2, rep2 = dataset.filter_records(dataset.load_dump(os.path.join(td, "d2.npz")))
        assert rep2["nonfinite"] >= 1 and rep2["not_unit"] >= 1 and rep2["below_horizon"] >= 1 and rep2["uv_out_of_range"] >= 1, rep2
        # structural errors
        base = {k: np.asarray(v) for k, v in np.load(p, allow_pickle=False).items()}

        def expect_error(mut, key):
            bad = dict(base)
            mut(bad)
            q = os.path.join(td, "bad.npz")
            np.savez(q, **bad)
            try:
                dataset.load_dump(q)
            except dataset.DumpError as e:
                return str(e)
            raise AssertionError(f"accepted a dump with {key}")
        expect_error(lambda b: b.pop("wo"), "missing wo")
        expect_error(lambda b: b.__setitem__("out", b["out"][:, :7]), "7 output columns")
        expect_error(lambda b: b.__setitem__("uv", b["uv"][:-1]), "short uv")
        expect_error(lambda b: b.__setitem__("format_version", np.int32(99)), "bad version")
        expect_error(lambda b: b.__setitem__("prims", b["prims"].astype(np.float32)), "float prims")
        world = dict(d, frame_local=False)
        dataset.save_dump(os.path.join(td, "w.npz"), **world)
        try:
            dataset.filter_records(dataset.load_dump(os.path.join(td, "w.npz")))
            raise AssertionError("accepted world-space dump")
        except dataset.DumpError:
            pass
        # a dump without tex_uv and without uv_kind=texture cannot index a latent grid
        nt = dict(d)
        nt.pop("tex_uv")
        nt["meta"] = {}
        dataset.save_dump(os.path.join(td, "nt.npz"), **nt)
        try:
            dataset.filter_records(dataset.load_dump(os.path.join(td, "nt.npz")))
            raise AssertionError("accepted a dump without texture coordinates")
        except dataset.DumpError:
            pass
        # several files concatenate
        rs3, reps, _ = dataset.load_records([p, p])
        assert rs3.n == 2 * rs.n and len(reps) == 2


@test
def split_is_deterministic_and_disjoint():
    a, b = dataset.split_indices(10000, 0.05, 262144, 0)
    a2, b2 = dataset.split_indices(10000, 0.05, 262144, 0)
    assert np.array_equal(a, a2) and np.array_equal(b, b2) and len(b) == 500 and len(a) == 9500
    assert len(np.intersect1d(a, b)) == 0 and len(np.union1d(a, b)) == 10000
    assert len(dataset.split_indices(10000, 0.5, 100, 0)[1]) == 100


@test
def sampling_distribution():
    rng = np.random.default_rng(0)
    wi, wo = sampling.sample_directions(rng, 50000, peak_frac=0.5, grazing_frac=0.1)
    for w in (wi, wo):
        assert np.abs(np.linalg.norm(w, axis=1) - 1).max() < 1e-9 and w[:, 2].min() >= sampling.MIN_COS - 1e-4
    graz = (wo[:, 2] < np.cos(np.radians(75))).mean()
    assert 0.14 < graz < 0.18, graz       # 0.9 * cos^2(75 deg) (cosine-weighted) + 0.1 (grazing mixture) = 0.16
    # peak targeting puts many wi close to the mirror direction of wo
    mirror = np.stack([-wo[:, 0], -wo[:, 1], wo[:, 2]], 1)
    near = (np.sum(wi * mirror, 1) > np.cos(np.radians(10))).mean()
    wi_u, _ = sampling.sample_directions(rng, 50000, peak_frac=0.0, grazing_frac=0.0)
    near_u = (np.sum(wi_u * mirror, 1) > np.cos(np.radians(10))).mean()
    assert near > 3 * near_u, (near, near_u)
    # uniform UV sampling on a mesh: two triangles tiling the unit square, one twice the size of the other's UV area
    tri = np.array([[[0, 0], [1, 0], [0, 1]], [[1, 0], [1, 1], [0, 1]]], np.float32)
    prims, bary, tex = sampling.uniform_uv_on_mesh(rng, 40000, tri)
    assert abs((prims == 0).mean() - 0.5) < 0.02 and abs(tex.mean() - 0.5) < 0.01
    t = tri[prims]
    rec = (1 - bary.sum(1))[:, None] * t[:, 0] + bary[:, :1] * t[:, 1] + bary[:, 1:] * t[:, 2]
    assert np.abs(rec - tex).max() < 1e-5 and (bary.sum(1) <= 1 + 1e-6).all()
    p2, b2 = sampling.locate_uv(tri, tex)                       # inverse map recovers an equivalent (prim, bary)
    r2 = (1 - b2.sum(1))[:, None] * tri[p2][:, 0] + b2[:, :1] * tri[p2][:, 1] + b2[:, 1:] * tri[p2][:, 2]
    assert np.abs(r2 - tex).max() < 1e-5


@test
def numpy_forward_matches_numpy_prototype():
    """np_model.forward_np (our export-side oracle) against the prototype's own forward, independent code."""
    rng = np.random.default_rng(4)
    for width in (16, 32):
        P = T.init_params(np.random.default_rng(width), 1, width)
        for k in P:
            P[k] = P[k] + 0.2 * rng.standard_normal(P[k].shape).astype(np.float32)       # move away from the init
        wi, wo = hemi(rng, 3000), hemi(rng, 3000)
        uv = rng.random((3000, 2))
        B = T.Batch(uv, wi, wo, np.zeros((3000, 3)))
        out, _ = T.forward(P, B, "lsmelu")
        want = np.exp(out[0])
        arrays = numpy_train_small.proto_to_arrays(P)
        cfg = numpy_train_small.proto_config(width)
        got = np_model.forward_np(arrays, cfg, uv, wi, wo)
        rel = (np.abs(got - want) / (np.abs(want) + 1e-3)).max()
        assert rel < 1e-4, (width, rel)


@test
def export_roundtrip_many_configs():
    rng = np.random.default_rng(5)
    combos = list(itertools.product((16, 32, 64), (1, 2, 3, 4), ("relu", "leaky_relu", "leaky_smelu")))
    rng.shuffle(combos)
    with tempfile.TemporaryDirectory() as td:
        for i, (w, nh, act) in enumerate(combos[:14]):
            c = (4, 8, 16)[i % 3]
            cfg = np_model.ModelConfig(width=w, num_hidden=nh, latent_channels=c, latent_height=9 + i, latent_width=7 + i,
                                       activation=act, beta=0.6, eps=0.0 if act == "relu" else 0.03).validate()
            arrays = np_model.random_arrays(cfg, seed=i)
            for l in range(nh + 1):                                            # non-trivial biases
                arrays[f"b{l}"] = arrays[f"b{l}"] + 0.1 * rng.standard_normal(arrays[f"b{l}"].shape).astype(np.float32)
            path = os.path.join(td, f"m{i}.nnm")
            npz = os.path.join(td, f"m{i}.npz")
            np_model.save_arrays_npz(npz, arrays, cfg)
            _, res = export_nnm.export(npz, path, num_tv=16, latent_levels=2, verbose=False)
            m = read_nnm(path)
            assert (m.width, m.num_hidden, m.latent_channels, m.input_dim) == (w, nh, c, 12 + c)
            assert res["fp16_rel_err_p99"] < 0.02, (w, nh, act, res)
        # the verifier must notice a mismatch between file and training model
        arrays = np_model.random_arrays(np_model.ModelConfig(width=32, num_hidden=2, latent_channels=8), seed=0)
        cfg = np_model.infer_config(arrays, np_model.ModelConfig())
        npz = os.path.join(td, "x.npz")
        np_model.save_arrays_npz(npz, arrays, cfg)
        export_nnm.export(npz, os.path.join(td, "x.nnm"), verbose=False)
        wrong = np_model.ModelConfig(**dict(cfg.to_dict(), beta=0.4))
        try:
            export_nnm.verify_file(os.path.join(td, "x.nnm"), arrays, wrong)
            raise AssertionError("verify_file accepted a wrong activation parameter")
        except AssertionError as e:
            assert "deviates" in str(e), e
        # swapped input order (latent first) is the classic bug: it must fail the check too
        sw = dict(arrays)
        sw["W0"] = np.concatenate([arrays["W0"][12:], arrays["W0"][:12]], 0)
        try:
            export_nnm.verify_file(os.path.join(td, "x.nnm"), sw, cfg)
            raise AssertionError("verify_file accepted a permuted first layer")
        except AssertionError as e:
            assert "deviates" in str(e), e


@test
def trained_numpy_model_exports_and_matches_prototype():
    """Few-step numpy training -> .nnm -> decode equals the prototype's forward (exp and clamp included)."""
    d, _ = make_synthetic_dump.make(20000, seed=2)
    with tempfile.TemporaryDirectory() as td:
        dataset.save_dump(os.path.join(td, "d.npz"), **d)
        rs, _, _ = dataset.load_records(os.path.join(td, "d.npz"))
        P, losses = numpy_train_small.train(rs, 16, steps=60, batch=2048, verbose=False)
        assert losses[-1] < 0.8 * losses[0], losses[::20]
        arrays, cfg = numpy_train_small.proto_to_arrays(P), numpy_train_small.proto_config(16)
        np_model.save_arrays_npz(os.path.join(td, "t.npz"), arrays, cfg)
        export_nnm.export(os.path.join(td, "t.npz"), os.path.join(td, "t.nnm"), verbose=False)
        m = read_nnm(os.path.join(td, "t.nnm"))
        sel = rs.take(np.arange(3000))
        B = T.Batch(sel.tex_uv.astype(np.float64), sel.wi, sel.wo, sel.target)
        want = np.exp(T.forward(P, B, "lsmelu")[0][0])
        got = evaluate.decode_records(m, sel.tex_uv, sel.wi, sel.wo)
        rel = np.abs(got - want) / (np.abs(want) + 1e-3)
        assert np.quantile(rel, 0.99) < 0.02, np.quantile(rel, 0.99)
        res, emap, _ = evaluate.evaluate_records(m, sel, 8)
        assert emap.shape == (8, 8) and res["rel_l1"] < 1.0


@test
def metrics_sanity():
    rng = np.random.default_rng(6)
    t = np.abs(rng.standard_normal((5000, 3))).astype(np.float32)
    wi, wo = hemi(rng, 5000), hemi(rng, 5000)
    m0 = metrics.compute_metrics(t, t, wi, wo)
    assert m0["pow_loss"] == 0 and m0["rel_l1"] == 0 and m0["rel_l1_top1"] == 0
    m2 = metrics.compute_metrics(2 * t, t, wi, wo)
    assert abs(m2["rel_l1"] - 1) < 1e-9 and abs(m2["peak_energy_top01"] - 2) < 1e-9 and 0 < m2["grazing_frac"] < 1
    assert abs(m2["rel_l1_grazing"] - 1) < 1e-9


@test
def sphere_records_and_png():
    rec = sphere.make_records(48)
    n = len(rec["wi"])
    assert n > 1000 and rec["tex_uv"].min() >= 0 and rec["tex_uv"].max() <= 1
    assert (rec["wi"][:, 2] > 0).all() and (rec["wo"][:, 2] > 0).all()
    _, tex = make_synthetic_dump.make(10)
    sd = make_synthetic_dump.make_sphere_dump(tex, 48)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.npz")
        dataset.save_dump(p, **sd)
        d = dataset.load_dump(p)
        assert sphere.check_dump_matches(rec, d) is None
        d.wi[5] += 0.1
        assert sphere.check_dump_matches(rec, d) is not None
        gt = sphere.shade(rec, d.out[:, 0:3])
        img = sphere.compare_image(rec["mask"], gt, gt)
        png_io.write_png(os.path.join(td, "i.png"), img)
        assert png_io.read_png_size(os.path.join(td, "i.png")) == (img.shape[1], img.shape[0])
        assert sphere.sphere_rel_l1(gt, gt) == 0


if __name__ == "__main__":
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all torch-free tests passed'}")
    sys.exit(1 if FAILS else 0)
