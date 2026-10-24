"""Residual bank, posterior samplers, leakage guard and the calibration statistic (study 4)."""

from __future__ import annotations

import numpy as np
import pytest

from msp.pose import (
    PosteriorSampler,
    apply_residuals,
    coverage,
    exp_so3,
    fit_bank,
    pose_residual,
    rotation_angle,
)


def _rand_rot(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def _bank(n: int = 400, seed: int = 0) -> tuple:
    """Object 1: error size grows with feature 0 (rot) and is constant-ish in translation."""
    rng = np.random.default_rng(seed)
    obj = np.ones(n, int)
    im = np.arange(n) * 3  # spans 24 blocks of 50 frames
    feats = np.stack([rng.uniform(0, 10, n), rng.uniform(0, 1, n)], 1)
    R_gt = np.stack([_rand_rot(rng) for _ in range(n)])
    t_gt = rng.standard_normal((n, 3)) * 0.1 + [0, 0, 0.8]
    ang = np.radians(feats[:, 0])  # error angle (deg) equals feature 0
    ax = rng.standard_normal((n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    R_hat = R_gt @ np.swapaxes(exp_so3(ax * ang[:, None]), 1, 2)
    t_hat = t_gt + rng.standard_normal((n, 3)) * 0.002
    syms = {1: np.eye(4)[None]}
    return fit_bank(obj, im, feats, R_hat, t_hat, R_gt, t_gt, syms), (R_hat, t_hat, R_gt, t_gt)


def test_residual_reproduces_the_annotation_exactly():
    bank, (R_hat, t_hat, R_gt, t_gt) = _bank(20)
    for i in range(20):
        R, t = apply_residuals(R_hat[i], t_hat[i], bank.dR[i][None], bank.dt[i][None])
        assert np.allclose(R[0], R_gt[i], atol=1e-10) and np.allclose(t[0], t_gt[i])
    dR, dt = pose_residual(R_hat[0], t_hat[0], R_gt[0], t_gt[0], np.eye(4)[None])
    assert np.allclose(dR, bank.dR[0]) and np.allclose(dt, bank.dt[0])


def test_knn_conditions_on_features():
    bank, _ = _bank()
    s = PosteriorSampler(bank, "knn", k=32)
    rng = np.random.default_rng(1)
    lo = np.degrees(rotation_angle(s.sample(1, np.array([0.5, 0.5]), 64, rng)[0]))
    hi = np.degrees(rotation_angle(s.sample(1, np.array([9.5, 0.5]), 64, rng)[0]))
    assert np.median(lo) < 2.5 and np.median(hi) > 7.5
    # the unconditional control ignores the features
    u = PosteriorSampler(bank, "object")
    a = np.degrees(rotation_angle(u.sample(1, np.array([0.5, 0.5]), 400, rng)[0]))
    assert 3.0 < np.median(a) < 7.0


def test_leakage_guard_excludes_own_and_adjacent_blocks():
    bank, _ = _bank()
    s = PosteriorSampler(bank, "knn", k=10_000, exclude_blocks=1)
    im = 500  # block 10
    rows = s.neighbours(1, np.array([5.0, 0.5]), im)
    blocks = bank.im_id[rows] // 50
    assert len(rows) > 0 and not np.isin(blocks, [9, 10, 11]).any()
    # every frame of blocks 9-11 is present in the bank, so the guard did remove something
    assert np.isin(bank.im_id // 50, [9, 10, 11]).sum() > 0
    with pytest.raises(KeyError):
        s.sample(2, np.zeros(2), 4, np.random.default_rng(0))


def test_gaussian_control_matches_second_moments():
    bank, _ = _bank()
    g = PosteriorSampler(bank, "gaussian")
    dR, dt = g.sample(1, np.zeros(2), 20000, np.random.default_rng(3))
    rms_bank = np.sqrt(np.mean(rotation_angle(bank.dR) ** 2))
    rms = np.sqrt(np.mean(rotation_angle(dR) ** 2))
    assert abs(rms - rms_bank) / rms_bank < 0.05
    assert np.allclose(dt.std(0), bank.dt.std(0), rtol=0.05)


def test_coverage_is_calibrated_when_truth_is_drawn_from_the_posterior():
    rng = np.random.default_rng(4)
    scale = rng.uniform(0.5, 5.0, 3000)
    samples = np.abs(rng.standard_normal((3000, 200))) * scale[:, None]
    truth = np.abs(rng.standard_normal(3000)) * scale
    cov = coverage(truth, samples)
    for lv, c in cov.items():
        assert abs(c - float(lv)) < 0.03


def test_second_view_mapping_carries_the_object_frame_error_across_cameras():
    """View j's estimate = annotation_j * E (object-frame error E). Mapped into camera i it must
    equal annotation_i * E: the error travels with the object, the camera motion is exact."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sr"))
    from second_view import map_to_view

    rng = np.random.default_rng(5)
    R_i, R_j = _rand_rot(rng)[None], _rand_rot(rng)[None]
    t_i, t_j = rng.standard_normal((1, 3)), rng.standard_normal((1, 3))
    E_R, E_t = exp_so3(np.array([0.05, -0.02, 0.1])), np.array([0.003, -0.001, 0.002])
    Rs_j = (R_j[0] @ E_R)[None, None]
    ts_j = (R_j[0] @ E_t + t_j[0])[None, None]
    R2, t2 = map_to_view(R_i, t_i, R_j, t_j, Rs_j, ts_j)
    assert np.allclose(R2[0, 0], R_i[0] @ E_R) and np.allclose(t2[0, 0], R_i[0] @ E_t + t_i[0])


def test_cross_object_sampler_never_draws_the_query_object():
    bank, _ = _bank()
    two = type(bank)(np.concatenate([bank.obj_id, np.full(len(bank.obj_id), 2)]),
                     np.concatenate([bank.im_id, bank.im_id]),
                     np.concatenate([bank.feats, bank.feats]),
                     np.concatenate([bank.dR, bank.dR * 0 + np.eye(3)]),
                     np.concatenate([bank.dt, bank.dt * 0]))
    s = PosteriorSampler(two, "knn_xo", k=16, xo_features=2)
    dR, dt = s.sample(1, np.array([9.5, 0.5]), 32, np.random.default_rng(0))
    assert np.allclose(dR, np.eye(3)) and np.allclose(dt, 0.0)  # all from object 2
