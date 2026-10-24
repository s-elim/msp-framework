"""LM-Grasp invariants: frames, proposals, splits, baseline geometry and the evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from msp.data.linemod import object_frame, propose_grasps, split_of, tcp_pose
from msp.models.lm_baselines import grasp_patches, rot6d_to_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sr"))


def test_tcp_pose_is_a_rotation_with_approach_along_minus_z():
    rng = np.random.default_rng(0)
    for _ in range(20):
        appr = rng.standard_normal(3)
        T = tcp_pose(np.zeros(3), appr, rng.uniform(0, np.pi))
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(-R[:, 2], appr / np.linalg.norm(appr))


def test_object_frame_inverts_the_pose():
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    R = q * np.sign(np.linalg.det(q))
    t = rng.standard_normal(3)
    T_cam = tcp_pose(rng.standard_normal(3), rng.standard_normal(3), 0.3)
    T_obj = object_frame(T_cam, R, t)
    T_co = np.eye(4)
    T_co[:3, :3], T_co[:3, 3] = R, t
    assert np.allclose(T_co @ T_obj, T_cam)


def test_proposals_depend_only_on_the_observation():
    """Same visible points and table plane -> same proposals; no pose argument exists."""
    pts = np.random.default_rng(2).normal(size=(500, 3)) * 0.03 + [0, 0, 0.8]
    c = pts.mean(0)
    n = np.array([0.0, -0.7, -0.7]) / np.sqrt(0.98)
    a = propose_grasps(pts, c, 16, np.random.default_rng(5), n_table=n, h_table=0.05)
    b = propose_grasps(pts, c, 16, np.random.default_rng(5), n_table=n, h_table=0.05)
    assert np.array_equal(a.action, b.action)
    assert a.action.shape == (16, 9)
    # approach directions within 45 degrees of straight down onto the table
    down = -n
    cosang = (-a.T[:, :3, 2]) @ down
    assert (cosang >= np.cos(np.radians(45.0)) - 1e-9).all()


def test_proposal_tcp_clears_the_table():
    pts = np.random.default_rng(3).normal(size=(300, 3)) * 0.02 + [0, 0, 0.8]
    c = pts.mean(0)
    n = np.array([0.0, -1.0, 0.0])
    p = propose_grasps(pts, c, 64, np.random.default_rng(0), n_table=n, h_table=0.0)
    h = (p.T[:, :3, 3] - c) @ n
    assert (h >= 0.04 - 1e-9).all()


def test_block_split_is_deterministic_and_blockwise():
    for oid in (1, 8):
        for blk in range(10):
            ids = range(blk * 50, blk * 50 + 50)
            assert len({split_of(oid, i) for i in ids}) == 1
    assert split_of(5, 123) == split_of(5, 123)


def test_rot6d_gives_rotations():
    R = rot6d_to_matrix(torch.randn(32, 6))
    eye = torch.eye(3).expand(32, 3, 3)
    assert torch.allclose(R.transpose(1, 2) @ R, eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(32), atol=1e-5)


def test_grasp_patches_shapes_and_centering():
    b, na, s = 2, 3, 96
    depth = torch.rand(b, 1, s, s)
    K = torch.tensor([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]]).expand(b, 3, 3)
    crop = torch.tensor([[275.0, 192.0, 100.0]]).expand(b, 3)
    T = torch.eye(4).expand(b, na, 4, 4).clone()
    T[..., 2, 3] = 0.8
    c = torch.tensor([[0.0, 0.0, 0.8]]).expand(b, 3)
    p, sc = grasp_patches(depth, T, K, crop, c)
    assert p.shape == (b * na, 1, 32, 32)
    assert sc.shape == (b * na, 4)
    assert torch.allclose(p.mean(dim=(1, 2, 3)), torch.zeros(b * na), atol=1e-5)


def test_auc_handles_ties_and_extremes():
    from evaluate_lm import auc_1d

    assert auc_1d(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0
    assert auc_1d(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
    assert np.isnan(auc_1d(np.array([0.5, 0.4]), np.array([1, 1])))


def test_certificate_is_valid_on_exchangeable_selected_pairs():
    """Selection-conditional calibration covers the selected candidate at >= 1 - alpha."""
    from evaluate_lm import certificate

    rng = np.random.default_rng(0)
    n, na = 4000, 8
    p = rng.uniform(size=(2 * n, na))
    y = (rng.uniform(size=p.shape) < p).astype(np.float32)
    out = certificate(p[:n], y[:n], p[n:], y[n:], alpha=0.1)
    assert out["coverage"] >= 0.88
    if out["certified_fraction"] > 0:
        assert out["certified_precision"] >= 0.88


@pytest.mark.parametrize("deg", [0.0, 10.0])
def test_noise_rotation_is_orthonormal(deg):
    from plan_with_pose import random_rotation

    R = random_rotation(np.random.default_rng(0), deg)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
