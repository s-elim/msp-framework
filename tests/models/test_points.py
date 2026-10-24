"""Point-cloud branch of the LM-Grasp scorer (study 6): frames, invariances, 3D / 4D layouts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from msp.data.linemod import tcp_pose
from msp.models.points import GraspLocalPointNet, PointBeliefEncoder, PointNetSet, tcp_frames
from msp.pose import exp_so3

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sr"))


def _actions(rng: np.random.Generator, na: int) -> torch.Tensor:
    rows = []
    for _ in range(na):
        T = tcp_pose(rng.normal(0, 0.3, 3), rng.standard_normal(3), rng.uniform(0, np.pi))
        rows.append(np.concatenate([T[:3, 3], T[:3, 0], T[:3, 1]]))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def test_tcp_frames_recover_the_tcp_rotation():
    rng = np.random.default_rng(0)
    T = tcp_pose(np.zeros(3), np.array([0.2, -0.3, -1.0]), 0.7)
    a = torch.tensor(np.concatenate([T[:3, 3], T[:3, 0], T[:3, 1]]), dtype=torch.float32)
    _, R = tcp_frames(a[None, None])
    assert torch.allclose(R[0, 0], torch.tensor(T[:3, :3], dtype=torch.float32), atol=1e-5)
    acts = _actions(rng, 8)[None]
    _, R = tcp_frames(acts)
    eye = R.transpose(-1, -2) @ R
    assert torch.allclose(eye, torch.eye(3).expand_as(eye), atol=1e-5)


def test_local_features_are_invariant_to_a_rigid_motion_of_scene_and_grasps():
    """Points and TCPs moved together: same TCP-frame neighbourhoods, hence same features."""
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    net = GraspLocalPointNet(k=64, out_dim=16).eval()
    pts = torch.cat([torch.randn(2, 500, 3) * 0.5, torch.rand(2, 500, 4)], -1)
    acts = torch.stack([_actions(rng, 6), _actions(rng, 6)])
    R = torch.tensor(exp_so3(np.array([0.3, -1.1, 0.4])), dtype=torch.float32)
    d = torch.tensor([0.4, -0.2, 0.9])
    pts2 = torch.cat([pts[..., :3] @ R.T + d, pts[..., 3:]], -1)
    acts2 = torch.cat([acts[..., :3] @ R.T + d, acts[..., 3:6] @ R.T, acts[..., 6:9] @ R.T], -1)
    with torch.no_grad():
        f1, f2 = net(pts, acts), net(pts2, acts2)
    assert f1.shape == (2, 6, 16)
    assert torch.allclose(f1, f2, atol=1e-4)
    # and they do depend on where the grasp is
    acts3 = acts.clone()
    acts3[..., :3] += 0.5
    with torch.no_grad():
        assert not torch.allclose(net(pts, acts3), f1, atol=1e-3)


def test_set_encoders_are_permutation_invariant_and_shapes_hold():
    torch.manual_seed(0)
    pts = torch.cat([torch.randn(3, 300, 3), torch.rand(3, 300, 4)], -1)
    perm = torch.randperm(300)
    g = PointNetSet(7).eval()
    with torch.no_grad():
        assert torch.allclose(g(pts), g(pts[:, perm]), atol=1e-6)
    enc = PointBeliefEncoder(latent_dim=8, pretrained=False, local_k=32, local_dim=4).eval()
    obs = torch.rand(3, 4, 96, 96)
    acts = _actions(np.random.default_rng(2), 5)[None].expand(3, -1, -1)
    with torch.no_grad():
        b1, b2 = enc(obs, pts), enc(obs, pts[:, perm])
        assert b1.mu.shape == (3, 8) and torch.allclose(b1.mu, b2.mu, atol=1e-5)
        a = enc.grasp_features(pts, torch.cat([acts, torch.zeros(3, 5, 1)], -1))
    assert a.shape == (3, 5, 9 + 1 + 4)
    with pytest.raises(ValueError):
        enc(obs)


def test_load_points_layouts(tmp_path):
    from train_lm import N_CUR_4D, T_SCALE, load_points

    n = 3
    cur = torch.rand(n, 2048, 6)
    ctx = torch.rand(n, 2, 512, 6)
    ok = torch.tensor([[True, True], [False, True], [False, False]])
    P = {"cur": cur.half(), "ctx": ctx.half(), "ctx_dt": torch.tensor([[-5, -10]] * n),
         "ctx_ok": ok, "obj_id": torch.tensor([1, 1, 2]), "im_id": torch.tensor([7, 8, 9])}
    torch.save(P, tmp_path / "points_val.pt")
    b = {"obj_id": torch.tensor([1, 1, 2]), "im_id": torch.tensor([7, 8, 9])}
    p3 = load_points(tmp_path / "val.pt", b, None, "3d").float()
    assert p3.shape == (n, 2048, 7) and (p3[..., 3] == 0).all()
    assert torch.allclose(p3[..., :3], cur.half().float()[..., :3] * T_SCALE, atol=1e-2)
    p4 = load_points(tmp_path / "val.pt", b, None, "4d").float()
    assert p4.shape == (n, 2048, 7)
    t = p4[..., 3]
    assert (t[:, :N_CUR_4D] == 0).all()
    assert torch.allclose(t[0, N_CUR_4D:N_CUR_4D + 512], torch.full((512,), -0.5))
    assert torch.allclose(t[0, N_CUR_4D + 512:], torch.full((512,), -1.0))
    assert (t[1, N_CUR_4D:N_CUR_4D + 512] == 0).all()  # missing frame -> current points, t = 0
    assert torch.allclose(t[1, N_CUR_4D + 512:], torch.full((512,), -1.0))
    assert (t[2] == 0).all()
    keep = torch.tensor([False, True, True])
    pk = load_points(tmp_path / "val.pt", {k: v[keep] for k, v in b.items()}, keep, "4d")
    assert torch.equal(pk, p4[keep].half())
    with pytest.raises(AssertionError, match="aligned"):
        load_points(tmp_path / "val.pt", {"obj_id": b["obj_id"], "im_id": b["im_id"] + 1},
                    None, "3d")
