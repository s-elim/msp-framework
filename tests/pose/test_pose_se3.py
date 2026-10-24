"""SO(3) helpers and BOP symmetry reduction (study 4)."""

from __future__ import annotations

import numpy as np

from msp.pose import (
    exp_so3,
    log_so3,
    random_rotation,
    reduce_symmetry,
    rotation_angle,
    symmetry_transforms,
)


def _rand_rot(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_exp_log_round_trip_including_small_and_near_pi_angles():
    rng = np.random.default_rng(0)
    ax = rng.standard_normal((200, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    ang = np.concatenate([np.full(50, 1e-9), rng.uniform(0, np.pi - 1e-3, 100),
                          np.full(50, np.pi - 1e-6)])
    w = ax * ang[:, None]
    R = exp_so3(w)
    assert np.allclose(np.swapaxes(R, 1, 2) @ R, np.eye(3), atol=1e-10)
    assert np.allclose(np.linalg.det(R), 1.0)
    assert np.allclose(rotation_angle(R), ang, atol=1e-6)
    assert np.allclose(exp_so3(log_so3(R)), R, atol=1e-5)


def test_random_rotation_rms_angle_matches_sigma():
    R = random_rotation(np.random.default_rng(1), 20000, np.radians(5.0))
    rms = np.sqrt(np.mean(rotation_angle(R) ** 2))
    assert abs(np.degrees(rms) - 5.0) < 0.1


def test_symmetry_reduction_recovers_the_equivalent_annotation():
    """An estimate equal to the symmetric twin of the annotation has zero reduced residual."""
    info = {"10": {"symmetries_discrete": [[-1, 0, 0, 2.0, 0, -1, 0, 3.0, 0, 0, 1, 0,
                                            0, 0, 0, 1]]}}
    syms = symmetry_transforms(info, 10)
    assert syms.shape == (2, 4, 4) and np.allclose(syms[1, :3, 3], [0.002, 0.003, 0.0])
    rng = np.random.default_rng(2)
    R, t = _rand_rot(rng), rng.standard_normal(3)
    R_twin, t_twin = R @ syms[1, :3, :3], R @ syms[1, :3, 3] + t
    Rr, tr = reduce_symmetry(R_twin, R, t, syms)
    assert np.allclose(Rr, R_twin) and np.allclose(tr, t_twin)
    # a non-symmetric object keeps its annotation
    Rr, tr = reduce_symmetry(R_twin, R, t, symmetry_transforms({"1": {}}, 1))
    assert np.allclose(Rr, R) and np.allclose(tr, t)
