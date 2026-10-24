"""Rotation and rigid-transform helpers used by the pose-uncertainty model.

Conventions (match `msp.data.linemod` and the BOP annotations)
    * A pose (R, t) maps model points into the camera frame: x_cam = R x_obj + t, metres.
    * Rotations are (..., 3, 3) float64 arrays; rotation vectors are (..., 3), radians.
    * BOP discrete symmetries are 4x4 transforms S in the MODEL frame (translation in mm in the
      file, metres here): (R S_R, R S_t + t) is an equivalent pose of a symmetric object.
"""

from __future__ import annotations

import numpy as np

__all__ = ["exp_so3", "log_so3", "make_T", "random_rotation", "reduce_symmetry",
           "rotation_angle", "symmetry_transforms"]


def _hat(w: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3, 3) skew-symmetric matrices."""
    z = np.zeros(w.shape[:-1])
    return np.stack([np.stack([z, -w[..., 2], w[..., 1]], -1),
                     np.stack([w[..., 2], z, -w[..., 0]], -1),
                     np.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vectors (..., 3) -> rotations (..., 3, 3)."""
    w = np.asarray(w, dtype=np.float64)
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = _hat(w)
    small = th < 1e-8
    th_safe = np.where(small, 1.0, th)
    a = np.where(small, 1.0, np.sin(th_safe) / th_safe)
    b = np.where(small, 0.5, (1.0 - np.cos(th_safe)) / th_safe**2)
    R: np.ndarray = np.eye(3) + a * K + b * (K @ K)
    return R


def rotation_angle(R: np.ndarray) -> np.ndarray:
    """Geodesic angle of rotations (..., 3, 3) from the identity, radians in [0, pi]."""
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.asarray(np.arccos(np.clip(c, -1.0, 1.0)))


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotations (..., 3, 3) -> rotation vectors (..., 3). Stable near 0; near pi the axis is
    taken from the symmetric part, which is what the residual statistics need."""
    R = np.asarray(R, dtype=np.float64)
    th = rotation_angle(R)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)
    out = np.where((th < 1e-8)[..., None], 0.5 * v,
                   v * (th / (2.0 * np.where(np.abs(s) < 1e-12, 1.0, s)))[..., None])
    near_pi = th > np.pi - 1e-4
    if np.any(near_pi):
        B = (R[near_pi] + np.eye(3)) / 2.0
        ax = np.sqrt(np.clip(np.diagonal(B, axis1=-2, axis2=-1), 0.0, None))
        k = np.argmax(ax, axis=-1)
        col = B[np.arange(len(k)), :, k]
        ax = col / np.linalg.norm(col, axis=-1, keepdims=True)
        out[near_pi] = ax * th[near_pi][..., None]
    return out


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.zeros((*R.shape[:-2], 4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def random_rotation(rng: np.random.Generator, n: int, sigma_rad: float) -> np.ndarray:
    """n rotations about uniform random axes by |N(0, sigma)| radians (RMS angle = sigma)."""
    ax = rng.standard_normal((n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    return exp_so3(ax * np.abs(rng.normal(0.0, sigma_rad, n))[:, None])


def symmetry_transforms(models_info: dict[str, dict[str, list[list[float]]]], obj_id: int
                        ) -> np.ndarray:
    """(S, 4, 4) model-frame symmetry transforms (identity first), translations in metres.

    Only `symmetries_discrete` is used: none of the 13 LM13 objects has a continuous symmetry
    (bowl, the one LM object with an axis symmetry, is excluded by the LM13 protocol)."""
    mats = [np.eye(4)]
    for s in models_info[str(obj_id)].get("symmetries_discrete", []):
        S = np.asarray(s, dtype=np.float64).reshape(4, 4).copy()
        S[:3, 3] /= 1000.0
        mats.append(S)
    return np.stack(mats)


def reduce_symmetry(R_hat: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray,
                    syms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The symmetry-equivalent ground-truth pose closest in rotation to the estimate.

    R_hat, R_gt (3, 3); t_gt (3,); syms (S, 4, 4) from `symmetry_transforms`.
    Returns (R_gt', t_gt') with R_gt' = R_gt S_R and t_gt' = R_gt S_t + t_gt.
    """
    Rs = R_gt[None] @ syms[:, :3, :3]
    k = int(np.argmin(rotation_angle(np.swapaxes(Rs, -1, -2) @ R_hat[None])))
    return Rs[k], R_gt @ syms[k, :3, 3] + t_gt
