"""Observable pose-quality features on a synthetic RGB-D frame of a box (study 4)."""

from __future__ import annotations

import numpy as np

from msp.pose import FEATURE_NAMES, ModelSurface, exp_so3, pose_features
from msp.pose.features import silhouette

K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
F = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _box_surface(half=(0.05, 0.04, 0.03), n=40000, seed=0) -> ModelSurface:
    rng = np.random.default_rng(seed)
    h = np.asarray(half)
    p = rng.uniform(-1, 1, (n, 3)) * h
    ax = rng.integers(0, 3, n)
    p[np.arange(n), ax] = np.sign(rng.standard_normal(n)) * h[ax]  # push onto a face
    return ModelSurface.from_points(p)


def _render_depth(surface: ModelSurface, R, t, shape=(480, 640)) -> np.ndarray:
    """z-buffered splat of the surface samples: depth of the visible surface."""
    pc = surface.points @ R.T + t
    u = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
    d = np.full(shape, np.inf)
    np.minimum.at(d, (v, u), pc[:, 2])
    d[np.isinf(d)] = 0.0
    return d


def _scene() -> tuple:
    surf = _box_surface()
    R = exp_so3(np.array([0.4, -0.3, 0.2]))
    t = np.array([0.01, -0.02, 0.7])
    depth = _render_depth(surf, R, t)
    mask = silhouette(surf, R, t, K, depth.shape)
    depth = np.where(mask, depth, 0.0)
    return surf, R, t, depth, mask


def test_correct_pose_fits_and_displaced_pose_does_not():
    surf, R, t, depth, mask = _scene()
    good = pose_features(depth, mask, K, R, t, surf)
    bad = pose_features(depth, mask, K, R, t + np.array([0.01, 0.0, 0.015]), surf)
    assert good.shape == (len(FEATURE_NAMES),) and np.isfinite(good).all()
    assert good[F["fit_med_mm"]] < 1.5
    assert bad[F["fit_med_mm"]] > 3.0
    assert good[F["sil_cover"]] > 0.95 and bad[F["sil_cover"]] < good[F["sil_cover"]]
    assert bad[F["mask_spill"]] > good[F["mask_spill"]]
    view = good[[F["view_x"], F["view_y"], F["view_z"]]]
    assert np.isclose(np.linalg.norm(view), 1.0)
    assert np.allclose(view, -R.T @ t / np.linalg.norm(t))


def test_occlusion_lowers_silhouette_cover_and_degenerate_frames_do_not_crash():
    surf, R, t, depth, mask = _scene()
    occ = mask.copy()
    cols = np.nonzero(mask.any(0))[0]
    occ[:, cols[: len(cols) // 2]] = False  # left half hidden
    f = pose_features(np.where(occ, depth, 0.0), occ, K, R, t, surf)
    assert f[F["sil_cover"]] < 0.7 and f[F["fit_med_mm"]] < 1.5
    empty = pose_features(np.zeros_like(depth), np.zeros_like(mask), K, R, t, surf)
    assert np.isfinite(empty).all() and empty[F["sil_cover"]] == 0.0
    no_depth = pose_features(np.zeros_like(depth), mask, K, R, t, surf)
    assert no_depth[F["valid_depth"]] == 0.0 and no_depth[F["outlier_frac"]] == 1.0
