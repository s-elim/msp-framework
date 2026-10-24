"""Observable quality features of an estimated object pose.

Every feature uses only what a deployed system has: the RGB-D frame, the detection mask, the
camera intrinsics, the object's CAD model and the ESTIMATED pose. None uses the annotated pose,
so the same function serves the residual bank (train split) and deployment.

Features (FEATURE_NAMES order)
    fit_med_mm     median distance of the visible points to the model surface at the estimate
    fit_p90_mm     90th percentile of that distance
    outlier_frac   fraction of visible points farther than `outlier_mm` from the surface
                   (segmentation leaks, depth artefacts, or a wrong pose)
    sil_cover      fraction of the estimated model silhouette covered by the mask (low when the
                   object is occluded, or when the estimate is displaced)
    mask_spill     fraction of the mask outside the estimated silhouette
    valid_depth    fraction of mask pixels with a depth reading
    log_area       log of the mask area in pixels (scale and distance)
    view_x/y/z     unit vector from the object origin to the camera, in the ESTIMATED object
                   frame (which side of the object is seen)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

__all__ = ["FEATURE_NAMES", "ModelSurface", "pose_features"]

FEATURE_NAMES = ("fit_med_mm", "fit_p90_mm", "outlier_frac", "sil_cover", "mask_spill",
                 "valid_depth", "log_area", "view_x", "view_y", "view_z")


@dataclass(frozen=True)
class ModelSurface:
    """Dense surface samples of one CAD model (metres, model frame) and their KD-tree."""

    points: np.ndarray  # (P, 3)
    tree: cKDTree

    @classmethod
    def from_points(cls, points: np.ndarray) -> ModelSurface:
        p = np.asarray(points, dtype=np.float64)
        assert p.ndim == 2 and p.shape[1] == 3, p.shape
        return cls(points=p, tree=cKDTree(p))

    @classmethod
    def from_mesh(cls, mesh_path: str, n: int = 30000, seed: int = 0, scale: float = 1e-3
                  ) -> ModelSurface:
        import trimesh

        m = trimesh.load(mesh_path, force="mesh")
        pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
        return cls.from_points(np.asarray(pts) * scale)


def _backproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    vs, us = np.nonzero(mask & (depth > 0))
    z = depth[vs, us]
    return np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], 1)


def silhouette(surface: ModelSurface, R: np.ndarray, t: np.ndarray, K: np.ndarray,
               shape: tuple[int, int], dilate: int = 1) -> np.ndarray:
    """Binary silhouette (H, W) of the model at pose (R, t), by splatting dense surface samples
    and closing the pin holes with a `dilate`-pixel dilation."""
    pc = surface.points @ R.T + t
    pc = pc[pc[:, 2] > 1e-3]
    u = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
    H, W = shape
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    sil = np.zeros(shape, bool)
    sil[v[ok], u[ok]] = True
    if dilate > 0:
        sil = ndimage.binary_dilation(sil, iterations=dilate)
    return sil


def pose_features(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, R_hat: np.ndarray,
                  t_hat: np.ndarray, surface: ModelSurface, max_points: int = 3000,
                  outlier_mm: float = 10.0, seed: int = 0) -> np.ndarray:
    """Feature vector (len(FEATURE_NAMES),) for one estimated pose.

    depth (H, W) metres with 0 = invalid; mask (H, W) bool; K (3, 3); R_hat (3, 3); t_hat (3,)
    metres. Degenerate inputs (empty mask, no valid depth) give the worst-case values (large
    fit distance, zero cover) rather than NaN, so they land among the bank's failures.
    """
    assert depth.shape == mask.shape and depth.ndim == 2, (depth.shape, mask.shape)
    assert R_hat.shape == (3, 3) and t_hat.shape == (3,) and K.shape == (3, 3)
    area = int(mask.sum())
    pts = _backproject(depth, mask, K)
    valid = len(pts) / area if area else 0.0
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    if len(pts):
        d = surface.tree.query((pts - t_hat) @ R_hat)[0] * 1000.0  # R^T (x - t), row form
        fit_med, fit_p90 = float(np.median(d)), float(np.percentile(d, 90))
        out_frac = float((d > outlier_mm).mean())
    else:
        fit_med = fit_p90 = 100.0
        out_frac = 1.0
    sil = silhouette(surface, R_hat, t_hat, K, mask.shape)
    n_sil = int(sil.sum())
    cover = float((sil & mask).sum() / n_sil) if n_sil else 0.0
    spill = float((mask & ~sil).sum() / area) if area else 1.0
    cam = -R_hat.T @ t_hat
    view = cam / max(np.linalg.norm(cam), 1e-9)
    return np.array([fit_med, fit_p90, out_frac, cover, spill, valid, np.log(max(area, 1)),
                     *view], dtype=np.float64)
