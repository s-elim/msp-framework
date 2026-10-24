"""A pose posterior for an estimator that reports none (FoundationPose), from its residuals.

FoundationPose returns one refined pose per frame and no covariance. Its scorer ranks
hypotheses but is not calibrated as a probability, so it is not used here. Instead, the
estimator's errors are measured where the annotated pose is known (the train split) and
resampled for a new frame from the training frames that look most alike under observable
features of the estimate (`msp.pose.features`). This is a conditional residual bootstrap.

Residual parametrisation (exact for the bank frame itself)
    R_gt = R_hat dR        dR in the OBJECT frame (shape ambiguities, e.g. about a symmetry
                           axis, stay attached to the object)
    t_gt = t_hat + dt      dt in the CAMERA frame (depth bias stays along the viewing ray)
For eggbox and glue the annotated pose is first replaced by its BOP symmetry-equivalent closest
to the estimate, so a correct estimate of a symmetric object has a near-zero residual.

Samplers (`PosteriorSampler.mode`)
    knn        residuals of the k nearest bank frames of the same object in standardised
               feature space (the method)
    object     residuals of any bank frame of the same object (no conditioning; control)
    gaussian   rotation about a uniform axis by |N(0, s_R)| and dt ~ N(0, diag(s_t^2)), with
               s_R and s_t each object's RMS residual angle and per-axis residual s.d.
               (matched second moments, no shape; control)
    knn_xo     cross-object: the k nearest bank frames of the OTHER objects, on the first
               `xo_features` features (the object-agnostic fit statistics, not the view
               direction) standardised over the whole bank; for an object without annotated
               frames (secondary check)

Leakage guard: bank frames of the query's own object within `exclude_blocks` 50-frame blocks of
the query frame are never drawn, whatever the split, so adjacent near-duplicate frames on either
side of a split boundary cannot hand a frame its own residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from msp.pose.se3 import random_rotation, reduce_symmetry, rotation_angle

__all__ = ["PosteriorSampler", "ResidualBank", "apply_residuals", "coverage", "fit_bank",
           "pose_residual"]

BLOCK = 50  # frames per split block (msp.data.linemod.BLOCK)
MODES = ("knn", "object", "gaussian", "knn_xo")


def pose_residual(R_hat: np.ndarray, t_hat: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray,
                  syms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(dR (3, 3) object frame, dt (3,) camera frame, metres) with R_gt' = R_hat dR and
    t_gt' = t_hat + dt, where (R_gt', t_gt') is the symmetry-reduced annotation."""
    Rg, tg = reduce_symmetry(R_hat, R_gt, t_gt, syms)
    return R_hat.T @ Rg, tg - t_hat


def apply_residuals(R_hat: np.ndarray, t_hat: np.ndarray, dR: np.ndarray, dt: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Pose samples (M, 3, 3), (M, 3) from one estimate and M residuals."""
    assert dR.ndim == 3 and dt.ndim == 2 and len(dR) == len(dt)
    return R_hat[None] @ dR, t_hat[None] + dt


@dataclass(frozen=True)
class ResidualBank:
    obj_id: np.ndarray  # (N,) int
    im_id: np.ndarray  # (N,) int, LM frame index
    feats: np.ndarray  # (N, F) observable features of the estimate
    dR: np.ndarray  # (N, 3, 3)
    dt: np.ndarray  # (N, 3) metres

    def __post_init__(self) -> None:
        n = len(self.obj_id)
        assert self.im_id.shape == (n,) and self.feats.shape[0] == n
        assert self.dR.shape == (n, 3, 3) and self.dt.shape == (n, 3)
        assert np.isfinite(self.feats).all(), "non-finite pose features in the bank"


def fit_bank(obj_id: np.ndarray, im_id: np.ndarray, feats: np.ndarray, R_hat: np.ndarray,
             t_hat: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray,
             syms: dict[int, np.ndarray]) -> ResidualBank:
    """Residuals of N estimates against their annotations. syms[obj] is (S, 4, 4)."""
    dR = np.empty((len(obj_id), 3, 3))
    dt = np.empty((len(obj_id), 3))
    for i in range(len(obj_id)):
        dR[i], dt[i] = pose_residual(R_hat[i], t_hat[i], R_gt[i], t_gt[i], syms[int(obj_id[i])])
    return ResidualBank(np.asarray(obj_id, int), np.asarray(im_id, int),
                        np.asarray(feats, np.float64), dR, dt)


class PosteriorSampler:
    """Draws M residuals for one frame; see the module docstring for the modes."""

    def __init__(self, bank: ResidualBank, mode: str = "knn", k: int = 64,
                 exclude_blocks: int = 1, xo_features: int = 7) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.bank, self.mode, self.k, self.excl = bank, mode, k, exclude_blocks
        self.xo_f = xo_features
        q25, q50, q75 = np.percentile(bank.feats[:, :xo_features], [25, 50, 75], axis=0)
        self._xo_loc, self._xo_scale = q50, np.maximum(q75 - q25, 1e-3)
        self._idx: dict[int, np.ndarray] = {}
        self._loc: dict[int, np.ndarray] = {}
        self._scale: dict[int, np.ndarray] = {}
        self._sR: dict[int, float] = {}
        self._st: dict[int, np.ndarray] = {}
        for o in np.unique(bank.obj_id):
            ix = np.flatnonzero(bank.obj_id == o)
            f = bank.feats[ix]
            q25, q50, q75 = np.percentile(f, [25, 50, 75], axis=0)
            self._idx[int(o)] = ix
            self._loc[int(o)] = q50
            self._scale[int(o)] = np.maximum(q75 - q25, 1e-3)
            self._sR[int(o)] = float(np.sqrt(np.mean(rotation_angle(bank.dR[ix]) ** 2)))
            self._st[int(o)] = bank.dt[ix].std(0)

    def candidates(self, obj: int, im_id: int | None) -> np.ndarray:
        """Bank rows the frame may draw from (the leakage guard applied)."""
        if obj not in self._idx:
            raise KeyError(f"object {obj} has no residuals in the bank")
        ix = self._idx[obj]
        if im_id is not None and self.excl >= 0:
            far = np.abs(self.bank.im_id[ix] // BLOCK - im_id // BLOCK) > self.excl
            ix = ix[far]
        if len(ix) == 0:
            raise ValueError(f"object {obj}: leakage guard removed every bank frame")
        return ix

    def neighbours(self, obj: int, feat: np.ndarray, im_id: int | None) -> np.ndarray:
        ix = self.candidates(obj, im_id)
        z = (self.bank.feats[ix] - self._loc[obj]) / self._scale[obj]
        zq = (np.asarray(feat, np.float64) - self._loc[obj]) / self._scale[obj]
        d = np.linalg.norm(z - zq, axis=1)
        return ix[np.argsort(d, kind="stable")[: self.k]]

    def sample(self, obj: int, feat: np.ndarray, m: int, rng: np.random.Generator,
               im_id: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """M residuals (dR (M, 3, 3), dt (M, 3)) for a frame of object `obj` with features
        `feat` (F,). `im_id` enables the leakage guard (pass it for every LM frame)."""
        obj = int(obj)
        if self.mode == "knn_xo":
            pool = np.flatnonzero(self.bank.obj_id != obj)
            z = (self.bank.feats[pool, :self.xo_f] - self._xo_loc) / self._xo_scale
            zq = (np.asarray(feat, np.float64)[:self.xo_f] - self._xo_loc) / self._xo_scale
            pool = pool[np.argsort(np.linalg.norm(z - zq, axis=1), kind="stable")[: self.k]]
            pick = rng.choice(pool, m, replace=len(pool) < m)
            return self.bank.dR[pick].copy(), self.bank.dt[pick].copy()
        if self.mode == "gaussian":
            self.candidates(obj, im_id)  # same error for unknown objects
            dR = random_rotation(rng, m, self._sR[obj])
            dt = rng.standard_normal((m, 3)) * self._st[obj]
            return dR, dt
        pool = (self.neighbours(obj, feat, im_id) if self.mode == "knn"
                else self.candidates(obj, im_id))
        pick = rng.choice(pool, m, replace=len(pool) < m)
        return self.bank.dR[pick].copy(), self.bank.dt[pick].copy()


def coverage(true_err: np.ndarray, sampled_err: np.ndarray,
             levels: tuple[float, ...] = (0.5, 0.8, 0.9)) -> dict[str, float]:
    """Fraction of frames whose true error magnitude is at most the `level` quantile of the
    posterior's error magnitudes. true_err (N,), sampled_err (N, M). Calibrated: coverage ~
    level."""
    assert sampled_err.ndim == 2 and len(true_err) == len(sampled_err)
    return {f"{lv:g}": float((true_err <= np.quantile(sampled_err, lv, axis=1)).mean())
            for lv in levels}
