"""Pose posteriors for FoundationPose estimates (study 4): residual bank, samples, calibration.

    python scripts/sr/pose_posterior.py --mode knn --seed 0 --splits val
        -> results/sr/posterior/knn_s0_<split>.npz   R (N, M, 3, 3), t (N, M, 3) float32
    python scripts/sr/pose_posterior.py --calibrate --split val
        -> results/sr/posterior/calibration_val.json

The bank is the TRAIN split's residuals (msp.pose.residuals). Every frame, train included, is
barred from residuals of its own object within one 50-frame block (`--exclude-blocks`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from msp.pose import (  # noqa: E402
    PosteriorSampler,
    apply_residuals,
    coverage,
    fit_bank,
    pose_residual,
    rotation_angle,
    symmetry_transforms,
)

log = logging.getLogger("pose_posterior")
OUT = REPO / "results/sr/posterior"


def load(feat_path: Path) -> tuple[dict, dict[int, np.ndarray]]:
    z = dict(np.load(feat_path))
    info = json.loads((REPO / "data/lm/models/models_info.json").read_text())
    syms = {int(o): symmetry_transforms(info, int(o)) for o in np.unique(z["train_obj_id"])}
    return z, syms


def bank_of(z: dict, syms: dict) -> object:
    return fit_bank(z["train_obj_id"], z["train_im_id"], z["train_feats"], z["train_R_hat"],
                    z["train_t_hat"], z["train_R_gt"], z["train_t_gt"], syms)


def sample_split(z: dict, sampler: PosteriorSampler, split: str, m: int, seed: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    oid, iid, F = z[f"{split}_obj_id"], z[f"{split}_im_id"], z[f"{split}_feats"]
    Rh, th = z[f"{split}_R_hat"], z[f"{split}_t_hat"]
    R = np.empty((len(oid), m, 3, 3))
    t = np.empty((len(oid), m, 3))
    for i in range(len(oid)):
        rng = np.random.default_rng([seed, 31337, int(oid[i]), int(iid[i])])
        dR, dt = sampler.sample(int(oid[i]), F[i], m, rng, im_id=int(iid[i]))
        R[i], t[i] = apply_residuals(Rh[i], th[i], dR, dt)
    return R, t


def calibrate(z: dict, syms: dict, split: str, m: int, k: int, excl: int) -> dict:
    """Coverage of posterior error-magnitude quantiles against the annotated error."""
    bank = bank_of(z, syms)
    oid = z[f"{split}_obj_id"]
    true = [pose_residual(z[f"{split}_R_hat"][i], z[f"{split}_t_hat"][i], z[f"{split}_R_gt"][i],
                          z[f"{split}_t_gt"][i], syms[int(oid[i])]) for i in range(len(oid))]
    a_true = np.degrees(rotation_angle(np.stack([d[0] for d in true])))
    t_true = 1000 * np.linalg.norm(np.stack([d[1] for d in true]), axis=1)
    res: dict = {"split": split, "m": m, "k": k, "exclude_blocks": excl,
                 "true_rot_deg_median": float(np.median(a_true)),
                 "true_trans_mm_median": float(np.median(t_true)), "modes": {}}
    for mode in ("knn", "object", "gaussian", "knn_xo"):
        s = PosteriorSampler(bank, mode, k=k, exclude_blocks=excl)
        a_s = np.empty((len(oid), m))
        t_s = np.empty((len(oid), m))
        for i in range(len(oid)):
            rng = np.random.default_rng([0, 31337, int(oid[i]), int(z[f"{split}_im_id"][i])])
            dR, dt = s.sample(int(oid[i]), z[f"{split}_feats"][i], m, rng,
                              im_id=int(z[f"{split}_im_id"][i]))
            a_s[i] = np.degrees(rotation_angle(dR))
            t_s[i] = 1000 * np.linalg.norm(dt, axis=1)
        # rank correlation between predicted spread and realised error: does the posterior
        # know WHICH frames are bad (sharpness that is informative, not just calibrated)?
        from scipy.stats import spearmanr

        res["modes"][mode] = {
            "rot_coverage": coverage(a_true, a_s), "trans_coverage": coverage(t_true, t_s),
            "rot_pred_median_deg": float(np.median(np.median(a_s, 1))),
            "trans_pred_median_mm": float(np.median(np.median(t_s, 1))),
            "spearman_rot_spread_vs_err": float(spearmanr(np.median(a_s, 1), a_true)[0]),
            "spearman_trans_spread_vs_err": float(spearmanr(np.median(t_s, 1), t_true)[0])}
        log.info("%s %s: rot cov %s trans cov %s", split, mode,
                 res["modes"][mode]["rot_coverage"],
                 res["modes"][mode]["trans_coverage"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(OUT / "features_fpose.npz"))
    ap.add_argument("--mode", default="knn", choices=["knn", "object", "gaussian", "knn_xo"])
    ap.add_argument("--splits", default="val")
    ap.add_argument("--m", type=int, default=16, help="pose samples per frame")
    ap.add_argument("--k", type=int, default=64, help="neighbours (knn mode)")
    ap.add_argument("--exclude-blocks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--split", default="val", help="split for --calibrate")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    z, syms = load(Path(a.features))
    OUT.mkdir(parents=True, exist_ok=True)
    if a.calibrate:
        res = calibrate(z, syms, a.split, a.m, a.k, a.exclude_blocks)
        (OUT / f"calibration_{a.split}.json").write_text(json.dumps(res, indent=1))
        return
    sampler = PosteriorSampler(bank_of(z, syms), a.mode, k=a.k, exclude_blocks=a.exclude_blocks)
    for split in a.splits.split(","):
        R, t = sample_split(z, sampler, split, a.m, a.seed)
        p = OUT / f"{a.mode}_s{a.seed}_{split}.npz"
        np.savez(p, R=R.astype(np.float32), t=t.astype(np.float32), obj_id=z[f"{split}_obj_id"],
                 im_id=z[f"{split}_im_id"], args=json.dumps(vars(a)))
        log.info("%s: %d frames x %d samples -> %s", split, len(R), a.m, p)


if __name__ == "__main__":
    main()
