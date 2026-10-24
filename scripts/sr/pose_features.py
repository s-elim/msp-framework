"""Observable quality features of every FoundationPose estimate in LM-Grasp (study 4).

    python scripts/sr/pose_features.py --workers 32
        -> results/sr/posterior/features_fpose.npz

Per split: obj_id, im_id (N,), feats (N, F) in msp.pose.FEATURE_NAMES order, the estimate
R_hat (N, 3, 3), t_hat (N, 3) and the annotation R_gt, t_gt (stored for the residual bank and
the calibration check only; the features never read them).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from msp.oracle.linemod_sim import LM13  # noqa: E402
from msp.pose import FEATURE_NAMES, ModelSurface, pose_features  # noqa: E402

log = logging.getLogger("pose_features")
ROOT = REPO / "data/lm"
_SURF: dict[int, ModelSurface] = {}


def _init() -> None:
    for o in LM13:
        _SURF[o] = ModelSurface.from_mesh(str(ROOT / "models" / f"obj_{o:06d}.ply"))


def _one(job: tuple) -> np.ndarray:
    from PIL import Image

    o, im, K, R, t = job
    sd = ROOT / "test" / f"{o:06d}"
    dep = np.asarray(Image.open(sd / "depth" / f"{im:06d}.png"), np.float64) / 1000.0
    msk = np.asarray(Image.open(sd / "mask_visib" / f"{im:06d}_000000.png")) > 0
    return pose_features(dep, msk, K, R, t, _SURF[o], seed=im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default=str(REPO / "results/sr/runs/fpose/scores.pt"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--splits", default="train,val,calib,test")
    ap.add_argument("--out", default=str(REPO / "results/sr/posterior/features_fpose.npz"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    import torch

    sc = torch.load(a.pose, map_location="cpu", weights_only=False)["scores"]
    out: dict[str, np.ndarray] = {"feature_names": np.array(FEATURE_NAMES)}
    with get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for split in a.splits.split(","):
            b = torch.load(REPO / "data/lm_grasp" / f"{split}.pt", map_location="cpu",
                           weights_only=False)
            oid, iid = b["obj_id"].numpy(), b["im_id"].numpy()
            K = b["K"].double().numpy()
            Rh = sc[split]["R_hat"].double().numpy()
            th = sc[split]["t_hat"].double().numpy()
            jobs = [(int(oid[i]), int(iid[i]), K[i], Rh[i], th[i]) for i in range(len(oid))]
            t0 = time.time()
            F = np.stack(pool.map(_one, jobs, chunksize=16))
            log.info("%s: %d frames in %.0fs; feature medians %s", split, len(F),
                     time.time() - t0, np.round(np.median(F, 0), 3).tolist())
            out |= {f"{split}_obj_id": oid, f"{split}_im_id": iid, f"{split}_feats": F,
                    f"{split}_R_hat": Rh, f"{split}_t_hat": th,
                    f"{split}_R_gt": b["R"].double().numpy(),
                    f"{split}_t_gt": b["t"].double().numpy()}
            del b
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    log.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
