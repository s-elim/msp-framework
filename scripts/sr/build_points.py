"""Point clouds for every LM-Grasp scene (study 6): current frame + two registered frames.

    python scripts/sr/build_points.py --workers 16
        -> data/lm_grasp/points_<split>.pt

Per scene (row order of data/lm_grasp/<split>.pt):
    cur      (N, 2048, 6) float16  current-frame points [x, y, z, r, g, b]; xyz in metres,
                                   camera frame, minus c
    ctx      (N, 2, 512, 6)        points of two other frames of the same 50-frame block,
                                   registered into the CURRENT camera frame, minus c
    ctx_dt   (N, 2) int            signed frame offsets (-5, -10; +5, +10 at a block start)
    ctx_ok   (N, 2) bool           False when the frame is missing or has no valid depth
    n_valid  (N,) int              valid masked depth pixels in the current frame
    c        (N, 3)                visible centroid used for centring (equals the corpus c)

Points are the masked depth pixels back-projected with K (the "known detection" assumption every
LM-Grasp method uses). Registration uses T_ci_cj = T_ci_o T_cj_o^-1 from the annotated poses
of the STATIC scene, i.e. the camera motion a robot knows from its kinematics; the object pose
itself is not given to the scorer. Points are sampled with replacement when a frame has fewer
pixels.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

log = logging.getLogger("build_points")
ROOT = REPO / "data/lm"
BLOCK, N_CUR, N_CTX, OFFSETS = 50, 2048, 512, ((-5, -10), (5, 10))
_GT: dict[int, dict] = {}


def _init() -> None:
    for d in sorted((ROOT / "test").iterdir()):
        _GT[int(d.name)] = {"gt": json.loads((d / "scene_gt.json").read_text()),
                            "cam": json.loads((d / "scene_camera.json").read_text())}


def frame_points(o: int, im: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frame im: points (M, 3) camera frame metres, rgb (M, 3) in [0, 1], R (3, 3), t (3,)."""
    from PIL import Image

    sd = ROOT / "test" / f"{o:06d}"
    g = _GT[o]["gt"][str(im)][0]
    K = np.array(_GT[o]["cam"][str(im)]["cam_K"], np.float64).reshape(3, 3)
    dep = np.asarray(Image.open(sd / "depth" / f"{im:06d}.png"), np.float64) / 1000.0
    msk = np.asarray(Image.open(sd / "mask_visib" / f"{im:06d}_000000.png")) > 0
    rgb = np.asarray(Image.open(sd / "rgb" / f"{im:06d}.png").convert("RGB"), np.float32) / 255
    vs, us = np.nonzero(msk & (dep > 0))
    z = dep[vs, us]
    P = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], 1)
    R = np.array(g["cam_R_m2c"], np.float64).reshape(3, 3)
    t = np.array(g["cam_t_m2c"], np.float64) / 1000.0
    return P, rgb[vs, us], R, t


def _sample(P: np.ndarray, C: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(P), n, replace=len(P) < n)
    return np.concatenate([P[idx], C[idx]], 1)


def scene(job: tuple[int, int]) -> dict:
    o, im = job
    rng = np.random.default_rng([o, im, 4])
    P, C, R_i, t_i = frame_points(o, im)
    cur = np.zeros((N_CUR, 6), np.float32)
    c = P.mean(0) if len(P) else t_i.copy()
    if len(P):
        cur = _sample(P - c, C, N_CUR, rng)
    ctx = np.zeros((2, N_CTX, 6), np.float32)
    ok = np.zeros(2, bool)
    have = _GT[o]["gt"]
    same_block = [p for p in OFFSETS
                  if all(str(im + d) in have and (im + d) // BLOCK == im // BLOCK for d in p)]
    offs = same_block[0] if same_block else OFFSETS[0]
    for k, d in enumerate(offs):
        j = im + d
        if str(j) not in have or j // BLOCK != im // BLOCK:
            continue
        Pj, Cj, R_j, t_j = frame_points(o, j)
        if not len(Pj):
            continue
        Pi = (Pj - t_j) @ R_j @ R_i.T + t_i  # R_i R_j^T (x - t_j) + t_i, row form
        ctx[k] = _sample(Pi - c, Cj, N_CTX, rng)
        ok[k] = True
    return {"cur": cur.astype(np.float16), "ctx": ctx.astype(np.float16),
            "ctx_dt": np.array(offs, np.int16), "ctx_ok": ok, "n_valid": len(P),
            "c": c.astype(np.float32)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--splits", default="train,val,calib,test")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    import torch

    with get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for split in a.splits.split(","):
            b = torch.load(REPO / "data/lm_grasp" / f"{split}.pt", map_location="cpu",
                           weights_only=False)
            oid, iid = b["obj_id"].numpy(), b["im_id"].numpy()
            n = a.limit or len(oid)
            t0 = time.time()
            rows = pool.map(scene, [(int(oid[i]), int(iid[i])) for i in range(n)], chunksize=8)
            out = {k: torch.from_numpy(np.stack([r[k] for r in rows]))
                   for k in ("cur", "ctx", "ctx_dt", "ctx_ok", "c")}
            out["n_valid"] = torch.tensor([r["n_valid"] for r in rows])
            dc = (out["c"] - b["c"][:n].float()).abs().max()
            assert dc < 1e-4, f"{split}: centroid differs from the corpus by {dc:.2e} m"
            out |= {"obj_id": b["obj_id"][:n], "im_id": b["im_id"][:n]}
            name = f"points_{split}.pt" if not a.limit else f"points_smoke_{split}.pt"
            torch.save(out, REPO / "data/lm_grasp" / name)
            log.info("%s: %d scenes in %.0fs; context frames present %.3f; median valid px %d",
                     split, n, time.time() - t0, float(out["ctx_ok"].float().mean()),
                     int(out["n_valid"].median()))
            del b


if __name__ == "__main__":
    main()
