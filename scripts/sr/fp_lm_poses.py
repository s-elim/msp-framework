"""FoundationPose (model-based, official pretrained weights) poses for the LM-Grasp scenes.

Runs in the separate FoundationPose env (baselines/fp-env, torch 2.7.1+cu128), not the repo
.venv. Same input contract as every LM-Grasp method: the real LM test image of the scene and its
ground-truth visible mask (the "known detection" assumption, FoundationPose's own LINEMOD
protocol with detect_type='mask'), plus the object's scanned CAD model and the camera K.
Default 5 refinement iterations, as in FoundationPose's run_linemod.py.

    $FPPY scripts/sr/fp_lm_poses.py --split test --shard 0 --n-shards 2
    $FPPY scripts/sr/fp_lm_poses.py --merge     # -> results/sr/runs/fpose/scores.pt

The merged file has the pose-run layout train_lm.py writes (scores[split] = {R_hat, t_hat}),
so evaluate_lm.py reports its ADD(-S) and plan_with_pose.py re-simulates its grasps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FP = REPO / "FoundationPose"
OUT = REPO / "results/sr/fp"
log = logging.getLogger("fp")


def run(a) -> None:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    sys.path.insert(0, str(FP))
    os.chdir(FP)
    import cv2
    import nvdiffrast.torch as dr
    import torch
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from Utils import set_seed, symmetry_tfs_from_info

    set_seed(0)
    b = torch.load(REPO / "data/lm_grasp" / f"{a.split}.pt", map_location="cpu",
                   weights_only=False)
    oid, iid = b["obj_id"].numpy(), b["im_id"].numpy()
    idx = np.array_split(np.arange(len(oid)), a.n_shards)[a.shard]
    if a.limit:  # smoke test: a few scenes of each object, written under a smoke_ name
        idx = np.concatenate([idx[oid[idx] == o][:a.limit] for o in np.unique(oid[idx])])
    root = REPO / "data/lm"
    info = json.loads((root / "models/models_info.json").read_text())
    glctx = dr.RasterizeCudaContext()
    est, cur = None, None
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    poses = np.zeros((len(idx), 4, 4), np.float64)
    secs = np.zeros(len(idx))
    for n, i in enumerate(idx):
        o, im = int(oid[i]), int(iid[i])
        if o != cur:
            mesh = trimesh.load(root / f"models/obj_{o:06d}.ply", force="mesh")
            mesh.vertices *= 1e-3
            sym = symmetry_tfs_from_info(info[str(o)], rot_angle_discrete=5)
            if est is None:
                est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                     symmetry_tfs=sym, mesh=mesh, scorer=scorer,
                                     refiner=refiner, glctx=glctx, debug=0,
                                     debug_dir=str(OUT / "debug"))
            else:
                est.reset_object(model_pts=mesh.vertices.copy(),
                                 model_normals=mesh.vertex_normals.copy(), symmetry_tfs=sym,
                                 mesh=mesh)
            cur = o
        sd = root / "test" / f"{o:06d}"
        cam = json.loads((sd / "scene_camera.json").read_text())[str(im)]
        K = np.array(cam["cam_K"], np.float64).reshape(3, 3)
        rgb = cv2.cvtColor(cv2.imread(str(sd / f"rgb/{im:06d}.png")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(sd / f"depth/{im:06d}.png"), -1).astype(np.float64)
        depth = depth * float(cam.get("depth_scale", 1.0)) / 1000.0
        depth[depth < 0.001] = 0
        mask = cv2.imread(str(sd / f"mask_visib/{im:06d}_000000.png"), -1) > 0
        t0 = time.time()
        poses[n] = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, ob_id=o, iteration=5)
        secs[n] = time.time() - t0
        if n % 50 == 0:
            log.info("%s shard %d: %d/%d obj %d im %d %.2fs", a.split, a.shard, n, len(idx), o,
                     im, secs[n])
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"smoke_{a.split}" if a.limit else f"{a.split}_{a.shard}of{a.n_shards}"
    np.savez(OUT / f"{name}.npz", idx=idx, poses=poses, secs=secs,
             gpu=torch.cuda.get_device_name(0))
    log.info("done %s shard %d: %d scenes, median %.2fs/scene", a.split, a.shard, len(idx),
             float(np.median(secs)))


def merge() -> None:
    import torch

    scores, secs = {}, []
    for split in ("val", "calib", "test"):
        parts = sorted(OUT.glob(f"{split}_*of*.npz"))
        if not parts:
            continue
        n = torch.load(REPO / "data/lm_grasp" / f"{split}.pt", map_location="cpu",
                       weights_only=False)["obj_id"].shape[0]
        P = np.full((n, 4, 4), np.nan)
        for p in parts:
            z = np.load(p)
            P[z["idx"]] = z["poses"]
            secs.append(z["secs"])
        assert not np.isnan(P).any(), f"{split}: missing shards"
        scores[split] = {"R_hat": torch.from_numpy(P[:, :3, :3]).float(),
                         "t_hat": torch.from_numpy(P[:, :3, 3]).float()}
    out = REPO / "results/sr/runs/fpose"
    out.mkdir(parents=True, exist_ok=True)
    s = np.concatenate(secs)
    torch.save({"scores": scores, "args": {"method": "foundationpose", "seed": 0, "holdout": ""},
                "wall_seconds": float(s.sum()), "median_sec_per_scene": float(np.median(s))},
               out / "scores.pt")
    log.info("merged %s, median %.2fs/scene", list(scores), float(np.median(s)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N scenes per object")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run(a)
