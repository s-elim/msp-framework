"""FoundationPose poses for every annotated LM-O instance (visib_fract >= 0.1) of the BOP-19 test
frames, same weights, settings and known-mask input as `fp_lm_poses.py`. Runs in baselines/fp-env.

    sbatch --job-name=fp_lmo --gres=gpu:1 -c 8 --mem=32G --time=02:00:00 \
        scripts/sr/fp_run_script.sbatch scripts/sr/fp_lmo.py
        -> results/sr/lmo/fp_poses.npz (im_id, obj_id, inst, pose (N, 4, 4), secs)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FP = REPO / "FoundationPose"
LMO = REPO / "data/lmo/test/000002"
OUT = REPO / "results/sr/lmo"
log = logging.getLogger("fp_lmo")


def instances(min_visib: float = 0.1) -> list[tuple[int, int, int]]:
    """(im_id, obj_id, instance index) of every annotated instance above the visibility cut."""
    gt = json.loads((LMO / "scene_gt.json").read_text())
    info = json.loads((LMO / "scene_gt_info.json").read_text())
    return [(int(f), int(g["obj_id"]), k) for f in sorted(gt, key=int)
            for k, (g, s) in enumerate(zip(gt[f], info[f], strict=True))
            if s["visib_fract"] >= min_visib]


def main() -> None:
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
    info = json.loads((REPO / "data/lm/models/models_info.json").read_text())
    cams = json.loads((LMO / "scene_camera.json").read_text())
    todo = sorted(instances(), key=lambda x: (x[1], x[0]))  # group by object: one mesh reset each
    glctx = dr.RasterizeCudaContext()
    est, cur = None, None
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    poses = np.zeros((len(todo), 4, 4))
    secs = np.zeros(len(todo))
    for n, (im, o, k) in enumerate(todo):
        if o != cur:
            mesh = trimesh.load(REPO / f"data/lm/models/obj_{o:06d}.ply", force="mesh")
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
        cam = cams[str(im)]
        K = np.array(cam["cam_K"], np.float64).reshape(3, 3)
        rgb = cv2.cvtColor(cv2.imread(str(LMO / f"rgb/{im:06d}.png")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(LMO / f"depth/{im:06d}.png"), -1).astype(np.float64)
        depth = depth * float(cam.get("depth_scale", 1.0)) / 1000.0
        depth[depth < 0.001] = 0
        mask = cv2.imread(str(LMO / f"mask_visib/{im:06d}_{k:06d}.png"), -1) > 0
        t0 = time.time()
        poses[n] = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, ob_id=o, iteration=5)
        secs[n] = time.time() - t0
        if n % 100 == 0:
            log.info("%d/%d obj %d im %d %.2fs", n, len(todo), o, im, secs[n])
    OUT.mkdir(parents=True, exist_ok=True)
    arr = np.array(todo)
    np.savez(OUT / "fp_poses.npz", im_id=arr[:, 0], obj_id=arr[:, 1], inst=arr[:, 2],
             pose=poses, secs=secs, gpu=torch.cuda.get_device_name(0))
    log.info("done: %d instances, median %.2fs", len(todo), float(np.median(secs)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    main()
