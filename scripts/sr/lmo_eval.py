"""LM-O transfer check (scientific_reports_paper/TODO_2026_10_25_lmo.md), CPU.

    python scripts/sr/lmo_eval.py --shard 0 --n-shards 2 --workers 24   # per Slurm job
    python scripts/sr/lmo_eval.py --merge                               # -> results/sr/lmo/

Every annotated LM-O instance with visib_fract >= 0.1: LM-Grasp observation (visible points,
fitted table plane, 16 observation-only proposals, seeds [0, obj, im]), outcomes simulated at the
annotated pose with the LM-Grasp physics prior, and three planners with the LM-Grasp settings:
FoundationPose point estimate (4 physics draws), U1 pose posterior (16 poses x 4 draws, residual
bank = LM-Grasp TRAIN split, k = 64) and the annotated pose (4 draws).
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
sys.path.insert(0, str(REPO / "scripts/sr"))

from msp.data.linemod import object_frame, propose_grasps, table_plane  # noqa: E402
from msp.grasp import score_under_posterior  # noqa: E402
from msp.oracle.linemod_sim import LinemodGraspSim, decompose_lm_models  # noqa: E402
from msp.pose import ModelSurface, apply_residuals, pose_features  # noqa: E402

LMO = REPO / "data/lmo/test/000002"
OUT = REPO / "results/sr/lmo"
log = logging.getLogger("lmo_eval")
_SIMS: dict[int, LinemodGraspSim] = {}
_SURF: dict[int, ModelSurface] = {}
_CTX: dict = {}


def _init(objs: tuple[int, ...]) -> None:
    for o in decompose_lm_models(REPO / "data/lm", REPO / "data/lm_grasp/coacd", obj_ids=objs):
        _SIMS[o.obj_id] = LinemodGraspSim(o)
        _SURF[o.obj_id] = ModelSurface.from_mesh(str(REPO / f"data/lm/models/obj_{o.obj_id:06d}.ply"))
    _CTX["gt"] = json.loads((LMO / "scene_gt.json").read_text())
    _CTX["info"] = json.loads((LMO / "scene_gt_info.json").read_text())
    _CTX["cam"] = json.loads((LMO / "scene_camera.json").read_text())
    from pose_posterior import bank_of, load

    z, syms = load(REPO / "results/sr/posterior/features_fpose.npz")
    from msp.pose import PosteriorSampler

    _CTX["sampler"] = PosteriorSampler(bank_of(z, syms), "knn", k=64)


def _instance(job: tuple) -> dict:
    from PIL import Image

    n, im, o, k, P_fp = job
    g = _CTX["gt"][str(im)][k]
    s = _CTX["info"][str(im)][k]
    K = np.array(_CTX["cam"][str(im)]["cam_K"], np.float64).reshape(3, 3)
    R = np.array(g["cam_R_m2c"], np.float64).reshape(3, 3)
    t = np.array(g["cam_t_m2c"], np.float64) / 1000.0
    dep = np.asarray(Image.open(LMO / "depth" / f"{im:06d}.png"), np.float32) / 1000.0
    msk = np.asarray(Image.open(LMO / "mask_visib" / f"{im:06d}_{k:06d}.png")) > 0
    vs, us = np.nonzero(msk & (dep > 0))
    z = dep[vs, us].astype(np.float64)
    pts = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], 1)
    c = pts.mean(0) if len(pts) else t.copy()
    n_tab, h_tab = table_plane(dep, msk, tuple(int(v) for v in s["bbox_visib"]), K, c,
                               np.random.default_rng([o, im, 3]))
    rng = np.random.default_rng([0, o, im])
    prop = propose_grasps(pts, c, 16, rng, n_table=n_tab, h_table=float(h_tab))
    sim = _SIMS[o]
    phys_scene = sim.sample_phys(rng)
    T_obj = object_frame(prop.T, R, t)
    out = np.array([sim.rollout(T, phys_scene) for T in T_obj])
    plan_rng = np.random.default_rng([0, 7919, n])
    phys = [sim.nominal_phys()] + [sim.sample_phys(plan_rng) for _ in range(3)]
    R_fp, t_fp = P_fp[:3, :3], P_fp[:3, 3]
    point = score_under_posterior(sim.rollout, prop.T, R_fp[None], t_fp[None], phys)
    annot = score_under_posterior(sim.rollout, prop.T, R[None], t[None], phys)
    feat = pose_features(dep.astype(np.float64), msk, K, R_fp, t_fp, _SURF[o], seed=im)
    dR, dt = _CTX["sampler"].sample(o, feat, 16, np.random.default_rng([0, 31337, o, im]))
    Rs, ts = apply_residuals(R_fp, t_fp, dR, dt)
    post = score_under_posterior(sim.rollout, prop.T, Rs, ts, phys)
    upright = float(np.degrees(np.arccos(np.clip(R[:, 2] @ n_tab, -1, 1))))
    return {"im": im, "obj": o, "inst": k, "visib": s["visib_fract"], "upright_deg": upright,
            "succ": out[:, 0].astype(np.float32), "exec": out[:, 2].astype(bool),
            "s_point": point.mean.astype(np.float32), "s_post": post.mean.astype(np.float32),
            "s_annot": annot.mean.astype(np.float32), "p_exec": post.p_exec.astype(np.float32),
            "s_pose": post.per_pose.astype(np.float16), "T": prop.T.astype(np.float32),
            "R": R, "t": t, "R_fp": R_fp, "t_fp": t_fp, "Rs": Rs.astype(np.float32),
            "ts": ts.astype(np.float32), "c": c, "n_table": n_tab, "h_table": float(h_tab)}


def run(a: argparse.Namespace) -> None:
    z = np.load(OUT / "fp_poses.npz")
    jobs = [(n, int(z["im_id"][n]), int(z["obj_id"][n]), int(z["inst"][n]), z["pose"][n])
            for n in range(len(z["im_id"]))][a.shard::a.n_shards]
    objs = tuple(sorted({j[2] for j in jobs}))
    t0 = time.time()
    with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(objs,)) as pool:
        rows = pool.map(_instance, jobs, chunksize=1)
    import torch

    torch.save(rows, OUT / f"shard_{a.shard}of{a.n_shards}.pt")
    log.info("shard %d: %d instances in %.0fs", a.shard, len(rows), time.time() - t0)


def merge(a: argparse.Namespace) -> None:
    import torch
    from evaluate_lm import paired_bootstrap, pick, selective, within_scene_auc

    rows = []
    for p in sorted(OUT.glob("shard_*of*.pt")):
        rows += torch.load(p, weights_only=False)
    rows.sort(key=lambda r: (r["im"], r["inst"]))
    torch.save(rows, OUT / "lmo_eval.pt")
    y = np.stack([r["succ"] for r in rows])
    rr = np.arange(len(y))
    up = np.array([r["upright_deg"] for r in rows]) <= 15.0
    res: dict = {"instances": len(rows), "frames": len({r["im"] for r in rows}),
                 "base_rate": float(y.mean()), "oracle": float(y.max(1).mean()),
                 "upright_fraction": float(up.mean()), "all": {}, "upright": {}}
    oks = {}
    for sub, m in (("all", np.ones(len(y), bool)), ("upright", up)):
        for key in ("s_point", "s_post", "s_annot"):
            s = np.stack([r[key] for r in rows])[m]
            ip = pick(s)
            ok = y[m][rr[: m.sum()], ip]
            oks[(sub, key)] = ok
            res[sub][key] = {"top1": float(ok.mean()),
                             "within_auc": within_scene_auc(s, y[m])[0],
                             "sel25": selective(s[rr[: m.sum()], ip], ok, 0.25), "n": int(m.sum())}
        res[sub]["post_minus_point"] = paired_bootstrap(oks[(sub, "s_post")],
                                                        oks[(sub, "s_point")])
        res[sub]["annot_minus_post"] = paired_bootstrap(oks[(sub, "s_annot")],
                                                        oks[(sub, "s_post")])
    (OUT / "lmo_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge(a) if a.merge else run(a)
