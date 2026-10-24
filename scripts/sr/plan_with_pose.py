"""Pose-then-plan scoring: re-simulate every candidate grasp under an ESTIMATED object pose.

The planner knows the object's identity, its scanned mesh and the simulator; it does not know
the scene's friction, mass or centre of mass, so it averages success over `--n-phys` draws
from the same prior the corpus was generated with (draw 0 is the nominal parameter set). With
the ground-truth pose this is a privileged reference: everything a pose pipeline could hope to
recover, planned with the true physics model.

    python scripts/sr/plan_with_pose.py --pose gt --split test --n-phys 4
    python scripts/sr/plan_with_pose.py --pose results/sr/runs/pose_s0/scores.pt --split test
    python scripts/sr/plan_with_pose.py --pose noise --rot-deg 10 --trans-mm 0 --split test

Study 4 (pose-uncertainty-marginalised planning): with --pose-samples, every candidate is
rolled out at each of the M posterior pose samples in that file (scripts/sr/pose_posterior.py)
under the same n-phys physics draws, and s is the posterior-mean success. Without it the
script is the original point-estimate planner, unchanged.

    python scripts/sr/plan_with_pose.py --pose results/sr/runs/fpose/scores.pt \
        --pose-samples results/sr/posterior/knn_s0_val.npz --split val --tag fposeU1_s0_p4

Output: results/sr/plan/<tag>_<split>.pt with s (N, Na) and the poses used; with
--pose-samples also p_exec (N, Na), q25 (N, Na) and s_pose (N, M, Na) float16.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from msp.data.linemod import object_frame  # noqa: E402
from msp.grasp import score_under_posterior  # noqa: E402
from msp.oracle.linemod_sim import LinemodGraspSim, PhysParams, decompose_lm_models  # noqa: E402

log = logging.getLogger("plan")
_SIMS: dict[int, LinemodGraspSim] = {}


def _init(root: str, cache: str) -> None:
    for o in decompose_lm_models(Path(root), Path(cache)):
        _SIMS[o.obj_id] = LinemodGraspSim(o)


def _phys(sim: LinemodGraspSim, n_phys: int, seed: int, i: int) -> list[PhysParams]:
    rng = np.random.default_rng([seed, 7919, i])
    return [sim.nominal_phys()] + [sim.sample_phys(rng) for _ in range(n_phys - 1)]


def _score(job) -> np.ndarray:
    i, oid, T_cam, R, t, n_phys, seed = job
    sim = _SIMS[oid]
    phys = _phys(sim, n_phys, seed, i)
    T_obj = object_frame(T_cam, R, t)
    s = np.zeros(len(T_obj), np.float32)
    for p in phys:
        for j, T in enumerate(T_obj):
            s[j] += sim.rollout(T, p)[0]
    return s / len(phys)


def _score_posterior(job) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(per-pose success (M, Na), p_exec (Na,), q25 (Na,)) under M pose samples x n_phys."""
    i, oid, T_cam, Rs, ts, n_phys, seed = job
    sim = _SIMS[oid]
    r = score_under_posterior(sim.rollout, T_cam, Rs, ts, _phys(sim, n_phys, seed, i))
    return r.per_pose.astype(np.float32), r.p_exec.astype(np.float32), r.quantile(0.25)


def random_rotation(rng: np.random.Generator, deg: float) -> np.ndarray:
    """Rotation about a uniformly random axis by |N(0, deg)| degrees."""
    ax = rng.standard_normal(3)
    ax /= np.linalg.norm(ax)
    ang = np.radians(abs(rng.normal(0.0, deg))) if deg > 0 else 0.0
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True, help="gt | noise | path to a pose scores.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-phys", type=int, default=4)
    ap.add_argument("--rot-deg", type=float, default=0.0)
    ap.add_argument("--trans-mm", type=float, default=0.0)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--data", default=str(REPO / "data/lm_grasp"))
    ap.add_argument("--root", default=str(REPO / "data/lm"))
    ap.add_argument("--cache", default=str(REPO / "data/lm_grasp/coacd"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--pose-samples", default="",
                    help="study 4: npz with R (N, M, 3, 3), t (N, M, 3) posterior samples")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N scenes only")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    import torch

    b = torch.load(Path(a.data) / f"{a.split}.pt", map_location="cpu", weights_only=False)
    n = b["obs"].shape[0]
    R = b["R"].double().numpy()
    t = b["t"].double().numpy()
    if a.pose == "gt":
        tag = a.tag or f"gt_p{a.n_phys}"
    elif a.pose == "noise":
        rng = np.random.default_rng([a.noise_seed, int(a.rot_deg * 10), int(a.trans_mm * 10)])
        R = np.stack([r @ random_rotation(rng, a.rot_deg) for r in R])
        t = t + rng.standard_normal(t.shape) * a.trans_mm / 1000.0
        tag = a.tag or f"noise_r{a.rot_deg:g}_t{a.trans_mm:g}_s{a.noise_seed}_p{a.n_phys}"
    else:
        sc = torch.load(a.pose, map_location="cpu", weights_only=False)["scores"][a.split]
        R = sc["R_hat"].double().numpy()
        t = sc["t_hat"].double().numpy()
        tag = a.tag or f"{Path(a.pose).parent.name}_p{a.n_phys}"

    T = b["T"].double().numpy()
    oid = b["obj_id"].numpy()
    if a.limit:
        n = min(n, a.limit)
    extra = {}
    t0 = time.time()
    if a.pose_samples:
        z = np.load(a.pose_samples)
        assert np.array_equal(z["obj_id"], oid) and np.array_equal(z["im_id"],
                                                                    b["im_id"].numpy()), \
            f"{a.pose_samples}: frame order differs from {a.split}.pt"
        Rs, ts = z["R"].astype(np.float64), z["t"].astype(np.float64)
        jobs = [(i, int(oid[i]), T[i], Rs[i], ts[i], a.n_phys, a.noise_seed) for i in range(n)]
        with get_context("spawn").Pool(a.workers, initializer=_init,
                                       initargs=(a.root, a.cache)) as pool:
            res = pool.map(_score_posterior, jobs, chunksize=2)
        s_pose = np.stack([r[0] for r in res])  # (N, M, Na)
        s = s_pose.mean(1)
        extra = {"p_exec": torch.from_numpy(np.stack([r[1] for r in res])),
                 "q25": torch.from_numpy(np.stack([r[2] for r in res]).astype(np.float32)),
                 "s_pose": torch.from_numpy(s_pose).half()}
    else:
        jobs = [(i, int(oid[i]), T[i], R[i], t[i], a.n_phys, a.noise_seed) for i in range(n)]
        with get_context("spawn").Pool(a.workers, initializer=_init,
                                       initargs=(a.root, a.cache)) as pool:
            s = np.stack(pool.map(_score, jobs, chunksize=8))
    out = REPO / "results/sr/plan"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"s": torch.from_numpy(s), "R": torch.from_numpy(R[:n]).float(),
                "t": torch.from_numpy(t[:n]).float(), "args": vars(a),
                "seconds": time.time() - t0, **extra},
               out / f"{tag}_{a.split}.pt")
    log.info("%s %s: %d scenes in %.0fs, mean planner score %.3f", tag, a.split, n,
             time.time() - t0, float(s.mean()))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    _ = PhysParams
    main()
