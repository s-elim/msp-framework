"""Build the LM-Grasp corpus: real LINEMOD crops + simulated grasp outcomes.

    MUJOCO_GL=egl python scripts/sr/build_lm_grasp.py --out data/lm_grasp --workers 32
    python scripts/sr/build_lm_grasp.py --pilot 12 --out results/sr/pilot   # base rates, timing

Writes one torch file per split with obs (N,4,S,S) fp16, actions (N,Na,9), T_cam_tcp
(N,Na,4,4), succ/slip/exec (N,Na), phys (N,5), obj_id, im_id, R, t, c, crop, K.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from msp.data.linemod import (  # noqa: E402
    SPLITS, crop_observation, index_lm, object_frame, propose_grasps, split_of)
from msp.oracle.linemod_sim import LM13, LinemodGraspSim, decompose_lm_models  # noqa: E402

log = logging.getLogger("lm_grasp")
_SIMS: dict[int, LinemodGraspSim] = {}
_ARGS: dict = {}


def _init(args: dict) -> None:
    _ARGS.update(args)
    objs = decompose_lm_models(Path(args["root"]), Path(args["cache"]))
    for o in objs:
        _SIMS[o.obj_id] = LinemodGraspSim(o)


def _scene(e) -> dict:
    root = Path(_ARGS["root"])
    rng = np.random.default_rng([_ARGS["seed"], e.obj_id, e.im_id])
    ob = crop_observation(root, e, size=_ARGS["size"])
    prop = propose_grasps(ob["points"], ob["c"], _ARGS["n_actions"], rng, n_table=ob["n_table"],
                          h_table=float(ob["h_table"]))
    sim = _SIMS[e.obj_id]
    phys = sim.sample_phys(rng)
    T_obj = object_frame(prop.T, e.R, e.t)
    succ = np.zeros(len(T_obj), np.float32)
    slip = np.zeros(len(T_obj), np.float32)
    exe = np.zeros(len(T_obj), bool)
    for j, T in enumerate(T_obj):
        succ[j], slip[j], exe[j] = sim.rollout(T, phys)
    return {"obj_id": e.obj_id, "im_id": e.im_id, "obs": ob["obs"], "c": ob["c"],
            "crop": ob["crop"], "d_med": ob["d_med"], "n_table": ob["n_table"], "h_table": ob["h_table"], "actions": prop.action,
            "T": prop.T.astype(np.float32), "succ": succ, "slip": slip, "exec": exe,
            "phys": phys.to_array(), "R": e.R.astype(np.float32), "t": e.t.astype(np.float32),
            "K": e.K.astype(np.float32), "visib": np.float32(e.visib_fract)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data/lm"))
    ap.add_argument("--cache", default=str(REPO / "data/lm_grasp/coacd"))
    ap.add_argument("--out", default=str(REPO / "data/lm_grasp"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n-actions", type=int, default=16)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot", type=int, default=0, help="images per object; 0 = all")
    ap.add_argument("--shard", type=int, default=-1, help="build shard i of --nshards only")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="merge shard files into splits")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    import torch

    args = {"root": a.root, "cache": a.cache, "seed": a.seed, "size": a.size,
            "n_actions": a.n_actions}
    _init(args)  # decompose once in the parent so workers only read the cache
    entries = index_lm(Path(a.root), LM13)
    if a.pilot:
        rng = np.random.default_rng(1)
        by = defaultdict(list)
        for e in entries:
            by[e.obj_id].append(e)
        entries = [by[o][i] for o in by for i in rng.choice(len(by[o]), a.pilot, replace=False)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.merge:
        rows = []
        for i in range(a.nshards):
            rows += torch.load(out / f"shard_{i:02d}_of_{a.nshards:02d}.pt", weights_only=False)
        _write(rows, out, a, args)
        return
    if a.shard >= 0:
        entries = entries[a.shard::a.nshards]
    log.info("%d scenes x %d grasps, %d workers", len(entries), a.n_actions, a.workers)

    t0 = time.time()
    rows = []
    with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(args,)) as pool:
        for i, r in enumerate(pool.imap(_scene, entries, chunksize=4)):
            rows.append(r)
            if (i + 1) % 500 == 0:
                log.info("  %d/%d  %.1f scenes/s", i + 1, len(entries), (i + 1) / (time.time() - t0))
    dt = time.time() - t0
    if a.shard >= 0:
        torch.save(rows, out / f"shard_{a.shard:02d}_of_{a.nshards:02d}.pt")
        log.info("shard %d: %d scenes in %.0f s", a.shard, len(rows), dt)
        return
    _write(rows, out, a, args, dt)


def _write(rows, out: Path, a, args: dict, dt: float = float("nan")) -> None:
    import torch

    per_obj = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        p = per_obj[r["obj_id"]]
        p[0] += len(r["succ"])
        p[1] += int(r["exec"].sum())
        p[2] += int(r["succ"].sum())
    summary = {"seconds": dt, "scenes": len(rows), "per_object": {
        str(k): {"grasps": v[0], "exec_rate": v[1] / v[0], "succ_rate": v[2] / v[0]}
        for k, v in sorted(per_obj.items())}}
    sims = {o: s.provenance() for o, s in _SIMS.items()}
    summary["provenance"] = sims
    summary["spec"] = args
    if a.pilot:
        (out / "pilot_summary.json").write_text(json.dumps(summary, indent=1))
        log.info(json.dumps(summary["per_object"], indent=1))
        log.info("%.1f s for %d scenes", dt, len(rows))
        return

    for split in SPLITS:
        sel = [r for r in rows if split_of(r["obj_id"], r["im_id"], a.seed) == split]
        blob = {k: torch.from_numpy(np.stack([r[k] for r in sel]))
                for k in ("obs", "c", "crop", "d_med", "n_table", "h_table", "actions", "T", "succ", "slip", "exec",
                          "phys", "R", "t", "K", "visib")}
        blob["obj_id"] = torch.tensor([r["obj_id"] for r in sel])
        blob["im_id"] = torch.tensor([r["im_id"] for r in sel])
        blob["spec"] = args
        blob["provenance"] = sims
        torch.save(blob, out / f"{split}.pt")
        summary[f"n_{split}"] = len(sel)
        log.info("%s: %d scenes, success %.3f, exec %.3f", split, len(sel),
                 float(blob["succ"].mean()), float(blob["exec"].float().mean()))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
