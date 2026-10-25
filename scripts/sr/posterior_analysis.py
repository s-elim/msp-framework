"""Test-split breakdowns of the pose-posterior planner for the manuscript (session 4).

    python scripts/sr/posterior_analysis.py --out results/sr/posterior_analysis.json

Per FoundationPose ADD(-S) error bin (fraction of the object diameter) and per object: scenes,
top-1 success of the point-estimate planner (fpose_p4), the posterior planner (fposeU1_s0_p4) and
the annotated-pose planner (gt_p4), with paired bootstrap CIs of posterior - point. Also the
per-scene planning cost of each planner (CPU seconds = wall seconds x workers / scenes).
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/sr"))

from evaluate_lm import add_metrics, model_points, paired_bootstrap, pick  # noqa: E402

PLAN = REPO / "results/sr/plan"
BINS = (0.0, 0.02, 0.05, 0.10, np.inf)
TAGS = {"point": "fpose_p4", "posterior": "fposeU1_s0_p4", "annotated": "gt_p4"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/sr/posterior_analysis.json"))
    a = ap.parse_args()
    b = torch.load(REPO / "data/lm_grasp/test.pt", map_location="cpu", weights_only=False)
    y, obj = b["succ"].numpy(), b["obj_id"].numpy()
    r = np.arange(len(y))
    plans = {k: torch.load(PLAN / f"{t}_test.pt", map_location="cpu", weights_only=False)
             for k, t in TAGS.items()}
    ok = {k: y[r, pick(p["s"].numpy())] for k, p in plans.items()}
    info = json.loads((REPO / "data/lm/models/models_info.json").read_text())
    diam = {int(k): v["diameter"] / 1000.0 for k, v in info.items()}
    fp = plans["point"]
    err = np.array(add_metrics(fp["R"].double().numpy(), fp["t"].double().numpy(),
                               b["R"].double().numpy(), b["t"].double().numpy(), obj,
                               model_points(REPO / "data/lm"), diam)["add_err_m"])
    rel = err / np.array([diam[int(o)] for o in obj])
    out: dict = {"n": len(y), "bins": [], "objects": {}, "cost": {}}
    for lo, hi in pairwise(BINS):
        m = (rel >= lo) & (rel < hi)
        row = {"lo": lo, "hi": None if np.isinf(hi) else hi, "n": int(m.sum())}
        row |= {k: float(v[m].mean()) for k, v in ok.items()} if m.any() else {}
        if m.sum() > 1:
            row["posterior_minus_point"] = paired_bootstrap(ok["posterior"][m], ok["point"][m])
        out["bins"].append(row)
    for o in np.unique(obj):
        m = obj == o
        out["objects"][int(o)] = {"n": int(m.sum()), **{k: float(v[m].mean()) for k, v in ok.items()},
                                  "add_err_rel_median": float(np.median(rel[m]))}
    for k, t in {**TAGS, "point_p64": "fpose_p64", "two_view": "fposeTV_s0_p4"}.items():
        p = torch.load(PLAN / f"{t}_test.pt", map_location="cpu", weights_only=False)
        n_sc = p["s"].shape[0]
        out["cost"][k] = {"cpu_seconds_per_scene": p["seconds"] * p["args"]["workers"] / n_sc,
                          "workers": p["args"]["workers"]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for row in out["bins"]:
        print(row)
    print(json.dumps(out["cost"], indent=1))


if __name__ == "__main__":
    main()
