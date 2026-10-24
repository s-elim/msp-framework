"""Where pose-then-plan loses grasp success, and what a pose costs on the wire.

    python scripts/sr/fp_analysis.py --out results/sr/fp_analysis.json

1. Strata of the TEST split (top-1 success of FoundationPose + planner, exact pose + planner,
   the deterministic outcome scorer seeds 0-4 and the 5-model ensemble seeds 0-4): per object,
   symmetric vs not, visible fraction (BOP visib_fract) tertiles, object diameter tertiles, and
   FoundationPose ADD(-S) error (in units of the diameter) bins.
2. Pose payload: the annotated test poses quantised on a uniform grid (translation step dt in
   mm per axis, rotation-vector step dr in degrees per axis) and entropy coded with per-coordinate
   symbol frequencies fitted on the calibration poses (add-one smoothing), the same code-length
   estimate used for the learned codes. The planner's success at that resolution is bracketed by
   the pose-noise sweep (a uniform step d has per-axis s.d. d / sqrt(12)).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/sr"))
sys.path.insert(0, str(REPO / "src"))

from evaluate_lm import LM_SYMMETRIC, add_metrics, model_points, pick  # noqa: E402

RUNS, PLAN = REPO / "results/sr/runs", REPO / "results/sr/plan"


def top1(s, y):
    return y[np.arange(len(y)), pick(s)]


def rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_rotvec()


def code_bits(cal: np.ndarray, te: np.ndarray, step: float) -> float:
    """Entropy-coded bits per pose: per-coordinate symbol frequencies from calib (add-one)."""
    tot = 0.0
    for j in range(cal.shape[1]):
        qc = np.round(cal[:, j] / step).astype(int)
        qt = np.round(te[:, j] / step).astype(int)
        lo, hi = min(qc.min(), qt.min()), max(qc.max(), qt.max())
        cnt = np.bincount(qc - lo, minlength=hi - lo + 1).astype(float) + 1.0
        p = cnt / cnt.sum()
        tot += float(-np.log2(p[qt - lo]).mean())
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/sr/fp_analysis.json"))
    a = ap.parse_args()
    te = torch.load(REPO / "data/lm_grasp/test.pt", map_location="cpu", weights_only=False)
    ca = torch.load(REPO / "data/lm_grasp/calib.pt", map_location="cpu", weights_only=False)
    y = te["succ"].numpy()
    obj, im = te["obj_id"].numpy(), te["im_id"].numpy()
    ok = {"fpose_plan": top1(torch.load(PLAN / "fpose_p4_test.pt", weights_only=False)["s"]
                             .numpy(), y),
          "exact_pose_plan": top1(torch.load(PLAN / "gt_p4_test.pt", weights_only=False)["s"]
                                  .numpy(), y)}
    ds = [torch.load(RUNS / f"direct_s{k}/scores.pt", weights_only=False)["scores"]["test"]["s"]
          .numpy() for k in range(5)]
    ok["direct"] = np.mean([top1(s, y) for s in ds], 0)
    ok["ensemble5"] = top1(np.mean(ds, 0), y)
    oracle = y.max(1)

    vis = np.zeros(len(obj))
    info_cache = {}
    for i, (o, k) in enumerate(zip(obj, im)):
        if o not in info_cache:
            info_cache[o] = json.loads((REPO / f"data/lm/test/{o:06d}/scene_gt_info.json")
                                       .read_text())
        vis[i] = info_cache[o][str(k)][0]["visib_fract"]
    models = json.loads((REPO / "data/lm/models/models_info.json").read_text())
    diam = {int(k): v["diameter"] / 1000 for k, v in models.items()}
    dia = np.array([diam[o] for o in obj])
    fp = torch.load(RUNS / "fpose/scores.pt", weights_only=False)["scores"]["test"]
    am = add_metrics(fp["R_hat"].double().numpy(), fp["t_hat"].double().numpy(),
                     te["R"].double().numpy(), te["t"].double().numpy(), obj,
                     model_points(REPO / "data/lm"), diam)
    err_d = np.array(am["add_err_m"]) / dia

    def strat(mask) -> dict:
        return {"n": int(mask.sum()), "oracle": float(oracle[mask].mean()),
                **{k: float(v[mask].mean()) for k, v in ok.items()}}

    res = {"overall": strat(np.ones(len(obj), bool)), "per_object": {}, "symmetry": {},
           "visib_tertile": {}, "diameter_tertile": {}, "fp_add_err_bins": {}}
    for o in np.unique(obj):
        res["per_object"][int(o)] = strat(obj == o)
    sym = np.isin(obj, LM_SYMMETRIC)
    res["symmetry"] = {"symmetric (eggbox, glue)": strat(sym), "asymmetric": strat(~sym)}
    for name, x, key in (("visib", vis, "visib_tertile"), ("diam", dia, "diameter_tertile")):
        q = np.quantile(x, [1 / 3, 2 / 3])
        for lab, m in ((f"low (<{q[0]:.3f})", x < q[0]),
                       (f"mid", (x >= q[0]) & (x < q[1])), (f"high (>={q[1]:.3f})", x >= q[1])):
            res[key][lab] = strat(m)
    for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, np.inf)):
        res["fp_add_err_bins"][f"[{lo}, {hi}) d"] = strat((err_d >= lo) & (err_d < hi))
    res["fp_add_err_over_diam_median"] = float(np.median(err_d))

    # pose payload
    tc, tt = ca["t"].double().numpy() * 1000, te["t"].double().numpy() * 1000  # mm
    rc = np.degrees(rotvec(ca["R"].double().numpy()))
    rt = np.degrees(rotvec(te["R"].double().numpy()))
    res["pose_payload"] = []
    for st in (1, 2, 5, 10):
        b = code_bits(tc, tt, st) + code_bits(rc, rt, st)
        res["pose_payload"].append({"step_mm_and_deg": st, "bits": b,
                                    "per_axis_sd": st / np.sqrt(12)})
    res["pose_payload_float32_bits"] = 6 * 32
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("overall", "symmetry", "visib_tertile",
                                           "diameter_tertile", "fp_add_err_bins",
                                           "pose_payload")}, indent=1))


if __name__ == "__main__":
    main()
