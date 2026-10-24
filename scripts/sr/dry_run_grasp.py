"""Dry run of the execution path on LM-Grasp val frames: decision -> base targets -> checks.

    python scripts/sr/dry_run_grasp.py --tag fposeU1_s0_p4 --rows 0 10 20

LM has no robot, so the base frame here is a TABLE frame built from the depth-fitted support
plane (z = plane normal, origin on the plane below the visible centroid, x = camera x projected
onto the plane). On a robot, T_base_cam comes from hand-eye calibration instead and `reachable`
from the arm's IK; without it the output says kinematics_checked = false. Nothing is sent to any
robot; the JSON is what an executor would consume after its own safety validation.
Writes results/sr/viz/dry_run_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from msp.grasp import Workspace, check_grasp, decide_summary, grasp_in_base, plan_execution  # noqa
from msp.pose import make_T  # noqa: E402


def table_frame(n_table: np.ndarray, h_table: float, c: np.ndarray) -> np.ndarray:
    """T_base_cam for a base frame on the fitted support plane (camera-frame inputs)."""
    z = n_table / np.linalg.norm(n_table)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    T_cam_base = make_T(np.stack([x, np.cross(z, x), z], 1), c - h_table * z)
    return np.linalg.inv(T_cam_base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fposeU1_s0_p4")
    ap.add_argument("--rows", type=int, nargs="+", default=[0, 100, 200, 300])
    ap.add_argument("--tau-success", type=float, default=0.5)
    ap.add_argument("--tau-exec", type=float, default=0.9)
    a = ap.parse_args()
    import torch

    b = torch.load(REPO / "data/lm_grasp/val.pt", map_location="cpu", weights_only=False)
    p = torch.load(REPO / f"results/sr/plan/{a.tag}_val.pt", map_location="cpu",
                   weights_only=False)
    # proposals are built with a 45 deg tilt bound and a 4 cm TCP clearance (linemod.py)
    ws = Workspace(lo=(-0.4, -0.4, 0.0), hi=(0.4, 0.4, 0.4), max_tilt_deg=45.0 + 1e-6,
                   table_z=0.0, min_tcp_height=0.02, standoff=0.10)
    T_tcp_flange = make_T(np.eye(3), np.array([0.0, 0.0, 0.085]))  # palm 85 mm behind the TCP
    out = []
    for i in a.rows:
        T_base_cam = table_frame(b["n_table"][i].double().numpy(), float(b["h_table"][i]),
                                 b["c"][i].double().numpy())
        T = b["T"][i].double().numpy()
        feas = [check_grasp(grasp_in_base(T_base_cam, T[j]), ws) for j in range(len(T))]
        q25 = p["q25"][i].numpy() if "q25" in p else None
        d = decide_summary(p["s"][i].numpy(), p["p_exec"][i].numpy() if "p_exec" in p
                           else np.ones(len(T)), q25, a.tau_success, a.tau_exec,
                           feasibility=feas)
        rec = {"row": i, "obj_id": int(b["obj_id"][i]), "im_id": int(b["im_id"][i]),
               "decision": d.index, "reasons": list(d.reasons),
               "posterior_mean": np.round(d.score, 3).tolist(),
               "outcome_of_choice": None if d.index is None else float(b["succ"][i, d.index])}
        if d.index is not None:
            plan = plan_execution(T_base_cam, T[d.index], ws, T_tcp_flange)
            rec["execution"] = plan.to_dict()
        out.append(rec)
        print(i, rec["decision"], rec["outcome_of_choice"],
              {r: rec["reasons"].count(r) for r in set(rec["reasons"])})
    dst = REPO / "results/sr/viz" / f"dry_run_{a.tag}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
