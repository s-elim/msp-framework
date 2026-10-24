"""Per-frame diagnostics of the pose-uncertainty planner (study 4), VAL split only.

    python scripts/sr/viz_pose_uncertainty.py --tag fposeU1_s0_p4 --samples knn_s0 --n 6

For each chosen val frame one row with three panels:
  (a) RGB with the detection mask outline;
  (b) model points projected at the FoundationPose estimate (orange) and the annotation
      (gray), and the object origin of each posterior pose sample (teal dots);
  (c) candidate TCPs with their closing axes, coloured by posterior-mean success; the planner's
      choice is circled and rejected candidates carry the `msp.grasp.decide` reason code.
Frames are picked deterministically among those where the two planners' chosen grasps have
DIFFERENT simulated outcomes (largest predicted pose spread first, one per object): n - 2 where
the posterior planner wins and 2 where the point estimate wins. Illustrative, not a statistic.
Writes results/sr/viz/<tag>_val.png (300 dpi) and .pdf.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/sr"))

from msp.grasp import decide_summary  # noqa: E402
from msp.pose import rotation_angle  # noqa: E402

ORANGE, TEAL, GRAY = "#E69F00", "#009E73", "#8C8C8C"  # Okabe-Ito
SHORT = {"accepted": "", "collision_prob": "C", "low_expected_success": "L",
         "low_success_quantile": "Q"}


def project(P: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2],
                     K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fposeU1_s0_p4")
    ap.add_argument("--point", default="fpose_p4")
    ap.add_argument("--samples", default="knn_s0")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tau-success", type=float, default=0.5)
    ap.add_argument("--tau-exec", type=float, default=0.9)
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from evaluate_lm import model_points, pick
    from PIL import Image

    b = torch.load(REPO / "data/lm_grasp/val.pt", map_location="cpu", weights_only=False)
    plan = torch.load(REPO / f"results/sr/plan/{a.tag}_val.pt", map_location="cpu",
                      weights_only=False)
    point = torch.load(REPO / f"results/sr/plan/{a.point}_val.pt", map_location="cpu",
                       weights_only=False)["s"].numpy()
    post = np.load(REPO / f"results/sr/posterior/{a.samples}_val.npz")
    s, pe = plan["s"].numpy(), plan["p_exec"].numpy()
    y = b["succ"].numpy()
    oid, iid = b["obj_id"].numpy(), b["im_id"].numpy()
    R_gt, t_gt = b["R"].double().numpy(), b["t"].double().numpy()
    Rh, th = plan["R"].double().numpy(), plan["t"].double().numpy()
    Rs, ts = post["R"].astype(np.float64), post["t"].astype(np.float64)
    pts = model_points(REPO / "data/lm", n=600)
    ip_u, ip_p = pick(s), pick(point)
    spread = np.median(np.degrees(rotation_angle(np.swapaxes(Rs, -1, -2) @ Rh[:, None])), 1)
    ok_u, ok_p = y[np.arange(len(y)), ip_u], y[np.arange(len(y)), ip_p]
    chosen, seen = [], set()
    # frames where the two planners' OUTCOMES differ, largest predicted spread first, one per
    # object; a.n - 2 where the posterior planner wins, then 2 where the point estimate wins
    for want, quota in ((1.0, a.n - 2), (0.0, 2)):
        cand = np.flatnonzero((ok_u != ok_p) & (ok_u == want))
        got = 0
        for i in cand[np.argsort(-spread[cand], kind="stable")]:
            if int(oid[i]) not in seen and got < quota:
                chosen.append(int(i))
                seen.add(int(oid[i]))
                got += 1
    fig = plt.figure(figsize=(9.6, 3.0 * len(chosen)))
    gs = fig.add_gridspec(len(chosen), 4, width_ratios=[1, 1, 1, 0.05], wspace=0.08)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(len(chosen))])
    log = []
    for r, i in enumerate(chosen):
        o, im = int(oid[i]), int(iid[i])
        sd = REPO / "data/lm/test" / f"{o:06d}"
        rgb = np.asarray(Image.open(sd / "rgb" / f"{im:06d}.png").convert("RGB"))
        msk = np.asarray(Image.open(sd / "mask_visib" / f"{im:06d}_000000.png")) > 0
        K = b["K"][i].double().numpy()
        u0, v0, side = b["crop"][i].numpy()
        box = (u0, u0 + side, v0 + side, v0)
        P = pts[o]
        uv_h = project(P @ Rh[i].T + th[i], K)
        uv_g = project(P @ R_gt[i].T + t_gt[i], K)
        uv_s = project(ts[i], K)
        T = b["T"][i].double().numpy()
        tcp = project(T[:, :3, 3], K)
        ax_end = project(T[:, :3, 3] + 0.02 * T[:, :3, 0], K)
        d = decide_summary(s[i], pe[i], tau_success=a.tau_success, tau_exec=a.tau_exec)
        for c in range(3):
            axes[r, c].imshow(rgb)
            axes[r, c].set_xlim(box[0], box[1])
            axes[r, c].set_ylim(box[2], box[3])
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].contour(msk, levels=[0.5], colors=[TEAL], linewidths=0.8)
        axes[r, 1].scatter(*uv_g.T, s=0.6, c=GRAY, alpha=0.6, linewidths=0)
        axes[r, 1].scatter(*uv_h.T, s=0.6, c=ORANGE, alpha=0.6, linewidths=0)
        axes[r, 1].scatter(*uv_s.T, s=6, c=TEAL, linewidths=0)
        sc = axes[r, 2].scatter(*tcp.T, s=18, c=s[i], cmap="viridis", vmin=0, vmax=1,
                                edgecolors="white", linewidths=0.4)
        for j in range(len(tcp)):
            axes[r, 2].plot([tcp[j, 0], ax_end[j, 0]], [tcp[j, 1], ax_end[j, 1]],
                            color=matplotlib.cm.viridis(s[i, j]), lw=0.8)
            code = SHORT.get(d.reasons[j], "")
            if code:
                axes[r, 2].annotate(code, tcp[j], fontsize=5, color="white",
                                    xytext=(2, 2), textcoords="offset points")
        axes[r, 2].scatter(*tcp[ip_u[i]], s=80, facecolors="none", edgecolors=TEAL, lw=1.2)
        axes[r, 2].scatter(*tcp[ip_p[i]], s=80, facecolors="none", edgecolors=ORANGE, lw=1.2,
                           linestyle="--")
        err = np.degrees(rotation_angle(Rh[i].T @ R_gt[i]))
        axes[r, 0].set_ylabel(f"obj {o} im {im}", fontsize=7)
        terr = 1000 * np.linalg.norm(th[i] - t_gt[i])
        axes[r, 1].set_title(f"FP err {err:.1f} deg, {terr:.1f} mm; spread {spread[i]:.1f} deg",
                             fontsize=7)
        axes[r, 2].set_title(f"U1 pick y={y[i, ip_u[i]]:.0f}, point pick y={y[i, ip_p[i]]:.0f}",
                             fontsize=7)
        log.append({"obj": o, "im": im, "fp_rot_err_deg": float(err),
                    "pred_spread_deg": float(spread[i]), "u1_pick": int(ip_u[i]),
                    "point_pick": int(ip_p[i]), "u1_outcome": float(y[i, ip_u[i]]),
                    "point_outcome": float(y[i, ip_p[i]]), "reasons": list(d.reasons),
                    "posterior_mean": s[i].round(3).tolist(),
                    "p_exec": pe[i].round(3).tolist()})
    fig.colorbar(sc, cax=fig.add_subplot(gs[:, 3]),
                 label="posterior-mean success")
    fig.text(0.01, 0.005, "(a) mask  (b) gray: annotation, orange: FoundationPose, teal: "
             "posterior samples  (c) teal circle: U1 choice, dashed orange: point-estimate "
             "choice; C collision prob., L low expected success", fontsize=6)
    out = REPO / "results/sr/viz"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{a.tag}_val.{ext}", dpi=300, bbox_inches="tight")
    (out / f"{a.tag}_val.json").write_text(json.dumps(log, indent=1))
    print("wrote", out / f"{a.tag}_val.png")


if __name__ == "__main__":
    main()
