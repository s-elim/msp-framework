"""Study 5: a triggered second view for the pose-uncertainty planner (pre-declared in the TODO).

LM test sequences show a static scene from a moving camera, and every frame has a
FoundationPose estimate, so a second view of frame i is another frame j of the same 50-frame
block (hence the same split). The camera motion T_ci_cj is computed from the annotated poses
(on a robot it comes from forward kinematics; this is the stated assumption). View choice: the
frame of the block whose viewing direction differs most from frame i's, up to --max-deg.

    python scripts/sr/second_view.py --split val           # two-view posterior, M = 2 x 16
        -> results/sr/posterior/twoview_s0_val.npz, second_view_val.json (chosen frames, angles)
    python scripts/sr/plan_with_pose.py ... \
        --pose-samples results/sr/posterior/twoview_s0_val.npz --tag fposeTV_s0_p4
    python scripts/sr/second_view.py --evaluate --split val
        -> results/sr/design/second_view_val.json

Evaluation (offline, trigger on the single-view U1 confidence = posterior-mean success of its
selected candidate; tau = the rate-quantile of that confidence on the same split): triggered
frames use the two-view scores, the others keep U1. Controls at the same rate: random trigger
(mean of 200 draws) and the same view with 2 x 16 samples (mean of U1 seeds 0 and 1, the same
planning budget as the two-view posterior).
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

POST, PLAN = REPO / "results/sr/posterior", REPO / "results/sr/plan"
BLOCK = 50


def view_dirs(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(N, 3) unit vectors from the object origin to the camera, in the object frame."""
    c = -np.einsum("nji,nj->ni", R, t)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def choose_views(oid: np.ndarray, iid: np.ndarray, R: np.ndarray, t: np.ndarray,
                 max_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Row index j (N,) of each frame's second view and the viewpoint change in degrees."""
    v = view_dirs(R, t)
    j_of = np.full(len(oid), -1)
    ang_of = np.full(len(oid), np.nan)
    key = oid * 100000 + iid // BLOCK
    for k in np.unique(key):
        rows = np.flatnonzero(key == k)
        if len(rows) < 2:
            continue
        A = np.degrees(np.arccos(np.clip(v[rows] @ v[rows].T, -1, 1)))
        np.fill_diagonal(A, -1.0)
        for a, i in enumerate(rows):
            ok = A[a] <= max_deg
            ok[a] = False
            b = int(np.argmax(np.where(ok, A[a], -1.0))) if ok.any() else int(np.argmax(A[a]))
            j_of[i], ang_of[i] = rows[b], A[a, b]
    return j_of, ang_of


def map_to_view(R_i: np.ndarray, t_i: np.ndarray, R_j: np.ndarray, t_j: np.ndarray,
                Rs_j: np.ndarray, ts_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pose samples of view j (N, M, 3, 3), (N, M, 3) expressed in camera i, using the camera
    motion T_ci_cj = T_ci_o T_cj_o^-1 of the static scene (annotated poses (N, 3, 3),
    (N, 3))."""
    R_ij = R_i @ np.swapaxes(R_j, -1, -2)
    t_ij = t_i - np.einsum("nab,nb->na", R_ij, t_j)
    return (np.einsum("nab,nmbc->nmac", R_ij, Rs_j),
            np.einsum("nab,nmb->nma", R_ij, ts_j) + t_ij[:, None])


def build(a: argparse.Namespace) -> None:
    import torch

    b = torch.load(REPO / "data/lm_grasp" / f"{a.split}.pt", map_location="cpu",
                   weights_only=False)
    oid, iid = b["obj_id"].numpy(), b["im_id"].numpy()
    R, t = b["R"].double().numpy(), b["t"].double().numpy()
    z = np.load(POST / f"knn_s{a.seed}_{a.split}.npz")
    assert np.array_equal(z["obj_id"], oid) and np.array_equal(z["im_id"], iid)
    Rs, ts = z["R"].astype(np.float64), z["t"].astype(np.float64)  # (N, M, ...), camera i
    j, ang = choose_views(oid, iid, R, t, a.max_deg)
    single = j < 0  # a block with one frame of this split: no second view, keep view i twice
    j = np.where(single, np.arange(len(oid)), j)
    R2, t2 = map_to_view(R, t, R[j], t[j], Rs[j], ts[j])
    np.savez(POST / f"twoview_s{a.seed}_{a.split}.npz",
             R=np.concatenate([Rs, R2], 1).astype(np.float32),
             t=np.concatenate([ts, t2], 1).astype(np.float32), obj_id=oid, im_id=iid,
             second_row=j, view_change_deg=ang, args=json.dumps(vars(a)))
    info = {"n": len(oid), "no_second_view": int(single.sum()),
            "view_change_deg_median": float(np.nanmedian(ang)),
            "view_change_deg_p10_p90": np.nanpercentile(ang, [10, 90]).tolist()}
    (POST / f"second_view_{a.split}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def evaluate(a: argparse.Namespace) -> None:
    import torch
    from evaluate_lm import paired_bootstrap, pick, selective

    b = torch.load(REPO / "data/lm_grasp" / f"{a.split}.pt", map_location="cpu",
                   weights_only=False)
    y = b["succ"].numpy()
    r = np.arange(len(y))

    def load(tag: str) -> np.ndarray:
        return torch.load(PLAN / f"{tag}_{a.split}.pt", map_location="cpu",
                          weights_only=False)["s"].numpy()

    s1, s1b, s2 = load("fposeU1_s0_p4"), load("fposeU1_s1_p4"), load("fposeTV_s0_p4")
    sdbl = (s1 + s1b) / 2
    ip1 = pick(s1)
    conf = s1[r, ip1]
    rng = np.random.default_rng(0)
    out: dict = {"split": a.split, "single_view_top1": float(y[r, ip1].mean()),
                 "rates": {}}
    # on any split but val, the trigger threshold is the one set on val (pre-declared)
    val_tau = None if a.split == "val" else {
        k: v["tau"] for k, v in json.loads((REPO / "results/sr/design/second_view_val.json")
                                           .read_text())["rates"].items()}
    for rate in (0.1, 0.25, 0.5, 1.0):
        trig = np.zeros(len(y), bool)
        if val_tau is None or rate == 1.0:
            n = round(rate * len(y))
            trig[np.argsort(conf, kind="stable")[:n]] = True
        else:
            if val_tau[f"{rate:g}"] >= 1.0:  # the val threshold saturated: not a trigger rate
                continue
            trig = conf <= val_tau[f"{rate:g}"]

        def oks(s_alt: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            s = np.where(mask[:, None], s_alt, s1)
            ip = pick(s)
            return y[r, ip], s[r, ip]

        def top1(s_alt: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
            ok, cf = oks(s_alt, mask)
            return float(ok.mean()), selective(cf, ok, 0.25)

        tv, tv_sel = top1(s2, trig)
        dbl, dbl_sel = top1(sdbl, trig)
        rand = [top1(s2, rng.permutation(trig))[0] for _ in range(200)]
        boot_dbl = paired_bootstrap(oks(s2, trig)[0], oks(sdbl, trig)[0])
        boot_single = paired_bootstrap(oks(s2, trig)[0], y[r, ip1])
        tau = (float(np.sort(conf)[max(int(trig.sum()) - 1, 0)]) if val_tau is None
               else val_tau.get(f"{rate:g}", 1.0))
        out["rates"][f"{rate:g}"] = {
            "triggered": int(trig.sum()), "rate_realised": float(trig.mean()), "tau": tau,
            "two_view_top1": tv, "two_view_sel25": tv_sel,
            "same_view_2x_samples_top1": dbl, "same_view_2x_samples_sel25": dbl_sel,
            "random_trigger_two_view_top1": float(np.mean(rand)),
            "random_trigger_two_view_top1_sd": float(np.std(rand)),
            "two_view_minus_same_view_2x": boot_dbl, "two_view_minus_single_view": boot_single}
        print(rate, json.dumps(out["rates"][f"{rate:g}"]))
    fp = np.concatenate([np.load(p)["secs"] for p in (REPO / "results/sr/fp").glob("*of*.npz")])
    out["fp_seconds_per_view_median"] = float(np.median(fp))
    d = REPO / "results/sr/design"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"second_view_{a.split}.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-deg", type=float, default=30.0)
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    evaluate(a) if a.evaluate else build(a)


if __name__ == "__main__":
    main()
