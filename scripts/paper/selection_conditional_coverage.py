"""Does the certificate cover the grasp the robot actually executes?

Reviewer 1.4 and Reviewer 2 both press the same point: the guarantee in the paper is
split conformal over randomly drawn (observation, action) PAIRS, while the robot runs
one action per scene, chosen by argmax over the same scores that were calibrated. Argmax
selection is data dependent, so the selected pair is not exchangeable with the
calibration pairs and Theorem 7 does not transfer to it.

This script measures the size of that gap and shows the fix, using only cached test
scores (results/libero/l1_scores.pt). Three procedures, all at the same target 1-alpha:

  pairwise      calibrate on random (o, a) pairs, evaluate on random (o, a) pairs.
                Reproduces the paper's number and should hold at 1-alpha.

  executed      calibrate on random (o, a) pairs, evaluate ONLY on the argmax action of
                each held-out scene. This is what the paper's certificate actually
                delivers on the executed grasp; nothing guarantees it equals 1-alpha.

  selective     calibrate on the argmax action of each CALIBRATION scene, evaluate on the
                argmax action of each held-out scene. Selection is applied identically on
                both folds, so exchangeability is restored at the scene level and the
                finite-sample guarantee covers the executed grasp.

The third row is the paper's answer to the reviewers: keep the theorem, restate it over
scenes rather than pairs, and recalibrate. No retraining, no new data.

    python scripts/paper/selection_conditional_coverage.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from msp.math.conformal import conformal_quantile, nonconformity_scores, prediction_set

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("selective")


def summarise(s: torch.Tensor, succ: torch.Tensor, q_hat: float) -> dict:
    """Coverage, certified fraction and certified precision at a given q_hat."""
    ps = prediction_set(s, q_hat)
    covered = torch.where(succ > 0.5, ps.contains_success, ps.contains_failure)
    certified = ps.contains_success & ~ps.contains_failure
    n_cert = int(certified.sum())
    return {
        "coverage": float(covered.float().mean()),
        "certified_fraction": n_cert / max(certified.numel(), 1),
        "certified_precision": float(succ[certified].mean()) if n_cert else float("nan"),
        "n": int(covered.numel()),
    }


def one_split(s: torch.Tensor, succ: torch.Tensor, ok: torch.Tensor,
              alpha: float, rng: np.random.Generator) -> dict:
    """One random half/half split of the scenes into calibration and evaluation."""
    n_scenes = s.shape[0]
    perm = rng.permutation(n_scenes)
    cal_i, ev_i = perm[: n_scenes // 2], perm[n_scenes // 2:]

    # argmax over executable actions, the action the controller would run
    masked = s.masked_fill(~ok, -float("inf"))
    pick = masked.argmax(dim=1)
    rows = torch.arange(n_scenes)
    s_pick, succ_pick = s[rows, pick], succ[rows, pick]

    # pooled pairs, restricted to executable actions
    cal_pairs = ok[cal_i]
    q_pair = conformal_quantile(
        nonconformity_scores(s[cal_i][cal_pairs], succ[cal_i][cal_pairs]), alpha)
    q_sel = conformal_quantile(
        nonconformity_scores(s_pick[cal_i], succ_pick[cal_i]), alpha)

    ev_pairs = ok[ev_i]
    return {
        "pairwise": summarise(s[ev_i][ev_pairs], succ[ev_i][ev_pairs], q_pair),
        "executed": summarise(s_pick[ev_i], succ_pick[ev_i], q_pair),
        "selective": summarise(s_pick[ev_i], succ_pick[ev_i], q_sel),
        "q_pair": q_pair,
        "q_sel": q_sel,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/libero/l1_scores.pt")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.05, 0.1, 0.2])
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--out", default="results/libero/l4_selection_conditional.json")
    args = ap.parse_args()

    d = torch.load(args.scores, map_location="cpu", weights_only=False)
    s, succ, ok = d["s_msp"], d["succ"], d["executable"]
    log.info("scenes=%d actions=%d executable=%.1f%%",
             s.shape[0], s.shape[1], 100 * ok.float().mean())

    report = []
    for alpha in args.alphas:
        runs = [one_split(s, succ, ok, alpha, np.random.default_rng(r))
                for r in range(args.repeats)]
        row = {"alpha": alpha, "target_coverage": 1 - alpha, "repeats": args.repeats}
        for name in ("pairwise", "executed", "selective"):
            for key in ("coverage", "certified_fraction", "certified_precision"):
                v = np.array([r[name][key] for r in runs], dtype=float)
                row[f"{name}/{key}"] = float(np.nanmean(v))
                row[f"{name}/{key}_sd"] = float(np.nanstd(v))
        row["q_pair"] = float(np.mean([r["q_pair"] for r in runs]))
        row["q_sel"] = float(np.mean([r["q_sel"] for r in runs]))
        report.append(row)

        log.info("\nalpha=%.2f  target %.3f   (%d random calib/eval splits)",
                 alpha, 1 - alpha, args.repeats)
        for name in ("pairwise", "executed", "selective"):
            log.info("  %-10s coverage %.4f +-%.4f   certified %.3f   precision %.3f",
                     name, row[f"{name}/coverage"], row[f"{name}/coverage_sd"],
                     row[f"{name}/certified_fraction"], row[f"{name}/certified_precision"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("\nwrote %s", out)


if __name__ == "__main__":
    main()
