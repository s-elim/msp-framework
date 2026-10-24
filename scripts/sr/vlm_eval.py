"""Study 3 V3: what the VLM knows at deployment time, without distillation.

    python scripts/sr/vlm_eval.py --out results/sr/vlm_eval.json

Rows (test split, all 13 objects, and the held-out objects 1, 6, 9, 15):
  vlm_direct        the VLM agreement score itself ranks the 16 candidates
  direct            label-trained direct scorer (seeds 0-4; held-out rows: the holdout runs)
  fuse_direct_vlm   logistic late fusion [logit s_direct, s_vlm], fit on VAL per seed
  fpose             FoundationPose + planner
  fuse_fpose_vlm    logistic late fusion [s_fpose, s_vlm], fit on VAL (needs fpose val plans)
Fusion weights are fit on val only (calib carries no VLM labels: budget, see TODO), and the
test split is touched once. Cost and latency of the VLM come from its labelling summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/sr"))
from evaluate_lm import paired_bootstrap, ranking_metrics, top1_ok  # noqa: E402

VLM = "gemini-robotics-er-2-preview"
HOLD = (1, 6, 9, 15)


def vlm_scores(split: str) -> np.ndarray:
    L = torch.load(REPO / f"results/sr/vlm/labels_{VLM}_{split}_v2.pt", map_location="cpu",
                   weights_only=False)
    r = L["ratings"].numpy() / 10.0
    return np.where(np.isnan(r), np.nanmean(r), r)  # parse failures: uninformative constant


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p) - np.log1p(-p)


def fuse(fv: list[np.ndarray], yv: np.ndarray, ft: list[np.ndarray]) -> np.ndarray:
    X = np.stack([f.reshape(-1) for f in fv], 1)
    m = LogisticRegression(C=1.0).fit(X, yv.reshape(-1))
    Xt = np.stack([f.reshape(-1) for f in ft], 1)
    return m.predict_proba(Xt)[:, 1].reshape(ft[0].shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/sr/vlm_eval.json"))
    a = ap.parse_args()
    d = {s: torch.load(REPO / f"data/lm_grasp/{s}.pt", map_location="cpu", weights_only=False)
         for s in ("val", "test")}
    y = {s: d[s]["succ"].numpy() for s in d}
    obj = {s: d[s]["obj_id"].numpy() for s in d}
    held = {s: np.isin(obj[s], HOLD) for s in d}
    v = {s: vlm_scores(s) for s in d}
    out: dict = {"model": VLM, "rows": {}, "comparisons": {}}
    for s in ("val", "test"):
        f = REPO / f"results/sr/vlm/summary_{VLM}_{s}_v2.json"
        if f.exists():
            out[f"cost_{s}"] = json.loads(f.read_text())

    def row(name: str, s_te: np.ndarray, mask: np.ndarray | None = None) -> dict:
        m = np.ones(len(s_te), bool) if mask is None else mask
        r = ranking_metrics(s_te[m], y["test"][m], obj["test"][m])
        r.pop("per_object")
        return r

    out["rows"]["vlm_direct"] = row("vlm", v["test"])
    out["rows"]["vlm_direct_heldout"] = row("vlm", v["test"], held["test"])
    ok_vlm = top1_ok(v["test"], y["test"])
    dir_rows, fus_rows, dir_ok, fus_ok = [], [], [], []
    for k in range(5):
        sc = torch.load(REPO / f"results/sr/runs/direct_s{k}/scores.pt", map_location="cpu",
                        weights_only=False)["scores"]
        sv, st = sc["val"]["s"].numpy(), sc["test"]["s"].numpy()
        fz = fuse([logit(sv), v["val"]], y["val"], [logit(st), v["test"]])
        dir_rows.append(row("direct", st))
        fus_rows.append(row("fuse", fz))
        dir_ok.append(top1_ok(st, y["test"]))
        fus_ok.append(top1_ok(fz, y["test"]))
    ho_rows, hof_rows, ho_ok, hof_ok = [], [], [], []
    for k in range(3):  # holdout-trained scorers: their "test" is the held-out objects only
        r = torch.load(REPO / f"results/sr/runs/direct_ho1-6-9-15_s{k}/scores.pt",
                       map_location="cpu", weights_only=False)["scores"]
        st = r["test"]["s"].numpy()
        sv = r["val"]["s"].numpy()  # seen objects of val: fusion fit on seen objects only
        mv = ~held["val"]
        fz = fuse([logit(sv), v["val"][mv]], y["val"][mv], [logit(st), v["test"][held["test"]]])
        ho_rows.append(ranking_metrics(st, y["test"][held["test"]], obj["test"][held["test"]]))
        hof_rows.append(ranking_metrics(fz, y["test"][held["test"]], obj["test"][held["test"]]))
        ho_ok.append(top1_ok(st, y["test"][held["test"]]))
        hof_ok.append(top1_ok(fz, y["test"][held["test"]]))

    def mean_rows(rs: list[dict]) -> dict:
        keys = ("within_scene_auc", "pooled_auc", "top1_success", "selective_25", "selective_50")
        return {k: {"mean": float(np.mean([r[k] for r in rs])),
                    "sd": float(np.std([r[k] for r in rs], ddof=1))} for k in keys}

    out["rows"]["direct"] = mean_rows(dir_rows)
    out["rows"]["fuse_direct_vlm"] = mean_rows(fus_rows)
    out["rows"]["direct_heldout"] = mean_rows(ho_rows)
    out["rows"]["fuse_direct_vlm_heldout"] = mean_rows(hof_rows)
    out["comparisons"]["fuse_direct_vlm-direct"] = paired_bootstrap(
        np.mean(fus_ok, 0), np.mean(dir_ok, 0))
    out["comparisons"]["fuse_direct_vlm-direct_heldout"] = paired_bootstrap(
        np.mean(hof_ok, 0), np.mean(ho_ok, 0))
    out["comparisons"]["vlm_direct-direct"] = paired_bootstrap(ok_vlm, np.mean(dir_ok, 0))
    fp_t = REPO / "results/sr/plan/fpose_p4_test.pt"
    fp_v = REPO / "results/sr/plan/fpose_p4_val.pt"
    if fp_t.exists():
        sft = torch.load(fp_t, weights_only=False)["s"].numpy()
        out["rows"]["fpose"] = row("fp", sft)
        out["rows"]["fpose_heldout"] = row("fp", sft, held["test"])
        if fp_v.exists():
            sfv = torch.load(fp_v, weights_only=False)["s"].numpy()
            fz = fuse([sfv, v["val"]], y["val"], [sft, v["test"]])
            out["rows"]["fuse_fpose_vlm"] = row("fz", fz)
            out["rows"]["fuse_fpose_vlm_heldout"] = row("fz", fz, held["test"])
            out["comparisons"]["fuse_fpose_vlm-fpose"] = paired_bootstrap(
                top1_ok(fz, y["test"]), top1_ok(sft, y["test"]))
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, r in out["rows"].items():
        t = r["top1_success"]
        t = t["mean"] if isinstance(t, dict) else t
        w = r["within_scene_auc"]
        w = w["mean"] if isinstance(w, dict) else w
        print(f"{k:28s} within-AUC {w:.3f} top-1 {t:.3f}")
    for k, c in out["comparisons"].items():
        print(k, c)


if __name__ == "__main__":
    main()
