"""Blind redesign study: validation-split metrics of every candidate and its matched control.

    python scripts/sr/select_val.py --out results/sr/design/val_selection.json

Reads val scores only (results/sr/design/*/scores.pt and the seed-0..2 baselines in
results/sr/runs). Never opens a test score. Per group (seeds pooled): val within-scene AUC,
val top-1 success, and, for quantised codes (qcode), the mean entropy-coded bits under the
learned prior. The selection rule is in scientific_reports_paper/TODO_2026_10_24.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/sr"))

from evaluate_lm import pick, within_scene_auc  # noqa: E402

BASELINES = ("msp_b300_d32_s{}", "direct_s{}", "msp_b1000_d32_s{}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/sr/design/val_selection.json"))
    a = ap.parse_args()
    vb = torch.load(REPO / "data/lm_grasp/val.pt", map_location="cpu", weights_only=False)
    yv = vb["succ"].numpy()
    row = {(int(o), int(i)): n for n, (o, i) in enumerate(zip(vb["obj_id"], vb["im_id"]))}

    def labels(res: dict, split: str) -> np.ndarray:
        """Outcomes aligned with a run's scored split (holdout runs score a subset of val)."""
        ix = res.get("split_index", {}).get(split)
        if ix is None:
            return yv
        return yv[[row[(int(o), int(i))] for o, i in zip(ix["obj_id"], ix["im_id"])]]
    files = sorted((REPO / "results/sr/design").glob("*/scores.pt"))
    files += [REPO / "results/sr/runs" / b.format(s) / "scores.pt" for b in BASELINES
              for s in range(3)]
    rows = {}
    for f in files:
        name = f.parent.name
        if name.startswith("smoke") or not f.exists():
            continue
        res = torch.load(f, map_location="cpu", weights_only=False)
        sc = res["scores"]["val"]
        s = sc["s"].numpy()
        y = labels(res, "val")
        ok = y[np.arange(len(y)), pick(s)]
        w, _ = within_scene_auc(s, y)
        rows[name] = {"within_scene_auc": w, "top1_success": float(ok.mean())}
        if "val_unseen" in res["scores"]:  # held-out objects: the secondary selection check
            su = res["scores"]["val_unseen"]["s"].numpy()
            yu = labels(res, "val_unseen")
            rows[name]["unseen_within_auc"] = within_scene_auc(su, yu)[0]
            rows[name]["unseen_top1"] = float(yu[np.arange(len(yu)), pick(su)].mean())
        if "bits" in sc:
            rows[name]["bits_learned_prior"] = float(sc["bits"].float().mean())
    groups = defaultdict(list)
    for k, v in rows.items():
        groups[re.sub(r"_s\d+$", "", k)].append(v)
    out = {}
    for g, rs in sorted(groups.items()):
        out[g] = {"n": len(rs)}
        for k in rs[0]:
            v = np.array([r[k] for r in rs])
            out[g][k] = {"mean": float(v.mean()),
                         "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                         "values": v.tolist()}
        print(f"{g:32s} n={len(rs)} " + " ".join(
            f"{k} {out[g][k]['mean']:.4f}+-{out[g][k]['sd']:.4f}" for k in rs[0]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"groups": out, "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
