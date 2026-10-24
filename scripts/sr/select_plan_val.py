"""Study 4 selection on the VALIDATION split: pose-uncertainty planners and their controls.

    python scripts/sr/select_plan_val.py --out results/sr/design/plan_val_selection.json

Reads results/sr/plan/*_val.pt only; never opens a test file. Groups strip a trailing _s<seed>
before _p<n> (fposeU1_s0_p4, fposeU1_s1_p4 -> fposeU1_p4). Per group: within-scene AUC, top-1,
selective success at 25 / 50 % coverage (confidence = score of the selected candidate) and, for
posterior planners, the same with the 25 % pose quantile as the confidence. Paired bootstrap
(10,000 resamples over val scenes) of the seed-averaged top-1 indicator against each control.
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

from evaluate_lm import paired_bootstrap, pick, selective, within_scene_auc  # noqa: E402

CONTROLS = ("fpose_p64", "fpose_p4", "fposeGauss_p4", "fposeObj_p4", "fposeXO_p4")
METHOD = "fposeU1_p4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/sr/design/plan_val_selection.json"))
    ap.add_argument("--plan", default=str(REPO / "results/sr/plan"))
    a = ap.parse_args()
    vb = torch.load(REPO / "data/lm_grasp/val.pt", map_location="cpu", weights_only=False)
    y = vb["succ"].numpy()
    r = np.arange(len(y))
    rows, oks = {}, defaultdict(list)
    for f in sorted(Path(a.plan).glob("*_val.pt")):
        tag = f.name[: -len("_val.pt")]
        d = torch.load(f, map_location="cpu", weights_only=False)
        s = d["s"].numpy()
        if s.shape != y.shape:
            print(f"skip {tag}: {s.shape} != {y.shape} (smoke run?)")
            continue
        ip = pick(s)
        ok = y[r, ip]
        row = {"within_scene_auc": within_scene_auc(s, y)[0], "top1_success": float(ok.mean()),
               "selective_25": selective(s[r, ip], ok, 0.25),
               "selective_50": selective(s[r, ip], ok, 0.5), "seconds": d.get("seconds")}
        if "q25" in d:
            q = d["q25"].numpy()[r, ip]
            row["selective_q25_25"] = selective(q, ok, 0.25)
            row["selective_q25_50"] = selective(q, ok, 0.5)
            row["mean_p_exec_selected"] = float(d["p_exec"].numpy()[r, ip].mean())
        rows[tag] = row
        g = re.sub(r"_s\d+(_p\d+)$", r"\1", tag)
        oks[g].append(ok)
    groups = defaultdict(list)
    for tag, row in rows.items():
        groups[re.sub(r"_s\d+(_p\d+)$", r"\1", tag)].append(row)
    out: dict = {"groups": {}, "runs": rows, "comparisons": {}}
    for g, rs in sorted(groups.items()):
        o = {"n": len(rs)}
        for k in rs[0]:
            if k == "seconds":
                continue
            v = np.array([x[k] for x in rs], float)
            o[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1
                    else float("nan"), "values": v.tolist()}
        out["groups"][g] = o
        print(f"{g:22s} n={len(rs)} " + " ".join(
            f"{k} {o[k]['mean']:.4f}" + (f"+-{o[k]['sd']:.4f}" if len(rs) > 1 else "")
            for k in o if k != "n"))
    if METHOD in oks:
        for c in (*CONTROLS, "gt_p4"):
            if c in oks:
                key = f"{METHOD} - {c}"
                out["comparisons"][key] = paired_bootstrap(np.mean(oks[METHOD], 0),
                                                           np.mean(oks[c], 0))
                print(key, json.dumps(out["comparisons"][key]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
