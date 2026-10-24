"""What no observation-level method can know: grasp success stratified by the hidden physics.

    python scripts/sr/physics_strata.py --out results/sr/physics_strata.json

Per test scene the simulator drew friction mu, a mass and a centre-of-mass offset that no
method sees. For each method (exact-pose planner, FoundationPose + planner, direct scorer,
POISE, VLM-direct) this reports top-1 success and within-scene AUC in terciles of mu, of the
mass multiplier (drawn mass / the object's nominal volume x density mass) and of the COM offset
norm, plus the oracle in each tercile. A method that reads the image well but cannot see physics
degrades with the oracle, not faster.

Also the VLM physics probe on every labelled split: Spearman correlation of the VLM's mass
estimate with the drawn mass (all scenes, and the mean within-object correlation, which removes
the object identity) and of its friction class with mu.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/sr"))
from evaluate_lm import top1_ok, within_scene_auc  # noqa: E402

VLM = "gemini-robotics-er-2-preview"


def load(p: Path) -> np.ndarray:
    return torch.load(p, map_location="cpu", weights_only=False)


def mean_seeds(pattern: str, n: int) -> np.ndarray | None:
    fs = [REPO / f"results/sr/runs/{pattern.format(k)}/scores.pt" for k in range(n)]
    fs = [f for f in fs if f.exists()]
    if not fs:
        return None
    return [load(f)["scores"]["test"]["s"].numpy() for f in fs]


def tercile_rows(x: np.ndarray, methods: dict, y: np.ndarray) -> list[dict]:
    q = np.quantile(x, [1 / 3, 2 / 3])
    bins = np.digitize(x, q)
    rows = []
    for b in range(3):
        m = bins == b
        r = {"tercile": b, "lo": float(x[m].min()), "hi": float(x[m].max()), "n": int(m.sum()),
             "oracle": float(y[m].max(1).mean())}
        for name, ss in methods.items():
            t1 = [float(top1_ok(s[m], y[m]).mean()) for s in ss]
            au = [within_scene_auc(s[m], y[m])[0] for s in ss]
            r[name] = {"top1": float(np.mean(t1)), "top1_sd": float(np.std(t1, ddof=1))
                       if len(t1) > 1 else 0.0, "within_auc": float(np.mean(au))}
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/sr/physics_strata.json"))
    a = ap.parse_args()
    b = load(REPO / "data/lm_grasp/test.pt")
    y = b["succ"].numpy()
    ph = b["phys"].numpy()  # mu, mass (kg), com xyz (m)
    obj = b["obj_id"].numpy()
    nominal = {o: np.median(ph[obj == o, 1]) for o in np.unique(obj)}
    mass_mult = ph[:, 1] / np.array([nominal[o] for o in obj])
    com = np.linalg.norm(ph[:, 2:5], axis=1)
    methods = {"exact_pose_plan": [load(REPO / "results/sr/plan/gt_p4_test.pt")["s"].numpy()],
               "fpose_plan": [load(REPO / "results/sr/plan/fpose_p4_test.pt")["s"].numpy()]}
    for name, pat in (("direct", "direct_s{}"), ("poise", "direct_plfpose_s{}")):
        ss = mean_seeds(pat, 5)
        if ss:
            methods[name] = ss
    vr = load(REPO / f"results/sr/vlm/labels_{VLM}_test_v2.pt")["ratings"].numpy()
    methods["vlm_direct"] = [np.where(np.isnan(vr), np.nanmean(vr), vr)]
    out = {"note": "nominal mass = per-object median of the drawn masses",
           "friction": tercile_rows(ph[:, 0], methods, y),
           "mass_multiplier": tercile_rows(mass_mult, methods, y),
           "com_offset_m": tercile_rows(com, methods, y), "vlm_probe": {}}
    for split in ("val", "train", "test"):
        f = REPO / f"results/sr/vlm/labels_{VLM}_{split}_v2.pt"
        if not f.exists():
            continue
        L, d = load(f), load(REPO / f"data/lm_grasp/{split}.pt")
        ms, fr = L["mass_g"].numpy(), L["friction"].numpy()
        p, o = d["phys"].numpy(), d["obj_id"].numpy()
        g = ~np.isnan(ms)
        within = [spearmanr(ms[g & (o == k)], p[g & (o == k), 1])[0] for k in np.unique(o)]
        gf = fr >= 0
        out["vlm_probe"][split] = {
            "n": int(g.sum()), "mass_spearman_all": float(spearmanr(ms[g], p[g, 1])[0]),
            "mass_spearman_within_object_mean": float(np.nanmean(within)),
            "friction_spearman_all": float(spearmanr(fr[gf], p[gf, 0])[0]),
            "friction_class_counts": {int(k): int((fr == k).sum()) for k in (-1, 0, 1, 2)}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k in ("friction", "mass_multiplier", "com_offset_m"):
        print(k)
        for r in out[k]:
            print(f"  [{r['lo']:.3f},{r['hi']:.3f}] oracle {r['oracle']:.3f} " + " ".join(
                f"{m} {r[m]['top1']:.3f}" for m in methods))
    print(json.dumps(out["vlm_probe"], indent=1))


if __name__ == "__main__":
    main()
