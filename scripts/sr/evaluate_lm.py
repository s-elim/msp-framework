"""Every LM-Grasp number in the manuscript comes from the JSON this script writes.

    python scripts/sr/evaluate_lm.py --out results/sr/metrics.json

It reads the corpus labels, the per-run score files (results/sr/runs/*/scores.pt) and the
pose-then-plan scores (results/sr/plan/*.pt), and computes, per run and aggregated over seeds:

    within-scene AUC   mean over test scenes containing both outcomes of the AUC among that
                       scene's candidates (ties count one half)
    pooled AUC         over all test (scene, candidate) pairs
    top-1 success      outcome of the highest-scoring candidate (ties broken by a fixed
                       pseudo-random jitter shared by all methods)
    selective success  top-1 success on the most confident 25% / 50% of scenes
    certificate        selection-conditional split conformal: calibrate on the selected
                       candidate of each calib scene, act on a test scene only if the
                       prediction set of its selected candidate is the singleton {success}
    pose               ADD(-S) < 0.1 d accuracy for pose runs (ADD-S for eggbox and glue)

Inexecutable candidates are failures and are kept in every metric.
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
sys.path.insert(0, str(REPO / "src"))

from msp.math.conformal import conformal_quantile, nonconformity_scores, prediction_set  # noqa

LM_SYMMETRIC = (10, 11)
ALPHAS = (0.05, 0.1, 0.2)


def jitter(shape, seed: int = 12345) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 1e-6, shape)


def auc_1d(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y > 0.5], s[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def within_scene_auc(s: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    vals = [auc_1d(s[i], y[i]) for i in range(len(s)) if 0 < y[i].sum() < y.shape[1]]
    return float(np.mean(vals)), len(vals)


def pick(s: np.ndarray) -> np.ndarray:
    return np.argmax(s + jitter(s.shape), axis=1)


def selective(conf: np.ndarray, ok: np.ndarray, cov: float) -> float:
    k = max(1, int(round(cov * len(conf))))
    idx = np.argsort(-(conf + jitter(conf.shape, 777)))[:k]
    return float(ok[idx].mean())


def certificate(s_cal, y_cal, s_te, y_te, alpha: float) -> dict:
    """Selection-conditional conformal on the selected candidates."""
    ic, it = pick(s_cal), pick(s_te)
    rc, rt = np.arange(len(s_cal)), np.arange(len(s_te))
    q = conformal_quantile(nonconformity_scores(torch.tensor(s_cal[rc, ic]),
                                                torch.tensor(y_cal[rc, ic])), alpha)
    ps = prediction_set(torch.tensor(s_te[rt, it]), q)
    yt = torch.tensor(y_te[rt, it])
    covered = torch.where(yt > 0.5, ps.contains_success, ps.contains_failure)
    cert = ps.contains_success & ~ps.contains_failure
    n = int(cert.sum())
    return {"alpha": alpha, "q_hat": q, "coverage": float(covered.float().mean()),
            "certified_fraction": n / len(yt),
            "certified_precision": float(yt[cert].mean()) if n else float("nan")}


def pairwise_certificate(s_cal, y_cal, s_te, y_te, alpha: float, seed: int = 0) -> dict:
    """Calibration on one random candidate per calib scene, the textbook split-conformal unit;
    evaluated on the SELECTED test candidate, which is what the robot executes."""
    rng = np.random.default_rng(seed)
    ic = rng.integers(0, s_cal.shape[1], len(s_cal))
    rc = np.arange(len(s_cal))
    q = conformal_quantile(nonconformity_scores(torch.tensor(s_cal[rc, ic]),
                                                torch.tensor(y_cal[rc, ic])), alpha)
    it, rt = pick(s_te), np.arange(len(s_te))
    ps = prediction_set(torch.tensor(s_te[rt, it]), q)
    yt = torch.tensor(y_te[rt, it])
    covered = torch.where(yt > 0.5, ps.contains_success, ps.contains_failure)
    cert = ps.contains_success & ~ps.contains_failure
    n = int(cert.sum())
    return {"alpha": alpha, "q_hat": q, "coverage": float(covered.float().mean()),
            "certified_fraction": n / len(yt),
            "certified_precision": float(yt[cert].mean()) if n else float("nan")}


def ranking_metrics(s: np.ndarray, y: np.ndarray, obj: np.ndarray, conf: np.ndarray | None = None
                    ) -> dict:
    ip = pick(s)
    ok = y[np.arange(len(y)), ip]
    conf = s[np.arange(len(s)), ip] if conf is None else conf
    w, n_mixed = within_scene_auc(s, y)
    out = {"within_scene_auc": w, "n_mixed_scenes": n_mixed,
           "pooled_auc": auc_1d(s.reshape(-1), y.reshape(-1)),
           "top1_success": float(ok.mean()),
           "selective_25": selective(conf, ok, 0.25), "selective_50": selective(conf, ok, 0.5),
           "per_object": {}}
    for o in np.unique(obj):
        m = obj == o
        wo, no = within_scene_auc(s[m], y[m]) if (0 < y[m].sum()) else (float("nan"), 0)
        out["per_object"][int(o)] = {"top1_success": float(ok[m].mean()),
                                     "within_scene_auc": wo, "n_mixed": no,
                                     "base_rate": float(y[m].mean()),
                                     "oracle": float(y[m].max(1).mean())}
    return out


def model_points(root: Path, n: int = 1000) -> dict[int, np.ndarray]:
    import trimesh

    out = {}
    rng = np.random.default_rng(0)
    for p in sorted((root / "models_eval").glob("obj_*.ply")):
        oid = int(p.stem.split("_")[1])
        v = np.asarray(trimesh.load(p, force="mesh").vertices) / 1000.0
        out[oid] = v[rng.choice(len(v), min(n, len(v)), replace=False)]
    return out


def add_metrics(Rh, th, R, t, obj, pts, diam) -> dict:
    from scipy.spatial import cKDTree

    errs, passed, rot_err, tr_err = [], [], [], []
    for i in range(len(R)):
        P = pts[int(obj[i])]
        a = P @ Rh[i].T + th[i]
        b = P @ R[i].T + t[i]
        if int(obj[i]) in LM_SYMMETRIC:
            e = float(cKDTree(b).query(a)[0].mean())
        else:
            e = float(np.linalg.norm(a - b, axis=1).mean())
        errs.append(e)
        passed.append(e < 0.1 * diam[int(obj[i])])
        c = np.clip((np.trace(Rh[i].T @ R[i]) - 1) / 2, -1, 1)
        rot_err.append(np.degrees(np.arccos(c)))
        tr_err.append(1000 * np.linalg.norm(th[i] - t[i]))
    passed = np.array(passed)
    per = {int(o): float(passed[obj == o].mean()) for o in np.unique(obj)}
    return {"add_01d": float(passed.mean()), "add_err_mm_median": 1000 * float(np.median(errs)),
            "rot_err_deg_median": float(np.median(rot_err)),
            "trans_err_mm_median": float(np.median(tr_err)), "add_01d_per_object": per,
            "add_err_m": errs}


def agg(rows: list[dict], keys: tuple[str, ...]) -> dict:
    out = {"n_seeds": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        out[k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1
                  else float("nan"), "values": v.tolist()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/lm_grasp"))
    ap.add_argument("--runs", default=str(REPO / "results/sr/runs"))
    ap.add_argument("--plan", default=str(REPO / "results/sr/plan"))
    ap.add_argument("--out", default=str(REPO / "results/sr/metrics.json"))
    a = ap.parse_args()
    data = Path(a.data)
    lab = {k: torch.load(data / f"{k}.pt", map_location="cpu", weights_only=False)
           for k in ("calib", "test")}
    y = {k: v["succ"].numpy() for k, v in lab.items()}
    obj = {k: v["obj_id"].numpy() for k, v in lab.items()}
    info = json.loads((REPO / "data/lm/models/models_info.json").read_text())
    diam = {int(k): v["diameter"] / 1000.0 for k, v in info.items()}
    pts = model_points(REPO / "data/lm")

    res: dict = {"corpus": {k: {"scenes": int(len(y[k])), "base_rate": float(y[k].mean()),
                                "executable": float(lab[k]["exec"].float().mean()),
                                "oracle_any_success": float(y[k].max(1).mean())}
                            for k in y},
                 "runs": {}, "plan": {}, "groups": {}}
    res["corpus"]["test"]["random_top1"] = float(y["test"].mean())

    for d in sorted(Path(a.runs).glob("*/scores.pt")):
        r = torch.load(d, map_location="cpu", weights_only=False)
        name = d.parent.name
        sc = r["scores"]
        held = r["args"].get("holdout", "")
        yt, yc = y["test"], y["calib"]
        ot = obj["test"]
        if held:  # unseen-object runs score only the held-out objects
            keep_t = np.isin(ot, [int(x) for x in held.split(",")])
            keep_c = np.isin(obj["calib"], [int(x) for x in held.split(",")])
            yt, ot, yc = yt[keep_t], ot[keep_t], yc[keep_c]
        row = {"method": r["args"]["method"], "seed": r["args"]["seed"], "holdout": held,
               "wall_seconds": r["wall_seconds"]}
        if "s" in sc["test"]:
            s_te, s_ca = sc["test"]["s"].numpy(), sc["calib"]["s"].numpy()
            row |= ranking_metrics(s_te, yt, ot)
            for al in ALPHAS:
                row[f"cert_{al}"] = certificate(s_ca, yc, s_te, yt, al)
                row[f"pairwise_{al}"] = pairwise_certificate(s_ca, yc, s_te, yt, al)
            if "logvar" in sc["test"] and r["args"]["method"] == "msp":
                mu, lv = sc["test"]["mu"], sc["test"]["logvar"]
                kl = 0.5 * (lv.exp() + mu ** 2 - 1 - lv).sum(-1)
                row["rate_nats"] = float(kl.mean())
                row["beta"] = r["args"]["beta"]
                row["latent"] = r["args"]["latent"]
        if "R_hat" in sc["test"]:
            row |= {k: v for k, v in add_metrics(
                sc["test"]["R_hat"].double().numpy(), sc["test"]["t_hat"].double().numpy(),
                lab["test"]["R"].double().numpy(), lab["test"]["t"].double().numpy(),
                obj["test"], pts, diam).items() if k != "add_err_m"}
        res["runs"][name] = row

    for p in sorted(Path(a.plan).glob("*_test.pt")):
        tag = p.name[: -len("_test.pt")]
        s_te = torch.load(p, map_location="cpu", weights_only=False)["s"].numpy()
        row = ranking_metrics(s_te, y["test"], obj["test"])
        pc = Path(a.plan) / f"{tag}_calib.pt"
        if pc.exists():
            s_ca = torch.load(pc, map_location="cpu", weights_only=False)["s"].numpy()
            for al in ALPHAS:
                row[f"cert_{al}"] = certificate(s_ca, y["calib"], s_te, y["test"], al)
        d = torch.load(p, map_location="cpu", weights_only=False)
        row |= {k: v for k, v in add_metrics(
            d["R"].double().numpy(), d["t"].double().numpy(), lab["test"]["R"].double().numpy(),
            lab["test"]["t"].double().numpy(), obj["test"], pts, diam).items()
            if k != "add_err_m"}
        res["plan"][tag] = row

    # seed groups: strip the trailing _s<seed>
    groups = defaultdict(list)
    for name, row in res["runs"].items():
        groups[re.sub(r"_s\d+$", "", name)].append(row)
    for name, row in res["plan"].items():
        m = re.match(r"(pose.*)_s\d+(_p\d+)$", name)
        if m:
            groups["plan:" + m.group(1) + m.group(2)].append(row)
    keys = ("within_scene_auc", "pooled_auc", "top1_success", "selective_25", "selective_50")
    for g, rows in groups.items():
        k = keys + tuple(x for x in ("rate_nats", "add_01d") if x in rows[0])
        out = agg(rows, k)
        for al in ALPHAS:
            if f"cert_{al}" in rows[0]:
                for f in ("coverage", "certified_fraction", "certified_precision"):
                    v = np.array([r[f"cert_{al}"][f] for r in rows], float)
                    out[f"cert_{al}_{f}"] = {"mean": float(np.nanmean(v)),
                                             "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1
                                             else float("nan")}
        res["groups"][g] = out
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    for g, o in sorted(res["groups"].items()):
        print(f"{g:40s} n={o['n_seeds']} within {o['within_scene_auc']['mean']:.3f}"
              f"+-{o['within_scene_auc']['sd']:.3f} top1 {o['top1_success']['mean']:.3f}"
              f" sel25 {o['selective_25']['mean']:.3f}")


if __name__ == "__main__":
    main()
