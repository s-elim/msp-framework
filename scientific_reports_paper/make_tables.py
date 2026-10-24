"""Write numbers.tex (one macro per number quoted in the text) and the LaTeX tables of the
Scientific Reports manuscript from the results files. No number in main.tex is typed by hand.

    cd /nas/Selim/selim_sarowar/msp_framework
    source scripts/sr/env.sh && $PY scientific_reports_paper/make_tables.py

Inputs: results/sr/metrics.json, rate_dims_msp.json, rate_dims_direct.json, latency.json,
results/sr/lm_upright.json, data/lm_grasp/summary.json, results/sr/plan/*.pt (seconds).
Outputs (beside this script): numbers.tex, table_main.tex, table_unseen.tex.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
R = REPO / "results/sr"
sys.path.insert(0, str(REPO / "scientific_reports_paper/figures"))


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def f3(x: float) -> str:
    return f"{x:.3f}"


def pm(v: dict, d: int = 3) -> str:
    return f"{v['mean']:.{d}f} $\\pm$ {v['sd']:.{d}f}" if np.isfinite(v["sd"]) else \
        f"{v['mean']:.{d}f}"


def macro(name: str, val: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{val}}}"


def frontier_at(rows: list[dict], budget: float) -> float:
    ok = [r["top1_success"] for r in rows if r["payload_bits"] <= budget]
    return max(ok) if ok else float("nan")


def bits_to_reach(rows: list[dict], target: float) -> float:
    ok = [r["payload_bits"] for r in rows if r["top1_success"] >= target]
    return min(ok) if ok else float("nan")


def main() -> None:
    m = load("metrics.json")
    g, plan, cmp_ = m["groups"], m["plan"], m["comparisons"]
    te = m["corpus"]["test"]
    summ = json.loads((REPO / "data/lm_grasp/summary.json").read_text())
    up = json.loads((R / "lm_upright.json").read_text())
    n = {k: m["corpus"][k]["scenes"] for k in ("calib", "test")}
    mac = []
    add = lambda k, v: mac.append(macro(k, v))  # noqa: E731

    # corpus
    import torch

    sizes = {s: int(torch.load(REPO / f"data/lm_grasp/{s}.pt", map_location="cpu",
                               weights_only=False)["obj_id"].shape[0])
             for s in ("train", "val")}
    add("NScenes", f"{summ['scenes']:,}")
    add("NTrain", f"{sizes['train']:,}")
    add("NVal", f"{sizes['val']:,}")
    add("NCalib", f"{n['calib']:,}")
    add("NTest", f"{n['test']:,}")
    add("NRollouts", f"{summ['scenes'] * 16:,}")
    add("TestBase", f3(te["base_rate"]))
    add("TestExec", f3(te["executable"]))
    add("TestOracle", f3(te["oracle_any_success"]))
    meds = [v["median_deg"] for v in up.values() if isinstance(v, dict) and "median_deg" in v]
    add("UprightDegLo", f"{min(meds):.1f}")
    add("UprightDegHi", f"{max(meds):.1f}")
    add("NMixed", f"{g['direct']['n_seeds'] and m['runs']['direct_s0']['n_mixed_scenes']:,}")

    # methods
    names = {"MSP": "msp_b300_d32", "Direct": "direct", "Ens": "ensemble5", "Gq": "gqcnn"}
    for k, grp in names.items():
        x = g[grp]
        add(f"{k}Top", pm(x["top1_success"]))
        add(f"{k}TopMean", f3(x["top1_success"]["mean"]))
        add(f"{k}Auc", pm(x["within_scene_auc"]))
        add(f"{k}SelQ", pm(x["selective_25"]))
        add(f"{k}SelQMean", f3(x["selective_25"]["mean"]))
        add(f"{k}CertFrac", f3(x["cert_0.1_certified_fraction"]["mean"]))
        add(f"{k}CertPrec", f3(x["cert_0.1_certified_precision"]["mean"]))
        add(f"{k}CertCov", f3(x["cert_0.1_coverage"]["mean"]))
        add(f"{k}N", str(x["n_seeds"]))
    for k, key in (("Gt", "gt_p4"), ("Fp", "fpose_p4")):
        if key in plan:
            add(f"{k}PlanTop", f3(plan[key]["top1_success"]))
            add(f"{k}PlanAuc", f3(plan[key]["within_scene_auc"]))
            add(f"{k}PlanSelQ", f3(plan[key]["selective_25"]))
            add(f"{k}Add", f3(plan[key]["add_01d"]))
            add(f"{k}RotErr", f"{plan[key]['rot_err_deg_median']:.1f}")
            add(f"{k}TransErr", f"{plan[key]['trans_err_mm_median']:.1f}")
            if "cert_0.1" in plan[key]:
                add(f"{k}CertFrac", f3(plan[key]["cert_0.1"]["certified_fraction"]))
                add(f"{k}CertPrec", f3(plan[key]["cert_0.1"]["certified_precision"]))
    if "plan:pose_p4" in g:
        x = g["plan:pose_p4"]
        add("PrPlanTop", pm(x["top1_success"]))
        add("PrPlanTopMean", f3(x["top1_success"]["mean"]))
        add("PrPlanAuc", pm(x["within_scene_auc"]))
        add("PrPlanSelQ", pm(x["selective_25"]))
        add("PrAdd", pm(g["pose"]["add_01d"]))
        add("PrAddMean", f3(g["pose"]["add_01d"]["mean"]))
        add("PrPlanCertFrac", f3(x["cert_0.1_certified_fraction"]["mean"]))
        add("PrPlanCertPrec", f3(x["cert_0.1_certified_precision"]["mean"]))
    for k, v in cmp_.items():
        tag = "Cmp" + "".join(w.capitalize() for w in
                              k.replace("plan:", "").replace("_p4", "").replace("msp_b300_d32",
                                                                                "msp")
                              .replace(" - ", " vs ").replace("_", " ").split())
        tag = "".join(ch for ch in tag if ch.isalpha())
        add(tag, f"{v['diff']:+.3f}")
        add(tag + "Ci", f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")

    # pose-noise study
    def nz(r, t):
        rows = [plan[f"noise_r{r:g}_t{t:g}_s{s}_p4"] for s in range(3)]
        return (np.mean([x["top1_success"] for x in rows]), np.mean([x["add_01d"] for x in rows]))
    for lab, (r, t) in {"RotFive": (5, 0), "RotTen": (10, 0), "RotTwo": (2, 0),
                        "TransTwo": (0, 2), "TransFive": (0, 5), "TransTen": (0, 10),
                        "TransTwenty": (0, 20), "RotTwenty": (20, 0)}.items():
        s_, a_ = nz(r, t)
        add(f"Noise{lab}Top", f3(s_))
        add(f"Noise{lab}Add", f3(a_))
    # largest success loss among settings with ADD(-S) pass rate >= 0.99
    sat = [(plan[k]["top1_success"], k) for k in plan
           if k.startswith("noise_") and plan[k]["add_01d"] >= 0.99]
    add("NoiseSatMinTop", f3(min(sat)[0]))
    add("NoiseSatDrop", f"{plan['gt_p4']['top1_success'] - min(sat)[0]:.3f}")

    # payload
    rm, rd = load("rate_dims_msp.json"), load("rate_dims_direct.json")
    b300 = [v for k, v in rm.items() if k.startswith("msp_b300_d32")]
    dirs = list(rd.values())
    for lab, runs in (("MSP", b300), ("Direct", dirs)):
        for bud in (4, 16, 32, 64):
            add(f"Pay{lab}At{ {4: 'Four', 16: 'Sixteen', 32: 'ThirtyTwo', 64: 'SixtyFour'}[bud]}",
                f3(np.mean([frontier_at(r["rows"], bud) for r in runs])))
        full = np.mean([max(x["top1_success"] for x in r["rows"]) for r in runs])
        add(f"Pay{lab}Full", f3(full))
        need = [bits_to_reach(r["rows"], 0.99 * max(x["top1_success"] for x in r["rows"]))
                for r in runs]
        add(f"Pay{lab}NinetyNine", f"{np.mean(need):.0f}")
    # beta sweep
    for b in (1, 10, 100, 300, 1000):
        key = f"msp_b{b}_d32"
        if key in g:
            add(f"BetaTop{ {1: 'One', 10: 'Ten', 100: 'Hundred', 300: 'ThreeHundred', 1000: 'Thousand'}[b]}",
                pm(g[key]["top1_success"]))
            if "rate_nats" in g[key]:
                add(f"BetaRate{ {1: 'One', 10: 'Ten', 100: 'Hundred', 300: 'ThreeHundred', 1000: 'Thousand'}[b]}",
                    f"{g[key]['rate_nats']['mean']:.1f}")
    for d, w in ((8, "Eight"), (64, "SixtyFour")):
        key = f"msp_b300_d{d}"
        if key in g:
            add(f"LatTop{w}", pm(g[key]["top1_success"]))

    # unseen objects
    ho = {k: g[k] for k in g if "_ho" in k}
    for k, v in ho.items():
        base = k.split("_ho")[0]
        w = {"msp_b300_d32": "MSP", "direct": "Direct", "gqcnn": "Gq"}.get(base)
        if w:
            add(f"Unseen{w}Top", pm(v["top1_success"]))
            add(f"Unseen{w}Auc", pm(v["within_scene_auc"]))

    # latency
    lat_p = R / "latency.json"
    if lat_p.exists():
        lat = json.loads(lat_p.read_text())
        for dev in ("cuda", "cpu"):
            if dev in lat:
                for k2, w in (("msp_k32", "MSP"), ("direct", "Direct"), ("ensemble5", "Ens"),
                              ("gqcnn", "Gq"), ("pose_regressor_only", "Pr")):
                    add(f"Lat{w}{dev.capitalize()}", f"{lat[dev][k2]:.1f}")
    if "fpose" in m["runs"]:
        pass
    fp_runs = REPO / "results/sr/runs/fpose/scores.pt"
    if fp_runs.exists():
        fr = torch.load(fp_runs, map_location="cpu", weights_only=False)
        add("FpSecPerScene", f"{fr['median_sec_per_scene']:.2f}")
    for key, w in (("gt_p4_test.pt", "Gt"),):
        d = torch.load(R / "plan" / key, map_location="cpu", weights_only=False)
        add("PlanSecPerScene", f"{d['seconds'] * d['args']['workers'] / n['test']:.2f}")

    (OUT / "numbers.tex").write_text("% generated by make_tables.py; do not edit\n" +
                                     "\n".join(mac) + "\n")
    print(len(mac), "macros")
    write_tables(m)


def write_tables(m: dict) -> None:
    g, plan = m["groups"], m["plan"]
    rows = []

    def row(label, top, auc, sel, cf, cp, extra=""):
        rows.append(f"{label} & {top} & {auc} & {sel} & {cf} & {cp}{extra} \\\\")

    for grp, lab in (("gqcnn", "Grasp-patch CNN (Dex-Net 2.0 architecture)"),
                     ("msp_b300_d32", "MSP, stochastic code ($\\beta=300$)"),
                     ("direct", "Outcome scorer, deterministic code"),
                     ("ensemble5", "Deep ensemble of 5 outcome scorers")):
        x = g[grp]
        row(f"{lab} (n={x['n_seeds']})", pm(x["top1_success"]), pm(x["within_scene_auc"]),
            pm(x["selective_25"]), f3(x["cert_0.1_certified_fraction"]["mean"]),
            f3(x["cert_0.1_certified_precision"]["mean"]))
    rows.append("\\midrule")
    if "plan:pose_p4" in g:
        x = g["plan:pose_p4"]
        row(f"Pose regressor + planner (n={x['n_seeds']})", pm(x["top1_success"]),
            pm(x["within_scene_auc"]), pm(x["selective_25"]),
            f3(x["cert_0.1_certified_fraction"]["mean"]),
            f3(x["cert_0.1_certified_precision"]["mean"]))
    for key, lab in (("fpose_p4", "FoundationPose + planner"),
                     ("gt_p4", "Ground-truth pose + planner (privileged)")):
        if key in plan:
            x = plan[key]
            c = x.get("cert_0.1", {})
            row(lab, f3(x["top1_success"]), f3(x["within_scene_auc"]), f3(x["selective_25"]),
                f3(c.get("certified_fraction", float("nan"))),
                f3(c.get("certified_precision", float("nan"))))
    te = m["corpus"]["test"]
    rows.append("\\midrule")
    rows.append(f"Random candidate & {f3(te['random_top1'])} & 0.500 & - & - & - \\\\")
    rows.append(f"Oracle (best candidate) & {f3(te['oracle_any_success'])} & 1.000 & - & - & - \\\\")
    (OUT / "table_main.tex").write_text("\n".join(rows) + "\n")

    import torch

    obj = torch.load(REPO / "data/lm_grasp/test.pt", map_location="cpu",
                     weights_only=False)["obj_id"].numpy()
    held = (1, 6, 9, 15)
    w = np.array([(obj == o).sum() for o in held], float)
    urows = []
    for base, lab in (("gqcnn", "Grasp-patch CNN"), ("msp_b300_d32", "MSP ($\\beta=300$)"),
                      ("direct", "Outcome scorer")):
        k = f"{base}_ho1-6-9-15"
        if k not in g or base not in g:
            continue
        # same four objects' test scenes, model trained WITH them (seen) vs WITHOUT (unseen)
        seen = [np.dot([r["per_object"][str(o)]["top1_success"] if str(o) in r["per_object"]
                        else r["per_object"][o]["top1_success"] for o in held], w) / w.sum()
                for name, r in m["runs"].items()
                if name.startswith(base + "_s") and int(name.rsplit("_s", 1)[1]) < 5]
        sv = {"mean": float(np.mean(seen)), "sd": float(np.std(seen, ddof=1))}
        x = g[k]
        urows.append(f"{lab} & {pm(sv)} & {pm(x['top1_success'])} & "
                     f"{pm(x['within_scene_auc'])} & {pm(x['selective_25'])} \\\\")
    (OUT / "table_unseen.tex").write_text("\n".join(urows) + "\n")


if __name__ == "__main__":
    main()
