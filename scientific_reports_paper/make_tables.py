"""Write numbers.tex (one macro per number quoted in the text) and the LaTeX tables of the
Scientific Reports manuscript from the results files. No number in main.tex is typed by hand.

    cd /nas/Selim/selim_sarowar/msp_framework
    source scripts/sr/env.sh && $PY scientific_reports_paper/make_tables.py

Inputs: results/sr/metrics_session4.json, posterior_analysis.json, posterior/calibration_test.json,
design/second_view_test.json, latency.json, lmo/lmo_metrics.json, data/lm_grasp/summary.json,
results/sr/plan/*_test.pt and data/lm_grasp/test.pt (paired bootstraps not stored elsewhere).
Outputs (beside this script): numbers.tex, table_main.tex, table_ablation.tex.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
R = REPO / "results/sr"
sys.path.insert(0, str(REPO / "scripts/sr"))


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def f3(x: float) -> str:
    return f"{x:.3f}"


def sgn(x: float) -> str:
    if abs(x) < 0.0005:
        return "0.000"
    return f"{x:+.3f}".replace("-", "$-$")


def ci(c: dict) -> str:
    lo, hi = c["ci95"]
    return f"[{sgn(lo)}, {sgn(hi)}]"


def macro(name: str, val: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{val}}}"


def paired(a_tag: str, b_tag: str, y: np.ndarray) -> dict:
    import torch
    from evaluate_lm import paired_bootstrap, pick

    r = np.arange(len(y))
    ok = [y[r, pick(torch.load(R / "plan" / f"{t}_test.pt", map_location="cpu",
                                    weights_only=False)["s"].numpy())] for t in (a_tag, b_tag)]
    return paired_bootstrap(ok[0], ok[1])


def main() -> None:
    import torch

    m = load("metrics_session4.json")
    P, G, C = m["plan"], m["groups"], m["comparisons"]
    an = load("posterior_analysis.json")
    cal = load("posterior/calibration_test.json")["modes"]
    sv = load("design/second_view_test.json")
    lat = load("latency.json")
    summ = json.loads((REPO / "data/lm_grasp/summary.json").read_text())
    y = torch.load(REPO / "data/lm_grasp/test.pt", map_location="cpu",
                   weights_only=False)["succ"].numpy()
    lines: list[str] = []

    def add(name: str, val: str) -> None:
        lines.append(macro(name, val))

    # corpus
    n_rollouts = sum(v["grasps"] for v in summ["per_object"].values())
    add("NScenes", f"{summ['scenes']:,}".replace(",", "{,}"))
    for k, nm in (("train", "NTrain"), ("val", "NVal"), ("calib", "NCalib"), ("test", "NTest")):
        add(nm, f"{summ[f'n_{k}']:,}".replace(",", "{,}"))
    add("NRollouts", f"{n_rollouts:,}".replace(",", "{,}"))
    add("TestBase", f3(m["corpus"]["test"]["base_rate"]))
    add("TestExec", f3(m["corpus"]["test"]["executable"]))
    add("TestOracle", f3(m["corpus"]["test"]["oracle_any_success"]))
    add("NMixed", f"{P['fposeU1_s0_p4']['n_mixed_scenes']:,}".replace(",", "{,}"))
    up = [v["median_deg"] for v in load("lm_upright.json").values()]
    add("UprightDegLo", f"{min(up):.1f}")
    add("UprightDegHi", f"{max(up):.1f}")
    # FoundationPose
    fp = P["fpose_p4"]
    add("FpAdd", f3(fp["add_01d"]))
    add("FpRotErr", f"{fp['rot_err_deg_median']:.1f}")
    add("FpTransErr", f"{fp['trans_err_mm_median']:.1f}")
    fp_secs = np.concatenate([np.load(p)["secs"] for p in (R / "fp").glob("*of*.npz")])
    add("FpSec", f"{np.median(fp_secs):.2f}")
    # planners
    for nm, tag in (("Point", "fpose_p4"), ("PointSixFour", "fpose_p64"),
                    ("Post", "fposeU1_s0_p4"), ("PostB", "fposeU1_s1_p4"),
                    ("TwoView", "fposeTV_s0_p4"), ("Gt", "gt_p4"), ("Gauss", "fposeGauss_s0_p4"),
                    ("Obj", "fposeObj_s0_p4"), ("Xo", "fposeXO_s0_p4")):
        add(f"{nm}Top", f3(P[tag]["top1_success"]))
        add(f"{nm}Auc", f3(P[tag]["within_scene_auc"]))
        add(f"{nm}Sel", f3(P[tag]["selective_25"]))
    for nm, tag in (("Point", "fpose_p4"), ("Post", "fposeU1_s0_p4"), ("Gt", "gt_p4")):
        c1 = P[tag]["cert_0.1"]
        add(f"{nm}CertFrac", f3(c1["certified_fraction"]))
        add(f"{nm}CertPrec", f3(c1["certified_precision"]))
        add(f"{nm}CertCov", f3(c1["coverage"]))
    cmp_keys = {"CmpPostPointSixFour": "plan:fposeU1_s0_p4 - plan:fpose_p64",
                "CmpPostPoint": "plan:fposeU1_s0_p4 - plan:fpose_p4",
                "CmpPostGauss": "plan:fposeU1_s0_p4 - plan:fposeGauss_s0_p4",
                "CmpPostObj": "plan:fposeU1_s0_p4 - plan:fposeObj_s0_p4",
                "CmpGtPost": "plan:gt_p4 - plan:fposeU1_s0_p4",
                "CmpPostXo": "plan:fposeU1_s0_p4 - plan:fposeXO_s0_p4",
                "CmpXoPointSixFour": "plan:fposeXO_s0_p4 - plan:fpose_p64",
                "CmpTwoViewPost": "plan:fposeTV_s0_p4 - plan:fposeU1_s0_p4",
                "CmpPoiseUPoise": "direct_plfposeU1_s0 - direct_plfpose",
                "CmpPoiseUPost": "direct_plfposeU1_s0 - plan:fposeU1_s0_p4",
                "CmpGtPoint": "plan:gt_p4 - plan:fpose_p4"}
    extra = {"CmpGaussPointSixFour": ("fposeGauss_s0_p4", "fpose_p64"),
             "CmpObjPointSixFour": ("fposeObj_s0_p4", "fpose_p64"),
             "CmpPointSixFourPoint": ("fpose_p64", "fpose_p4")}
    comps = {k: C[v] for k, v in cmp_keys.items()}
    comps |= {k: paired(a, b, y) for k, (a, b) in extra.items()}
    g = comps["CmpGtPost"]  # also as posterior - annotated, the sign the text reads naturally
    comps["CmpPostGt"] = {"diff": -g["diff"], "ci95": [-g["ci95"][1], -g["ci95"][0]]}
    for k, v in comps.items():
        add(k, sgn(v["diff"]))
        add(k + "Ci", ci(v))
        add(k + "Abs", f"{abs(v['diff']):.3f}")  # for sentences that state the direction
    # learned scorers and fusion
    for nm, g in (("Gq", "gqcnn"), ("Direct", "direct"), ("Ens", "ensemble5"),
                  ("Poise", "direct_plfpose"), ("PoiseU", "direct_plfposeU1_s0"),
                  ("Regr", "plan:pose_p4")):
        add(f"{nm}Top", f3(G[g]["top1_success"]["mean"]))
        add(f"{nm}Sel", f3(G[g]["selective_25"]["mean"]))
    add("LatDirect", f"{lat['cpu']['direct']:.1f}")
    add("LatEns", f"{lat['cpu']['ensemble5']:.1f}")
    # error bins and per-object
    for nm, b in zip(("Lo", "Mid", "Hi"), an["bins"][:3], strict=True):
        add(f"Bin{nm}N", f"{b['n']:,}".replace(",", "{,}"))
        for k in ("point", "posterior", "annotated"):
            add(f"Bin{nm}{k.capitalize()}", f3(b[k]))
    r = np.arange(len(y))
    from evaluate_lm import pick

    ip = {t: pick(torch.load(R / "plan" / f"{t}_test.pt", map_location="cpu",
                             weights_only=False)["s"].numpy()) for t in ("fpose_p4", "fposeU1_s0_p4")}
    ok_p, ok_u = y[r, ip["fpose_p4"]], y[r, ip["fposeU1_s0_p4"]]
    add("NDiffer", str(int((ip["fpose_p4"] != ip["fposeU1_s0_p4"]).sum())))
    add("NWins", str(int(((ok_u > 0.5) & (ok_p < 0.5)).sum())))
    add("NLosses", str(int(((ok_u < 0.5) & (ok_p > 0.5)).sum())))
    # calibration (test)
    for nm, mode in (("Knn", "knn"), ("Obj", "object"), ("Gauss", "gaussian"), ("Xo", "knn_xo")):
        add(f"Cal{nm}RotNine", f3(cal[mode]["rot_coverage"]["0.9"]))
        add(f"Cal{nm}TransNine", f3(cal[mode]["trans_coverage"]["0.9"]))
        add(f"Cal{nm}RotFive", f3(cal[mode]["rot_coverage"]["0.5"]))
        add(f"Cal{nm}TransFive", f3(cal[mode]["trans_coverage"]["0.5"]))
        add(f"Rho{nm}Rot", f"{cal[mode]['spearman_rot_spread_vs_err']:.2f}")
        add(f"Rho{nm}Trans", f"{cal[mode]['spearman_trans_spread_vs_err']:.2f}")
    # cost
    cost = an["cost"]
    for nm, k in (("Point", "point"), ("Post", "posterior"), ("TwoView", "two_view")):
        add(f"Cpu{nm}", f"{cost[k]['cpu_seconds_per_scene']:.1f}")
    # second view (val threshold applied on test)
    add("SvViewDeg", f"{load('posterior/second_view_test.json')['view_change_deg_median']:.0f}")
    q = sv["rates"]["0.25"]
    add("SvRate", f"{100 * q['rate_realised']:.0f}")
    add("SvGain", sgn(q["two_view_minus_single_view"]["diff"]))
    add("SvGainCi", ci(q["two_view_minus_single_view"]))
    add("SvVsDouble", sgn(q["two_view_minus_same_view_2x"]["diff"]))
    add("SvVsDoubleCi", ci(q["two_view_minus_same_view_2x"]))
    add("SvAllGain", sgn(sv["rates"]["1"]["two_view_minus_single_view"]["diff"]))
    add("SvAllGainCi", ci(sv["rates"]["1"]["two_view_minus_single_view"]))
    # pose-noise study (annotated pose perturbed)
    for kind, nm in (("r5_t0", "RotFive"), ("r0_t5", "TransFive"), ("r10_t0", "RotTen"),
                     ("r0_t10", "TransTen")):
        rows = [P[f"noise_{kind}_s{s}_p4"] for s in range(3)]
        add(f"Noise{nm}Top", f3(np.mean([x["top1_success"] for x in rows])))
        add(f"Noise{nm}Add", f3(np.mean([x["add_01d"] for x in rows])))
    # LM-O transfer check
    lmo_p = R / "lmo/lmo_metrics.json"
    if lmo_p.exists():
        lm = json.loads(lmo_p.read_text())
        add("LmoInst", f"{lm['instances']:,}".replace(",", "{,}"))
        add("LmoFrames", str(lm["frames"]))
        add("LmoUpright", f3(lm["upright_fraction"]))
        for sub, nm in (("all", ""), ("upright", "Up")):
            for k, lab in (("s_point", "Point"), ("s_post", "Post"), ("s_annot", "Gt")):
                add(f"Lmo{nm}{lab}Top", f3(lm[sub][k]["top1"]))
                add(f"Lmo{nm}{lab}Sel", f3(lm[sub][k]["sel25"]))
            add(f"Lmo{nm}Cmp", sgn(lm[sub]["post_minus_point"]["diff"]))
            add(f"Lmo{nm}CmpCi", ci(lm[sub]["post_minus_point"]))
            add(f"Lmo{nm}GtMinusPost", sgn(lm[sub]["annot_minus_post"]["diff"]))
            add(f"Lmo{nm}GtMinusPostCi", ci(lm[sub]["annot_minus_post"]))
            add(f"Lmo{nm}GtMinusPostAbs", f"{abs(lm[sub]['annot_minus_post']['diff']):.3f}")
            gap = lm[sub]["s_annot"]["top1"] - lm[sub]["s_point"]["top1"]
            add(f"Lmo{nm}Recovered", f"{100 * lm[sub]['post_minus_point']['diff'] / gap:.0f}")
    (OUT / "numbers.tex").write_text("% written by make_tables.py; do not edit\n"
                                     + "\n".join(lines) + "\n")
    write_main(P, G, m)
    write_ablation(P, cal, comps)
    print(f"{len(lines)} macros")


def _cells(vals: list[float | None], best_idx: list[int], d: int = 3) -> list[str]:
    order = sorted({v for i, v in enumerate(vals) if v is not None and i in best_idx},
                   reverse=True)
    out = []
    for i, v in enumerate(vals):
        if v is None:
            out.append("--")
            continue
        s = f"{v:.{d}f}"
        if i in best_idx and order and v == order[0]:
            s = "\\best " + s
        elif i in best_idx and len(order) > 1 and v == order[1]:
            s = "\\second " + s
        out.append(s)
    return out


def write_main(P: dict, G: dict, m: dict) -> None:
    """Table 1. Families: learned outcome scorers, learned fusion, pose-then-plan, reference."""
    def g(key: str, f: str) -> tuple[float, float, int]:
        v = G[key][f]
        return v["mean"], v["sd"], G[key]["n_seeds"]

    rows = []  # (family, label, n, top, top_sd, auc, sel, cert_frac, cert_prec, ours)
    for fam, lab, key in (("learn", "Dex-Net 2.0 architecture\\cite{mahler2017dexnet}", "gqcnn"),
                          ("learn", "RGB-D outcome scorer", "direct"),
                          ("learn", "Deep ensemble of 5\\cite{lakshminarayanan2017}", "ensemble5"),
                          ("fuse", "POISE (scorer + point score)", "direct_plfpose"),
                          ("fuse", "POISE-U (scorer + posterior score)",
                           "direct_plfposeU1_s0"),
                          ("plan", "Pose regressor + planner", "plan:pose_p4")):
        top, sd, n = g(key, "top1_success")
        cf = G[key].get("cert_0.1_certified_fraction", {}).get("mean")
        cp = G[key].get("cert_0.1_certified_precision", {}).get("mean")
        rows.append((fam, lab, n, top, sd, g(key, "within_scene_auc")[0],
                     g(key, "selective_25")[0], cf, cp, False))
    for lab, tag, ours in (("FoundationPose\\cite{wen2024foundationpose} + planner", "fpose_p4", False),
                           ("FoundationPose + planner, 64 draws", "fpose_p64", False),
                           ("\\textbf{Pose posterior (ours)}", "fposeU1_s0_p4", True),
                           ("\\textbf{+ second view (ours)}", "fposeTV_s0_p4", True)):
        r = P[tag]
        c1 = r.get("cert_0.1", {})
        rows.append(("plan", lab, 1, r["top1_success"], float("nan"), r["within_scene_auc"],
                     r["selective_25"], c1.get("certified_fraction"),
                     c1.get("certified_precision"), ours))
    r = P["gt_p4"]
    rows.append(("ref", "Annotated pose + planner", 1, r["top1_success"], float("nan"),
                 r["within_scene_auc"], r["selective_25"], r["cert_0.1"]["certified_fraction"],
                 r["cert_0.1"]["certified_precision"], False))
    rows.append(("ref", "Oracle (best candidate)", 1, m["corpus"]["test"]["oracle_any_success"],
                 float("nan"), None, None, None, None, False))
    comp = [i for i, r_ in enumerate(rows) if r_[0] != "ref"]
    cols = {k: _cells([r_[k] for r_ in rows], comp) for k in (3, 5, 6, 8)}
    fam_col = {"learn": "famlearn", "fuse": "famfuse", "plan": "fampose", "ref": "famref"}
    fam_name = {"learn": "Learned outcome scorers (no object model)",
                "fuse": "Learned fusion with the planner score",
                "plan": "Pose-then-plan (mesh, simulator and physics prior at decision time)",
                "ref": "Privileged references"}
    out, last = [], None
    for i, r_ in enumerate(rows):
        fam, lab, n, _top, sd, _auc, _sel, cf, _cp, ours = r_
        if fam != last:
            if last is not None:
                out.append("\\midrule")
            out.append(f"\\multicolumn{{6}}{{l}}{{\\textcolor{{{fam_col[fam]}}}"
                       f"{{\\textbf{{{fam_name[fam]}}}}}}}\\\\")
            last = fam
        topc = cols[3][i] + (f" $\\pm$ {sd:.3f}" if np.isfinite(sd) else "")
        cert = "--" if cf is None or not np.isfinite(cf) or cf == 0 else (
            f"{cf:.3f} / " + cols[8][i])
        pre = "\\rowcolor{oursrow}" if ours else ""
        out.append(f"{pre}{lab} & {n} & {topc} & {cols[5][i]} & {cols[6][i]} & {cert}\\\\")
    head = ["\\begin{tabular}{lccccc}", "\\toprule",
            "Method & n & Top-1 & AUC & Sel@25\\% & Cert. (fraction / precision)\\\\",
            "\\midrule"]
    (OUT / "table_main.tex").write_text("\n".join(head + out + ["\\bottomrule",
                                                               "\\end{tabular}"]) + "\n")


def write_ablation(P: dict, cal: dict, comps: dict) -> None:
    """Table 2: what the posterior needs (test split)."""
    rows = (("Point estimate, 4 draws", "fpose_p4", None, "CmpPointSixFourPoint", True),
            ("Point estimate, 64 draws", "fpose_p64", None, None, False),
            ("Gaussian, matched moments", "fposeGauss_s0_p4", "gaussian", "CmpGaussPointSixFour",
             False),
            ("Bootstrap, unconditional", "fposeObj_s0_p4", "object", "CmpObjPointSixFour", False),
            ("kNN, other 12 objects", "fposeXO_s0_p4", "knn_xo", "CmpXoPointSixFour", False),
            ("\\textbf{kNN, same object (ours)}", "fposeU1_s0_p4", "knn", "CmpPostPointSixFour",
             False))
    out = []
    for lab, tag, mode, cmp, rev in rows:
        r = P[tag]
        cells = ([f"{cal[mode]['rot_coverage']['0.9']:.2f}", f"{cal[mode]['trans_coverage']['0.9']:.2f}",
                  f"{cal[mode]['spearman_rot_spread_vs_err']:.2f}",
                  f"{cal[mode]['spearman_trans_spread_vs_err']:.2f}"] if mode else ["--"] * 4)
        if cmp is None:
            d = "ref."
        else:
            c = comps[cmp]
            diff, lo, hi = c["diff"], *c["ci95"]
            if rev:  # stored as 64 draws - 4 draws
                diff, lo, hi = -diff, -hi, -lo
            d = f"{sgn(diff)} {ci({'ci95': [lo, hi]})}"
        pre = "\\rowcolor{oursrow}" if "ours" in lab else ""
        out.append(f"{pre}{lab} & " + " & ".join(cells) + f" & {r['top1_success']:.3f} & {d} & "
                   f"{r['selective_25']:.3f}\\\\")
    head = ["\\begin{tabular}{lccccccc}", "\\toprule",
            " & \\multicolumn{2}{c}{Coverage@0.9} & \\multicolumn{2}{c}{$\\rho$(spread, error)} & & & \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
            "Pose distribution & rot. & trans. & rot. & trans. & Top-1 & $\\Delta$ vs 64 draws [95\\% CI] & Sel@25\\%\\\\",
            "\\midrule"]
    (OUT / "table_ablation.tex").write_text("\n".join(head + out + ["\\bottomrule",
                                                                   "\\end{tabular}"]) + "\n")


if __name__ == "__main__":
    main()
