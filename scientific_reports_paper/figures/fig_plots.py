"""Figs. 3-5: data plots from results/sr/metrics_session4.json and the session-4 analysis files.

    $PY scientific_reports_paper/figures/fig_plots.py
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
from common import (
    BLUE,
    BLUE_DARK,
    GREY,
    GREY_LIGHT,
    MM,
    ORANGE,
    PURPLE,
    TEAL,
    TEAL_DARK,
    TEAL_LIGHT,
    metrics,
    panel_label,
    result,
    save,
)

COV = np.round(np.linspace(0.05, 1.0, 20), 2)


def _noise(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """levels, top-1 mean/sd and ADD(-S) pass mean/sd for rotation ('r') or translation ('t')."""
    m = metrics()["plan"]
    by: dict[float, list] = {}
    for k, v in m.items():
        g = re.match(r"noise_r([\d.]+)_t([\d.]+)_s\d+_p4$", k)
        if not g:
            continue
        r, t = float(g.group(1)), float(g.group(2))
        if (kind == "r" and t == 0) or (kind == "t" and r == 0):
            by.setdefault(r if kind == "r" else t, []).append((v["top1_success"], v["add_01d"]))
    lv = np.array([0.0, *sorted(by)])
    gt = m["gt_p4"]
    top = np.array([gt["top1_success"]] + [np.mean([x[0] for x in by[k]]) for k in lv[1:]])
    tsd = np.array([0.0] + [np.std([x[0] for x in by[k]], ddof=1) for k in lv[1:]])
    add = np.array([1.0] + [np.mean([x[1] for x in by[k]]) for k in lv[1:]])
    asd = np.array([0.0] + [np.std([x[1] for x in by[k]], ddof=1) for k in lv[1:]])
    return lv, top, tsd, add, asd


def fig3_pose_gap(name: str = "fig3_pose_gap") -> None:
    fig, ax = plt.subplots(1, 3, figsize=(180 * MM, 46 * MM),
                           gridspec_kw={"width_ratios": [1, 1, 1.25], "wspace": 0.38})
    fp = metrics()["plan"]["fpose_p4"]
    for a, kind, unit in ((ax[0], "r", "rotation error s.d. (deg)"),
                          (ax[1], "t", "translation error s.d. per axis (mm)")):
        lv, top, tsd, add, asd = _noise(kind)
        x = np.where(lv == 0, lv + (0.5 if kind == "r" else 1.0), lv)
        a.plot(x, add, color=GREY, ls="--", lw=1.0, marker="s", ms=2.5, label="ADD(-S) pass")
        a.fill_between(x, add - asd, add + asd, color=GREY, alpha=0.15, lw=0)
        a.plot(x, top, color=ORANGE, lw=1.4, marker="o", ms=3, label="top-1 grasp success")
        a.fill_between(x, top - tsd, top + tsd, color=ORANGE, alpha=0.2, lw=0)
        fp_err = fp["rot_err_deg_median"] if kind == "r" else fp["trans_err_mm_median"]
        a.axvline(fp_err, color=ORANGE, lw=0.6, ls=":")
        a.set_xscale("log")
        a.set_xlabel(unit)
        a.set_ylim(0, 1.02)
        ticks = [0.5, 1, 2, 5, 10, 20, 45] if kind == "r" else [1, 2, 5, 10, 20, 40]
        a.set_xticks(ticks, ["0" if (t == 0.5 and kind == "r") or (t == 1 and kind == "t")
                             else f"{t:g}" for t in ticks])
        a.minorticks_off()
    ax[0].set_ylabel("fraction of test scenes")
    ax[0].legend(loc="lower left", handlelength=1.6)
    bins = result("posterior_analysis.json")["bins"][:3]
    xs = np.arange(len(bins))
    w = 0.26
    for k, (key, col, hatch) in enumerate((("point", ORANGE, ""), ("posterior", TEAL, ""),
                                           ("annotated", "white", "////"))):
        ax[2].bar(xs + (k - 1) * w, [b[key] for b in bins], w * 0.92, color=col,
                  edgecolor=col if key != "annotated" else GREY, hatch=hatch, lw=0.7,
                  label={"point": "FoundationPose point", "posterior": "pose posterior",
                         "annotated": "annotated pose"}[key])
    ax[2].set_xticks(xs, [f"< 2 %\nn={bins[0]['n']}", f"2-5 %\nn={bins[1]['n']}",
                          f"5-10 %\nn={bins[2]['n']}"])
    ax[2].set_xlabel("FoundationPose error (% of diameter)")
    ax[2].set_ylabel("top-1 grasp success")
    ax[2].set_ylim(0.5, 1.02)
    ax[2].legend(loc="upper left", ncol=1, handlelength=1.2, bbox_to_anchor=(0.0, 1.04))
    for k, lab in enumerate("abc"):
        panel_label(ax[k], lab, x=-0.12)
    save(fig, name)


METHODS = (  # (label, source, key, colour, hatch)
    ("Dex-Net 2.0 arch.", "group", "gqcnn", BLUE, ""),
    ("RGB-D scorer", "group", "direct", BLUE, ""),
    ("ensemble (5)", "group", "ensemble5", BLUE_DARK, ""),
    ("POISE", "group", "direct_plfpose", PURPLE, ""),
    ("regressor + planner", "group", "plan:pose_p4", "#F3C877", ""),
    ("FoundationPose point", "plan", "fpose_p4", ORANGE, ""),
    ("FoundationPose point, 64 draws", "plan", "fpose_p64", ORANGE, "////"),
    ("pose posterior (ours)", "plan", "fposeU1_s0_p4", TEAL, ""),
    ("+ second view (ours)", "plan", "fposeTV_s0_p4", TEAL_DARK, ""),
    ("annotated pose", "plan", "gt_p4", "white", "////"),
)


def _val(src: str, key: str, field: str) -> tuple[float, float, list]:
    m = metrics()
    if src == "plan":
        return m["plan"][key][field], 0.0, [m["plan"][key][field]]
    g = m["groups"][key][field]
    return g["mean"], g["sd"] if np.isfinite(g["sd"]) else 0.0, g.get("values", [g["mean"]])


def fig4_results(name: str = "fig4_results") -> None:
    fig = plt.figure(figsize=(180 * MM, 52 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.42)
    a = fig.add_subplot(gs[0, 0])
    ys = np.arange(len(METHODS))[::-1]
    for y, (_lab, src, key, col, hatch) in zip(ys, METHODS, strict=True):
        mu, sd, vals = _val(src, key, "top1_success")
        a.barh(y, mu, color=col, edgecolor=col if col != "white" else GREY, hatch=hatch,
               lw=0.6, height=0.7, xerr=sd if sd else None,
               error_kw=dict(lw=0.6, capsize=1.5))
        if len(vals) > 1:
            a.scatter(vals, y + np.linspace(-0.18, 0.18, len(vals)), s=2.5, c="black",
                      zorder=3, linewidths=0)
        a.text(mu + (sd or 0) + 0.004, y, f"{mu:.3f}", va="center", fontsize=5.6)
    a.set_yticks(ys, [m[0] for m in METHODS])
    a.set_xlim(0.6, 0.9)
    a.set_xlabel("top-1 grasp success")
    a.tick_params(axis="y", length=0)
    b = fig.add_subplot(gs[0, 1])
    m = metrics()
    curves = (("FoundationPose point", m["plan"]["fpose_p4"]["selective_curve"], ORANGE, "-"),
              ("ensemble (5)", m["groups"]["ensemble5"]["selective_curve"]["mean"], BLUE_DARK, "-"),
              ("POISE", m["groups"]["direct_plfpose"]["selective_curve"]["mean"], PURPLE, "-"),
              ("annotated pose", m["plan"]["gt_p4"]["selective_curve"], GREY, "--"),
              ("pose posterior (ours)", m["plan"]["fposeU1_s0_p4"]["selective_curve"], TEAL, "-"),
              ("+ second view", m["plan"]["fposeTV_s0_p4"]["selective_curve"], TEAL_DARK, ":"))
    for lab, c, col, ls in curves:
        b.plot(COV, c, color=col, ls=ls, lw=1.2, label=lab)
    b.set_xlabel("fraction of scenes acted on")
    b.set_ylabel("success of executed grasps")
    b.set_ylim(0.68, 1.005)
    b.legend(loc="lower left", fontsize=5.4, handlelength=1.5)
    c = fig.add_subplot(gs[0, 2])
    for key, col, _lab in (("fpose_p4", ORANGE, "FoundationPose point"),
                          ("fposeU1_s0_p4", TEAL, "pose posterior (ours)"),
                          ("gt_p4", GREY, "annotated pose")):
        r = m["plan"][key]
        pts = [(r[f"cert_{al}"]["certified_fraction"], r[f"cert_{al}"]["certified_precision"],
                mk) for al, mk in ((0.05, "o"), (0.1, "s"), (0.2, "^"))]
        pts = [p for p in pts if p[0] > 0 and np.isfinite(p[1])]
        c.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=0.8,
               ls="--" if key == "gt_p4" else "-")
        for x_, y_, mk in pts:
            c.scatter(x_, y_, marker=mk, s=16, color=col, edgecolors="white", lw=0.4, zorder=3)
    from matplotlib.lines import Line2D

    c.legend([Line2D([], [], ls="", marker=mk, color="#555555", ms=3.5) for mk in "os^"],
             [r"$\alpha=0.05$", r"$\alpha=0.1$", r"$\alpha=0.2$"], loc="lower left",
             handletextpad=0.2, fontsize=5.5)
    c.axhline(0.9, color=GREY_LIGHT, lw=0.6, ls=":")
    c.set_xlabel("certified fraction of scenes")
    c.set_ylabel("success of certified grasps")
    c.set_ylim(0.88, 1.0)
    for k, ax_ in enumerate((a, b, c)):
        panel_label(ax_, "abc"[k], x=-0.05 if k else -0.55)
    save(fig, name)


def fig5_analysis(name: str = "fig5_analysis") -> None:
    fig, ax = plt.subplots(1, 3, figsize=(180 * MM, 47 * MM),
                           gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 1.1]})
    cal = result("posterior/calibration_test.json")["modes"]
    lv = np.array([0.5, 0.8, 0.9])
    style = {"knn": (TEAL, "ours"), "knn_xo": (TEAL_LIGHT, "ours, other objects"),
             "object": (GREY, "unconditional"), "gaussian": (PURPLE, "Gaussian")}
    for a, key in ((ax[0], "rot_coverage"), (ax[1], "trans_coverage")):
        a.plot([0.45, 0.95], [0.45, 0.95], color=GREY_LIGHT, lw=0.8)
        for mode, (col, lab) in style.items():
            a.plot(lv, [cal[mode][key][f"{x:g}"] for x in lv], color=col, marker="o", ms=3,
                   lw=1.2, label=lab)
        a.set_xlim(0.45, 0.95)
        a.set_ylim(0.25, 0.97)
        a.set_xlabel("nominal posterior quantile")
        a.set_aspect("equal", adjustable="box")
    ax[0].set_ylabel("empirical coverage")
    ax[0].legend(loc="lower right", fontsize=5.4, handlelength=1.4)
    sv = result("design/second_view_test.json")
    rates = list(sv["rates"])
    xr = np.array([0.0] + [sv["rates"][k]["rate_realised"] for k in rates])
    base = sv["single_view_top1"]
    a = ax[2]
    a.axhline(0, color=GREY_LIGHT, lw=0.6)
    gain = np.array([0.0] + [sv["rates"][k]["two_view_minus_single_view"]["diff"] for k in rates])
    lo = np.array([0.0] + [sv["rates"][k]["two_view_minus_single_view"]["ci95"][0] for k in rates])
    hi = np.array([0.0] + [sv["rates"][k]["two_view_minus_single_view"]["ci95"][1] for k in rates])
    a.fill_between(xr, 100 * lo, 100 * hi, color=TEAL, alpha=0.18, lw=0)
    a.plot(xr, 100 * gain, color=TEAL_DARK, marker="o", ms=3, lw=1.3,
           label="triggered second view")
    a.plot(xr, 100 * (np.array([base] + [sv["rates"][k]["random_trigger_two_view_top1"]
                                         for k in rates]) - base),
           color=GREY, marker="s", ms=2.5, lw=1.0, label="random trigger")
    a.plot(xr, 100 * (np.array([base] + [sv["rates"][k]["same_view_2x_samples_top1"]
                                         for k in rates]) - base),
           color=ORANGE, marker="^", ms=3, lw=1.0, label="same view, 2x samples")
    a.set_xlabel("fraction of scenes re-observed")
    a.set_ylabel("top-1 gain over one view (pp)")
    a.legend(loc="upper left", fontsize=5.4, handlelength=1.5)
    for k, lab in enumerate("abc"):
        panel_label(ax[k], lab, x=-0.1)
    save(fig, name)


if __name__ == "__main__":
    fig3_pose_gap()
    fig4_results()
    fig5_analysis()
