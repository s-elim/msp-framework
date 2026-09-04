"""Figures for the reviewer-driven experiments. Reads results/rebuttal/*.json only.

    python scripts/paper/make_rebuttal_figures.py --out "ICRA'27_MSP-FRAMEWORK/figures"

fig_rate_sweep    measured payload bits vs decision quality and coverage (R1)
fig_packet_loss   what a lossy uplink does to action rate and to success (R1)
fig_selective     coverage on the executed grasp under three calibrations (R4)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL, DCOL = 3.5, 7.16
plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 8,
                     "savefig.dpi": 400, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#1B7F4B", "#5F6E85"


def fig_rate(res: dict, out: Path) -> None:
    sweep = sorted(res["sweep"], key=lambda r: r["entropy_bits"])
    bits = [r["entropy_bits"] for r in sweep]
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    ax.plot(bits, [r["within_scene_auc"] for r in sweep], "o-", color=BLUE,
            label="within-scene AUC")
    ax.plot(bits, [r["top1_success"] for r in sweep], "s-", color=ORANGE,
            label="top-1 success")
    ax.set_xscale("log")
    ax.set_xlabel("payload (bits/scene, entropy-coded)")
    ax.set_ylabel("decision quality")
    ax2 = ax.twinx()
    ax2.plot(bits, [r["coverage"] for r in sweep], "^--", color=GREEN, label="coverage")
    ax2.axhline(0.9, color=GREY, lw=0.7, ls=":")
    ax2.set_ylabel("coverage", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    for s in ("top",):
        ax.spines[s].set_visible(False)
    fig.savefig(out / "fig_rate_sweep.pdf")
    fig.savefig(out / "fig_rate_sweep.png")


def fig_loss(res: dict, out: Path) -> None:
    link = res["link"]
    p = [100 * r["p_loss"] for r in link]
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    ax.plot(p, [r["acted_fraction"] for r in link], "o-", color=BLUE,
            label="scenes acted on")
    ax.plot(p, [r["success_given_acted"] for r in link], "s-", color=GREEN,
            label="success | acted")
    ax.set_xlabel("packet loss (\\%)")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="center left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out / "fig_packet_loss.pdf")
    fig.savefig(out / "fig_packet_loss.png")


def fig_selective(rows: list, out: Path) -> None:
    names = ["pairwise", "executed", "selective"]
    labels = ["pairwise\n(published)", "executed grasp\n(pairwise $\\hat q$)",
              "executed grasp\n(selection-calibrated)"]
    alphas = [r["alpha"] for r in rows]
    x = np.arange(len(alphas))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(DCOL, 2.3))
    w = 0.26
    for k, (n, c) in enumerate(zip(names, (GREY, ORANGE, BLUE))):
        a0.bar(x + (k - 1) * w, [r[n]["coverage"] for r in rows], w, color=c, label=labels[k])
        a1.bar(x + (k - 1) * w, [r[n]["certified_fraction"] for r in rows], w, color=c)
    for i, r in enumerate(rows):
        a0.plot([i - 1.6 * w, i + 1.6 * w], [1 - r["alpha"]] * 2, color="k", lw=0.9, ls=":")
    a0.set_xticks(x); a0.set_xticklabels([f"$\\alpha$={a}" for a in alphas])
    a1.set_xticks(x); a1.set_xticklabels([f"$\\alpha$={a}" for a in alphas])
    a0.set_ylabel("coverage"); a1.set_ylabel("certified fraction")
    a0.set_ylim(0.7, 1.0)
    a0.legend(frameon=False, fontsize=6.5, loc="lower left")
    for ax in (a0, a1):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_selective_coverage.pdf")
    fig.savefig(out / "fig_selective_coverage.png")


def fig_frontier_vs_baseline(res: dict, out: Path) -> None:
    """Rate-quality frontier: the same quantizer applied to both representations."""
    sw = res["sweep"]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for key, colour, marker, label in (
        ("msp", BLUE, "o", "MSP (per-scene statistic)"),
        ("gqcnn", ORANGE, "s", "GQ-CNN (per-action features)"),
    ):
        pts = sorted(((r[key]["payload_bits"], r[key]["within_scene_auc"]) for r in sw))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker + "-", color=colour,
                label=label, ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("payload per scene (bits, entropy-coded)")
    ax.set_ylabel("within-scene AUC")
    ax.axhline(0.5, color=GREY, lw=0.7, ls=":")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out / "fig_rate_frontier.pdf")
    fig.savefig(out / "fig_rate_frontier.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/rebuttal")
    ap.add_argument("--out", default="ICRA'27_MSP-FRAMEWORK/figures")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (res / "r1.json").exists():
        r1 = json.loads((res / "r1.json").read_text())
        fig_rate(r1, out); fig_loss(r1, out)
        print("wrote fig_rate_sweep, fig_packet_loss")
    if (res / "r6.json").exists():
        fig_frontier_vs_baseline(json.loads((res / "r6.json").read_text()), out)
        print("wrote fig_rate_frontier")
    if (res / "r4.json").exists():
        fig_selective(json.loads((res / "r4.json").read_text()), out)
        print("wrote fig_selective_coverage")


if __name__ == "__main__":
    main()
