"""Turn results/*.json into camera-ready PDF figures.

    python scripts/paper/make_figures.py --results results/ --out paper/figures/

Reads only JSON. Never trains, never touches a GPU, never invents a number. If a result file
is missing, the corresponding figure is skipped with a warning rather than silently drawing
placeholder data -- a figure in a paper must be traceable to an experiment that ran.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("figures")

# Single-column IEEE width is 3.5in; double is 7.16in.
COL, DCOL = 3.5, 7.16
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
})

INK = "#14181F"
ACCENT = "#17557F"
WARN = "#A3160B"


def _load(results: Path, name: str):
    p = results / name
    if not p.exists():
        log.warning("SKIP %s (run scripts/paper/run_experiments.py first)", name)
        return None
    return json.loads(p.read_text())


# ======================================================================================


def fig_frontier(results: Path, out: Path) -> None:
    """Figure: the rate-distortion frontier. The minimality evidence (V2 Ch.6)."""
    rows = _load(results, "e1_frontier.json")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["beta"])
    rate = np.array([r["rate"] for r in rows])
    dist = np.array([r["distortion"] for r in rows])
    beta = np.array([r["beta"] for r in rows])

    fig, ax = plt.subplots(figsize=(COL, 2.4))
    ax.plot(rate, dist, "-o", color=ACCENT, zorder=3)
    for r, d, b in zip(rate, dist, beta):
        ax.annotate(f"$\\beta$={b:g}", (r, d), textcoords="offset points",
                    xytext=(4, 4), fontsize=6, color=INK, alpha=0.8)

    ax.set_xlabel("Rate  $R = \\mathbb{E}_o\\,\\mathrm{KL}(q_\\theta(z|o)\\,\\|\\,r(z))$   [nats]")
    ax.set_ylabel("Outcome distortion  $D$   [nats]")
    ax.set_title("Rate-distortion frontier (Eq. 7-10)")
    fig.savefig(out / "fig_frontier.pdf")
    fig.savefig(out / "fig_frontier.png")
    plt.close(fig)
    log.info("wrote fig_frontier")

    # Companion: the DIRECTION of beta. This panel exists because the reference
    # implementation had beta inverted, and a reader is entitled to see that
    # beta -> inf buys sufficiency (Theorem 3), not compression.
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    ax.semilogx(beta, dist, "-o", color=ACCENT, label="distortion $D$ (sufficiency)")
    ax2 = ax.twinx()
    ax2.semilogx(beta, rate, "-s", color=WARN, label="rate $R$ (compression)")
    ax2.grid(False)
    ax.set_xlabel("$\\beta$  (multiplies the relevance term)")
    ax.set_ylabel("$D$", color=ACCENT)
    ax2.set_ylabel("$R$", color=WARN)
    ax.set_title("Theorem 3:  $\\beta \\to \\infty \\Rightarrow$ sufficiency")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    fig.savefig(out / "fig_beta_direction.pdf")
    plt.close(fig)
    log.info("wrote fig_beta_direction")


def fig_identifiability(results: Path, out: Path) -> None:
    """Figure: contribution C2. Principal angles between the predicted ker J(x) and the
    independently MEASURED grasp-invariant subspace. The result the T-RO review identifies
    as the one that moved Reviewer A to Accept."""
    res = _load(results, "e2_identifiability.json")
    if not res:
        return
    angles = np.array(res["principal_angles_deg"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DCOL, 2.3), width_ratios=[1.1, 1])

    a1.hist(angles, bins=30, color=ACCENT, edgecolor="white", linewidth=0.4)
    a1.axvline(float(angles.mean()), color=WARN, ls="--", lw=1.0,
               label=f"mean = {angles.mean():.2f}$^\\circ$")
    a1.set_xlabel("Principal angle between $\\ker J(x)$ and the measured\n"
                  "grasp-invariant subspace  [degrees]")
    a1.set_ylabel("count")
    a1.set_title(f"Theorem 4 verified at {res['n_states']} states")
    a1.legend()

    labels = ["true\n$\\dim\\ker J$", "estimated\n$\\dim\\ker J$",
              "true\nrank $J$", "estimated\nrank $J$"]
    vals = [res["true_null_dim"], res["estimated_null_dim_mean"],
            res["true_rank"], res["estimated_rank_mean"]]
    colors = [ACCENT, "#6FB3E0", INK, "#6B7684"]
    a2.bar(range(4), vals, color=colors, width=0.62)
    a2.set_xticks(range(4))
    a2.set_xticklabels(labels, fontsize=6)
    a2.set_ylabel("dimension")
    a2.set_title("Recovered dimensions ($d_X = %d$)" % res["state_dim"])
    for i, v in enumerate(vals):
        a2.text(i, v + 0.05, f"{v:g}", ha="center", fontsize=6.5)

    fig.savefig(out / "fig_identifiability.pdf")
    fig.savefig(out / "fig_identifiability.png")
    plt.close(fig)
    log.info("wrote fig_identifiability  (max angle %.3f deg)", res["max_angle_deg"])


def fig_coverage(results: Path, out: Path) -> None:
    """Figure: contribution C4. Empirical coverage vs the nominal 1 - alpha (Theorem 7)."""
    rows = _load(results, "e3_coverage.json")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["alpha"])
    tgt = np.array([r["target_coverage"] for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    prec = np.array([r["certified_precision"] for r in rows])
    abst = np.array([r["abstention_rate"] for r in rows])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DCOL, 2.3))

    lims = [min(tgt.min(), cov.min()) - 0.03, 1.005]
    a1.plot(lims, lims, "--", color=INK, lw=0.8, label="nominal $1-\\alpha$")
    a1.plot(tgt, cov, "o-", color=ACCENT, label="empirical coverage")
    a1.set_xlim(lims)
    a1.set_ylim(lims)
    a1.set_xlabel("target coverage  $1-\\alpha$")
    a1.set_ylabel("empirical coverage")
    a1.set_title("Theorem 7: marginal coverage")
    a1.legend(loc="lower right")

    # The operational trade-off the certificate creates.
    a2.plot(tgt, prec, "o-", color=ACCENT, label="precision | certified")
    a2.plot(tgt, abst, "s-", color=WARN, label="abstention rate")
    a2.set_xlabel("target coverage  $1-\\alpha$")
    a2.set_ylabel("rate")
    a2.set_title("Cost of the certificate")
    a2.legend()

    fig.savefig(out / "fig_coverage.pdf")
    fig.savefig(out / "fig_coverage.png")
    plt.close(fig)
    log.info("wrote fig_coverage")


def fig_latent_dim(results: Path, out: Path) -> None:
    """Figure: the 'lower-dimensional' clause of the hypothesis. Distortion should saturate
    once d reaches the true manipulation-relevant dimension -- extra capacity is wasted."""
    rows = _load(results, "e5_latent_dim.json")
    if not rows:
        return
    e2 = _load(results, "e2_identifiability.json")
    rows = sorted(rows, key=lambda r: r["latent_dim"])
    d = np.array([r["latent_dim"] for r in rows])
    dist = np.array([r["distortion"] for r in rows])

    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.plot(d, dist, "-o", color=ACCENT)
    if e2:
        ax.axvline(e2["true_rank"], color=WARN, ls="--", lw=1.0,
                   label=f"$\\mathrm{{rank}}\\,J(x) = {e2['true_rank']}$")
        ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xlabel("latent dimension $d$")
    ax.set_ylabel("outcome distortion $D$  [nats]")
    ax.set_title("Sufficiency saturates at the intrinsic dimension")
    fig.savefig(out / "fig_latent_dim.pdf")
    plt.close(fig)
    log.info("wrote fig_latent_dim")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig_identifiability(args.results, args.out)
    fig_frontier(args.results, args.out)
    fig_coverage(args.results, args.out)
    fig_latent_dim(args.results, args.out)
    log.info("figures -> %s", args.out)


if __name__ == "__main__":
    main()
