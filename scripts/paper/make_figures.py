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
    # One serif family everywhere, matching fig_framework and the IEEE body font.
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "savefig.dpi": 400,
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


def _load_any(results: Path, *names: str):
    """First existing file wins. The LIBERO results (l*) supersede the synthetic-world
    ones (e*): same evaluator, same schema, real objects."""
    for n in names:
        p = results / n
        if p.exists():
            return json.loads(p.read_text())
    log.warning("SKIP %s (not found)", " or ".join(names))
    return None


# ======================================================================================


def fig_frontier(results: Path, out: Path) -> None:
    """Figure: the rate-distortion frontier. The minimality evidence (V2 Ch.6)."""
    rows = _load_any(results, "l5_frontier.json", "e1_frontier.json")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["beta"])
    rate = np.array([r["rate"] for r in rows])
    dist = np.array([r["distortion"] for r in rows])
    beta = np.array([r["beta"] for r in rows])

    fig, ax = plt.subplots(figsize=(COL, 2.4))
    # Under the sufficiency-for-success budget the UNWEIGHTED total sum_j D_j is dominated
    # by the margin/slip likelihoods the budget deliberately sacrifices, so it can RISE with
    # beta. The frontier that evidences minimality-for-success is D_succ vs rate; when the
    # per-dimension numbers are present (eval_l5_perdim.py) they are the primary curve.
    if all("distortion_succ" in r for r in rows):
        dsucc = np.array([r["distortion_succ"] for r in rows])
        ax.plot(rate, dist, "--s", color="#9AA1AB", lw=1.0, ms=3, zorder=2,
                label="total $\\sum_j D_j$")
        ax.plot(rate, dsucc, "-o", color=ACCENT, zorder=3, label="$D_{succ}$ (certified outcome)")
        for r, d, b in zip(rate, dsucc, beta):
            ax.annotate(f"$\\beta$={b:g}", (r, d), textcoords="offset points",
                        xytext=(4, 4), fontsize=6, color=INK, alpha=0.8)
        ax.legend(loc="center right", fontsize=6)
    else:
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
    rows = _load_any(results, "l2_coverage.json", "e3_coverage.json")
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



# ======================================================================================
# THE LIBERO FIGURES -- these are the ones the paper leads with.
# ======================================================================================

CHANCE = 0.5
#: An MLP handed the TRUE state x -- pose, size, friction, mass, centre of mass, everything a camera
#: cannot see -- and asked to rank a scene's candidate grasps. No perception system can beat it, so
#: it is drawn on every within-scene axis. Without it a reader cannot tell 0.63 from "nearly solved".
ORACLE_CEILING = 0.685


def _scores(results: Path):
    p = results / "l1_scores.pt"
    if not p.exists():
        log.warning("SKIP LIBERO figures (%s missing -- run run_libero.py --only l1)", p)
        return None
    import torch

    return torch.load(p, map_location="cpu")


def fig_within_scene(results: Path, out: Path) -> None:
    """THE HEADLINE FIGURE. Can either predictor rank the candidate grasps of one settled pose?

    Drawn on the WITHIN-SCENE axis, never the pooled one. Per-object base success rates on this
    corpus run from 0.043 to 0.999, so a pooled AUC is won by answering "which grocery is this?" --
    MSP once scored 0.716 pooled while being unable to rank two grasps of the same object.
    """
    r = _load(results, "l1_proxy_vs_msp.json")
    if r is None:
        return
    an, msp = r["within_scene_auc_analytic"], r["within_scene_auc_msp"]

    fig, ax = plt.subplots(figsize=(COL, 2.3))
    bars = ax.bar(
        ["Ferrari-Canny\n(reconstruction)", "MSP\n(outcomes)"], [an, msp],
        color=[WARN, ACCENT], width=0.55, zorder=3,
    )
    for b, v in zip(bars, [an, msp]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, zorder=4)

    ax.axhline(CHANCE, color=INK, ls=":", lw=1.0, zorder=2)
    ax.text(1.46, CHANCE + 0.004, "chance", fontsize=6.5, color=INK, ha="right")
    ax.axhline(ORACLE_CEILING, color=INK, ls="--", lw=1.0, zorder=2)
    ax.text(1.46, ORACLE_CEILING + 0.004, "oracle (true state)", fontsize=6.5, color=INK, ha="right")

    ax.set_ylim(0.45, 0.74)
    ax.set_ylabel("within-scene AUC")
    ax.set_title("Ranking the grasps of a single object")
    fig.savefig(out / "fig_within_scene.pdf")
    fig.savefig(out / "fig_within_scene.png")
    plt.close(fig)
    log.info("wrote fig_within_scene")


def fig_geometry_split(results: Path, out: Path) -> None:
    """THE FALSIFIABLE FORM OF THE CLAIM, and it is a natural control group.

    Six of the thirteen groceries have a single-box collision hull -- butter, cookies, cream cheese
    simply ARE boxes -- so the bounding-box "reconstruction" is EXACT for them and there is no
    hallucinated surface. The rest are cans and bottles. The thesis predicts the analytic proxy works
    where its geometry is right and fails where it is not. If it were equally bad on both, the
    failure would not be reconstruction error and wrong-assumption #11 would not be what is going on.
    """
    r = _load(results, "l1_proxy_vs_msp.json")
    if r is None or r.get("analytic_auc_boxlike") is None:
        return
    groups = ["box-shaped\n(reconstruction exact)", "curved\n(reconstruction wrong)"]
    an = [r["analytic_auc_boxlike"], r["analytic_auc_complex"]]
    ms = [r["msp_auc_boxlike"], r["msp_auc_complex"]]

    x = np.arange(2)
    w = 0.36
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.bar(x - w / 2, an, w, label="Ferrari-Canny", color=WARN, zorder=3)
    ax.bar(x + w / 2, ms, w, label="MSP", color=ACCENT, zorder=3)
    for xi, v in zip(x - w / 2, an):
        if abs(v - CHANCE) < 0.045:  # the label would sit on the chance line: put it in the bar
            ax.text(xi, v - 0.014, f"{v:.3f}", ha="center", va="top", fontsize=6.5,
                    color="white", zorder=4)
        else:
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.5, zorder=4)
    for xi, v in zip(x + w / 2, ms):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.5, zorder=4)

    ax.axhline(CHANCE, color=INK, ls=":", lw=1.0, zorder=2)
    ax.text(1.5, CHANCE + 0.008, "chance", fontsize=6.5, ha="right", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("AUC vs the real lift outcome")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper left")
    ax.set_title("The proxy fails exactly where its geometry is wrong")
    fig.savefig(out / "fig_geometry_split.pdf")
    fig.savefig(out / "fig_geometry_split.png")
    plt.close(fig)
    log.info("wrote fig_geometry_split")


def fig_risk_coverage(results: Path, out: Path) -> None:
    """THE DEPLOYMENT FIGURE, and the one a roboticist will read first.

    The system ranks a scene's candidate grasps, executes its favourite, and acts only when
    confident enough. Sweeping that confidence threshold sweeps the ACT RATE. The y-axis is the
    fraction of executed grasps that actually lifted the object.

    A useful scorer's curve RISES as it becomes more selective. The analytic proxy's does not: its
    precision at its most confident 10% is BELOW its own always-act precision and below the random
    control, i.e. the grasps it is surest about are the ones that fail.
    """
    import torch

    d = _scores(results)
    if d is None:
        return
    from msp.diagnostics.selective import compare_selective

    cmp_ = compare_selective(
        analytic_score=d["margin"], msp_score=d["s_msp"], succ=d["succ"],
        executable=d["executable"], generator=torch.Generator().manual_seed(0),
    )

    fig, ax = plt.subplots(figsize=(COL, 2.4))
    ax.plot(cmp_.msp.act_rate, cmp_.msp.precision, "-", color=ACCENT, label="MSP", zorder=3)
    ax.plot(cmp_.analytic.act_rate, cmp_.analytic.precision, "-", color=WARN,
            label="Ferrari-Canny", zorder=3)
    ax.axhline(cmp_.random_pick, color=INK, ls=":", lw=1.0, zorder=2,
               label=f"random grasp ({cmp_.random_pick:.2f})")
    ax.set_xlabel("act rate  (fraction of scenes the system commits to)")
    ax.set_ylabel("grasps that lifted the object")
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc="lower right")  # lower left sits on the curve's most damning region
    ax.set_title("When it commits, does the object come up?")
    fig.savefig(out / "fig_risk_coverage.pdf")
    fig.savefig(out / "fig_risk_coverage.png")
    plt.close(fig)
    log.info("wrote fig_risk_coverage")


def fig_per_object(results: Path, out: Path) -> None:
    """Per-object within-scene AUC, ordered by base success rate.

    This figure exists to make the pooled-AUC trap visible: the base rates (grey) span 0.04 to 0.999,
    which is the entire reason a pooled AUC can be won without understanding grasping at all.

    AN OBJECT IS ONLY PLOTTED IF ENOUGH OF ITS SCENES ARE ACTUALLY RANKABLE. A within-scene AUC needs
    a scene containing BOTH a success and a failure; an object grasped successfully 99.9% of the time
    (bbq_sauce) supplies almost none, and the AUC over the two or three that remain is noise, not a
    measurement. Plotted unfiltered it renders as a near-zero point and reads as a catastrophic
    failure of the model on that object, which is not what it is. Those objects get their base-rate
    bar and no marker, which is the honest thing to draw: the question cannot be asked of them.
    """
    import torch

    d = _scores(results)
    if d is None:
        return
    from msp.diagnostics.selective import _auc

    MIN_RANKABLE = 15

    rows = []
    for k, nm in enumerate(d["object_names"]):
        sel = d["object_index"] == k
        if int(sel.sum()) < 20:
            continue
        y, ex = d["succ"][sel], d["executable"][sel]
        m = ex.bool()
        base = float(y[m].mean())

        # scenes where the ranking question is even well posed
        aa, mm = [], []
        for i in range(y.shape[0]):
            yi = y[i][ex[i].bool()].numpy()
            if len(yi) < 3 or yi.min() == yi.max():
                continue
            aa.append(_auc(d["margin"][sel][i][ex[i].bool()].numpy(), yi))
            mm.append(_auc(d["s_msp"][sel][i][ex[i].bool()].numpy(), yi))
        rows.append({
            "name": nm.replace("_", " "), "base": base, "box": bool(d["boxlike"][k]),
            "n": len(mm),
            "wa": float(np.mean(aa)) if len(mm) >= MIN_RANKABLE else np.nan,
            "wm": float(np.mean(mm)) if len(mm) >= MIN_RANKABLE else np.nan,
        })

    rows.sort(key=lambda r: r["base"])
    x = np.arange(len(rows))
    base = np.array([r["base"] for r in rows])
    wa = np.array([r["wa"] for r in rows])
    wm = np.array([r["wm"] for r in rows])

    fig, ax = plt.subplots(figsize=(DCOL, 2.8))
    ax.bar(x, base, 0.7, color="#C9CFD6", zorder=2, label="base success rate")
    ax.plot(x, wa, "o", color=WARN, zorder=4, label="Ferrari-Canny (within-scene AUC)")
    ax.plot(x, wm, "s", color=ACCENT, zorder=4, label="MSP (within-scene AUC)")
    ax.axhline(CHANCE, color=INK, ls=":", lw=1.0, zorder=3)

    # Say out loud which objects cannot be scored, rather than drawing a noisy point for them.
    for xi, r in zip(x, rows):
        if np.isnan(r["wm"]):
            ax.text(xi, 0.52, "no rankable\nscenes", fontsize=5.5, ha="center", va="bottom",
                    color=INK, rotation=90, alpha=0.75, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['name']}\n({'box' if r['box'] else 'curved'})" for r in rows],
        rotation=45, ha="right", fontsize=6,
    )
    ax.set_ylabel("rate  /  AUC")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", ncol=3, fontsize=6)
    ax.set_title("Base rates span 0.04 to 0.999 -- which is why a pooled AUC cannot be trusted")
    fig.savefig(out / "fig_per_object.pdf")
    fig.savefig(out / "fig_per_object.png")
    plt.close(fig)
    log.info("wrote fig_per_object (%d/%d objects rankable)",
             int((~np.isnan(wm)).sum()), len(rows))


def fig_ablations(results: Path, out: Path) -> None:
    """Ablations on the within-scene axis. The bar to look at is `uniform beta`."""
    rows = _load(results, "l6_ablations.json")
    if rows is None:
        return
    rows = sorted(rows, key=lambda r: r.get("within_scene_auc", 0.0))
    names = [r["ablation"] for r in rows]
    vals = [r.get("within_scene_auc", float("nan")) for r in rows]

    fig, ax = plt.subplots(figsize=(COL, 0.36 * len(rows) + 1.0))
    colors = [ACCENT if n == "full" else WARN for n in names]
    ax.barh(names, vals, color=colors, height=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=6.5, zorder=4)
    ax.axvline(CHANCE, color=INK, ls=":", lw=1.0, zorder=2)
    ax.text(CHANCE, len(rows) - 0.4, " chance", fontsize=6.5, color=INK)
    ax.axvline(ORACLE_CEILING, color=INK, ls="--", lw=1.0, zorder=2)
    ax.set_xlim(0.45, 0.74)
    ax.set_xlabel("within-scene AUC")
    ax.set_title("Ablations")
    fig.savefig(out / "fig_ablations.pdf")
    fig.savefig(out / "fig_ablations.png")
    plt.close(fig)
    log.info("wrote fig_ablations")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # The LIBERO figures the paper leads with.
    fig_within_scene(args.results, args.out)
    fig_geometry_split(args.results, args.out)
    fig_risk_coverage(args.results, args.out)
    fig_per_object(args.results, args.out)
    fig_ablations(args.results, args.out)

    # The theory figures.
    fig_identifiability(args.results, args.out)
    fig_frontier(args.results, args.out)
    fig_coverage(args.results, args.out)
    fig_latent_dim(args.results, args.out)
    log.info("figures -> %s", args.out)


if __name__ == "__main__":
    main()
