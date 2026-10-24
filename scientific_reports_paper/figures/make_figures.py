"""Regenerate every figure of the Scientific Reports manuscript from results files.

    cd /nas/Selim/selim_sarowar/msp_framework
    source scripts/sr/env.sh && $PY scientific_reports_paper/figures/make_figures.py [--only fig2]

Inputs: data/lm_grasp/{train,val,calib,test}.pt, data/lm (real RGB), results/sr/metrics.json,
results/sr/rate.json, results/sr/plan/*.pt, results/sr/runs/*/scores.pt.
Outputs: fig*.pdf (vector, fonts embedded) and fig*.png (300 dpi) beside this script.

Colour semantics (Okabe-Ito, colourblind safe): teal/bluish green = MSP (ours), orange =
pose-then-plan, sky blue = grasp-aligned patch scorer (GQ-CNN architecture), grey = other.
Grasp outcome colours: bluish green = lifted, vermillion = dropped, grey dashed = not executable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/sr"))

OK = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7",
      "grey": "#8C8C8C"}
C_MSP, C_POSE, C_GQ, C_DIRECT, C_OTHER = OK["green"], OK["orange"], OK["sky"], OK["blue"], OK["grey"]
C_SUCC, C_FAIL = OK["green"], OK["verm"]
NAMES = {1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat", 8: "driller", 9: "duck",
         10: "eggbox", 11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"}
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("wrote", name)


def load_split(name: str) -> dict:
    return torch.load(REPO / "data/lm_grasp" / f"{name}.pt", map_location="cpu",
                      weights_only=False)


def project(K: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]])


def draw_scene(ax, b: dict, i: int, highlight: dict | None = None, half_w: float = 0.035,
               show_all: bool = True) -> None:
    """Real RGB crop at native resolution with candidate jaw axes coloured by outcome."""
    oid, im = int(b["obj_id"][i]), int(b["im_id"][i])
    rgb = np.asarray(Image.open(REPO / f"data/lm/test/{oid:06d}/rgb/{im:06d}.png").convert("RGB"))
    u0, v0, side = (float(x) for x in b["crop"][i])
    pad = int(side)
    img = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), constant_values=255)
    crop = img[int(v0) + pad:int(v0) + pad + int(side), int(u0) + pad:int(u0) + pad + int(side)]
    ax.imshow(crop, extent=(0, side, side, 0))
    K = b["K"][i].numpy().astype(float)
    T = b["T"][i].numpy().astype(float)
    y = b["succ"][i].numpy()
    ex = b["exec"][i].numpy()
    order = np.argsort(y)  # draw successes last
    for j in order:
        if not show_all and (highlight is None or j not in highlight):
            continue
        p, x = T[j, :3, 3], T[j, :3, 0]
        a, c = project(K, p - half_w * x) - [u0, v0], project(K, p + half_w * x) - [u0, v0]
        if not ex[j]:
            ax.plot(*((a + c) / 2), "o", ms=2.2, color="white", mec=OK["grey"], mew=0.6)
            continue
        col = C_SUCC if y[j] > 0.5 else C_FAIL
        ax.plot([a[0], c[0]], [a[1], c[1]], color=col, lw=1.1, solid_capstyle="round")
        ax.plot(*((a + c) / 2), "o", ms=1.6, color=col)
    if highlight:
        for j, (col, mk) in highlight.items():
            p = T[j, :3, 3]
            q = project(K, p) - [u0, v0]
            ax.plot(q[0], q[1], mk, ms=7, mfc="none", mec=col, mew=1.6)
    ax.set_xlim(0, side)
    ax.set_ylim(side, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
        s.set_color("#999999")


# ------------------------------------------------------------------------------------ fig 2
def fig2_corpus() -> None:
    """LM-Grasp: real crops with simulated outcomes, per-object rates."""
    b = load_split("test")
    allb = [load_split(s) for s in ("train", "val", "calib", "test")]
    obj = np.concatenate([x["obj_id"].numpy() for x in allb])
    succ = np.concatenate([x["succ"].numpy() for x in allb])
    exe = np.concatenate([x["exec"].numpy() for x in allb])
    rng = np.random.default_rng(3)
    picks = []
    for oid in (8, 5, 1, 13):  # driller, can, ape, iron: a spread of shapes
        idx = np.nonzero(b["obj_id"].numpy() == oid)[0]
        mixed = [i for i in idx if 3 <= b["succ"][i].sum() <= 10]
        picks.append(int(rng.choice(mixed)))
    fig = plt.figure(figsize=(180 * MM, 82 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.32)
    gl = gs[0].subgridspec(2, 2, wspace=0.05, hspace=0.14)
    for k, i in enumerate(picks):
        ax = fig.add_subplot(gl[k // 2, k % 2])
        draw_scene(ax, b, i)
        n_s, n_e = int(b["succ"][i].sum()), int(b["exec"][i].sum())
        ax.set_title(NAMES[int(b["obj_id"][i])], fontsize=7, pad=2)
        ax.text(0.03, 0.03, f"lifted {n_s}, dropped {n_e - n_s}, blocked {16 - n_e}",
                transform=ax.transAxes, fontsize=5.6, va="bottom",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85})
        if k == 0:
            ax.text(-0.06, 1.12, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
    h = [plt.Line2D([], [], color=C_SUCC, lw=1.2, label="lifted"),
         plt.Line2D([], [], color=C_FAIL, lw=1.2, label="dropped"),
         plt.Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec=OK["grey"],
                    label="blocked (open jaws in contact)")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.06, -0.035), ncol=3,
               frameon=False, title="simulated outcome of each candidate grasp",
               title_fontsize=6.5)

    ax = fig.add_subplot(gs[1])
    ids = sorted(NAMES)
    srate = [succ[obj == o].mean() for o in ids]
    erate = [exe[obj == o].mean() for o in ids]
    yy = np.arange(len(ids))
    ax.barh(yy, erate, color="#D9D9D9", height=0.72, label="executable")
    ax.barh(yy, srate, color=C_SUCC, height=0.72, label="lifted")
    ax.set_yticks(yy, [NAMES[o] for o in ids])
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of candidate grasps (all 15,800 scenes)")
    ax.set_xlim(0, 0.9)
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.99))
    ax.text(-0.36, 1.03, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
    save(fig, "fig2_lmgrasp")


FIGS = {"fig2": fig2_corpus}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    for k, f in FIGS.items():
        if a.only is None or k in a.only:
            f()


if __name__ == "__main__":
    main()
