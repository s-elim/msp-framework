"""Fig. 2 (method): pose-posterior grasp planning, one real test scene through every stage.

    $PY scientific_reports_paper/figures/fig_method.py [--scene 627]

All inset content is real: the scene's RGB-D crop, its FoundationPose estimate, its neighbours in
the residual bank, its 16 posterior samples, its per-pose simulated success matrix and the
selected grasp.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from common import (
    CMAP,
    GREY,
    GREY_LIGHT,
    MM,
    ORANGE,
    REPO,
    TEAL,
    TEAL_DARK,
    Scene,
    choices,
    plan,
    posterior,
    save,
)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCENE = 627
STAGES = (  # (title, face colour, edge colour)
    ("Observation", "#F2F2F2", "#BDBDBD"),
    ("Pose estimate", "#FDF1DC", ORANGE),
    ("Pose posterior", "#E3F4EE", TEAL),
    ("Posterior planning", "#D3EEE5", TEAL_DARK),
    ("Decision and execution", "#E8F1F8", "#0072B2"),
)
X = (0.005, 0.135, 0.285, 0.535, 0.79)  # panel left edges (figure fraction)
W = (0.122, 0.142, 0.242, 0.247, 0.205)
Y0, H = 0.04, 0.80


def _panel(bg: plt.Axes, k: int) -> None:
    title, fc, ec = STAGES[k]
    bg.add_patch(FancyBboxPatch((X[k], Y0), W[k], H, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.9, transform=bg.transAxes))
    bg.text(X[k] + 0.014, Y0 + H - 0.04, str(k + 1), color="white", fontsize=6.2,
            fontweight="bold", ha="center", va="center", transform=bg.transAxes,
            bbox=dict(boxstyle="circle,pad=0.18", fc=ec, ec="none"))
    bg.text(X[k] + 0.03, Y0 + H - 0.04, title, fontsize=6.8, fontweight="bold", va="center",
            transform=bg.transAxes)


def _arrow(bg: plt.Axes, x0: float, x1: float, y: float = Y0 + H / 2, col: str = "#4D4D4D",
           rad: float = 0.0) -> None:
    bg.add_patch(FancyArrowPatch((x0, y), (x1, y), transform=bg.transAxes, arrowstyle="-|>",
                                 mutation_scale=7, lw=1.0, color=col,
                                 connectionstyle=f"arc3,rad={rad}"))


def _eq(bg: plt.Axes, x: float, y: float, s: str) -> None:
    bg.text(x, y, s, fontsize=6.2, ha="center", va="center", transform=bg.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GREY_LIGHT, lw=0.5))


def _inset(fig: plt.Figure, x: float, y: float, w: float, h: float) -> plt.Axes:
    a = fig.add_axes([x, y, w, h])
    a.set_xticks([])
    a.set_yticks([])
    return a


def draw(scene: int = SCENE, name: str = "fig2_method") -> None:
    from msp.pose import PosteriorSampler, fit_bank, symmetry_transforms

    sc = Scene(scene, pad=1.05)
    i = sc.i
    fp = plan("point")
    R_fp, t_fp = fp["R"][i].double().numpy(), fp["t"][i].double().numpy()
    post = posterior("test")
    Rs, ts = post["R"][i].astype(float), post["t"][i].astype(float)
    s_pose = plan("posterior")["s_pose"][i].float().numpy()  # (M, Na)
    s_mean = plan("posterior")["s"][i].numpy()
    j = int(choices("posterior")[i])

    fig = plt.figure(figsize=(180 * MM, 62 * MM))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.axis("off")
    for k in range(5):
        _panel(bg, k)
    for k in range(4):
        _arrow(bg, X[k] + W[k] + 0.001, X[k + 1] - 0.001)
    # second view: from the decision back to the observation, over the top
    bg.add_patch(FancyArrowPatch((X[4] + 0.10, Y0 + H + 0.005), (X[0] + 0.06, Y0 + H + 0.005),
                                 transform=bg.transAxes, arrowstyle="-|>", mutation_scale=7,
                                 lw=0.9, color=TEAL_DARK, ls=(0, (3, 2)),
                                 connectionstyle="arc3,rad=0.12"))
    bg.text(0.5, 0.955, "low confidence: second view", color=TEAL_DARK, fontsize=6.2,
            ha="center", va="center", transform=bg.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    # 1 observation: RGB and depth crops
    a = _inset(fig, X[0] + 0.008, 0.43, W[0] - 0.016, 0.30)
    a.imshow(sc.rgb, interpolation="bilinear")
    a.contour(sc.mask, levels=[0.5], colors=["white"], linewidths=0.5)
    u0, u1, v1, v0 = sc.box
    a.add_patch(plt.Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, ec=TEAL, lw=0.8))
    a = _inset(fig, X[0] + 0.008, 0.09, W[0] - 0.016, 0.30)
    d = np.where(sc.depth > 0, sc.depth, np.nan)
    a.imshow(d, cmap="Greys_r", vmin=np.nanpercentile(d, 2), vmax=np.nanpercentile(d, 98))

    # 2 FoundationPose estimate (frozen network)
    a = _inset(fig, X[1] + 0.012, 0.20, W[1] - 0.024, 0.46)
    sc.show(a)
    sc.draw_box(a, R_fp, t_fp, ORANGE, lw=1.2)
    bg.text(X[1] + W[1] / 2, 0.13, "❄  FoundationPose", fontsize=6.2, ha="center",
            color="#8A5A00", transform=bg.transAxes)
    _eq(bg, X[1] + W[1] / 2, 0.075, r"$(\hat R, \hat t)$")

    # 3 residual bank neighbours -> posterior samples
    z = dict(np.load(REPO / "results/sr/posterior/features_fpose.npz"))
    info = __import__("json").loads((REPO / "data/lm/models/models_info.json").read_text())
    syms = {int(o): symmetry_transforms(info, int(o)) for o in np.unique(z["train_obj_id"])}
    bank = fit_bank(z["train_obj_id"], z["train_im_id"], z["train_feats"], z["train_R_hat"],
                    z["train_t_hat"], z["train_R_gt"], z["train_t_gt"], syms)
    smp = PosteriorSampler(bank, "knn", k=64)
    feat = z["test_feats"][i]
    nb = smp.neighbours(sc.obj, feat, sc.im)
    rows = np.flatnonzero(bank.obj_id == sc.obj)
    F = (bank.feats[rows] - smp._loc[sc.obj]) / smp._scale[sc.obj]
    F0 = F - F.mean(0)
    _, _, Vt = np.linalg.svd(F0, full_matrices=False)
    P2 = F0 @ Vt[:2].T
    q = ((feat - smp._loc[sc.obj]) / smp._scale[sc.obj] - F.mean(0)) @ Vt[:2].T
    a = _inset(fig, X[2] + 0.012, 0.20, 0.095, 0.46)
    isn = np.isin(rows, nb)
    a.scatter(*P2[~isn].T, s=1.2, c=GREY_LIGHT, linewidths=0)
    a.scatter(*P2[isn].T, s=2.2, c=TEAL, linewidths=0)
    a.scatter(*q, s=40, marker="*", c=ORANGE, edgecolors="white", linewidths=0.5, zorder=3)
    lo, hi = np.percentile(P2, [2, 98], axis=0)
    pad = 0.15 * (hi - lo)
    a.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    a.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    for s_ in a.spines.values():
        s_.set_color(GREY_LIGHT)
    _arrow(bg, X[2] + 0.112, X[2] + 0.126, y=0.43, col=TEAL_DARK)
    a = _inset(fig, X[2] + 0.128, 0.20, 0.104, 0.46)
    sc.show(a)
    sc.draw_heat(a, sc.occupancy(Rs, ts))
    for R, t in zip(Rs, ts, strict=True):
        sc.draw_box(a, R, t, TEAL, lw=0.45, alpha=0.5)
    _eq(bg, X[2] + W[2] / 2, 0.10,
        r"$T_m=(\hat R\,\delta R_m,\ \hat t+\delta t_m),\ \ (\delta R_m,\delta t_m)\sim\mathcal{N}_k$")

    # 4 per-pose simulated success and the posterior mean
    a = _inset(fig, X[3] + 0.03, 0.33, W[3] - 0.045, 0.33)
    order = np.argsort(-s_mean, kind="stable")
    a.imshow(s_pose[:, order], cmap=CMAP, vmin=0, vmax=1, aspect="auto",
             interpolation="nearest")
    a.set_ylabel("pose $m$", fontsize=5.8, labelpad=1)
    jj = int(np.flatnonzero(order == j)[0])
    a.add_patch(plt.Rectangle((jj - 0.5, -0.5), 1, s_pose.shape[0], fill=False, ec=TEAL,
                              lw=1.4))
    b = _inset(fig, X[3] + 0.03, 0.19, W[3] - 0.045, 0.12)
    cols = [TEAL if k == jj else GREY for k in range(len(order))]
    b.bar(np.arange(len(order)), s_mean[order], color=cols, width=0.8)
    b.set_xlim(-0.5, len(order) - 0.5)
    b.set_ylim(0, 1.05)
    b.set_xlabel("candidate grasp", fontsize=5.8, labelpad=1)
    for s_ in ("top", "right"):
        b.spines[s_].set_visible(False)
    _eq(bg, X[3] + W[3] / 2, 0.09,
        r"$\bar s(a)=\frac{1}{MP}\sum_{m,p} y(a;\,T_m,\theta_p)$")

    # 5 decision on the image, then frames to the robot
    a = _inset(fig, X[4] + 0.012, 0.24, 0.10, 0.42)
    sc.show(a, dim=0.25)
    sc.draw_candidates(a, s_mean, lw=1.0)
    sc.mark_choice(a, j, TEAL, size=55)
    tags = (("accept", TEAL), ("collision", GREY), ("low $\\bar s$", GREY), ("certify", TEAL_DARK))
    for k, (t_, c_) in enumerate(tags):
        bg.text(X[4] + 0.165, 0.60 - 0.085 * k, t_, fontsize=5.8, ha="center", va="center",
                color="white" if c_ != GREY else "black", transform=bg.transAxes,
                bbox=dict(boxstyle="round,pad=0.22", fc=c_ if c_ != GREY else GREY_LIGHT,
                          ec="none"))
    _eq(bg, X[4] + W[4] / 2, 0.10, r"$T_{\mathrm{base}}^{\mathrm{flange}}=T_{\mathrm{base}}^{\mathrm{cam}}\,T_{\mathrm{cam}}^{\mathrm{tcp}}\,T_{\mathrm{tcp}}^{\mathrm{flange}}$")
    save(fig, name, jpeg=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=SCENE)
    a = ap.parse_args()
    draw(a.scene)
